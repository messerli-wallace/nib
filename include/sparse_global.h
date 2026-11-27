#pragma once

#include "inkling_engine.h"

#include <cstddef>

namespace inkling {

// ============================================================================
// Quest-Style Query-Aware Block-Sparse Global Attention (Step 7C)
// ============================================================================
//
// Global layers (8 KV slots) attend over the full 256k context.  Tokens are
// grouped into fixed pages of SPARSE_PAGE_SIZE = 64 consecutive positions
// (which matches the WMMA prefill key tile BC = 64 exactly).
//
// Per (KV head, page) the KV store keeps a *conservative* per-dimension
// elementwise min / max of the stored (FWHT-rotated, clipped, dequantized) K
// vector, quantized to int8 with ONE shared FP16 scale:
//
//   k_min_q[d] = clamp(floor(min_d / s) - 1, -127, 127)  (rounds toward -inf)
//   k_max_q[d] = clamp(ceil (max_d / s) + 1, -127, 127)  (rounds toward +inf)
//   k_mm_d     = __float2half_ru(s),  s = max|k| / 127 * (1 + 2^-10), s >= 2^-24
//
// The one-LSB widening absorbs the f32 rounding of the q_i8 * d products
// computed by the scoring kernel, and __float2half_ru + the 2^-24 floor keep
// the scale from ever underflowing or rounding down.  Hence for every d and
// every stored key value k_d in [min_d, max_d] we have k_min_q[d]*d <= k_d <=
// k_max_q[d]*d regardless of the sign of q_d, hence
//
//   bound(Q, page) = sum_d max(Q_d * k_min_q[d] * s, Q_d * k_max_q[d] * s)
//                  >= max_{j in page} (Q . K_j)                       (exact)
//
// which is a mathematically strict upper bound on the largest logit inside the
// page.  RelBias only touches distances < N_REL_EXTENT = 1024 (16 pages), and
// every query always keeps the last SPARSE_RECENT_PAGES*64 = 1536 tokens as
// mandatory pages, so historical pages contribute Q.K only and the bound
// covers the *entire* attention logit of every page that may be dropped.
//
// Metadata cost: N_HEAD_KV * max_pages * (2*128 + 2) B
//   @ 256k (max_pages = 4096): 8 * 4096 * 258 B = 8,454,144 B = 8.06 MiB/slot
//   across the 8 Global slots: 67.6 MB (64.5 MiB).
//
// Scratch cost (shared by all layers):
//   page_scores      8 * 8 * 4096 * 4 B          = 1,048,576 B   ([qb][N_HEAD_KV][max_pages])
//   q_min / q_max    8 * 8 * 3 * 128 * 4 B each  =  98,304 B each
//   selected_pages   8 * 8 * 128 * 4 B           =  32,768 B
//   n_selected       8 * 8 * 4 B                 =     256 B
//                                          total = 1,278,208 B (1.22 MiB)
// ============================================================================

constexpr int SPARSE_PAGE_SIZE        = 64;
constexpr int SPARSE_MAX_PAGES        = 2097152 / SPARSE_PAGE_SIZE;        // 32768 pages: the table is an index, not a capability limit; VRAM bounds the pool
constexpr int SPARSE_MAX_SELECTED     = 256;                              // per (q_block, kv_head); allows budgets up to 254
constexpr int SPARSE_SINK_PAGES       = 2;                                // pages [0, 1]
constexpr int SPARSE_RECENT_PAGES     = 24;                               // 1536 tokens >= 1024 RelBias window
constexpr int SPARSE_PREFILL_LOCAL_TOKENS = 1536;
constexpr int SPARSE_MIN_PAGES        = SPARSE_SINK_PAGES + SPARSE_RECENT_PAGES; // 26
constexpr int SPARSE_MAX_QB           = (MAX_UBATCH + SPARSE_PAGE_SIZE - 1) / SPARSE_PAGE_SIZE; // 8
constexpr int SPARSE_DECODE_SCORE_CHUNKS = 24;                            // 24*8 = 192 CTAs
constexpr int SPARSE_PREFILL_SCORE_CHUNKS = 8;
constexpr int SPARSE_DECODE_SPLITS    = 16;                               // 16*8 = 128 CTAs
constexpr int SPARSE_DEFAULT_PAGES    = 64;
constexpr int SPARSE_TOPK_THREADS     = 256;
constexpr int SPARSE_TOPK_SLOTS       = (SPARSE_MAX_PAGES + SPARSE_TOPK_THREADS - 1) / SPARSE_TOPK_THREADS;

// Step 8G: keep prefill exact-dense up to 64K context so multi-turn agentic chats
// never bake sparse-prefill selection errors into downstream KV vectors, and use a
// wider sparse prefill budget (>= 224 pages = 14,336 tokens) beyond 64K.
constexpr int SPARSE_DENSE_PREFILL_LIMIT = 65536;
constexpr int SPARSE_PREFILL_MIN_PAGES   = 224;

// Damped Quest radius weight (Step 8F + Step 8G):
//   Metadata stores lo = k_mean - k_rad and hi = k_mean + k_rad (true page centroid
//   k_mean with half-range k_rad = 0.5*(k_max - k_min)), so every scoring kernel
//   automatically evaluates:
//     score(q, page) = <q, k_mean> + alpha * <|q|, k_rad>
//                    = <0.5*(q + alpha*|q|), hi> + <0.5*(q - alpha*|q|), lo>.
// Replacing the (k_min+k_max)/2 midpoint with true k_mean eliminates single-outlier
// centroid skew while alpha = 0.25 preserves single-token needle spikes.
constexpr float SPARSE_QUEST_ALPHA    = 0.25f;

// Metadata formats (Step 7K).  `SPARSE_META_Q8` is the Step 7C int8 min/max
// layout (258 B/page/head).  `SPARSE_META_Q4` (Q4_MM) packs the outward-rounded
// min and max of one dimension into one byte (low nibble = q_min + 8, high
// nibble = q_max + 8, both in 1..15) next to ONE shared FP16 scale, i.e.
// 130 B/page/head -- exactly half the DRAM traffic and 32 MiB less VRAM at 256k.
// The bound stays strictly conservative: with d = __float2half_ru(amax/7*(1+2^-9))
// (amax = max_d max(|min_d|,|max_d|), so |k_d| <= 7*d for every dimension),
// q_min = clamp(floor(k_min/d), -7, 7) and q_max = clamp(ceil(k_max/d), -7, 7)
// give  q_min*d <= k_min <= k_max <= q_max*d  for every dimension.
constexpr int SPARSE_META_Q8 = 0;
constexpr int SPARSE_META_Q4 = 1;

// Per-Global-slot page metadata (SoA, 16-byte aligned rows: 128 int8 per page).
struct SparseGlobalPageMeta {
    int8_t * k_min_q  = nullptr;  // Q8 only: [N_HEAD_KV, max_pages, HEAD_DIM]
    int8_t * k_max_q  = nullptr;  // Q8 only: [N_HEAD_KV, max_pages, HEAD_DIM]
    uint8_t * k_qmm   = nullptr;  // Q4 only: [N_HEAD_KV, max_pages, HEAD_DIM] packed
    __half * k_mm_d   = nullptr;  // [N_HEAD_KV, max_pages]
    int32_t  max_pages = 0;
    int32_t  fmt       = SPARSE_META_Q8;
};

// Layer-independent scratch shared by every Global slot (streams are serialized).
struct SparseGlobalScratch {
    float   * page_scores    = nullptr; // [SPARSE_MAX_QB * N_HEAD_KV * SPARSE_MAX_PAGES]
    float   * q_min          = nullptr; // [SPARSE_MAX_QB * N_HEAD_KV * GQA_RATIO * HEAD_DIM]
    float   * q_max          = nullptr; // [SPARSE_MAX_QB * N_HEAD_KV * GQA_RATIO * HEAD_DIM]
    int32_t * selected_pages = nullptr; // [SPARSE_MAX_QB * N_HEAD_KV * SPARSE_MAX_SELECTED]
    int32_t * n_selected     = nullptr; // [SPARSE_MAX_QB * N_HEAD_KV]
};

inline size_t sparse_global_page_meta_bytes(int max_pages, int fmt = SPARSE_META_Q8) {
    const size_t dims = (fmt == SPARSE_META_Q4) ? (size_t)HEAD_DIM
                                                : (size_t)2 * HEAD_DIM;
    return (size_t)N_HEAD_KV * (size_t)max_pages * (dims + sizeof(__half));
}

inline size_t sparse_global_scratch_bytes() {
    return (size_t)SPARSE_MAX_QB * N_HEAD_KV * SPARSE_MAX_PAGES * sizeof(float)
         + 2 * (size_t)SPARSE_MAX_QB * N_HEAD_KV * GQA_RATIO * HEAD_DIM * sizeof(float)
         + (size_t)SPARSE_MAX_QB * N_HEAD_KV * SPARSE_MAX_SELECTED * sizeof(int32_t)
         + (size_t)SPARSE_MAX_QB * N_HEAD_KV * sizeof(int32_t);
}

// Both allocators zero-fill on `stream` (stream-ordered): hipMalloc'd VRAM is
// not guaranteed to be clean (issue #1).
bool sparse_global_page_meta_alloc(SparseGlobalPageMeta & m, int max_pages,
                                   int fmt, hipStream_t stream);
void sparse_global_page_meta_free(SparseGlobalPageMeta & m);
bool sparse_global_scratch_alloc(SparseGlobalScratch & s, hipStream_t stream);
void sparse_global_scratch_free(SparseGlobalScratch & s);

// First mandatory local page for a prefill query block starting at q_pos_start:
// pages [max(2, floor((q_pos_start - 1536) / 64)) .. q_pos_end / 64] are always
// selected, which covers the full 1024-wide RelBias window of every query.
inline __host__ __device__ int sparse_local_start_page(int q_pos_start) {
    const int v = q_pos_start - SPARSE_PREFILL_LOCAL_TOKENS;
    const int p = (v >= 0) ? (v / SPARSE_PAGE_SIZE)
                           : -((-v + SPARSE_PAGE_SIZE - 1) / SPARSE_PAGE_SIZE);
    return (p > SPARSE_SINK_PAGES) ? p : SPARSE_SINK_PAGES;
}

// ============================================================================
// Page metadata maintenance (call AFTER the KV store wrote the new tokens)
// ============================================================================

// Recompute every page touched by [pos_start, pos_start + n_tokens) from the
// already-stored K rows (partial first / last pages are re-derived exactly).
void launch_kv_update_page_minmax_prefill(
        const KVCacheSlot & kv_slot,
        const SparseGlobalPageMeta & meta,
        int pos_start, int n_tokens,
        hipStream_t stream);

// Recompute the single active page (d_meta->pos / 64) for all 8 KV heads.
// 100% static HIP-graph safe: every input comes from d_meta on the device.
void launch_kv_update_page_minmax_decode(
        const KVCacheSlot & kv_slot,
        const SparseGlobalPageMeta & meta,
        const StepMeta * d_meta,
        hipStream_t stream);

// ============================================================================
// Page selection
// ============================================================================

// Decode: q is the *stored* FWHT-rotated Q [N_HEAD, HEAD_DIM] (buf_q_prep_).
//   Stage 1 (sparse_score_pages_decode_kernel): page_scores[h][p]
//   Stage 2 (sparse_topk_pages_decode_kernel): selected_pages[0][h][*], n_selected[0][h]
// Output layout: [sinks 0,1][top-K historical][last 24 pages], ascending within
// each group.  sparse_pages is clamped to [26, 126] and MUST be constant for a
// captured decode graph.
void launch_sparse_global_select_pages_decode(
        const float * q,
        const SparseGlobalPageMeta & meta,
        SparseGlobalScratch & scratch,
        const StepMeta * d_meta,
        int sparse_pages,
        hipStream_t stream);

// Prefill: per (q_block, kv_head) selection with the 4-product interval bound.
//   q_min / q_max are derived internally from q (stored, rotated).
// Output layout: selected_pages[qb][h][*], n_selected[qb][h].
void launch_sparse_global_select_pages_prefill(
        const float * q,
        const SparseGlobalPageMeta & meta,
        SparseGlobalScratch & scratch,
        int n_tokens, int pos_start,
        int sparse_pages,
        hipStream_t stream);

// ============================================================================
// Sparse FlashAttention
// ============================================================================

// Decode (n_tokens == 1) sparse Global FA over the selected pages.
//   rel: either rel_table ([1][N_HEAD][N_REL_EXTENT]) or the rel_table-free
//        path r_prep ([1][N_HEAD][N_REL], clipped+tau) + attn_rel_bias
//        ([N_REL_EXTENT][N_REL]); pass rel_table == nullptr to select it.
//        sinks_in / qgate_in keep the exact dense strides (n_tokens == 1).
//   fa_done (>= N_HEAD_KV zeroed int32s): in-kernel self-resetting atomic
//        combine; otherwise tmp_o/tmp_meta + inkling_fa_decode_sparse_reduce.
//   n_splits is clamped to [1, 32]; grid = (n_splits, N_HEAD_KV).
void launch_fa_sparse_global_decode(
        const float * q,
        KVCacheSlot kv_slot,
        const float * sinks_in, int sinks_stride,
        const float * rel_table,
        const float * r_prep,
        const float * attn_rel_bias,
        const float * qgate_in, int qgate_stride,
        float * out,                                   // [N_HEAD, HEAD_DIM]
        const int32_t * selected_pages,                // [N_HEAD_KV, SPARSE_MAX_SELECTED]
        const int32_t * n_selected_pages,              // [N_HEAD_KV]
        float * tmp_o, float2 * tmp_meta,              // [N_HEAD * n_splits] each
        int n_splits,
        const StepMeta * d_meta,
        hipStream_t stream,
        int32_t * fa_done = nullptr);

// Prefill sparse Global FA (BM = 64 q rows, BC = 64 = exactly one page):
//   grid = (ceil(n_tokens / 64), N_HEAD_KV), 12 warps, deepest q_block first.
//   rel: either rel_table ([n_tokens][N_HEAD][N_REL_EXTENT]) or the
//        rel_table-free path r_prep ([n_tokens][N_HEAD][N_REL]) +
//        attn_rel_bias ([N_REL_EXTENT][N_REL]).
void launch_fa_sparse_global_prefill(
        const float * q,
        KVCacheSlot kv_slot,
        const float * sinks_in, int sinks_stride,
        const float * rel_table,
        const float * r_prep,
        const float * attn_rel_bias,
        const float * qgate_in, int qgate_stride,
        float * out,                                   // [n_tokens, N_HEAD, HEAD_DIM]
        const int32_t * selected_pages,                // [qb][N_HEAD_KV][SPARSE_MAX_SELECTED]
        const int32_t * n_selected_pages,              // [qb][N_HEAD_KV]
        int n_tokens, int pos_start, int n_kv,
        hipStream_t stream);

} // namespace inkling
