# nib on RDNA4 — gfx1201 (Radeon AI PRO R9700) port

Status: **builds and is believed correct; not yet runtime-validated on silicon.**
This directory documents the `messerli-wallace/rdna4` branch: what was changed to
make the engine compile and produce *correct* math on gfx1200/gfx1201, what is
verified statically, what is still slow, and how to run the first real test.

Target: **AMD Radeon AI PRO R9700** — `gfx1201` / Navi 48, **64 CU** (32 WGP),
32 GB GDDR6 @ 640 GB/s, 64 MB Infinity Cache. ROCm 7.2.0 in WSL2 on Windows 11
(Adrenalin 26.2.2, `/dev/dxg`, ROCDXG), Ubuntu 24.04.

## Why it did not build, and what was done about it

| Blocker (see `../nib-gfx1201-assessment.md`) | Fix in this branch |
|---|---|
| 17 `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32` sites (RDNA3-only, `wmma-256b-insts`) | `include/nib_mma.h` supplies a layout-independent 16x16x16 tile (`v_dot2_f32_f16`, FP32 accumulate). gfx11 keeps the native builtin, untouched. |
| Prefill FA compiled to `(void)q;` on non-gfx11 (silent garbage) | `kernels_fa_relbias.hip` and `kernels_sparse_global.hip` now compute K·Qᵀ and P·Vᵀ for real on gfx12; P is rebuilt in ascending-key order from the two half-waves instead of the RDNA3-packed P fragment. |
| `kernels_core.hip` ctrl_down / ctrl_apply / router prefill had empty gfx12 bodies | All three are real kernels on gfx12 now. |
| 96-block cooperative kernels deadlock on 64 CU | CU-aware co-residency gates: `decode_fused_prime_occupancy()` / `s7i_a_v3_coresident()`. Where a grid cannot be fully resident, the launchers fall back to the multi-launch composition (same math, no cross-block barrier). gfx11 short-circuits to the old unconditional fused launch. |
| gfx11-only cache-flush asm (`s_waitcnt_vscnt`, `buffer_gl0_inv/gl1_inv`) | Verified absent on gfx12 (they do not assemble) and that the scoped fences the kernels already use emit the RDNA4 equivalents: `fence(RELEASE, AGENT)` → `s_waitcnt` + `global_wb scope:SCOPE_DEV`, acquire-scope atomics → `global_inv scope:SCOPE_DEV`. No hand-written cache asm is needed; documented at all 13 helper copies. |
| gfx12 has no bare `s_barrier` | `nib_lds_fence_barrier()` in `include/nib_target.h` (identical asm on gfx11, `s_waitcnt 0` + `__builtin_amdgcn_s_barrier()` → `s_barrier_signal/wait` on gfx12). |
| Split-K / decode-split heuristics claimed 96 CU | Grids are 64 CTAs × split, so 384/192/256 are exact multiples of 64 too; only the comments were wrong. Re-labelled. |

## The tile primitive (`include/nib_mma.h`)

The native RDNA3 wave32 16x16x16 f16→f32 fragment layout (documented and unit
tested in `kernels_wmma_prefill.hip`) is

```
A (M x K): lane l holds row (l % 16) as 16 contiguous K halves; lanes 16..31 replicate
B (K x N): lane l holds row (l % 16) of the row-major [n][k] tile, 16 contiguous K halves
D        : lane l, element i -> row (2*i + l/16), column (l % 16)
```

so one tile needs **no cross-lane traffic**: each lane reads the 8 A rows it owns
as D rows (`m = 2*i + l/16`) from the *same shared-memory tile* the RDNA3 path
stages, and reuses the B row it already holds. `dot16()` is 8 `v_dot2_f32_f16`
(FP16 inputs, FP32 accumulation — the tensor-core operand semantics), so the
numerics differ from RDNA3 WMMA only by ~1e-5 relative, which the upstream header
already documents for the native path.

Because it consumes rows rather than fragments, no LDS staging, XOR swizzle or
lane→element mapping was re-derived, and every epilogue (online softmax, the
in-register FWHT-128, the coalesced stores) is unchanged.

## Verified statically (no GPU run)

* `cmake -DGPU_TARGET=gfx1201` builds clean (ROCm 7.2.0, clang 22), and so does
  `-DGPU_TARGET=gfx1100`.
* **gfx1100 is unchanged**: all 18 kernel objects of the gfx1100 build are
  instruction-for-instruction identical to upstream `93507de` (v0.1.2, built with
  the same toolchain), compared object by object with `llvm-objdump -d`.
* Every ported kernel has a real body on gfx1201 with the expected amount of
  `v_dot2_f32_f16` and zero `v_wmma`:
  FA prefill 3072, ctrl_apply 1024, ctrl_down 128, router 256, dense GEMM 1024,
  MoE gate/up 4096, Quest scorer 4096, vision FA 2048, patch-embed 64.
* `rocminfo` reports `gfx1201`, 64 compute units, `AMD Radeon AI PRO R9700`.
* gfx12 WMMA exists and assembles: `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12`
  emits `v_wmma_f32_16x16x16_f16 v[0:7], v[8:11], v[8:11], 0` (8-half fragments).
  It is now wired up as an opt-in fast path (`NIB_GFX12_WMMA=1`, next section);
  the v_dot2 emulation stays the default until `nib wmma-selftest` passes on
  silicon.

## The gfx12 WMMA fast path (`include/nib_mma.h`, opt-in)

The emulation above is correct but slow (64 `v_dot2_f32_f16` + 8 `ds_load_b128` per
16x16x16 tile per lane). `include/nib_mma.h` now also implements the same tile on
the RDNA4 tensor core and dispatches to it at run time:

```
NIB_GFX12_WMMA=1 ./build/nib <mode>      # tensor core
./build/nib <mode>                       # default: validated v_dot2 emulation
```

### The instruction and its fragment layout

`nib_f16x8 __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(nib_f16x8 a,
nib_f16x8 b, nib_f32x8 c)` → `v_wmma_f32_16x16x16_f16`; A/B are 8 halves (4 VGPRs)
per lane, no RDNA3-style lane duplication, C/D are 8 f32. With `laneCol = lane % 16`
and `laneGrp = lane / 16`:

```
A: a[i] = A[m = laneCol][k = 8*laneGrp + i]      (K-major, 8 contiguous K per lane)
B: b[i] = B[k = 8*laneGrp + i][n = laneCol]      (K-major, 8 contiguous K per lane)
D: d[i] = D[m = 8*laneGrp + i][n = laneCol]
```

Sources (all agree): the builtin declaration in
`/opt/rocm/include/ck/utility/amd_wmma.hpp` + `ck_tile/ops/gemm/warp/` (ROCm 7.2.0),
AMD GPUOpen *Using the Matrix Cores of AMD RDNA 4 architecture GPUs* (its sample
kernel loads `a_frag`/`b_frag` and stores `c` with exactly those index formulas,
verified against hipBLAS), the CK `wmma_gemm.hpp` wave32 lane diagram, and an
independent R9700 measurement of the D map.

Only D differs from the nib/RDNA3 layout, and it differs by a **row permutation
inside the tile**, never by cross-lane traffic: hardware keeps D row `8*laneGrp + i`
in `acc[i]`, nib wants row `2*i + laneGrp`. Feeding the A fragment of hw row `r`
from nib row

```
perm(r) = 2*(r % 8) + r / 8     ->  0,2,4,6,8,10,12,14, 1,3,5,7,9,11,13,15
```

makes the hardware compute `D'[r] = D[perm(r)]`, i.e. `acc[i] = D'[8*laneGrp + i] =
D[2*i + laneGrp]` — nib's accumulator layout exactly, so **no shuffles and no
epilogue changes**. B needs no permutation either: a lane group just takes the low
or the high 8 halves of the 16-half B fragment it already holds.

Cost per tile per lane: `1 ds_load_b128 + 1 v_wmma` (+2 address ops) instead of
`8 ds_load_b128 + 64 v_dot2 + 8 v_add_f32`.

### Fragment loads: 8 halves, not 16

The instruction consumes 8 halves per lane, but the shared (RDNA3-compatible) staging
path produces 16-half fragments. Loading all 16 and then halving them with `pick8()`
reads 32 B of LDS to use 16 B. Each lane's half is exactly one 16 B block of the
XOR-swizzled row - block `(2*kc + laneGrp) ^ (row & 7)` - so it can be fetched alone:

```cpp
nib_f16x8 load8_xor(const _Float16 * row, int row_low3, int kc, int lane) {
    return d4[(2 * kc + (lane >> 4)) ^ row_low3];   // one ds_load_b128
}
```

The values are identical (greedy output unchanged) and B-fragment LDS traffic halves.
On the grouped MoE and dense tiles together this measured **+14.3% prefill at 32K**
(3,826 -> 4,373 tok/s pp2560) and +17.8% at depth 0, matching the bandwidth arithmetic:
fragment reads had been consuming ~87% of the MMA issue time.

### Where it is wired in

`nib_mma::mma_stride`, `mma_xor64` and `mma_rows` keep their signatures and their nib
layout semantics; each is now `_emu` (unchanged emulation) + `_hw` (tensor core, or
the native gfx11 builtin on RDNA3) + a dispatcher. All 16 tile call sites in
`kernels_core`, `kernels_vision`, `kernels_fa_relbias`, `kernels_sparse_global` and
`kernels_wmma_prefill` therefore get the fast path with no source change;
`kernels_vision`'s patch-embed `dot16` (a single row·col, not a tile) stays on the
emulation.

The gate is one `__device__ int` flag **per translation unit** (nib links one code
object per `.hip` file, no `-fgpu-rdc`, so a cross-TU device symbol cannot resolve).
Each TU registers a setter callback from a static initializer, and
`nib_mma::apply_gate_appliers()` (called from `InklingEngine::init` and from
`nib wmma-selftest`) launches that TU's tiny setter kernel once the runtime is up.
The setter kernel's store is load-bearing: a `static __device__` flag with no
device-side writer is constant-folded to 0 at -O3 and the whole path disappears
(measured). With it, the read compiles to one `s_load_b32`, hoisted into a VGPR and
tested with one `v_cmp`+branch per tile. With `NIB_GFX12_WMMA` unset nothing is
registered and no GPU call is made.

### Validate it: `nib wmma-selftest`

```bash
NIB_GFX12_WMMA=1 ./build/nib wmma-selftest
```

No model needed. Fills 16 random 16x16x16 tiles whose values are *exact* in FP16
(small integers, so FP32 accumulation is exact too), computes each tile with the
emulation and with the tensor core from the same LDS staging and the same nib
fragments (`mma_stride` on a plain tile and `mma_xor64` on the XOR-swizzled layout,
4 K chunks), and compares them in the nib D layout. Pass looks like

```
[WMMA] tile selftest: gfx12 (RDNA4)  emulated tile=v_dot2_f32_f16  hardware tile=v_wmma_f32_16x16x16_f16 (gfx12 builtin)  tiles=16
[WMMA] NIB_GFX12_WMMA gate: 1 (kernels use the tensor-core tile)
[WMMA] mma_stride (plain 16-half stride)             max|diff|=0  cos=1.000000000  mismatches=0     -> PASS
[WMMA] mma_xor64  (XOR-swizzled 64-half, 4 K chunks) max|diff|=0  cos=1.000000000  mismatches=0     -> PASS
[WMMA] PASS (hardware tile == emulated tile)
```

exit code 0. On failure it also prints the cosine against the *transposed* hardware
tile and a 4x4 corner of both results, which distinguishes "wrong row permutation"
from "the D map is transposed" — the only thing that cannot be settled statically.

Then the end-to-end gate is unchanged: `./build/nib selftest -m <gguf>` with the env
var set must still print `cos_sim > 0.999`, and PPL must stay at the RDNA3 number.

## What is slow (expected)

* `spill_audit.sh GPU_TARGET=gfx1201` fails with VGPR spills in the emulated
  kernels — the tile math needs more live registers than the WMMA fragments:
  `kernels_fa.o` 1600 spills total over its 6 prefill instantiations (~240–290
  each, 920 B private), `kernels_core.o` 493 (ctrl_apply prefill, 1976 B
  private), `kernels_wmma.o` 278 (dense GEMM 65, MoE 32–40), `kernels_vision.o`
  0, `kernels_sparse_global.o` 2 (pre-existing). This is the price of "correct
  first"; register budgets and LDS blocking need a re-tune pass.
* Prefill GEMM/attention throughput will be far below the RDNA3 numbers
  (wmma → 64 dot2 + 16 LDS loads per tile per lane). Decode should be much less
  affected: the decode path (GEMV / FA decode v2-v4 / MoE decode) uses no WMMA.
* The A step of decode (dual mix + ctrl QKV + norm) runs as 2–3 launches on
  64 CU instead of 1, and MTP verify rows run one launch per row.

## First runtime test

```bash
cd ~/nib
# model pack (~17.3 GiB) in models/:
#   Inkling-2-Mini-NVFP4-Mixed.gguf, mtp.safetensors, mmproj-Inkling-2-Mini-Q8_0.gguf
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGPU_TARGET=gfx1201
cmake --build build -j 8

# 0. static checks, no GPU kernels involved (safe next to a busy GPU)
./scripts/verify_rdna4.sh

# 1. THE correctness gate: nib prefills 5 tokens with the prefill kernels, then
#    runs the same 5 tokens one-by-one through the (unmodified) decode kernels and
#    compares the logits. Pass = cos_sim > 0.999, exit code 0:
#    [SELFTEST] Prefill vs Decode (5 tokens): next_pf=N next_dec=N | max_diff=... | cos_sim=0.9999999
#    A low cos_sim means the gfx12 tile emulation is wrong, not the model.
./build/nib selftest -m models/Inkling-2-Mini-NVFP4-Mixed.gguf

# 2. decode-only generation (exercises the GEMV/MoE/FA-decode path, ~50 tokens)
./build/nib gen -m models/Inkling-2-Mini-NVFP4-Mixed.gguf \
    --prompt-file tests/prompts/prose.txt -n 32 -c 4096

# 3. PREFILL gate: a ~2k-token prompt pushes every token through the gfx12 tile
#    math (dense GEMM, MoE grouped GEMM, prefill FA). If the tile emulation were
#    wrong the continuation would be obvious garbage / repeated tokens.
for i in $(seq 40); do cat tests/prompts/prose.txt; done > /tmp/long_prompt.txt
./build/nib gen -m models/Inkling-2-Mini-NVFP4-Mixed.gguf \
    --prompt-file /tmp/long_prompt.txt -n 32 -c 8192

# 4. optional: MTP drafter path (verify-batch exercises the m-row fallbacks)
./build/nib selftest -m models/Inkling-2-Mini-NVFP4-Mixed.gguf \
    --mtp models/mtp.safetensors --bench-mtp
```

There is no `--spill` flag in v0.1.2; spills are audited with
`GPU_TARGET=gfx1201 ./scripts/spill_audit.sh`.

Sanity signals to watch:
* `[step7i] occupancy blocks/WGP (wgp=64): ... a_v3=...` — on the R9700 the A
  step should report the fallback path (`occ * 64 < 96`), not the fused launch.
* `NIB_DUMP_OCC=1` prints the decode_fused occupancy cache at init.
* Prefill logits/perplexity must match the RDNA3 numbers (`PPL 3.5788` on the
  reference pack); if prefill is wrong the tile emulation is wrong, not the model.

## Honest limitations

* Nothing here has been executed on the GPU by the author of this port; all
  correctness claims are by construction (layout argument above) plus static
  verification. A `ppl`/`oracle` diff against a gfx1100 run is the real gate.
* The gfx12 WMMA fast path is implemented and wired in, but its fragment layout
  could not be executed here (no GPU runs allowed). It is derived from AMD's own
  GPUOpen sample code and CK's gfx12 traits and is checked by `nib wmma-selftest`,
  which must be run on silicon before `NIB_GFX12_WMMA=1` is made the default. See
  the diagnostics that selftest prints if it fails.
* `kernels_step7j_fdot2.hip`'s opt-in 192/384-block MoE fdot2 kernels and
  `kernels_step7h_fused.hip`'s v3 kernels are cooperative and sized for 96 CU;
  they are off by default (`fdot2_moe = 0`, `b_v3 = 0`, `c_v3 = 0`) and would
  need the same gate before they are safe on 64 CU.


## Measured on gfx1201 (Radeon AI PRO R9700, 32 GB, ROCm 7.2 WSL2)

The port is validated end-to-end with the tensor-core tile as the compile-time default
(the earlier per-TU runtime gate silently left the emulation active, which made it look
neutral; a forced build showed 3.0-4.0x):

| prompt tokens | served prefill tok/s | gate |
|---:|---:|---|
| 16,404 | 3,998.9 | - |
| 32,782 | 3,759.1 | - |
| 65,556 | 3,340.3 | needles 3/3 |
| 133,460 | 3,406.1 | needles 3/3 |
| 230,060 | 3,400.8 | needles 3/3 |

`pp2560` bench: 3828.4 at depth 0, 3319.2 at 32K. Decode 93.3 tok/s. Vision 5/5 markers.
Numerics vs the `v_dot2` fallback: greedy 32/32 tokens identical, WikiText-2 PPL
6.5295 vs 6.5071. Configuration: `-c 262144 --kv-type q4_0 --sparse-pages 224`,
`MAX_UBATCH` 2560, MoE tile BN=64 (16 KB LDS -> 4 CTAs/CU).

## Measured and rejected on gfx1201

All numbers are `pp2560` on an R9700, `-c 131072 --kv-type q4_0 --sparse-pages 224`,
against the shipped build in the same session (r=3; the depth-32K column is the
low-variance one). Recorded so nobody re-runs them:

| variant | @32K | verdict |
|---|---:|---|
| MoE tile TM=128 (FM=2, 8 accum chains) | 4098 (-6.4%) | per-expert rows average ~160, so a 128-row tile stages 256 rows for 160 of work |
| MoE tile TM=192 (one CTA, B staged once) | 3599 (-6.2%) | 40 KB LDS drops it to 1 CTA/CU |
| MoE tile BN=64 (16 KB, 4 CTAs/CU) | -7.7% | occupancy is not the limiter |
| MoE tile BN=256 (16 warps, 1 CTA/CU) | -4.5% | halves A re-staging but loses occupancy |
| NVFP4 super-step 8/6 (deeper prefetch) | 4313 (-1.5%) | +18 VGPRs |
| A/B fragment hoist out of the MMA loop | 4341 (-1.0%) | LDS is no longer the limiter after the 8-half fix |
| dense `<64,128,4,2>` preferred for M>512 | 4304 (-2.4%) | 128x128 amortises better at M=2560 |
| FA prefill NWARPS_GLOB=18 / 9 | -1.1% / -1.2% | 12 is a strict optimum |
| FA prefill BC=32 / 64 | -1.7% / -3.1% | 48 is the optimum |
| `--amdgpu-unroll-threshold=800 / 1600` | -0.4% / -0.7% | already unrolled |
| Queued-launch K+V fusion | +0.09% | launch count is not the cost |

Two structural findings behind those rows:

* The grouped MoE runs at **~59 TFLOPS** against a 70-78 TFLOPS rocBLAS reference at the
  same aggregate shape - and 25 TFLOPS for the same work as 128 separate per-expert GEMMs,
  which is why the grouping matters. Once the per-tile padding is accounted for (below) the
  kernel is **at vendor efficiency: 59 x (25088/20480) = 72.3 TFLOPS per full tile**.
* That padding is measured, not assumed. Instrumenting the routing on a 2560-token chunk
  (20,480 slot rows over 128 experts) gives `padded=25088, waste=18.4%`, with a heavy tail:
  36 experts have <=32 rows, 24 have >256 (max 2095). It cannot be scheduled away - packing
  rows across experts would need `experts + tiles - 1 = 448` weight stagings instead of 392,
  because a straddling tile must stage both experts' weights, and weight staging is 2/3 of
  the per-tile staging volume. A single big GEMM pays none of this because it has no expert
  boundaries; a grouped MoE cannot avoid it.

## Context depth: the page table is not a context limit

`SPARSE_MAX_PAGES` is `262144 / SPARSE_PAGE_SIZE`, i.e. the sparse page table is sized to exactly
the model's nominal context. With it, any `-c` above 263,680 is rejected:

```
InklingEngine: --sparse-pages supports max_ctx <= 263680 (requested 393216 = 6144 pages > 4096 scored + 24 recent)
```

That reads like an architectural limit. It is not - the table is an index, and the scratch cost
is only ~4 B per page per KV head. Raising it to `2097152 / SPARSE_PAGE_SIZE` (2M tokens of
table, ~16 MiB of page scores) leaves VRAM as the only bound.

Raw depth on a 32 GB R9700 (3-code needle at 25/50/75% depth, cold):

| prompt tokens | result |
|---:|---|
| 203,124 | 3/3 PASS |
| 462,830 | 3/3 PASS |
| 693,830 | 3/3 PASS |
| 925,531 | 3/3 PASS |
| 959,823 | empty reply |

So retrieval holds to ~925K tokens (3.5x the 262,144 training context, thanks to sliding-window
layers + a relative-position bias + sparse global attention) and then fails abruptly with no
output. VRAM: a 1M-token pool is 28.4 GiB resident (q4_0 KV, ~9.3 KiB/token); 1.25M leaves
0.07 GiB free; 1.5M aborts. The sparse page budget does not matter at depth - 112/224/320 pages
all pass at 693K and speed varies by 2%.

**Gotcha:** `SPARSE_MAX_QB` must stay `>= (MAX_UBATCH + SPARSE_PAGE_SIZE - 1) / SPARSE_PAGE_SIZE`
(= 41). The `// 8` comment next to it is stale (it predates the 64-token page size), and
shrinking it to the decode-slot count silently breaks prefill attention at *every* depth.
