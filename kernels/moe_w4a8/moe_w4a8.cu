// W4A8 routed-MoE decode kernels for Qwen3.8-Flash (sm_120a), 10-28.
// Weights: the checkpoint's NVFP4 tensors as vLLM prepares them for b12x (w13 [E, 2N, K/2] gate rows then up rows,
// w2 [E, K, N/2]; e4m3 block scales per 16 in the 128x4 swizzled layout; one fp32 global scale per expert).
// Activations: MXFP8 (e4m3 per 32 with a power-of-two scale, chosen so nothing saturates), for x and for silu(g)*u.
// Math: weight = e2m1 * e4m3 scale and the activations are exact in f16; m16n8k16 f16 MMA, fp32 accumulation.
// The MMA k index is permuted: thread quad q owns 32 contiguous k of each 128-k tile (16 contiguous weight bytes,
// one coalesced 16-byte load per row; one MXFP8 block of the activations) for both operands, so neither operand
// is staged in shared memory; the next k-tile of both is prefetched into registers.
// Kernels: prep (routing + x quantization, once per call), gate_up (FC1 split over K by 4 warps, deterministic
// partial sum, silu*mul, h quantization), down (FC2 * router weight per pair), reduce (a token's 10 pairs in slot
// order -> deterministic).
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>

namespace {

constexpr int kHidden = 2560;          // K of FC1, rows of FC2
constexpr int kInter = 640;            // N of FC1 (per half), K of FC2
constexpr int kTopK = 10;
constexpr int kMaxTok = 64;            // tokens per call (decode / verify steps); larger calls go to b12x
constexpr int kMaxPairs = kMaxTok * kTopK;
constexpr int kExperts = 512;
constexpr int kWarps = 4;
constexpr int kThreads = kWarps * 32;
constexpr int kXBlk = kHidden / 32;    // MXFP8 blocks per x row
constexpr int kHBlk = kInter / 32;     // MXFP8 blocks per h row
constexpr int kGroup = 8;              // tokens per work group (one m16n8 token tile)

// workspace (int32 words). A work group = one expert x up to 8 of its tokens; experts ascending, tokens ascending.
constexpr int kMetaNwg = 0,
              kMetaWgE = 1,                              // expert of work group
              kMetaWgOff = kMetaWgE + kMaxPairs,         // first pair-list index of the group
              kMetaWgCnt = kMetaWgOff + kMaxPairs,       // tokens in the group (1..8)
              kMetaPairs = kMetaWgCnt + kMaxPairs,       // pairs grouped by expert, ascending
              kMetaQueue = kMetaPairs + kMaxPairs,       // work-queue head
              kMetaGcnt = kMetaQueue + 1,                // finished FC1 items per work group
              kMetaTcnt = kMetaGcnt + kMaxPairs,         // finished FC2 items per token
              kMetaWords = kMetaTcnt + kMaxTok;
constexpr int kGateItems = kInter / 32;          // FC1 items per expert (32 intermediates each)
constexpr int kDownItems = kHidden / 256;        // FC2 items per expert (256 rows each)
constexpr int kRedItems = 4;
constexpr int kDownItems13 = kHidden / 128;   // v13 FC2 items per work group (128 rows = 2 slabs)                     // reduce items per token (640 columns each)

__device__ __forceinline__ void cvt_e2m1x8(uint32_t w, uint32_t (&o)[4]) {
  asm("{\n .reg .b8 b0, b1, b2, b3;\n mov.b32 {b0, b1, b2, b3}, %4;\n"
      " cvt.rn.f16x2.e2m1x2 %0, b0;\n cvt.rn.f16x2.e2m1x2 %1, b1;\n"
      " cvt.rn.f16x2.e2m1x2 %2, b2;\n cvt.rn.f16x2.e2m1x2 %3, b3;\n}"
      : "=r"(o[0]), "=r"(o[1]), "=r"(o[2]), "=r"(o[3]) : "r"(w));
}

__device__ __forceinline__ uint32_t cvt_e4m3x2(uint16_t s) {
  uint32_t o;
  asm("{\n .reg .b16 t;\n mov.b16 t, %1;\n cvt.rn.f16x2.e4m3x2 %0, t;\n}" : "=r"(o) : "h"(s));
  return o;
}

__device__ __forceinline__ uint32_t hmul2(uint32_t a, uint32_t b) {
  uint32_t o;
  asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(o) : "r"(a), "r"(b));
  return o;
}

__device__ __forceinline__ uint32_t bcast_lo(uint32_t s) { return __byte_perm(s, s, 0x1010); }
__device__ __forceinline__ uint32_t bcast_hi(uint32_t s) { return __byte_perm(s, s, 0x3232); }

__device__ __forceinline__ float pow2f(int e);
__device__ __forceinline__ uint32_t h2_pow2(int e) {   // f16x2 {2^e, 2^e}; e <= 0 (0 for e < -24)
  const __half2 m = __half2half2(__float2half_rn(pow2f(e)));
  return *reinterpret_cast<const uint32_t*>(&m);
}

__device__ __forceinline__ void mma16816(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint4 ldg_stream(const void* p) {
  uint4 v;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}

// byte offset of the (row r, scale column c) e4m3 scale in one expert's 128x4-swizzled [rows, cols] grid
__device__ __forceinline__ int sf_off(int r, int c, int cols) {
  return ((r >> 7) * (cols >> 2) + (c >> 2)) * 512 + (r & 31) * 16 + ((r >> 5) & 3) * 4 + (c & 3);
}

// power-of-two exponent e with amax / 2^e <= 448 (MXFP8 e4m3 without saturation); amax == 0 -> -126
__device__ __forceinline__ int mx_exp(float amax) {
  if (!(amax > 0.f)) return -126;
  const float r = amax * (1.f / 448.f);
  const uint32_t b = __float_as_uint(r), E = (b >> 23) & 0xffu;
  if (E == 0) {   // subnormal: rare, exact path
    int e;
    const float m = frexpf(r, &e);   // r = m * 2^e, m in [0.5, 1)
    return m == 0.5f ? e - 1 : e;
  }
  return (b & 0x7fffffu) ? (int)E - 126 : (int)E - 127;
}
__device__ __forceinline__ float pow2f(int e) {   // 2^e, exact
  return e >= -126 ? __uint_as_float((uint32_t)(e + 127) << 23) : exp2f((float)e);
}

__device__ __forceinline__ uint32_t pack_e4m3x4(float a, float b, float c, float d) {
  const __nv_fp8_e4m3 q0(a), q1(b), q2(c), q3(d);
  return (uint32_t)q0.__x | ((uint32_t)q1.__x << 8) | ((uint32_t)q2.__x << 16) | ((uint32_t)q3.__x << 24);
}

// ---- prep: CTA 0 routes (active experts ascending, each expert's pairs ascending), CTA 1+t quantizes token t ----
// xq [T][kHidden] e4m3, xe [T][kXBlk] block exponents, xemax [T] row max exponent
__global__ void __launch_bounds__(256) prep_kernel(const __nv_bfloat16* __restrict__ x, const int* __restrict__ ids,
                                                   int T, int* __restrict__ meta, uint8_t* __restrict__ xq,
                                                   int8_t* __restrict__ xe, int* __restrict__ xemax) {
  const int tid = threadIdx.x;
  if (blockIdx.x == 0) {
    __shared__ int cnt[kExperts], off[kExperts], wgo[kExperts];
    __shared__ unsigned long long mask[kExperts];
    __shared__ int wsum[8][2];
    const int np = T * kTopK;
    for (int i = tid; i < kMetaWords - kMetaQueue; i += 256) meta[kMetaQueue + i] = 0;
    for (int e = tid; e < kExperts; e += 256) { cnt[e] = 0; mask[e] = 0ull; }
    __syncthreads();
    for (int p = tid; p < np; p += 256) {
      const int e = min(max(ids[p], 0), kExperts - 1);   // out-of-range ids (padding) cannot hang the queue
      atomicAdd(&cnt[e], 1);
      atomicOr(&mask[e], 1ull << (p / kTopK));
    }
    __syncthreads();
    // exclusive scans over experts (thread i: experts 2i, 2i+1) of pair counts and of group counts
    const int e0 = 2 * tid, c0 = cnt[e0], c1 = cnt[e0 + 1];
    const int g0 = (c0 + kGroup - 1) / kGroup, g1 = (c1 + kGroup - 1) / kGroup;
    int sc = c0 + c1, sg = g0 + g1;
    const int lane = tid & 31, wid = tid >> 5;
    for (int o = 1; o < 32; o <<= 1) {
      const int vc = __shfl_up_sync(0xffffffffu, sc, o), vg = __shfl_up_sync(0xffffffffu, sg, o);
      if (lane >= o) { sc += vc; sg += vg; }
    }
    if (lane == 31) { wsum[wid][0] = sc; wsum[wid][1] = sg; }
    __syncthreads();
    int bc = 0, bg = 0;
    for (int w = 0; w < wid; ++w) { bc += wsum[w][0]; bg += wsum[w][1]; }
    const int ec = bc + sc - (c0 + c1), eg = bg + sg - (g0 + g1);   // exclusive prefix at expert e0
    off[e0] = ec; off[e0 + 1] = ec + c0;
    wgo[e0] = eg; wgo[e0 + 1] = eg + g0;
    if (tid == 255) meta[kMetaNwg] = bg + sg;
    __syncthreads();
    for (int e = tid; e < kExperts; e += 256) {
      const int c = cnt[e];
      for (int g = 0; g * kGroup < c; ++g) {
        const int wg = wgo[e] + g;
        meta[kMetaWgE + wg] = e;
        meta[kMetaWgOff + wg] = off[e] + g * kGroup;
        meta[kMetaWgCnt + wg] = min(kGroup, c - g * kGroup);
      }
    }
    for (int p = tid; p < np; p += 256) {
      const int e = min(max(ids[p], 0), kExperts - 1), t = p / kTopK;
      int rank;
      if (__popcll(mask[e]) == cnt[e]) {
        rank = __popcll(mask[e] & ((1ull << t) - 1ull));   // tokens < t routed to e
      } else {   // a token lists e more than once (dummy routing): count earlier pairs one by one
        rank = 0;
        for (int p2 = 0; p2 < p; ++p2) rank += min(max(ids[p2], 0), kExperts - 1) == e;
      }
      meta[kMetaPairs + off[e] + rank] = p;
    }
    return;
  }
  // token t: 80 blocks of 32, 256 threads -> thread handles block tid (tid < 80)
  const int t = blockIdx.x - 1;
  __shared__ int s_e[kXBlk];
  if (tid < kXBlk) {
    const uint4* p = reinterpret_cast<const uint4*>(x + (size_t)t * kHidden + tid * 32);
    uint4 v[4];
    float am = 0.f;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      v[j] = p[j];
      const __nv_bfloat162* h2 = reinterpret_cast<const __nv_bfloat162*>(&v[j]);
#pragma unroll
      for (int t2 = 0; t2 < 4; ++t2) {
        const float2 f2 = __bfloat1622float2(h2[t2]);
        am = fmaxf(am, fmaxf(fabsf(f2.x), fabsf(f2.y)));
      }
    }
    const int ex = mx_exp(am);
    const float inv = pow2f(-ex);
    uint32_t pk[8];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const __nv_bfloat162* h2 = reinterpret_cast<const __nv_bfloat162*>(&v[j]);
      const float2 a = __bfloat1622float2(h2[0]), b = __bfloat1622float2(h2[1]);
      const float2 c = __bfloat1622float2(h2[2]), d = __bfloat1622float2(h2[3]);
      pk[2 * j] = pack_e4m3x4(a.x * inv, a.y * inv, b.x * inv, b.y * inv);
      pk[2 * j + 1] = pack_e4m3x4(c.x * inv, c.y * inv, d.x * inv, d.y * inv);
    }
    uint4* d = reinterpret_cast<uint4*>(xq + (size_t)t * kHidden + tid * 32);
    d[0] = make_uint4(pk[0], pk[1], pk[2], pk[3]);
    d[1] = make_uint4(pk[4], pk[5], pk[6], pk[7]);
    xe[t * kXBlk + tid] = (int8_t)ex;
    s_e[tid] = ex;
  }
  __syncthreads();
  if (tid == 0) {
    int m = -126;
    for (int i = 0; i < kXBlk; ++i) m = max(m, s_e[i]);
    xemax[t] = m;
  }
}

// B operand from global MXFP8 rows: per (tok, k-tile) NB blocks of 32 e4m3 bytes owned by this thread (k-tile =
// NB*128 k; thread quad q owns k [q*NB*32, (q+1)*NB*32)) + each block's multiplier 2^(e_blk - e_row_max).
// fetch() = raw loads (prefetched one k-tile ahead), convert(j) = 16 f16x2 words of block j.
template <int NB>
struct BRaw {
  uint4 v[2 * NB];
  int e[NB];
};
struct BMx {
  const uint8_t* q;     // [rows][row_len]
  const int8_t* e;      // [rows][row_len / 32]
  int row_len;
  int nrow;             // valid rows of this CTA (tokens)
  const int* row_of;    // CTA token -> global row
  const int* rmax;      // CTA token -> row max exponent
  bool coherent;        // rows written earlier in the same kernel: L2 loads (ld.cg), not the non-coherent path
  template <int NB>
  __device__ __forceinline__ void fetch(int tok, int k0, BRaw<NB>& r) const {   // k0 = first k of the tile
    const int q4 = threadIdx.x & 3;
    if (tok < nrow) {
      const int row = row_of[tok];
      const int kk = k0 + q4 * NB * 32;
      const uint4* src = reinterpret_cast<const uint4*>(q + (size_t)row * row_len + kk);
#pragma unroll
      for (int j = 0; j < 2 * NB; ++j) r.v[j] = coherent ? __ldcg(src + j) : __ldg(src + j);
#pragma unroll
      for (int j = 0; j < NB; ++j) {
        const int8_t* ep = e + row * (row_len / 32) + kk / 32 + j;
        r.e[j] = (coherent ? __ldcg(reinterpret_cast<const signed char*>(ep)) : *ep) - rmax[tok];
      }
    } else {
#pragma unroll
      for (int j = 0; j < 2 * NB; ++j) r.v[j] = make_uint4(0, 0, 0, 0);
#pragma unroll
      for (int j = 0; j < NB; ++j) r.e[j] = 0;
    }
  }
  template <int NB>
  __device__ __forceinline__ void convert(const BRaw<NB>& r, int j, uint32_t (&b)[16]) const {
    const uint32_t m = h2_pow2(r.e[j]);
#pragma unroll
    for (int jj = 0; jj < 2; ++jj) {
      const uint4 v = r.v[2 * j + jj];
      const uint32_t ws[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
      for (int t = 0; t < 4; ++t) {
        b[8 * jj + 2 * t] = hmul2(cvt_e4m3x2((uint16_t)(ws[t] & 0xffffu)), m);
        b[8 * jj + 2 * t + 1] = hmul2(cvt_e4m3x2((uint16_t)(ws[t] >> 16)), m);
      }
    }
  }
};

// One warp: RG row groups of 16 weight rows (rows[rg] + 0..15) x NT*8 tokens over KT k-tiles starting at k0.
// k-tile = NB*128 k: per row the thread reads NB*16 contiguous bytes (the quad: NB*64 B, NB=2 -> a full 128-B line)
// and NB*2 scale groups (2 or 4 contiguous swizzled bytes).
template <int RG, int NB, int KT, int NT, typename Mid>
__device__ __forceinline__ void warp_gemm(const uint8_t* __restrict__ w, int row_bytes, const uint8_t* __restrict__ sf,
                                          int sf_cols, const int (&rows)[RG], const BMx& bsrc, int k0,
                                          float (&acc)[RG][NT][4], Mid mid) {
  const int lane = threadIdx.x & 31, g = lane >> 2, q = lane & 3;
  constexpr int TK = NB * 128;
#pragma unroll
  for (int rg = 0; rg < RG; ++rg)
#pragma unroll
    for (int nt = 0; nt < NT; ++nt)
#pragma unroll
      for (int i = 0; i < 4; ++i) acc[rg][nt][i] = 0.f;

  const uint8_t* wp[RG][2];
  int sr[RG][2];
#pragma unroll
  for (int rg = 0; rg < RG; ++rg)
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int r = rows[rg] + g + 8 * h;
      wp[rg][h] = w + (size_t)r * row_bytes + k0 / 2 + q * NB * 16;
      sr[rg][h] = r;
    }
  uint4 wall[KT][RG][2][NB];   // every weight load of the warp's k range is issued up front
  uint32_t scur[RG][2], snxt[RG][2];
  BRaw<NB> bcur[NT], bnxt[NT];
  auto loadA = [&](int kt, uint32_t (&s)[RG][2]) {
#pragma unroll
    for (int rg = 0; rg < RG; ++rg)
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int c = (k0 + kt * TK) / 16 + q * NB * 2;
        const uint8_t* sp = sf + sf_off(sr[rg][h], c, sf_cols);
        if constexpr (NB == 2) s[rg][h] = __ldg(reinterpret_cast<const uint32_t*>(sp));
        else s[rg][h] = __ldg(reinterpret_cast<const uint16_t*>(sp));
      }
  };
  auto loadB = [&](int kt, BRaw<NB> (&br)[NT]) {
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) bsrc.fetch<NB>(nt * 8 + g, k0 + kt * TK, br[nt]);
  };
#pragma unroll
  for (int kt = 0; kt < KT; ++kt)
#pragma unroll
    for (int rg = 0; rg < RG; ++rg)
#pragma unroll
      for (int h = 0; h < 2; ++h)
#pragma unroll
        for (int j = 0; j < NB; ++j) wall[kt][rg][h][j] = ldg_stream(wp[rg][h] + kt * (TK / 2) + j * 16);
  loadA(0, scur);
  mid();   // token metadata (and dependency waits) while the first weight loads are in flight
  loadB(0, bcur);
#pragma unroll
  for (int kt = 0; kt < KT; ++kt) {
    if (kt + 1 < KT) { loadA(kt + 1, snxt); loadB(kt + 1, bnxt); }
    auto& cur = wall[kt];
#ifdef MOE_NOCOMPUTE
#pragma unroll
    for (int rg = 0; rg < RG; ++rg)
#pragma unroll
      for (int h = 0; h < 2; ++h)
#pragma unroll
        for (int j = 0; j < NB; ++j)
          acc[rg][0][0] += __uint_as_float((cur[rg][h][j].x ^ cur[rg][h][j].y ^ cur[rg][h][j].z ^ cur[rg][h][j].w ^
                                            scur[rg][h] ^ bcur[0].v[j].x) & 0x3fffffffu);
    if (false)
#endif
#pragma unroll
    for (int j = 0; j < NB; ++j) {      // one 32-k block (16 weight bytes per row) at a time
      uint32_t b[NT][16];
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) bsrc.convert<NB>(bcur[nt], j, b[nt]);
#pragma unroll
      for (int rg = 0; rg < RG; ++rg) {
        uint32_t f[2][16];   // [row half][byte] f16x2 of (e2m1 * scale)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const uint32_t s = cvt_e4m3x2((uint16_t)(scur[rg][h] >> (16 * j)));
          const uint32_t s0 = bcast_lo(s), s1 = bcast_hi(s);
          const uint32_t ws[4] = {cur[rg][h][j].x, cur[rg][h][j].y, cur[rg][h][j].z, cur[rg][h][j].w};
#pragma unroll
          for (int jj = 0; jj < 4; ++jj) {
            uint32_t o[4];
            cvt_e2m1x8(ws[jj], o);
#pragma unroll
            for (int t = 0; t < 4; ++t) f[h][4 * jj + t] = hmul2(o[t], jj < 2 ? s0 : s1);
          }
        }
#pragma unroll
        for (int i = 0; i < 8; ++i)
#pragma unroll
          for (int nt = 0; nt < NT; ++nt)
            mma16816(acc[rg][nt], f[0][2 * i], f[1][2 * i], f[0][2 * i + 1], f[1][2 * i + 1], b[nt][2 * i],
                     b[nt][2 * i + 1]);
      }
    }
    if (kt + 1 < KT) {
#pragma unroll
      for (int rg = 0; rg < RG; ++rg)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          scur[rg][h] = snxt[rg][h];
        }
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) bcur[nt] = bnxt[nt];
    }
  }
}

// ---- FC1: grid (kInter/32, max active experts); CTA = 32 intermediate indices; warp w: 16 intermediates
// (ig = w & 1: gate rows + up rows) x K half (kh = w >> 1), 5 k-tiles of 256; K halves summed in fixed order ----
template <int NT, typename Mid>
__device__ void gateup_body(const uint8_t* __restrict__ xq, const int8_t* __restrict__ xe,
                            const uint8_t* __restrict__ w13, const uint8_t* __restrict__ s13, float gs, int e,
                            int ntok, const int* s_pairs, const int* s_row, const int* s_rmax, float* s_red,
                            uint8_t* __restrict__ h8, int8_t* __restrict__ hexp, int tile, int gate_row0,
                            const Mid& mid) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, q = lane & 3;
  const int ig = warp & 1, kh = warp >> 1;
  const int i0 = tile * 32;
  const int rows[2] = {gate_row0 + i0 + 16 * ig, (kInter - gate_row0) + i0 + 16 * ig};   // gate rows, up rows
  const BMx bsrc{xq, xe, kHidden, ntok, s_row, s_rmax, false};
  float acc[2][NT][4];
  warp_gemm<2, 2, kHidden / 2 / 256, NT>(w13 + (size_t)e * (2 * kInter) * (kHidden / 2), kHidden / 2,
                                         s13 + (size_t)e * (2 * kInter) * (kHidden / 16), kHidden / 16, rows, bsrc,
                                         kh * (kHidden / 2), acc, mid);
  // K-half partials: warps 2, 3 -> smem; warps 0, 1 add them (fixed order) and finish their 16 intermediates
  constexpr int RED = 2 * NT * 4 * 32;
  if (kh == 1) {
#pragma unroll
    for (int rg = 0; rg < 2; ++rg)
#pragma unroll
      for (int nt = 0; nt < NT; ++nt)
#pragma unroll
        for (int i = 0; i < 4; ++i) s_red[ig * RED + ((rg * NT + nt) * 4 + i) * 32 + lane] = acc[rg][nt][i];
  }
  __syncthreads();
  if (kh == 1) return;   // (the caller re-converges all warps with __syncthreads)
#pragma unroll
  for (int rg = 0; rg < 2; ++rg)
#pragma unroll
    for (int nt = 0; nt < NT; ++nt)
#pragma unroll
      for (int i = 0; i < 4; ++i) acc[rg][nt][i] += s_red[ig * RED + ((rg * NT + nt) * 4 + i) * 32 + lane];
  // silu(g) * u, then the 32-block (both warps' 16) amax through smem
  __shared__ float s_am[2][2 * NT * 8];   // [ig][token]
  float hv[NT][2][2];                      // [nt][c][hh]: intermediate i0 + 16 ig + g + 8 hh, token nt*8 + 2q + c
#pragma unroll
  for (int nt = 0; nt < NT; ++nt)
#pragma unroll
    for (int c = 0; c < 2; ++c) {
      const int tok = nt * 8 + 2 * q + c;
      const float sc = tok < ntok ? gs * pow2f(s_rmax[tok]) : 0.f;
      float am = 0.f;
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const float gv = acc[0][nt][2 * hh + c] * sc, uv = acc[1][nt][2 * hh + c] * sc;
        hv[nt][c][hh] = gv / (1.f + __expf(-gv)) * uv;
        am = fmaxf(am, fabsf(hv[nt][c][hh]));
      }
      am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 4));
      am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 8));
      am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 16));
      if (g == 0) s_am[ig][tok] = am;
    }
  asm volatile("bar.sync 1, 64;");   // warps 0 and 1 only
#pragma unroll
  for (int nt = 0; nt < NT; ++nt)
#pragma unroll
    for (int c = 0; c < 2; ++c) {
      const int tok = nt * 8 + 2 * q + c;
      if (tok >= ntok) continue;
      const int ex = mx_exp(fmaxf(s_am[0][tok], s_am[1][tok]));
      const float inv = pow2f(-ex);
      const int pair = s_pairs[tok];
      uint8_t* dst = h8 + (size_t)pair * kInter + i0 + 16 * ig + g;
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const __nv_fp8_e4m3 qv(hv[nt][c][hh] * inv);
        dst[8 * hh] = qv.__x;
      }
      if (g == 0 && ig == 0) hexp[pair * kHBlk + i0 / 32] = (int8_t)ex;
    }
}

// ---- FC2: grid (kHidden/64/kWarps, max active experts); warp = 64 output rows; writes y[pair][row] * weight ----
template <int NT, typename Mid>
__device__ void down_body(const uint8_t* __restrict__ w2, const uint8_t* __restrict__ s2, float gs, int e,
                          const uint8_t* __restrict__ h8, const int8_t* __restrict__ hexp, int ntok,
                          const int* s_pairs, const int* s_rmax, const float* s_wt, float* __restrict__ y,
                          int tile, const Mid& mid) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, q = lane & 3;
  const int r0 = (tile * kWarps + warp) * 64;
  const int rows[4] = {r0, r0 + 16, r0 + 32, r0 + 48};
  const BMx bsrc{h8, hexp, kInter, ntok, s_pairs, s_rmax, true};
  float acc[4][NT][4];
  warp_gemm<4, 1, kInter / 128, NT>(w2 + (size_t)e * kHidden * (kInter / 2), kInter / 2,
                                    s2 + (size_t)e * kHidden * (kInter / 16), kInter / 16, rows, bsrc, 0, acc, mid);
#pragma unroll
  for (int nt = 0; nt < NT; ++nt)
#pragma unroll
    for (int c = 0; c < 2; ++c) {
      const int tok = nt * 8 + 2 * q + c;
      if (tok >= ntok) continue;
      const float sc = gs * pow2f(s_rmax[tok]) * s_wt[tok];
      float* dst = y + (size_t)s_pairs[tok] * kHidden;
#pragma unroll
      for (int rg = 0; rg < 4; ++rg)
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) dst[rows[rg] + g + 8 * hh] = acc[rg][nt][2 * hh + c] * sc;
    }
}

// spin until *p >= target; a broken queue traps (engine error) instead of hanging the GPU
__device__ __forceinline__ void wait_geq(const int* p, int target);
__device__ __forceinline__ int ld_acquire(const int* p) {
  int v;
  asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}

__device__ __forceinline__ void wait_geq(const int* p, int target) {
  for (long long n = 0; ld_acquire(p) < target; ++n) {
    if (n > (1ll << 26)) {
#ifndef MOE_NOPRINTF
      printf("[moe_w4a8] wait timeout: cta %d warp %d counter word %d value %d target %d\n", blockIdx.x, threadIdx.x >> 5,
             (int)(p - (const int*)0) , ld_acquire(p), target);
#endif
      __trap();
    }
    __nanosleep(64);
  }
}

// ---- persistent kernel: CTAs pull items from one queue: FC1 items (expert-major), then FC2 items (each waits for
// its expert's FC1 items), then reduce items (each waits for its token's FC2 items). Every wait points at items
// claimed earlier, so the queue always drains; FC2 items fill the FC1 tail. ----
__global__ void __launch_bounds__(kThreads) moe_kernel(
    int* __restrict__ meta, int T, const uint8_t* __restrict__ xq, const int8_t* __restrict__ xe,
    const int* __restrict__ xemax, const float* __restrict__ topk_w, const uint8_t* __restrict__ w13,
    const uint8_t* __restrict__ s13, const float* __restrict__ g13, const uint8_t* __restrict__ w2,
    const uint8_t* __restrict__ s2, const float* __restrict__ g2, uint8_t* __restrict__ h8,
    int8_t* __restrict__ hexp, float* __restrict__ y, __nv_bfloat16* __restrict__ out, int gate_row0) {
  __shared__ int s_item, s_pairs[kGroup], s_row[kGroup], s_rmax[kGroup];
  __shared__ float s_wt[kGroup];
  __shared__ float s_red[2 * 2 * 2 * 4 * 32];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int nwg = meta[kMetaNwg];
  const int n_gate = nwg * kGateItems, n_down = nwg * kDownItems, n_all = n_gate + n_down + T * kRedItems;
  if (tid == 0) s_item = atomicAdd(&meta[kMetaQueue], 1);
  __syncthreads();
  int item = s_item;
  for (;;) {
    if (item >= n_all) break;
    int nxt = 0;
    if (tid == 0) nxt = atomicAdd(&meta[kMetaQueue], 1);   // claimed now, used after this item
    if (item < n_gate) {
      const int wg = item / kGateItems, tile = item % kGateItems;
      const int e = meta[kMetaWgE + wg], off = meta[kMetaWgOff + wg], ntok = meta[kMetaWgCnt + wg];
      auto mid = [&] {
        if (tid < ntok) {
          const int pair = meta[kMetaPairs + off + tid], row = pair / kTopK;
          s_pairs[tid] = pair; s_row[tid] = row; s_rmax[tid] = xemax[row];
        }
        __syncthreads();
      };
      gateup_body<1>(xq, xe, w13, s13, g13[e], e, ntok, s_pairs, s_row, s_rmax, s_red, h8, hexp, tile, gate_row0, mid);
      __threadfence();
      __syncthreads();
      if (tid == 0) atomicAdd(&meta[kMetaGcnt + wg], 1);
    } else if (item < n_gate + n_down) {
      const int wg = (item - n_gate) / kDownItems, tile = (item - n_gate) % kDownItems;
      const int e = meta[kMetaWgE + wg], off = meta[kMetaWgOff + wg], ntok = meta[kMetaWgCnt + wg];
      auto mid = [&] {
        if (tid < ntok) { const int pair = meta[kMetaPairs + off + tid]; s_pairs[tid] = pair; s_wt[tid] = topk_w[pair]; }
        if (tid == 0)
          wait_geq(&meta[kMetaGcnt + wg], kGateItems);
        __syncthreads();
        for (int t = warp; t < ntok; t += kWarps) {   // row max exponent of each pair's h
          const int pair = s_pairs[t];
          int m = lane < kHBlk ? (int)__ldcg(reinterpret_cast<const signed char*>(hexp + pair * kHBlk + lane)) : -126;
#pragma unroll
          for (int o = 16; o; o >>= 1) m = max(m, __shfl_xor_sync(0xffffffffu, m, o));
          if (lane == 0) s_rmax[t] = m;
        }
        __syncthreads();
      };
      down_body<1>(w2, s2, g2[e], e, h8, hexp, ntok, s_pairs, s_rmax, s_wt, y, tile, mid);
      __threadfence();
      __syncthreads();
      if (tid < ntok) atomicAdd(&meta[kMetaTcnt + s_pairs[tid] / kTopK], 1);
    } else {
      const int r = item - n_gate - n_down, t = r / kRedItems, c0 = (r % kRedItems) * (kHidden / kRedItems);
      if (tid == 0)
        wait_geq(&meta[kMetaTcnt + t], kTopK * kDownItems);
      __syncthreads();
      for (int c = c0 + tid * 4; c < c0 + kHidden / kRedItems; c += kThreads * 4) {
        float4 sum = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
        for (int j = 0; j < kTopK; ++j) {
          const float4 v = __ldcg(reinterpret_cast<const float4*>(y + (size_t)(t * kTopK + j) * kHidden + c));
          sum.x += v.x; sum.y += v.y; sum.z += v.z; sum.w += v.w;
        }
        __nv_bfloat162* o = reinterpret_cast<__nv_bfloat162*>(out + (size_t)t * kHidden + c);
        o[0] = __floats2bfloat162_rn(sum.x, sum.y);
        o[1] = __floats2bfloat162_rn(sum.z, sum.w);
      }
    }
    __syncthreads();   // smem reuse
    if (tid == 0) s_item = nxt;
    __syncthreads();
    item = s_item;
  }
}

#ifdef MOE13_TRACE
__device__ unsigned long long g_trace[188 * 64 * 4];   // [cta][event][wait_start, data_ready, compute_done, item_end|kind]
__device__ unsigned long long g_trace2[188 * 8];
__device__ __forceinline__ unsigned long long gtime() { unsigned long long t; asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t) :: "memory"); return t; }
#endif
// ======================= v13: producer / consumer persistent kernel (1 CTA per SM) =======================
// Producer warp: claims items, streams each item's weight slabs (contiguous 20 KB: 16 FC1 rows or 64 FC2 rows) into a
// shared-memory ring with 1D bulk async copies, plus the slab's scale words (4-byte cp.async) into a compact layout.
// 4 consumer warps: FC1 slab = K quarter per warp (split-K summed in fixed order), FC2 slab = 16 rows per warp.
constexpr int kSlabBytes = 20480;            // 16 x 1280 B (FC1) or 64 x 320 B (FC2)
constexpr int kSlabScale = 2560;             // 16 x 160 or 64 x 40 scale bytes
constexpr int kSlots = 4;
constexpr int kSlotBytes = kSlabBytes + kSlabScale;
constexpr int kConsumers = 4;
constexpr int kThreads13 = (kConsumers + 1) * 32;
enum : int { kDescStop = 0, kDescGate = 1, kDescDown = 2, kDescRed = 3 };
struct Desc {   // written by the producer; slab 0 of an item carries its token metadata
  int kind, item, slab, e, ntok, wg;
  int pairs[kGroup];
};
__device__ __forceinline__ void red_release_add(int* p, int v) {
  asm volatile("red.release.gpu.global.add.s32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void mbar_init(uint64_t* b, int count) {
  asm volatile("mbarrier.init.shared.b64 [%0], %1;" ::"r"(smem_u32(b)), "r"(count));
}
__device__ __forceinline__ void mbar_arrive(uint64_t* b) {
  asm volatile("{ .reg .b64 st; mbarrier.arrive.release.cta.shared::cta.b64 st, [%0]; }" ::"r"(smem_u32(b)) : "memory");
}
__device__ __forceinline__ void mbar_arrive_tx(uint64_t* b, int bytes) {
  asm volatile("{ .reg .b64 st; mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 st, [%0], %1; }" ::"r"(smem_u32(b)),
               "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* b, int parity) {
  uint32_t done;
  do {
    asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 p, [%1], %2; selp.u32 %0, 1, 0, p; }"
                 : "=r"(done) : "r"(smem_u32(b)), "r"(parity) : "memory");
  } while (!done);
}
__device__ __forceinline__ bool mbar_test(uint64_t* b, int parity) {
  uint32_t done;
  asm volatile("{ .reg .pred p; mbarrier.test_wait.parity.acquire.cta.shared::cta.b64 p, [%1], %2; selp.u32 %0, 1, 0, p; }"
               : "=r"(done) : "r"(smem_u32(b)), "r"(parity) : "memory");
  return done;
}
__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, int bytes, uint64_t* bar) {
#ifdef MOE13_BULK_CLUSTER
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
#else   // .shared::cluster makes the driver reserve ~3.7 GB on first launch; the CTA-local form does not
  asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
#endif
               ::"r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ void cp4_g2s(void* dst, const void* src) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" ::"r"(smem_u32(dst)), "l"(src) : "memory");
}
__device__ __forceinline__ void cp_arrive_noinc(uint64_t* bar) {
  asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" ::"r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ void cbar() { asm volatile("bar.sync 1, %0;" ::"n"(kConsumers * 32) : "memory"); }

// One consumer warp: 16 rows (local rows lr0 + g, lr0 + g + 8) of a slab x 8 tokens over KT k-tiles of 128.
// sw: slab weights [rows][rowb]; ssc: compact scales [rows][scs]; byte0 / grp0: the warp's k offset; b: the item's
// activation fragments (f16, staged once per item, shared by all its slabs).
template <int KT>
__device__ __forceinline__ void slab_mma(const uint8_t* sw, int rowb, const uint8_t* ssc, int scs, int lr0, int byte0,
                                         int grp0, const uint32_t (&b)[KT][16], float (&acc)[4]) {
  const int lane = threadIdx.x & 31, g = lane >> 2, q = lane & 3;
  const uint8_t* wr[2] = {sw + (lr0 + g) * rowb + byte0 + q * 16, sw + (lr0 + g + 8) * rowb + byte0 + q * 16};
  const uint8_t* sr[2] = {ssc + (lr0 + g) * scs + grp0 + 2 * q, ssc + (lr0 + g + 8) * scs + grp0 + 2 * q};
  uint4 v[KT][2];
  uint16_t sc[KT][2];
#pragma unroll
  for (int kt = 0; kt < KT; ++kt)
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      v[kt][h] = *reinterpret_cast<const uint4*>(wr[h] + kt * 64);
      sc[kt][h] = *reinterpret_cast<const uint16_t*>(sr[h] + kt * 8);
    }
#pragma unroll
  for (int kt = 0; kt < KT; ++kt) {
    uint32_t f[2][16];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const uint32_t s = cvt_e4m3x2(sc[kt][h]);
      const uint32_t s0 = bcast_lo(s), s1 = bcast_hi(s);
      const uint32_t ws[4] = {v[kt][h].x, v[kt][h].y, v[kt][h].z, v[kt][h].w};
#pragma unroll
      for (int jj = 0; jj < 4; ++jj) {
        uint32_t o[4];
        cvt_e2m1x8(ws[jj], o);
#pragma unroll
        for (int t = 0; t < 4; ++t) f[h][4 * jj + t] = hmul2(o[t], jj < 2 ? s0 : s1);
      }
    }
#pragma unroll
    for (int i = 0; i < 8; ++i)
      mma16816(acc, f[0][2 * i], f[1][2 * i], f[0][2 * i + 1], f[1][2 * i + 1], b[kt][2 * i], b[kt][2 * i + 1]);
  }
}

// raw B of one MXFP8 row for the warp's KT tiles (exponents unadjusted), and its conversion with the row max
template <int KT>
struct BPre {
  uint4 v[KT][2];
  int e[KT];
};
template <int KT, bool COH>
__device__ __forceinline__ void b_fetch_raw(const uint8_t* q, const int8_t* ex, int row_len, int row, int k0, BPre<KT>& r) {
  const int q4 = threadIdx.x & 3;
#pragma unroll
  for (int kt = 0; kt < KT; ++kt) {
    const int kk = k0 + kt * 128 + q4 * 32;
    if (row >= 0) {
      const uint4* src = reinterpret_cast<const uint4*>(q + (size_t)row * row_len + kk);
      r.v[kt][0] = COH ? __ldcg(src) : __ldg(src);
      r.v[kt][1] = COH ? __ldcg(src + 1) : __ldg(src + 1);
      const signed char* ep = reinterpret_cast<const signed char*>(ex + row * (row_len / 32) + kk / 32);
      r.e[kt] = COH ? __ldcg(ep) : *ep;
    } else {
      r.v[kt][0] = r.v[kt][1] = make_uint4(0, 0, 0, 0);
      r.e[kt] = -126;
    }
  }
}
template <int KT>
__device__ __forceinline__ void b_convert(const BPre<KT>& r, int rmax, uint32_t (&b)[KT][16]) {
  const BMx cv{};
#pragma unroll
  for (int kt = 0; kt < KT; ++kt) {
    BRaw<1> t;
    t.v[0] = r.v[kt][0]; t.v[1] = r.v[kt][1]; t.e[0] = r.e[kt] - rmax;
    cv.convert<1>(t, 0, b[kt]);
  }
}

// the item's B fragments for KT tiles from one MXFP8 row (row < 0: zero), converted to f16 once
template <int KT, bool COH>
__device__ __forceinline__ void stage_b_row(const uint8_t* q, const int8_t* ex, int row_len, int row, int rmax, int k0,
                                            uint32_t (&b)[KT][16]) {
  const int q4 = threadIdx.x & 3;
  BRaw<1> r[KT];
#pragma unroll
  for (int kt = 0; kt < KT; ++kt) {
    const int kk = k0 + kt * 128 + q4 * 32;
    if (row >= 0) {
      const uint4* src = reinterpret_cast<const uint4*>(q + (size_t)row * row_len + kk);
      r[kt].v[0] = COH ? __ldcg(src) : __ldg(src);
      r[kt].v[1] = COH ? __ldcg(src + 1) : __ldg(src + 1);
      const signed char* ep = reinterpret_cast<const signed char*>(ex + row * (row_len / 32) + kk / 32);
      r[kt].e[0] = (COH ? __ldcg(ep) : *ep) - rmax;
    } else {
      r[kt].v[0] = r[kt].v[1] = make_uint4(0, 0, 0, 0);
      r[kt].e[0] = 0;
    }
  }
  const BMx cv{};
#pragma unroll
  for (int kt = 0; kt < KT; ++kt) cv.convert<1>(r[kt], 0, b[kt]);
}

// the item's B fragments for KT tiles starting at k0 (token g), converted to f16 once
template <int KT>
__device__ __forceinline__ void stage_b(const BMx& bsrc, int k0, uint32_t (&b)[KT][16]) {
  const int g = (threadIdx.x & 31) >> 2;
  BRaw<1> r[KT];
#pragma unroll
  for (int kt = 0; kt < KT; ++kt) bsrc.fetch<1>(g, k0 + kt * 128, r[kt]);
#pragma unroll
  for (int kt = 0; kt < KT; ++kt) bsrc.convert<1>(r[kt], 0, b[kt]);
}

__global__ void __launch_bounds__(kThreads13, 1) moe13_kernel(
    int* __restrict__ meta, int T, const uint8_t* __restrict__ xq, const int8_t* __restrict__ xe,
    const int* __restrict__ xemax, const float* __restrict__ topk_w, const uint8_t* __restrict__ w13,
    const uint8_t* __restrict__ s13, const float* __restrict__ g13, const uint8_t* __restrict__ w2,
    const uint8_t* __restrict__ s2, const float* __restrict__ g2, uint8_t* __restrict__ h8,
    int8_t* __restrict__ hexp, float* __restrict__ y, __nv_bfloat16* __restrict__ out, int gate_row0) {
  extern __shared__ __align__(128) uint8_t ring[];
  __shared__ __align__(8) uint64_t full[kSlots], empty[kSlots];
  __shared__ Desc desc[kSlots];
  // per-item consumer state (indexed by item parity)
  __shared__ int s_pairs[2][kGroup], s_row[2][kGroup], s_rmax[2][kGroup];
  __shared__ float s_wt[2][kGroup];
  // one s_red suffices: every item's slab 0 has a consumer barrier, so warps 1-3 cannot reach the next item's
  // split-K exchange before warp 0 finished this item's epilogue
  __shared__ float s_red[kConsumers - 1][4][4][32];   // non-owner split-K partials, slot (warp - owner - 1) & 3
  __shared__ __align__(16) uint8_t s_h[kConsumers][kGroup][32];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  if (tid == 0) {
    for (int i = 0; i < kSlots; ++i) { mbar_init(&full[i], 1 + 32); mbar_init(&empty[i], kConsumers); }
#ifndef MOE13_NOCLUSTERFENCE
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
#endif
  }
  __syncthreads();
  const int nwg = meta[kMetaNwg];
  const int n_gate = nwg * kGateItems, n_down = nwg * kDownItems13, n_all = n_gate + n_down + T * kRedItems;

  if (warp == kConsumers) {   // ---------------- producer ----------------
    // software-pipelined: the next item is claimed at the top and its metadata loads are issued between this
    // item's slab posts, so the dependent L2 chain (queue -> work group -> pairs) never stalls the copies
    int slot = 0, ph = 0;
    auto kind_of = [&](int it) {
      return it >= n_all ? kDescStop : it < n_gate ? kDescGate : it < n_gate + n_down ? kDescDown : kDescRed;
    };
    auto wg_of = [&](int it, int k) { return k == kDescGate ? it / kGateItems : k == kDescDown ? (it - n_gate) / kDownItems13 : 0; };
    int cur = 0;
    if (lane == 0) cur = atomicAdd(&meta[kMetaQueue], 1);
    cur = __shfl_sync(0xffffffffu, cur, 0);
    int ck = kind_of(cur), cwg = wg_of(cur, ck), ce = 0, coff = 0, cn = 0, cpair = 0;
    if (ck == kDescGate || ck == kDescDown) { ce = meta[kMetaWgE + cwg]; coff = meta[kMetaWgOff + cwg]; cn = meta[kMetaWgCnt + cwg]; }
    if (lane < cn) cpair = meta[kMetaPairs + coff + lane];
    for (;;) {
      int nraw = 0;
      if (ck != kDescStop && lane == 0) nraw = atomicAdd(&meta[kMetaQueue], 1);
      const int nslab = ck == kDescGate ? 4 : ck == kDescDown ? 2 : 1;
      const int tile = ck == kDescGate ? cur % kGateItems : ck == kDescDown ? (cur - n_gate) % kDownItems13 : 0;
      int nxt = 0, nk = kDescStop, nwgi = 0, ne = 0, noff = 0, nn = 0, npair = 0;
      for (int sl = 0; sl < nslab; ++sl) {
        mbar_wait(&empty[slot], ph ^ 1);
        uint8_t* dst = ring + slot * kSlotBytes;
        const uint8_t* src = nullptr;
        if (ck == kDescGate) {
          const int r0 = (sl < 2 ? gate_row0 : kInter - gate_row0) + tile * 32 + 16 * (sl & 1);
          src = w13 + ((size_t)ce * (2 * kInter) + r0) * (kHidden / 2);
          const uint8_t* sfe = s13 + (size_t)ce * (2 * kInter) * (kHidden / 16);
          {   // lane: row lr = lane & 15, column groups cg = (lane >> 4) + 2j; swizzled address = row base + cg * 512
            const int lr = lane & 15;
            const uint8_t* rb = sfe + sf_off(r0 + lr, 0, kHidden / 16);
            uint8_t* db = dst + kSlabBytes + lr * 160;
#pragma unroll
            for (int j = 0; j < 20; ++j) {
              const int cg = (lane >> 4) + 2 * j;
              cp4_g2s(db + cg * 4, rb + cg * 512);
            }
          }
        } else if (ck == kDescDown) {
          const int r0 = tile * 128 + 64 * sl;
          src = w2 + ((size_t)ce * kHidden + r0) * (kInter / 2);
          const uint8_t* sfe = s2 + (size_t)ce * kHidden * (kInter / 16);
#pragma unroll
          for (int h = 0; h < 2; ++h) {   // lane: rows lane, lane + 32; 10 column groups each
            const int lr = lane + 32 * h;
            const uint8_t* rb = sfe + sf_off(r0 + lr, 0, kInter / 16);
            uint8_t* db = dst + kSlabBytes + lr * 40;
#pragma unroll
            for (int cg = 0; cg < 10; ++cg) cp4_g2s(db + cg * 4, rb + cg * 512);
          }
        }
        if (lane < kGroup) desc[slot].pairs[lane] = cpair;
        __syncwarp();
        if (lane == 0) {
          desc[slot].kind = ck; desc[slot].item = cur; desc[slot].slab = sl;
          desc[slot].e = ce; desc[slot].ntok = cn; desc[slot].wg = cwg;
#ifdef MOE13_NOBULK
          mbar_arrive(&full[slot]);
#else
          if (src) { mbar_arrive_tx(&full[slot], kSlabBytes); bulk_g2s(dst, src, kSlabBytes, &full[slot]); }
          else mbar_arrive(&full[slot]);
#endif
        }
        cp_arrive_noinc(&full[slot]);
        if (++slot == kSlots) { slot = 0; ph ^= 1; }
        // next item's metadata, one dependent level per posted slab
        if (sl == 0 && ck != kDescStop) {
          nxt = __shfl_sync(0xffffffffu, nraw, 0);
          nk = kind_of(nxt); nwgi = wg_of(nxt, nk);
          if (nk == kDescGate || nk == kDescDown) { ne = meta[kMetaWgE + nwgi]; noff = meta[kMetaWgOff + nwgi]; nn = meta[kMetaWgCnt + nwgi]; }
        }
        if (sl == nslab - 1 && ck != kDescStop) { if (lane < nn) npair = meta[kMetaPairs + noff + lane]; }
      }
      if (ck == kDescStop) break;
      cur = nxt; ck = nk; cwg = nwgi; ce = ne; coff = noff; cn = nn; cpair = npair;
    }
    return;
  }

  // ---------------- consumers ----------------
  // No consumer-wide barrier except the FC1 split-K exchange; each warp reads its item metadata from the slot
  // descriptor. The FC1 epilogue rotates over the warps (item gi -> warp gi % 4); completion signals (release adds)
  // are deferred while the next slab is already present. On an item's last slab the next item's activation rows and
  // exponents are prefetched if its first slab is already in the ring.
  int slot = 0, ph = 0, gi = -1;
  float acc[4][4];
  float gs = 0.f;
  uint32_t bst[5][16];
  int my_pair[2] = {0, 0}, my_rmax[2] = {0, 0}, ntok = 0, e = 0, wg = 0;
  float my_wt[2] = {0.f, 0.f};
  int pend = 0, pend_ptr_off = 0, pend_n = 0, pend_tok = 0;
  const int g = lane >> 2, q = lane & 3;
  // prefetch state for the next item
  int pf_item = -1, pf_rowmax_g = 0, pf_aux[2] = {0, 0}, pf_mt[kGroup];
  BPre<5> pf;
  auto flush = [&]() {
    if (pend == 1) { if (lane == 0) red_release_add(&meta[kMetaGcnt + pend_ptr_off], 1); }
    else if (pend == 2) { if (lane < pend_n) red_release_add(&meta[kMetaTcnt + pend_tok], 1); }
    pend = 0;
  };
  // issue the next item's loads: FC1: x row of token g + x row max exponents; FC2 (if its FC1 is done): h row of
  // pair g, every token's h block exponents, router weights
  auto prefetch_next = [&](int cur_slot, int cur_ph) {
    const int ns = cur_slot + 1 == kSlots ? 0 : cur_slot + 1, nph = cur_slot + 1 == kSlots ? cur_ph ^ 1 : cur_ph;
    pf_item = -1;
    if (!__all_sync(0xffffffffu, mbar_test(&full[ns], nph))) return;
    const Desc& nd = desc[ns];
    if (nd.slab != 0) return;
    const int nn = nd.ntok;
    if (nd.kind == kDescGate) {
      const int rowg = g < nn ? nd.pairs[g] / kTopK : -1;
      b_fetch_raw<5, false>(xq, xe, kHidden, rowg, warp * 640, pf);
      pf_rowmax_g = rowg >= 0 ? xemax[rowg] : 0;
#pragma unroll
      for (int c = 0; c < 2; ++c) pf_aux[c] = 2 * q + c < nn ? xemax[nd.pairs[2 * q + c] / kTopK] : 0;
      pf_item = nd.item;
    } else if (nd.kind == kDescDown) {
      int ok = 0;
      if (lane == 0) ok = ld_acquire(&meta[kMetaGcnt + nd.wg]) >= kGateItems;
      if (!__shfl_sync(0xffffffffu, ok, 0)) return;
#pragma unroll
      for (int t = 0; t < kGroup; ++t)
        pf_mt[t] = (t < nn && lane < kHBlk) ? (int)__ldcg(reinterpret_cast<const signed char*>(hexp + nd.pairs[t] * kHBlk + lane)) : -126;
      b_fetch_raw<5, true>(h8, hexp, kInter, g < nn ? nd.pairs[g] : -1, 0, pf);
#pragma unroll
      for (int c = 0; c < 2; ++c) pf_aux[c] = 2 * q + c < nn ? __float_as_int(topk_w[nd.pairs[2 * q + c]]) : 0;
      pf_item = nd.item;
    }
  };
#ifdef MOE13_TRACE
  int ev = 0;
#endif
  for (;;) {
#ifdef MOE13_TRACE
    unsigned long long t_w = gtime();
#endif
    // a deferred signal is kept only while the next slab is already here; never block holding one (other CTAs'
    // waits may need it while this CTA's producer waits for their progress)
    if (pend && !__all_sync(0xffffffffu, mbar_test(&full[slot], ph))) flush();
    mbar_wait(&full[slot], ph);
#ifdef MOE13_TRACE
    unsigned long long t_r = gtime();
#endif
    const Desc& d = desc[slot];
    const int kind = d.kind, sl = d.slab, item = d.item;
    if (kind == kDescStop) break;
    const uint8_t* sw = ring + slot * kSlotBytes;
    const uint8_t* ssc = sw + kSlabBytes;
    const bool have_pf = sl == 0 && pf_item == item;
    if (sl == 0 && kind != kDescRed) {
      e = d.e; ntok = d.ntok; wg = d.wg;
#pragma unroll
      for (int c = 0; c < 2; ++c) my_pair[c] = d.pairs[2 * q + c];
    }
    if (kind == kDescGate) {
      const int tile = item % kGateItems;
      if (sl == 0) {
        ++gi;
        gs = g13[e];
        if (have_pf) {
          my_rmax[0] = pf_aux[0]; my_rmax[1] = pf_aux[1];
          b_convert<5>(pf, pf_rowmax_g, bst);
        } else {
          const int rowg = g < ntok ? d.pairs[g] / kTopK : -1;
          BPre<5> r;
          b_fetch_raw<5, false>(xq, xe, kHidden, rowg, warp * 640, r);
          const int rm = rowg >= 0 ? xemax[rowg] : 0;
#pragma unroll
          for (int c = 0; c < 2; ++c) my_rmax[c] = 2 * q + c < ntok ? xemax[my_pair[c] / kTopK] : 0;
          b_convert<5>(r, rm, bst);
        }
      }
      float a4[4] = {0.f, 0.f, 0.f, 0.f};
      slab_mma<5>(sw, kHidden / 2, ssc, kHidden / 16, 0, warp * 320, warp * 40, bst, a4);
      __syncwarp();
      if (lane == 0) mbar_arrive(&empty[slot]);
      flush();
#pragma unroll
      for (int rg = 0; rg < 4; ++rg)   // static indexing keeps acc in registers
        if (rg == sl) {
#pragma unroll
          for (int i = 0; i < 4; ++i) acc[rg][i] = a4[i];
        }
      if (sl == 3) {   // split-K: every warp's partials -> s_red[gi & 1]; the owner sums in fixed warp order
        prefetch_next(slot, ph);
        const int owner = gi % kConsumers;
        if (warp != owner) {
          const int rs = (warp - owner - 1) & 3;
#pragma unroll
          for (int rg = 0; rg < 4; ++rg)
#pragma unroll
            for (int i = 0; i < 4; ++i) s_red[rs][rg][i][lane] = acc[rg][i];
        }
        cbar();
        if (warp == owner) {   // fixed summation order: warp 0, 1, 2, 3
#pragma unroll
          for (int rg = 0; rg < 4; ++rg)
#pragma unroll
            for (int i = 0; i < 4; ++i) {
              float v[kConsumers];
#pragma unroll
              for (int w = 0; w < kConsumers; ++w) v[w] = w == owner ? acc[rg][i] : s_red[(w - owner - 1) & 3][rg][i][lane];
              acc[rg][i] = ((v[0] + v[1]) + v[2]) + v[3];
            }
        }
        cbar();   // s_red read before anyone writes the next item's partials
        if (warp == owner) {
          const int i0 = tile * 32;
#pragma unroll
          for (int c = 0; c < 2; ++c) {
            const int tok = 2 * q + c;
            const float sc = tok < ntok ? gs * pow2f(my_rmax[c]) : 0.f;
            float hv[4];   // intermediate i0 + g, i0 + g + 8, i0 + 16 + g, i0 + 24 + g
#pragma unroll
            for (int j = 0; j < 4; ++j) {
              const int rg = j >> 1, hh = j & 1;
              const float gv = acc[rg][2 * hh + c] * sc, uv = acc[rg + 2][2 * hh + c] * sc;
              hv[j] = gv / (1.f + __expf(-gv)) * uv;
            }
            float am = fmaxf(fmaxf(fabsf(hv[0]), fabsf(hv[1])), fmaxf(fabsf(hv[2]), fabsf(hv[3])));
            am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 4));
            am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 8));
            am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 16));
            const int ex = mx_exp(am);
            const float inv = pow2f(-ex);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
              const __nv_fp8_e4m3 qv(hv[j] * inv);
              s_h[warp][tok][(j >> 1) * 16 + (j & 1) * 8 + g] = qv.__x;
            }
            if (g == 0 && tok < ntok) hexp[my_pair[c] * kHBlk + i0 / 32] = (int8_t)ex;
          }
          __syncwarp();
          {   // 16 B per lane: token lane/2 (pair held by lane tok/2, i.e. q = tok/2, in my_pair[tok&1]), half lane&1
            const int tok = (lane >> 1) & 7, hf = lane & 1, src = tok >> 1;
            const int p0 = __shfl_sync(0xffffffffu, my_pair[0], src), p1 = __shfl_sync(0xffffffffu, my_pair[1], src);
            const int pair = (tok & 1) ? p1 : p0;
            if (lane < 2 * kGroup && tok < ntok)
              *reinterpret_cast<uint4*>(h8 + (size_t)pair * kInter + i0 + 16 * hf) =
                  *reinterpret_cast<const uint4*>(&s_h[warp][tok][16 * hf]);
          }
          __syncwarp();
          pend = 1; pend_ptr_off = wg;
        }
      }
    } else if (kind == kDescDown) {
      const int tile = (item - n_gate) % kDownItems13;
      if (sl == 0) {
        gs = g2[e];
        int rm_g = 0;
        if (have_pf) {
#pragma unroll
          for (int c = 0; c < 2; ++c) my_wt[c] = __int_as_float(pf_aux[c]);
#pragma unroll
          for (int t = 0; t < kGroup; ++t) {
            int m = pf_mt[t];
#pragma unroll
            for (int o = 16; o; o >>= 1) m = max(m, __shfl_xor_sync(0xffffffffu, m, o));
            if (t == g) rm_g = m;
            if (t == 2 * q) my_rmax[0] = m;
            if (t == 2 * q + 1) my_rmax[1] = m;
          }
          b_convert<5>(pf, rm_g, bst);
        } else {
          flush();   // never wait on a counter while holding a deferred signal
          if (lane == 0) wait_geq(&meta[kMetaGcnt + wg], kGateItems);
          __syncwarp();
#pragma unroll
          for (int c = 0; c < 2; ++c) my_wt[c] = 2 * q + c < ntok ? topk_w[my_pair[c]] : 0.f;
          int mt[kGroup];
#pragma unroll
          for (int t = 0; t < kGroup; ++t)   // all loads first, then the reductions
            mt[t] = (t < ntok && lane < kHBlk) ? (int)__ldcg(reinterpret_cast<const signed char*>(hexp + d.pairs[t] * kHBlk + lane)) : -126;
          BPre<5> r;
          b_fetch_raw<5, true>(h8, hexp, kInter, g < ntok ? d.pairs[g] : -1, 0, r);
#pragma unroll
          for (int t = 0; t < kGroup; ++t) {
            int m = mt[t];
#pragma unroll
            for (int o = 16; o; o >>= 1) m = max(m, __shfl_xor_sync(0xffffffffu, m, o));
            if (t == g) rm_g = m;
            if (t == 2 * q) my_rmax[0] = m;
            if (t == 2 * q + 1) my_rmax[1] = m;
          }
          b_convert<5>(r, rm_g, bst);
        }
      }
      float a4[4] = {0.f, 0.f, 0.f, 0.f};
      slab_mma<5>(sw, kInter / 2, ssc, kInter / 16, 16 * warp, 0, 0, bst, a4);
      __syncwarp();
      if (lane == 0) mbar_arrive(&empty[slot]);
      flush();
      if (sl == 1) prefetch_next(slot, ph);
      const int rbase = tile * 128 + 64 * sl + 16 * warp + g;
#pragma unroll
      for (int c = 0; c < 2; ++c) {
        const int tok = 2 * q + c;
        if (tok >= ntok) continue;
        const float sc = gs * pow2f(my_rmax[c]) * my_wt[c];
        float* dst = y + (size_t)my_pair[c] * kHidden + rbase;
        dst[0] = a4[c] * sc;
        dst[8] = a4[2 + c] * sc;
      }
      if (sl == 1) {
        __syncwarp();
        const int src = (lane & 7) >> 1;   // pair of token `lane` is held by lane lane/2 (q = lane/2) in my_pair[lane&1]
        const int p0 = __shfl_sync(0xffffffffu, my_pair[0], src), p1 = __shfl_sync(0xffffffffu, my_pair[1], src);
        pend = 2; pend_n = ntok; pend_tok = ((lane & 1) ? p1 : p0) / kTopK;
      }
    } else {   // reduce: token t, 640 columns; each warp its own 160
      __syncwarp();
      if (lane == 0) mbar_arrive(&empty[slot]);
      flush();   // (before the counter wait below)
      const int r = item - n_gate - n_down, t = r / kRedItems, c0 = (r % kRedItems) * (kHidden / kRedItems);
#ifndef MOE13_STREAM_ONLY
      if (lane == 0) wait_geq(&meta[kMetaTcnt + t], kTopK * kDownItems13 * kConsumers);
      __syncwarp();
#endif
      for (int c = c0 + (warp * 32 + lane) * 4; c < c0 + kHidden / kRedItems; c += kConsumers * 32 * 4) {
        float4 sum = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
        for (int j = 0; j < kTopK; ++j) {
          const float4 v = __ldcg(reinterpret_cast<const float4*>(y + (size_t)(t * kTopK + j) * kHidden + c));
          sum.x += v.x; sum.y += v.y; sum.z += v.z; sum.w += v.w;
        }
        __nv_bfloat162* o = reinterpret_cast<__nv_bfloat162*>(out + (size_t)t * kHidden + c);
        o[0] = __floats2bfloat162_rn(sum.x, sum.y);
        o[1] = __floats2bfloat162_rn(sum.z, sum.w);
      }
    }
#ifdef MOE13_TRACE
    if (tid == 0 && ev < 64) {
      unsigned long long* tr = g_trace + ((size_t)blockIdx.x * 64 + ev) * 4;
      tr[0] = t_w; tr[1] = t_r; tr[2] = gtime(); tr[3] = (unsigned long long)(kind * 16 + sl);
    }
    ++ev;
#endif
    if (++slot == kSlots) { slot = 0; ph ^= 1; }
  }
  flush();
}

}  // namespace

int64_t workspace_bytes() {
  return (int64_t)kMetaWords * 4 + (int64_t)kMaxTok * (kHidden + kXBlk + 4) + (int64_t)kMaxPairs * (kInter + kHBlk) +
         (int64_t)kMaxPairs * kHidden * 4 + 8 * 256;
}

// ws: one uint8 buffer of workspace_bytes(); gate_first: w13 rows = [gate; up] (else [up; gate])
void moe_w4a8(torch::Tensor x, torch::Tensor topk_ids, torch::Tensor topk_w, torch::Tensor w13, torch::Tensor s13,
              torch::Tensor g13, torch::Tensor w2, torch::Tensor s2, torch::Tensor g2, torch::Tensor ws,
              torch::Tensor out, bool gate_first) {
  const int T = x.size(0);
  TORCH_CHECK(T >= 1 && T <= kMaxTok, "moe_w4a8: 1..64 tokens, got ", T);
  TORCH_CHECK(x.size(1) == kHidden && topk_ids.size(1) == kTopK, "moe_w4a8: shape");
  TORCH_CHECK(w13.size(1) == 2 * kInter && w13.size(2) == kHidden / 2 && w2.size(1) == kHidden &&
                  w2.size(2) == kInter / 2, "moe_w4a8: weight shape");
  TORCH_CHECK(topk_ids.dtype() == torch::kInt32 && topk_w.dtype() == torch::kFloat32, "moe_w4a8: topk dtypes");
  TORCH_CHECK(ws.numel() >= workspace_bytes(), "moe_w4a8: workspace too small");
  auto st = c10::cuda::getCurrentCUDAStream();
  uint8_t* base = reinterpret_cast<uint8_t*>(ws.data_ptr());
  int* meta = reinterpret_cast<int*>(base);
  auto al = [](uint8_t* q) { return reinterpret_cast<uint8_t*>(((uintptr_t)q + 255) & ~(uintptr_t)255); };
  uint8_t* p = al(base + (size_t)kMetaWords * 4);
  int* xemax = reinterpret_cast<int*>(p); p = al(p + kMaxTok * 4);
  uint8_t* xq = p; p = al(p + kMaxTok * kHidden);
  int8_t* xe = reinterpret_cast<int8_t*>(p); p = al(p + kMaxTok * kXBlk);
  uint8_t* h8 = p; p = al(p + kMaxPairs * kInter);
  int8_t* hexp = reinterpret_cast<int8_t*>(p); p = al(p + kMaxPairs * kHBlk);
  float* y = reinterpret_cast<float*>(p);
  TORCH_CHECK(w13.size(0) == kExperts, "moe_w4a8: 512 experts");
  prep_kernel<<<1 + T, 256, 0, st>>>(reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), topk_ids.data_ptr<int>(),
                                     T, meta, xq, xe, xemax);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  // producer/consumer kernel from 4 tokens up; below that the per-SM producer latency dominates and the
  // all-warps persistent kernel is faster
  static const int v13_min = getenv("MOE_W4A8_V13") ? (atoi(getenv("MOE_W4A8_V13")) ? 1 : 1 << 30) : 4;
  if (T >= v13_min) {
    static int sms = 0;
    constexpr int smem = kSlots * kSlotBytes;
    if (!sms) {
      int dev = 0;
      C10_CUDA_CHECK(cudaGetDevice(&dev));
      C10_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
      C10_CUDA_CHECK(cudaFuncSetAttribute(moe13_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    }
    moe13_kernel<<<sms, kThreads13, smem, st>>>(
        meta, T, xq, xe, xemax, topk_w.data_ptr<float>(), reinterpret_cast<const uint8_t*>(w13.data_ptr()),
        reinterpret_cast<const uint8_t*>(s13.data_ptr()), g13.data_ptr<float>(),
        reinterpret_cast<const uint8_t*>(w2.data_ptr()), reinterpret_cast<const uint8_t*>(s2.data_ptr()),
        g2.data_ptr<float>(), h8, hexp, y, reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), gate_first ? 0 : kInter);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }
  static int grid = 0;
  if (!grid) {
    int per_sm = 0, sms = 0, dev = 0;
    C10_CUDA_CHECK(cudaGetDevice(&dev));
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, moe_kernel, kThreads, 0));
    grid = sms * per_sm;
    if (const char* g = getenv("MOE_W4A8_GRID")) grid = atoi(g);   // test knob
  }
  moe_kernel<<<grid, kThreads, 0, st>>>(
      meta, T, xq, xe, xemax, topk_w.data_ptr<float>(), reinterpret_cast<const uint8_t*>(w13.data_ptr()),
      reinterpret_cast<const uint8_t*>(s13.data_ptr()), g13.data_ptr<float>(),
      reinterpret_cast<const uint8_t*>(w2.data_ptr()), reinterpret_cast<const uint8_t*>(s2.data_ptr()),
      g2.data_ptr<float>(), h8, hexp, y, reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), gate_first ? 0 : kInter);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

#ifdef MOE13_TRACE
void read_trace(torch::Tensor out) { C10_CUDA_CHECK(cudaMemcpyFromSymbol(out.data_ptr(), g_trace, sizeof(g_trace))); }
void read_trace2(torch::Tensor out) { C10_CUDA_CHECK(cudaMemcpyFromSymbol(out.data_ptr(), g_trace2, sizeof(g_trace2))); }
#endif
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
#ifdef MOE13_TRACE
  m.def("read_trace", &read_trace);
  m.def("read_trace2", &read_trace2);
#endif
  m.def("moe_w4a8", &moe_w4a8);
  m.def("workspace_bytes", &workspace_bytes);
}
