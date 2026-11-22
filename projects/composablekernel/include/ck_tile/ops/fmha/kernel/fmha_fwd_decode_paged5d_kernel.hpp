// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/core/arch/amd_cluster_load.hpp"
#include "ck_tile/core/arch/amd_tdm_descriptor.hpp"

// gfx125 fp8 decode attention over a paged KV cache, cache pages of page_size = 16, 64 or 128
// tokens, in one of three layouts (FmhaDecodePagedLayout):
//   shuffle_5d (sglang / aiter vectorized_5d):
//     K [num_blocks][nhead_k][hdim / 16][page_size tokens][16 B]
//     V [num_blocks][nhead_k][page_size / 16][hdim][16 tokens]
//   hnd: K and V [num_blocks][nhead_k][page_size tokens][hdim]
//   nhd: K and V [num_blocks][page_size tokens][nhead_k][hdim]
// qlen query tokens per sequence (MTP: token i sees the keys before ctx - qlen + 1 + i), hdim 64,
// 128 or 256, fp8 e4m3 K/V with per-tensor descales, bf16 or fp16 Q (quantized per row in the
// kernel) and output of the same type. The query rows of one kv head (qlen tokens x its q heads,
// token-major) are the columns of S^T = K Q^T, 16 per tile: a workgroup computes kMTiles tiles from
// each K/V page it reads, and further rows go to further row blocks of the grid. Below, a page is
// 16 tokens; a cache page holds page_size / 16.
//
// Grid (kNumSplits * groups, nhead_k * row blocks, batch). Each kNumSplits consecutive splits are
// one workgroup cluster; each stores O / l in bf16 and (m, l), and after a cluster barrier every
// split merges its share of the rows (with several groups, see MergeGroups). The grid is sized for
// max_seqlen_k; each (batch, kv head) spreads its own key range over it, and clusters left
// without pages exit at once.
//
// K and V pages are copied to LDS by the TDM. In shuffle_5d the LDS bytes are WMMA operands as
// they are (S^T = K Q^T and O^T += V^T P^T); in hnd / nhd a page is kept as token rows (padded
// against bank conflicts), K rows are read as they are and V^T with transposing LDS loads.

namespace ck_tile {

enum class FmhaDecodePagedLayout : index_t
{
    shuffle_5d = 0,
    hnd        = 1,
    nhd        = 2,
};

struct FmhaDecodePaged5dHostArgs
{
    const void* q_ptr;           // [batch][qlen][nhead_q][hdim] bf16, or fp16 with qo_fp16
    const void* k_ptr;           // fp8, in layout
    const void* v_ptr;           // fp8, in layout
    const int32_t* block_table;  // [batch][max_blocks_per_seq]
    const int32_t* seqlen_k_ptr; // [batch]
    const float* k_descale_ptr;  // one float
    const float* v_descale_ptr;  // one float
    void* o_ptr;                 // [batch][qlen][nhead_q][hdim], the type of Q
    void* workspace_ptr;         // fmha_decode_paged5d_workspace_bytes, no initialization needed
    // fmha_decode_paged5d_num_counters int32; zero them once, every launch leaves them zero, so
    // launches of any shape can share them as long as they do not run concurrently
    int32_t* counters_ptr;
    index_t batch;
    index_t nhead_q;
    index_t nhead_k;
    index_t max_blocks_per_seq;
    index_t page_size;       // tokens per cache page: 16, 64 or 128
    index_t window;          // keys attended, the query's own included; 0 = no window
    index_t pages_per_split; // lengthened in the kernel if the grid would not cover a range
    index_t num_groups;      // clusters per (batch, kv head)
    float scale_s;
    bool qo_fp16                 = false; // Q and O are fp16 instead of bf16
    FmhaDecodePagedLayout layout = FmhaDecodePagedLayout::shuffle_5d;
    index_t hdim                 = 128; // 64, 128 or 256
    index_t qlen                 = 1;   // query tokens per sequence, the last ones of the context
    // optional: per q head logit that joins every softmax denominator (unscaled, with no value;
    // -inf: none; fp32, bf16 or fp16 by sink_dtype 0, 1, 2), and fp32 [batch][qlen][nhead_q]
    // output of the log-sum-exp of the scaled scores (natural log, sink included)
    const void* sink_ptr = nullptr;
    index_t sink_dtype   = 0;
    float* lse_ptr       = nullptr;
};

// num_splits workgroups per cluster, num_groups clusters per (batch, kv head)
struct FmhaDecodePaged5dConfig
{
    index_t num_splits;
    index_t pages_per_split;
    bool wave_split;
    index_t num_groups = 1;
    // 16-row tiles per workgroup (1 or 2) and row blocks per (batch, kv head)
    index_t m_tiles    = 1;
    index_t row_blocks = 1;
};

// The most 16-token pages the key range of any context up to max_seqlen_k covers: the windows of
// qlen query tokens span W + qlen - 1 keys, which touch up to ceil(that / 16) + 1 pages, and with
// larger cache pages the range starts on a 4-page boundary (see FmhaDecodePaged5dKernel::Run), up
// to 3 pages earlier.
CK_TILE_HOST index_t fmha_decode_paged5d_max_span(index_t max_seqlen_k,
                                                  index_t window,
                                                  index_t page_size = 16,
                                                  index_t qlen      = 1)
{
    index_t span = (max_seqlen_k + 15) / 16;
    if(window > 0)
        span =
            std::min<index_t>(span, (window + qlen - 1 + 15) / 16 + 1 + (page_size > 16 ? 3 : 0));
    return std::max<index_t>(span, 1);
}

// Up to 240 pages: one cluster of cooperative splits, as many as 4-page steps allow (at most 15),
// at least 16 pages each once the grid would exceed about 640 workgroups. Up to 960 pages: one
// wave-split cluster while the grid has 64 to 512 workgroups. Longer: at hdim 128, several
// clusters (groups) of the shortest 8-, 12- or 16-page cooperative splits within 512 workgroups;
// else wave-split splits for about 384 workgroups. The wave-split grid limits scale with
// 128 / hdim: a wave-split workgroup's LDS (and so its residency) scales with hdim. rows query
// rows per kv head (qlen x q heads per kv head) take m_tiles (1 or 2) tiles of 16 per workgroup and
// as many row blocks as needed. A second tile per workgroup reads K/V once for twice the rows but
// serializes their compute, so 2 tiles (m_tiles = 0) only once the (tile, page) tasks or the grid
// of single tiles are large.
CK_TILE_HOST FmhaDecodePaged5dConfig fmha_decode_paged5d_config(index_t batch,
                                                                index_t nhead_k,
                                                                index_t max_seqlen_k,
                                                                index_t window,
                                                                index_t page_size = 16,
                                                                index_t hdim      = 128,
                                                                index_t rows      = 1,
                                                                index_t qlen      = 1,
                                                                index_t m_tiles   = 0)
{
    const index_t span  = fmha_decode_paged5d_max_span(max_seqlen_k, window, page_size, qlen);
    const auto round_up = [](index_t x, index_t m) { return (x + m - 1) / m * m; };
    const auto ceil_div = [](index_t x, index_t y) { return (x + y - 1) / y; };
    if(m_tiles == 0)
    {
        const auto one = fmha_decode_paged5d_config(
            batch, nhead_k, max_seqlen_k, window, page_size, hdim, rows, qlen, 1);
        const long_index_t tasks = long_index_t{batch} * nhead_k * span * one.row_blocks;
        const long_index_t wgs =
            long_index_t{batch} * nhead_k * one.row_blocks * one.num_splits * one.num_groups;
        if(rows <= 16 || (tasks < 49152 && wgs < 2048))
            return one;
        m_tiles = 2;
    }
    const index_t row_blocks = ceil_div(rows, 16 * m_tiles);
    const index_t heads      = batch * nhead_k * row_blocks;
    const auto make          = [&](index_t pps, bool wave_split) {
        const index_t total  = ceil_div(span, pps);
        const index_t groups = ceil_div(total, 15);
        return FmhaDecodePaged5dConfig{
            ceil_div(total, groups), pps, wave_split, groups, m_tiles, row_blocks};
    };
    if(span <= 15 * 16)
    {
        index_t pps = round_up(ceil_div(span, std::min<index_t>(15, ceil_div(span, 4))), 4);
        if(heads * ceil_div(span, pps) > 640)
            pps = std::max<index_t>(pps, 16);
        return make(pps, false);
    }
    const index_t wave_pps = round_up(ceil_div(span, 15), 16);
    const index_t wave_wgs = heads * ceil_div(span, wave_pps);
    if(span <= 15 * 64 && wave_wgs <= 512 * 128 / hdim && (span <= 512 || wave_wgs >= 64))
        return make(wave_pps, true);
    if(hdim == 128)
        for(index_t pps : {8, 12, 16})
            if(ceil_div(span, pps) <= 192 && heads * ceil_div(span, pps) <= 512)
                return make(pps, false);
    return make(std::max<index_t>(32, round_up(ceil_div(span * heads * hdim, 384 * 128), 16)),
                true);
}

// block_rows: query rows per (batch, kv head) in whole row blocks, m_tiles * 16 * row_blocks
CK_TILE_HOST size_t fmha_decode_paged5d_workspace_bytes(index_t batch,
                                                        index_t nhead_k,
                                                        index_t num_splits,
                                                        index_t num_groups = 1,
                                                        index_t hdim       = 128,
                                                        index_t block_rows = 16)
{
    // per split: bf16 O / l [batch][nhead_k][block_rows / rows per block][groups][splits][rows per
    // block][hdim] and fp32 (m, l) [..][2]; per group the same with one split
    const size_t heads = static_cast<size_t>(batch) * nhead_k;
    const size_t rows  = heads * num_groups * num_splits * block_rows;
    const size_t grows = num_groups > 1 ? heads * num_groups * block_rows : 0;
    return (rows + grows) * (static_cast<size_t>(hdim) * 2 + 2 * sizeof(float));
}

// one per (batch, kv head, row block) and merging split, for any configuration with at most
// row_blocks row blocks
CK_TILE_HOST size_t fmha_decode_paged5d_num_counters(index_t batch,
                                                     index_t nhead_k,
                                                     index_t row_blocks = 1)
{
    return static_cast<size_t>(batch) * nhead_k * row_blocks * 15;
}

template <bool kWaveSplit_,
          index_t kNumSplits_,
          bool kGroups_                  = false,
          index_t kPageSize_             = 16,
          bool kQoFp16_                  = false,
          FmhaDecodePagedLayout kLayout_ = FmhaDecodePagedLayout::shuffle_5d,
          index_t kHdim_                 = 128,
          index_t kMTiles_               = 1>
struct FmhaDecodePaged5dKernel
{
    static constexpr FmhaDecodePagedLayout kLayout = kLayout_;
    // hnd / nhd: pages as token rows in LDS
    static constexpr bool kRowMajor     = kLayout != FmhaDecodePagedLayout::shuffle_5d;
    static constexpr bool kQoFp16       = kQoFp16_; // Q and O are fp16 instead of bf16
    static constexpr bool kWaveSplit    = kWaveSplit_;
    static constexpr index_t kNumSplits = kNumSplits_;
    static constexpr index_t kPageSize  = kPageSize_; // tokens per cache page
    // several clusters per (batch, kv head), merged by MergeGroups
    static constexpr bool kGroups       = kGroups_;
    static constexpr index_t kBlockSize = 128;
    static constexpr index_t kNumWaves  = 4;
    static constexpr index_t kHdim      = kHdim_;
    static_assert(kHdim == 64 || kHdim == 128 || kHdim == 256);
    // 16-byte head-dim chunks of a K token row; QK as kQkSteps WMMAs over kQkChunks chunks each
    // (16x16x128, or 16x16x64 at hdim 64); 16-wide O^T blocks
    static constexpr index_t kChunks    = kHdim / 16;
    static constexpr index_t kQkChunks  = kHdim >= 128 ? 8 : 4;
    static constexpr index_t kQkSteps   = kChunks / kQkChunks;
    static constexpr index_t kOutBlocks = kHdim / 16;
    // threads per row in the split merges, two head dims each
    static constexpr index_t kMergeTpr = kHdim / 2;
    static constexpr index_t kPage     = 16;
    static constexpr index_t kRows     = 16;
    // P scaled by 448 (the row max lands on e4m3's largest normal): exp2(s * scl + log2(448) - m)
    static constexpr float kLog2P = 8.807354922057604f;
    static constexpr float kLog2e = 1.4426950408889634f;
    static constexpr float kLn2   = 0.6931471805599453f;
    // 16-row tiles per workgroup
    static constexpr index_t kMTiles = kMTiles_;
    static constexpr index_t kMRows  = kRows * kMTiles;
    static_assert(kMTiles == 1 || kMTiles == 2);
    static constexpr index_t kPageBytes = kPage * kHdim;
    // pages per wave per step: one (cooperative) or four (wave split)
    static constexpr index_t kWavePages = kWaveSplit ? 4 : 1;
    static constexpr index_t kStages    = kWaveSplit ? 2 : 3;
    // 16-token pages per cache page
    static constexpr index_t kSubPages = kPageSize / kPage;
    static_assert(1 <= kNumSplits && kNumSplits <= 15);
    static_assert(kPageSize == 16 || kPageSize == 64 || kPageSize == 128);
    // LDS bytes of a token row (hnd / nhd): hdim plus 16 bytes of padding against bank conflicts
    static constexpr index_t kRowPitch = kHdim + 16;
    // LDS bytes of one page
    static constexpr index_t kPageBuf = kRowMajor ? kPage * kRowPitch : kPageBytes;
    // shuffle_5d LDS K of a wave-split step: four page-16 images, or with larger cache pages the
    // 64 tokens as they lie in the cache ([hdim / 16][64 tokens][16 B]). Offsets of a page, a
    // token and a 16-byte head-dim chunk.
    static constexpr index_t kKPageStride =
        kRowMajor ? kPageBuf : (kWaveSplit && kSubPages > 1 ? 256 : kPageBytes);
    static constexpr index_t kKTokStride = kRowMajor ? kRowPitch : 16;
    static constexpr index_t kKChunkStride =
        kRowMajor ? 16 : (kWaveSplit && kSubPages > 1 ? 1024 : 256);
    // TDM instructions per step and wave (the waits count them): hnd / nhd copy each page of a
    // page-16 wave-split step separately
    static constexpr index_t kTdmPerStep = kRowMajor ? (kWaveSplit && kSubPages == 1 ? 8 : 2)
                                                     : (kSubPages == 1 ? 2 : (kChunks + 7) / 8 + 1);

    struct Kargs
    {
        const uint16_t* q;
        const uint8_t* k;
        const uint8_t* v;
        const int32_t* block_table;
        const int32_t* seqlen_k;
        const float* k_descale;
        const float* v_descale;
        uint16_t* o;
        uint16_t* ws_o;
        float* ws_ml;
        index_t nhead_q;
        index_t nhead_k;
        index_t gqa; // nhead_q / nhead_k, so the kernel does not divide
        index_t max_blocks_per_seq;
        index_t window;
        index_t qlen; // next to window: the key range needs both
        index_t pages_per_split;
        index_t num_groups; // read with kGroups only
        // ceil(2^32 / (kNumSplits * num_groups)), for kGroups: the high half of a product with it
        // divides by the split count
        uint32_t splits_rcp;
        // ceil(2^31 / x) for Div (exact also for x = 1) by gqa and row_blocks
        uint32_t gqa_rcp;
        uint32_t row_blocks_rcp;
        index_t rows;       // query rows per kv head, qlen * gqa
        index_t row_blocks; // ceil(rows / kMRows), along y with the kv heads
        float scale_s;
        const void* sink; // nullptr: none
        float* lse;       // nullptr: not stored
        index_t sink_dtype;
        // used with kGroups only; last, so that the arguments the prologue needs come first
        uint16_t* gws_o;
        float* gws_ml;
        int32_t* counters; // per (batch, kv head, row block)
    };

    CK_TILE_HOST static uint32_t rcp(index_t x)
    {
        return static_cast<uint32_t>(((uint64_t{1} << 31) + x - 1) / x);
    }
    CK_TILE_HOST static index_t RowBlocks(const FmhaDecodePaged5dHostArgs& h)
    {
        return (h.nhead_q / h.nhead_k * h.qlen + kMRows - 1) / kMRows;
    }

    CK_TILE_HOST static Kargs MakeKargs(const FmhaDecodePaged5dHostArgs& h)
    {
        const size_t heads = static_cast<size_t>(h.batch) * h.nhead_k * RowBlocks(h);
        const size_t rows  = heads * h.num_groups * kNumSplits * kMRows;
        const size_t grows = heads * h.num_groups * kMRows;
        auto* ws           = static_cast<uint8_t*>(h.workspace_ptr);
        auto* gws          = ws == nullptr ? nullptr : ws + rows * (kHdim * 2 + 8);
        return {static_cast<const uint16_t*>(h.q_ptr),
                static_cast<const uint8_t*>(h.k_ptr),
                static_cast<const uint8_t*>(h.v_ptr),
                h.block_table,
                h.seqlen_k_ptr,
                h.k_descale_ptr,
                h.v_descale_ptr,
                static_cast<uint16_t*>(h.o_ptr),
                reinterpret_cast<uint16_t*>(ws),
                ws == nullptr ? nullptr : reinterpret_cast<float*>(ws + rows * kHdim * 2),
                h.nhead_q,
                h.nhead_k,
                h.nhead_q / h.nhead_k,
                h.max_blocks_per_seq,
                h.window,
                h.qlen,
                h.pages_per_split,
                h.num_groups,
                static_cast<uint32_t>(((uint64_t{1} << 32) + kNumSplits * h.num_groups - 1) /
                                      (kNumSplits * h.num_groups)),
                rcp(h.nhead_q / h.nhead_k),
                rcp(RowBlocks(h)),
                h.nhead_q / h.nhead_k * h.qlen,
                RowBlocks(h),
                h.scale_s,
                h.sink_ptr,
                h.lse_ptr,
                h.sink_dtype,
                reinterpret_cast<uint16_t*>(gws),
                gws == nullptr ? nullptr : reinterpret_cast<float*>(gws + grows * kHdim * 2),
                h.counters_ptr};
    }

    CK_TILE_HOST static dim3 GridSize(const FmhaDecodePaged5dHostArgs& h)
    {
        return dim3(kNumSplits * h.num_groups, h.nhead_k * RowBlocks(h), h.batch);
    }

    CK_TILE_HOST static constexpr dim3 BlockSize() { return dim3(kBlockSize); }

#if defined(__gfx125__)
    using v4i  = int32_t __attribute__((ext_vector_type(4)));
    using v8i  = int32_t __attribute__((ext_vector_type(8)));
    using v16i = int32_t __attribute__((ext_vector_type(16)));
    using v8f  = float __attribute__((ext_vector_type(8)));

    struct alignas(16) SmemCoop
    {
        alignas(16) uint8_t k[kStages][kNumWaves][kPageBuf];
        alignas(16) uint8_t v[kStages][kNumWaves][kPageBuf];
        alignas(16) uint8_t p[kMTiles][kRows][kNumWaves * kPage];
        alignas(16) float red[2][kNumWaves][kMRows]; // per-wave row max / row sum
    };

    struct alignas(16) SmemWaveSplit
    {
        union
        {
            struct
            {
                alignas(16) uint8_t k[kStages][kNumWaves][kWavePages][kPageBuf];
                alignas(16) uint8_t v[kStages][kNumWaves][kWavePages][kPageBuf];
            } kv;
            // the waves' O for the intra-workgroup merge; padded rows keep the column-wise
            // stores of 16 rows off one bank
            alignas(16) float o[kNumWaves][kRows][kHdim + 4];
        };
        alignas(16) float ml[2][kNumWaves][kRows];
    };

    template <typename T, typename P>
    CK_TILE_DEVICE static T* ptr_as(P* p)
    {
        return static_cast<T*>(static_cast<void*>(p));
    }
    template <typename T, typename P>
    CK_TILE_DEVICE static const T* ptr_as(const P* p)
    {
        return static_cast<const T*>(static_cast<const void*>(p));
    }
    CK_TILE_DEVICE static bool is_neg_inf(float x) { return __builtin_isinf(x) && x < 0.f; }
    CK_TILE_DEVICE static float bf16_to_f32(uint32_t x) { return __uint_as_float(x << 16); }
    CK_TILE_DEVICE static uint32_t f32_to_bf16(float f)
    {
        const uint32_t u = __float_as_uint(f);
        return (u + 0x7fff + ((u >> 16) & 1)) >> 16;
    }
    CK_TILE_DEVICE static uint32_t pack_bf16(float a, float b)
    {
        return f32_to_bf16(a) | (f32_to_bf16(b) << 16);
    }
    CK_TILE_DEVICE static float fp16_to_f32(uint32_t x)
    {
        return static_cast<float>(__builtin_bit_cast(_Float16, static_cast<uint16_t>(x)));
    }
    CK_TILE_DEVICE static uint32_t pack_fp16(float a, float b)
    {
        return __builtin_bit_cast(uint16_t, static_cast<_Float16>(a)) |
               (static_cast<uint32_t>(__builtin_bit_cast(uint16_t, static_cast<_Float16>(b)))
                << 16);
    }
    CK_TILE_DEVICE static float exp2(float x) { return __builtin_amdgcn_exp2f(x); }

    // TDM gather of kCount rows of kRowBytes (row indices up to 2^31) to consecutive LDS rows
    template <index_t kRowBytes, index_t kCount>
    CK_TILE_DEVICE static void
    tdm_rows(const uint8_t* base, uint8_t* lds, const int32_t (&rows)[kCount])
    {
        static_assert(kCount <= 8 && kRowBytes % 8 == 0);
        TDM_GROUP0 g0{static_cast<uint32_t>(reinterpret_cast<uintptr_t>(to_lds(lds))),
                      reinterpret_cast<uintptr_t>(base),
                      static_cast<uint32_t>(TDMGatherIndexSize::Row32bit_Index),
                      1};
        TDM_GROUP1 g1;
        g1.data_size = 3; // 8-byte units
        g1.tensorDim0(kRowBytes / 8);
        g1.tensorDim1(0x7fffffff);
        g1.tensorDim0Stride(kRowBytes / 8);
        g1.tile_dim0 = kRowBytes / 8;
        g1.tile_dim1 = kCount; // row indices
        TDM_GROUP2 g2;
        TDM_GROUP3 g3;
        for(index_t i = 0; i < kCount; ++i)
        {
            if(i < 4)
                g2.mode1.setRowIndex32(i, static_cast<uint32_t>(rows[i]));
            else
                g3.mode1.setRowIndex32(i - 4, static_cast<uint32_t>(rows[i]));
        }
        __builtin_amdgcn_tensor_load_to_lds(
            bit_cast<uint32x4_t>(amd_wave_read_first_lane(g0.bitfield)),
            amd_wave_read_first_lane(g1.bitfield),
            amd_wave_read_first_lane(g2.bitfield),
            amd_wave_read_first_lane(g3.bitfield),
            int32x8_t{},
            0);
    }
    // TDM copy of kRows rows of kHdim bytes, pitch bytes apart in memory, to LDS rows of kRowPitch
    template <index_t kTileRows>
    CK_TILE_DEVICE static void tdm_tile(const uint8_t* base, uint8_t* lds, long_index_t pitch)
    {
        static_assert(kRowPitch == kHdim + 16);
        TDM_GROUP0 g0{static_cast<uint32_t>(reinterpret_cast<uintptr_t>(to_lds(lds))),
                      reinterpret_cast<uintptr_t>(base),
                      0,
                      0};
        TDM_GROUP1 g1;
        g1.data_size = 3; // 8-byte units
        g1.tensorDim0(kHdim / 8);
        g1.tensorDim1(kTileRows);
        g1.tensorDim0Stride(static_cast<uint64_t>(pitch) / 8);
        g1.tile_dim0  = kHdim / 8;
        g1.tile_dim1  = kTileRows;
        g1.pad_enable = 1;
        // every row of kHdim bytes: 2^(pad_interval + 1) dwords
        g1.pad_interval = kHdim == 64 ? 3 : (kHdim == 128 ? 4 : 5);
        g1.pad_amount   = 3; // 4 dwords = 16 bytes
        __builtin_amdgcn_tensor_load_to_lds(
            bit_cast<uint32x4_t>(amd_wave_read_first_lane(g0.bitfield)),
            amd_wave_read_first_lane(g1.bitfield),
            int32x4_t{},
            int32x4_t{},
            int32x8_t{},
            0);
    }
    // V^T operand of head-dim rows d0 + (lane % 16) from token rows in LDS: load q gives the
    // 8 tokens row(q, j), j = 0..7, of this lane's row. ds_load_tr8_b64 transposes, per half
    // wave, the 8-byte loads of its 16 lanes: output lane i < 8 gets byte i of the loads of
    // lanes {0-3, 8-11}, lane i >= 8 byte i - 8 of lanes {4-7, 12-15}.
    template <typename RowAddr>
    CK_TILE_DEVICE static v8i v_tr8(index_t lane16, RowAddr row_addr)
    {
        using v2i          = int32_t __attribute__((ext_vector_type(2)));
        const index_t j    = (lane16 & 3) + ((lane16 >> 3) << 2); // token of this lane's load
        const index_t dcol = ((lane16 >> 2) & 1) * 8;
        v8i r{};
        for(index_t q = 0; q < 4; ++q)
        {
            const v2i x =
                __builtin_amdgcn_ds_load_tr8_b64_v2i32(to_lds(ptr_as<v2i>(row_addr(q, j) + dcol)));
            r[2 * q]     = x.x;
            r[2 * q + 1] = x.y;
        }
        return r;
    }
    // index of query row r (token qi = r / gqa) of kv head g in [batch][qlen][nhead_q]
    CK_TILE_DEVICE static long_index_t
    RowIndex(const Kargs& a, index_t b, index_t g, index_t r, index_t qi)
    {
        return (static_cast<long_index_t>(b) * a.qlen + qi) * a.nhead_q + g * a.gqa +
               (r - qi * a.gqa);
    }
    // its offset in Q and O
    CK_TILE_DEVICE static long_index_t
    RowOffset(const Kargs& a, index_t b, index_t g, index_t r, index_t qi)
    {
        return RowIndex(a, b, g, r, qi) * kHdim;
    }
    // y / x for y * x < 2^30, given rcp = ceil(2^31 / x)
    CK_TILE_DEVICE static index_t Div(index_t y, uint32_t rcp)
    {
        return __umulhi(2 * static_cast<uint32_t>(y), rcp);
    }
    CK_TILE_DEVICE static index_t QueryToken(const Kargs& a, index_t r)
    {
        return Div(r, a.gqa_rcp);
    }
    CK_TILE_DEVICE static long_index_t RowOffset(const Kargs& a, index_t b, index_t g, index_t r)
    {
        return RowOffset(a, b, g, r, QueryToken(a, r));
    }
    // sink of query row r of kv head g in log2 units; -inf without sinks
    CK_TILE_DEVICE static float SinkLog2(const Kargs& a, index_t g, index_t r)
    {
        if(a.sink == nullptr)
            return -numeric<float>::infinity();
        const index_t hq = g * a.gqa + r - QueryToken(a, r) * a.gqa;
        const float x    = a.sink_dtype == 0   ? ptr_as<float>(a.sink)[hq]
                           : a.sink_dtype == 1 ? bf16_to_f32(ptr_as<uint16_t>(a.sink)[hq])
                                               : fp16_to_f32(ptr_as<uint16_t>(a.sink)[hq]);
        return x * kLog2e;
    }
    // Last normalization of query row r of kv head g, running max m and sum l in P units (see
    // kLog2P): the sink of its q head joins l; the log-sum-exp is stored when requested (one
    // writer per row). Returns the factor of the accumulated O. Without either, the branch keeps
    // the plain division off the sink's loads.
    CK_TILE_DEVICE static float
    Finish(const Kargs& a, index_t b, index_t g, index_t r, float m, float l, float vd, bool writer)
    {
        if(a.sink == nullptr && a.lse == nullptr)
            return l > 0.f ? vd / l : 0.f;
        const float mu        = is_neg_inf(m) ? 0.f : m;
        const float sink_log2 = SinkLog2(a, g, r);
        if(!is_neg_inf(sink_log2))
            l += exp2(sink_log2 + kLog2P - mu);
        if(writer && a.lse != nullptr)
            a.lse[RowIndex(a, b, g, r, QueryToken(a, r))] =
                l > 0.f ? (mu + __log2f(l) - kLog2P) * kLn2 : -numeric<float>::infinity();
        return l > 0.f ? vd / l : 0.f;
    }
    // S^T tile += K rows (kQkChunks 16-byte chunks per lane) x Q^T
    CK_TILE_DEVICE static v8f wmma_qk(const int32_t* k, const int32_t* q, v8f c)
    {
        if constexpr(kQkChunks == 8)
            return __builtin_amdgcn_wmma_f32_16x16x128_fp8_fp8(v16i{k[0],
                                                                    k[1],
                                                                    k[2],
                                                                    k[3],
                                                                    k[4],
                                                                    k[5],
                                                                    k[6],
                                                                    k[7],
                                                                    k[8],
                                                                    k[9],
                                                                    k[10],
                                                                    k[11],
                                                                    k[12],
                                                                    k[13],
                                                                    k[14],
                                                                    k[15]},
                                                               v16i{q[0],
                                                                    q[1],
                                                                    q[2],
                                                                    q[3],
                                                                    q[4],
                                                                    q[5],
                                                                    q[6],
                                                                    q[7],
                                                                    q[8],
                                                                    q[9],
                                                                    q[10],
                                                                    q[11],
                                                                    q[12],
                                                                    q[13],
                                                                    q[14],
                                                                    q[15]},
                                                               0,
                                                               c,
                                                               false,
                                                               false);
        else
            return __builtin_amdgcn_wmma_f32_16x16x64_fp8_fp8(
                v8i{k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]},
                v8i{q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7]},
                0,
                c,
                false,
                false);
    }
    // pairwise: shorter dependency chains than a running max or sum
    template <index_t N>
    CK_TILE_DEVICE static float tree_max(const float* x)
    {
        if constexpr(N == 1)
            return x[0];
        else
            return fmaxf(tree_max<N / 2>(x), tree_max<N / 2>(x + N / 2));
    }
    template <index_t N>
    CK_TILE_DEVICE static float tree_sum(const float* x)
    {
        if constexpr(N == 1)
            return x[0];
        else
            return tree_sum<N / 2>(x) + tree_sum<N / 2>(x + N / 2);
    }

    CK_TILE_DEVICE void operator()(const Kargs& a) const
    {
        if constexpr(kWaveSplit)
        {
            __shared__ SmemWaveSplit sm;
            Run(a, sm);
        }
        else
        {
            __shared__ SmemCoop sm;
            Run(a, sm);
        }
    }

    template <typename Smem>
    CK_TILE_DEVICE void Run(const Kargs& a, Smem& sm) const
    {
        const index_t l = threadIdx.x % 32;
        const index_t w = __builtin_amdgcn_readfirstlane(threadIdx.x) / 32;
        const index_t n = l % 16, h = l / 16;
        // groups of kNumSplits along x, one cluster each
        const index_t gi = kGroups ? blockIdx.x / kNumSplits : 0;
        const index_t s  = kGroups ? blockIdx.x % kNumSplits : blockIdx.x;
        // y: kv head g, row block rb
        const index_t g  = Div(blockIdx.y, a.row_blocks_rcp);
        const index_t rb = blockIdx.y - g * a.row_blocks;
        const index_t b  = blockIdx.z;
        const index_t sg = gi * kNumSplits + s; // split index within the (batch, kv head)

        constexpr index_t kStepPages = kNumWaves * kWavePages;
        const index_t ctx            = a.seqlen_k[b];
        // query token i sees keys [hi - window, hi) with hi = ctx - qlen + 1 + i: the block reads
        // the union of the tokens' ranges and skips the mask only where every token sees the page
        const index_t hi_min = ctx - a.qlen + 1;
        const index_t lo     = a.window > 0 ? max(0, hi_min - a.window) : 0;
        const index_t lo_max = a.window > 0 ? max(0, ctx - a.window) : 0;
        const index_t pg_hi  = (ctx + kPage - 1) / kPage;
        // with larger cache pages the range starts on a 4-page boundary, so that a wave-split
        // step's four pages of a wave lie in one cache page (positions before lo are masked)
        const index_t pg_lo = kSubPages > 1 ? lo / kPage / 4 * 4 : lo / kPage;
        const index_t span  = max(pg_hi - pg_lo, 1);
        // this range over all splits of the grid, in whole steps and at most the configured size
        constexpr index_t kMinPages = kWaveSplit ? 16 : (kGroups ? 8 : 4);
        index_t per_split;
        if constexpr(kGroups)
            per_split =
                __umulhi(static_cast<uint32_t>(span + kNumSplits * a.num_groups - 1), a.splits_rcp);
        else
            per_split = (span + kNumSplits - 1) / kNumSplits;
        // the configured size, but never too short to cover the range with this grid
        const index_t cover = (per_split + kStepPages - 1) / kStepPages * kStepPages;
        const index_t pps   = max(cover, min(a.pages_per_split, max(kMinPages, cover)));
        // groups without pages take no part in the merge
        if(gi > 0 && gi * kNumSplits * pps >= span)
            return;
        const index_t pg0   = pg_lo + sg * pps;
        const index_t pend  = min(pg_hi, pg_lo + (sg + 1) * pps);
        const index_t steps = pend > pg0 ? (pend - pg0 + kStepPages - 1) / kStepPages : 0;

        // block ids of the split's cache pages, 32 per register (lane = cache page relative to
        // the split's first), picked with readlane
        const int32_t* bt  = a.block_table + static_cast<long_index_t>(b) * a.max_blocks_per_seq;
        const index_t blk0 = pg0 / kSubPages;
        const auto load_bt = [&](index_t chunk) {
            return bt[min(blk0 + chunk * 32 + l, max(pend - 1, 0) / kSubPages)];
        };
        int32_t bt_cur = load_bt(0), bt_next = load_bt(1);
        index_t cur_chunk = 0;
        // the descales go out with the prologue loads instead of after the Q quantization
        const float kds = *a.k_descale, vd = *a.v_descale;
        // loads return in order: the block ids, which the first copies wait for, go out first
        __builtin_amdgcn_sched_barrier(0);

        // column n of tile i: query row rb * kMRows + i * 16 + n (padding columns repeat the last
        // row), its query token and the keys [lo_c, hi_c) it sees
        index_t lo_c[kMTiles], hi_c[kMTiles];
        // Q^T operand of each tile's column n: 16-byte d chunks 2i + h, quantized per row below
        v4i qraw[kMTiles][kHdim / 16];
#pragma unroll
        for(index_t i = 0; i < kMTiles; ++i)
        {
            const index_t r      = min(rb * kMRows + i * kRows + n, a.rows - 1);
            const index_t qi     = QueryToken(a, r);
            hi_c[i]              = hi_min + qi;
            lo_c[i]              = a.window > 0 ? max(0, hi_c[i] - a.window) : 0;
            const uint16_t* qrow = a.q + RowOffset(a, b, g, r, qi);
            for(index_t c2 = 0; c2 < kHdim / 32; ++c2)
            {
                const index_t c     = 2 * c2 + h;
                qraw[i][2 * c2]     = *ptr_as<v4i>(qrow + c * 16);
                qraw[i][2 * c2 + 1] = *ptr_as<v4i>(qrow + c * 16 + 8);
            }
        }

        // LDS page buffers of wave w
        const auto kbuf = [&](index_t st, index_t q) -> uint8_t* {
            if constexpr(kWaveSplit)
                return &sm.kv.k[st][w][0][0] + q * kKPageStride;
            else
            {
                ignore = q;
                return &sm.k[st][w][0];
            }
        };
        const auto vbuf = [&](index_t st, index_t q) -> uint8_t* {
            if constexpr(kWaveSplit)
                return &sm.kv.v[st][w][q][0];
            else
            {
                ignore = q;
                return &sm.v[st][w][0];
            }
        };
        // page index (within the split) of wave w's q-th page of step t
        const auto page_idx = [&](index_t t, index_t q) {
            return (t * kNumWaves + w) * kWavePages + q;
        };
        // block id of the cache page holding page idx (relative to pg0)
        const auto block_of = [&](index_t idx) {
            const index_t cb = (pg0 + idx) / kSubPages - blk0;
            return __builtin_amdgcn_readlane((cb >> 5) == cur_chunk ? bt_cur : bt_next, cb & 31);
        };
        // K and V of step t, one TDM gather each; a page past the split repeats the last one (its V
        // is zeroed once it lands)
        const auto issue = [&](index_t t) {
            const index_t last = pend - 1 - pg0;
            if constexpr(kRowMajor)
            {
                // a 16-token page (or, wave split with larger cache pages, the wave's four pages of
                // one cache page) as token rows: global offset of its first row and the row pitch
                const long_index_t pitch =
                    kLayout == FmhaDecodePagedLayout::hnd ? kHdim : a.nhead_k * long_index_t{kHdim};
                const auto first_row = [&](index_t idx) {
                    const long_index_t blk = block_of(idx);
                    const index_t tok      = (pg0 + idx) % kSubPages * kPage;
                    return kLayout == FmhaDecodePagedLayout::hnd
                               ? ((blk * a.nhead_k + g) * kPageSize + tok) * kHdim
                               : ((blk * kPageSize + tok) * a.nhead_k + g) * kHdim;
                };
                if constexpr(kWaveSplit && kSubPages > 1)
                {
                    const long_index_t off = first_row(min(page_idx(t, 0), last / 4 * 4));
                    tdm_tile<4 * kPage>(a.k + off, kbuf(t % kStages, 0), pitch);
                    tdm_tile<4 * kPage>(a.v + off, vbuf(t % kStages, 0), pitch);
                }
                else
                {
                    for(index_t q = 0; q < kWavePages; ++q)
                    {
                        const long_index_t off = first_row(min(page_idx(t, q), last));
                        tdm_tile<kPage>(a.k + off, kbuf(t % kStages, q), pitch);
                        tdm_tile<kPage>(a.v + off, vbuf(t % kStages, q), pitch);
                    }
                }
            }
            else if constexpr(kSubPages == 1)
            {
                int32_t rows[kWavePages];
                for(index_t q = 0; q < kWavePages; ++q)
                    rows[q] = block_of(min(page_idx(t, q), last)) * a.nhead_k + g;
                tdm_rows<kPageBytes>(a.k, kbuf(t % kStages, 0), rows);
                tdm_rows<kPageBytes>(a.v, vbuf(t % kStages, 0), rows);
            }
            else if constexpr(kWaveSplit)
            {
                // the wave's four pages are a quarter of a 64-token cache page: K as eight rows
                // of 64 tokens x 16 B, V as one row of four page-16 images
                const index_t idx = min(page_idx(t, 0), last / 4 * 4);
                const index_t qtr = (pg0 + idx) % kSubPages / 4;
                const index_t row = block_of(idx) * a.nhead_k + g;
                for(index_t c0 = 0; c0 < kChunks; c0 += 8)
                {
                    constexpr index_t kN = kChunks < 8 ? kChunks : 8;
                    int32_t krows[kN];
                    for(index_t c = 0; c < kN; ++c)
                        krows[c] = (row * kChunks + c0 + c) * (kSubPages / 4) + qtr;
                    tdm_rows<4 * 256>(a.k, kbuf(t % kStages, 0) + c0 * 4 * 256, krows);
                }
                const int32_t vrows[1] = {row * (kSubPages / 4) + qtr};
                tdm_rows<4 * kPageBytes>(a.v, vbuf(t % kStages, 0), vrows);
            }
            else
            {
                // one page: K as eight rows of 16 tokens x 16 B (a page-16 image), V as one row
                const index_t idx = min(page_idx(t, 0), last);
                const index_t sub = (pg0 + idx) % kSubPages;
                const index_t row = block_of(idx) * a.nhead_k + g;
                for(index_t c0 = 0; c0 < kChunks; c0 += 8)
                {
                    constexpr index_t kN = kChunks < 8 ? kChunks : 8;
                    int32_t krows[kN];
                    for(index_t c = 0; c < kN; ++c)
                        krows[c] = (row * kChunks + c0 + c) * kSubPages + sub;
                    tdm_rows<256>(a.k, kbuf(t % kStages, 0) + c0 * 256, krows);
                }
                const int32_t vrows[1] = {row * kSubPages + sub};
                tdm_rows<kPageBytes>(a.v, vbuf(t % kStages, 0), vrows);
            }
        };
        index_t issued = min(steps, kStages);
        for(index_t t = 0; t < issued; ++t)
            issue(t);
        asm volatile("" ::"s"(kds), "s"(vd));

        // per tile: Q^T operand bytes (int 4i + e holds bytes 4e..4e+3 of chunk 2i + h) and the
        // scale of column n of S^T in log2 units
        int32_t qf[kMTiles][kHdim / 8];
        float scl[kMTiles];
#pragma unroll
        for(index_t i = 0; i < kMTiles; ++i)
        {
            float qv[kHdim / 2];
            for(index_t j = 0; j < kHdim / 16; ++j)
                for(index_t e = 0; e < 4; ++e)
                {
                    const uint32_t u = static_cast<uint32_t>(qraw[i][j][e]);
                    if constexpr(kQoFp16)
                    {
                        qv[j * 8 + 2 * e]     = fp16_to_f32(u & 0xffff);
                        qv[j * 8 + 2 * e + 1] = fp16_to_f32(u >> 16);
                    }
                    else
                    {
                        qv[j * 8 + 2 * e]     = bf16_to_f32(u & 0xffff);
                        qv[j * 8 + 2 * e + 1] = bf16_to_f32(u >> 16);
                    }
                }
            float qmax = 1e-20f;
            for(index_t j = 0; j < kHdim / 2; ++j)
                qmax = fmaxf(qmax, fabsf(qv[j]));
            qmax             = fmaxf(qmax, __shfl_xor(qmax, 16, 32));
            const float qinv = 448.f / qmax;
            for(index_t j = 0; j < kHdim / 8; ++j)
            {
                const int32_t x = __builtin_amdgcn_cvt_pk_fp8_f32(
                    qv[4 * j] * qinv, qv[4 * j + 1] * qinv, 0, false);
                qf[i][j] = __builtin_amdgcn_cvt_pk_fp8_f32(
                    qv[4 * j + 2] * qinv, qv[4 * j + 3] * qinv, x, true);
            }
            scl[i] = qmax / 448.f * kds * a.scale_s * kLog2e;
        }

        // step t's copies of this wave are in LDS; zero the V bytes of tokens past the context
        const auto land = [&](index_t t) {
            if(issued > t + 2)
                __builtin_amdgcn_s_wait_tensorcnt(2 * kTdmPerStep);
            else if(issued > t + 1)
                __builtin_amdgcn_s_wait_tensorcnt(kTdmPerStep);
            else
                __builtin_amdgcn_s_wait_tensorcnt(0);
            for(index_t q = 0; q < kWavePages; ++q)
            {
                const index_t pg    = pg0 + page_idx(t, q);
                const index_t valid = __builtin_amdgcn_readfirstlane(
                    pg < pend ? min(max(ctx - pg * kPage, 0), kPage) : 0);
                uint8_t* vd8 = vbuf(t % kStages, q);
                if constexpr(kRowMajor)
                {
                    // token rows valid..15: 16 rows x kChunks chunks of 16 bytes over the wave
                    if(valid < kPage)
                        for(index_t i = 0; i < kHdim / 32; ++i)
                        {
                            const index_t c = i * 32 + l;
                            if(c / kChunks >= valid)
                                *ptr_as<v4i>(vd8 + c / kChunks * kRowPitch + c % kChunks * 16) =
                                    v4i{0, 0, 0, 0};
                        }
                }
                else if(valid == 0)
                    for(index_t i = 0; i < kHdim / 32; ++i)
                        *ptr_as<v4i>(vd8 + (i * 32 + l) * 16) = v4i{0, 0, 0, 0};
                else if(valid < kPage)
                    for(index_t i = 0; i < kHdim / 32; ++i)
                        for(index_t tok = valid; tok < kPage; ++tok)
                            vd8[(i * 32 + l) * 16 + tok] = 0;
            }
        };
        const auto advance_bt = [&](index_t t) {
            // the block-id chunk of step t's pages; the next one is loaded a chunk ahead
            const index_t chunk = ((pg0 + t * kStepPages) / kSubPages - blk0) / 32;
            if(chunk != cur_chunk)
            {
                cur_chunk = chunk;
                bt_cur    = bt_next;
                bt_next   = load_bt(chunk + 1);
            }
        };

        float m_run[kMTiles], l_run[kMTiles];
        // 16-wide head-dim blocks per wave and tile: all of them, or every fourth (cooperative)
        constexpr index_t kAcc = kWaveSplit ? kOutBlocks : kOutBlocks / kNumWaves;
        v8f acc[kMTiles][kAcc];
#pragma unroll
        for(index_t i = 0; i < kMTiles; ++i)
        {
            m_run[i] = -numeric<float>::infinity();
            l_run[i] = 0.f;
            for(index_t j = 0; j < kAcc; ++j)
                acc[i][j] = v8f{0, 0, 0, 0, 0, 0, 0, 0};
        }

        // S^T of 16 key rows of stage st (row m: page kq(m), token kt(m) of this wave's pages) for
        // every tile, from one read of the K rows; lane (n, h) gets rows h*8 + r of column n,
        // unscaled, -inf outside the column's keys when masked (first page pg_first)
        const auto qk_tile = [&](index_t st,
                                 index_t kq,
                                 index_t kt,
                                 index_t pg_first,
                                 index_t tok_h,
                                 float(&sv)[kMTiles][8],
                                 auto masked) {
            const uint8_t* kb = kbuf(st, kq) + kt * kKTokStride;
            v8f st8[kMTiles];
#pragma unroll
            for(index_t i = 0; i < kMTiles; ++i)
                st8[i] = v8f{0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
            for(index_t ks = 0; ks < kQkSteps; ++ks)
            {
                // chunks ks * kQkChunks + 2i + h on both operands
                int32_t kf[kQkChunks * 2];
                for(index_t c2 = 0; c2 < kQkChunks / 2; ++c2)
                {
                    const v4i x = *ptr_as<v4i>(kb + (ks * kQkChunks + 2 * c2 + h) * kKChunkStride);
                    for(index_t e = 0; e < 4; ++e)
                        kf[4 * c2 + e] = x[e];
                }
#pragma unroll
                for(index_t i = 0; i < kMTiles; ++i)
                    st8[i] = wmma_qk(kf, qf[i] + ks * kQkChunks * 2, st8[i]);
            }
#pragma unroll
            for(index_t i = 0; i < kMTiles; ++i)
                for(index_t r = 0; r < 8; ++r)
                {
                    sv[i][r] = st8[i][r];
                    if constexpr(decltype(masked)::value)
                    {
                        const index_t pos = pg_first * kPage + tok_h + r;
                        if(!(pos >= lo_c[i] && pos < hi_c[i] && pos < pend * kPage))
                            sv[i][r] = -numeric<float>::infinity();
                    }
                }
        };
        // pages [pg, pg + count) need no mask for any column
        const auto inside = [&](index_t pg, index_t count) {
            return pg * kPage >= lo_max && (pg + count) * kPage <= hi_min && pg + count <= pend;
        };

        if constexpr(kWaveSplit)
        {
            for(index_t t = 0; t < steps; ++t)
            {
                const index_t st = t % kStages;
                land(t);
                // tile j: rows 0-7 are tokens (j%2)*8 + m of page j/2, rows 8-15 those of page
                // 2 + j/2, so lane (n, h) gets all tokens of pages 2h and 2h + 1: byte j*8 + r of
                // its P^T operand is token (j%2)*8 + r of page 2h + j/2, as on the V^T operand
                const index_t pgw = pg0 + page_idx(t, 0);
                float sv[kMTiles][kWavePages][8];
                const auto qk4 = [&](auto masked) {
                    for(index_t j = 0; j < kWavePages; ++j)
                    {
                        float sj[kMTiles][8];
                        qk_tile(st,
                                (n < 8 ? 0 : 2) + j / 2,
                                (j % 2) * 8 + (n & 7),
                                pgw + 2 * h + j / 2,
                                (j % 2) * 8,
                                sj,
                                masked);
#pragma unroll
                        for(index_t i = 0; i < kMTiles; ++i)
                            for(index_t r = 0; r < 8; ++r)
                                sv[i][j][r] = sj[i][r];
                    }
                };
                if(inside(pgw, kWavePages))
                    qk4(bool_constant<false>{});
                else
                    qk4(bool_constant<true>{});
                v8i pf[kMTiles];
#pragma unroll
                for(index_t i = 0; i < kMTiles; ++i)
                {
                    float mx          = tree_max<kWavePages * 8>(&sv[i][0][0]);
                    mx                = fmaxf(mx, __shfl_xor(mx, 16, 32)) * scl[i];
                    const float m_new = fmaxf(m_run[i], mx);
                    const float m_use = is_neg_inf(m_new) ? 0.f : m_new;
                    const float c     = kLog2P - m_use;
                    float pv[kWavePages][8];
                    int32_t pk[8];
                    for(index_t j = 0; j < kWavePages; ++j)
                        for(index_t r2 = 0; r2 < 4; ++r2)
                        {
                            const float p0    = exp2(fmaf(sv[i][j][2 * r2], scl[i], c));
                            const float p1    = exp2(fmaf(sv[i][j][2 * r2 + 1], scl[i], c));
                            pv[j][2 * r2]     = p0;
                            pv[j][2 * r2 + 1] = p1;
                            const index_t e   = j * 8 + 2 * r2; // operand byte
                            pk[e / 4] =
                                (e & 3) == 0
                                    ? __builtin_amdgcn_cvt_pk_fp8_f32(p0, p1, 0, false)
                                    : __builtin_amdgcn_cvt_pk_fp8_f32(p0, p1, pk[e / 4], true);
                        }
                    float ps = tree_sum<kWavePages * 8>(&pv[0][0]);
                    ps       = ps + __shfl_xor(ps, 16, 32);
                    // rescale only when a row max grew
                    if(__builtin_amdgcn_ballot_w32(m_new > m_run[i]) != 0)
                    {
                        const float alpha = exp2(m_run[i] - m_use);
                        l_run[i] *= alpha;
                        for(index_t db = 0; db < kAcc; ++db)
                            acc[i][db] *= alpha;
                    }
                    l_run[i] += ps;
                    m_run[i] = m_new;
                    pf[i]    = v8i{pk[0], pk[1], pk[2], pk[3], pk[4], pk[5], pk[6], pk[7]};
                }
                for(index_t db = 0; db < kAcc; ++db)
                {
                    v8i vf;
                    if constexpr(kRowMajor)
                    {
                        // operand byte 8q + j: token 32h + 8q + j, as on P^T
                        vf = v_tr8(n, [&](index_t q, index_t j) {
                            return vbuf(st, 2 * h + q / 2) + ((q % 2) * 8 + j) * kRowPitch +
                                   db * 16;
                        });
                    }
                    else
                    {
                        const index_t off = (db * 16 + n) * 16;
                        const v4i x       = *ptr_as<v4i>(&sm.kv.v[st][w][2 * h][off]);
                        const v4i y       = *ptr_as<v4i>(&sm.kv.v[st][w][2 * h + 1][off]);
                        vf                = v8i{x[0], x[1], x[2], x[3], y[0], y[1], y[2], y[3]};
                    }
#pragma unroll
                    for(index_t i = 0; i < kMTiles; ++i)
                        acc[i][db] = __builtin_amdgcn_wmma_f32_16x16x64_fp8_fp8(
                            vf, pf[i], 0, acc[i][db], false, false);
                }
                // this wave is done with stage st: it takes step t + kStages
                if(t + kStages < steps)
                {
                    advance_bt(t + kStages);
                    issue(t + kStages);
                    ++issued;
                }
            }
        }
        else
        {
            if(steps > 0)
                land(0);
            __syncthreads();
            for(index_t t = 0; t < steps; ++t)
            {
                const index_t st = t % kStages;
                const index_t pg = pg0 + page_idx(t, 0);
                float sv[kMTiles][8];
                if(inside(pg, 1))
                    qk_tile(st, 0, n, pg, h * 8, sv, bool_constant<false>{});
                else
                    qk_tile(st, 0, n, pg, h * 8, sv, bool_constant<true>{});
#pragma unroll
                for(index_t i = 0; i < kMTiles; ++i)
                {
                    float mx = tree_max<8>(sv[i]);
                    mx       = fmaxf(mx, __shfl_xor(mx, 16, 32)) * scl[i];
                    if(h == 0)
                        sm.red[0][w][i * kRows + n] = mx;
                }
                __syncthreads();
                // every wave is past step t-1's PV: its stage takes step t + 2
                if(t >= 1 && t + 2 < steps)
                {
                    advance_bt(t + 2);
                    issue(t + 2);
                    ++issued;
                }
                float m_new[kMTiles], m_use[kMTiles];
#pragma unroll
                for(index_t i = 0; i < kMTiles; ++i)
                {
                    float m_step = sm.red[0][0][i * kRows + n];
                    for(index_t k = 1; k < kNumWaves; ++k)
                        m_step = fmaxf(m_step, sm.red[0][k][i * kRows + n]);
                    m_new[i]      = fmaxf(m_run[i], m_step);
                    m_use[i]      = is_neg_inf(m_new[i]) ? 0.f : m_new[i];
                    const float c = kLog2P - m_use[i];
                    float pv[8];
                    for(index_t r = 0; r < 8; ++r)
                        pv[r] = exp2(fmaf(sv[i][r], scl[i], c));
                    float ps   = tree_sum<8>(pv);
                    ps         = ps + __shfl_xor(ps, 16, 32);
                    int32_t p0 = __builtin_amdgcn_cvt_pk_fp8_f32(pv[0], pv[1], 0, false);
                    p0         = __builtin_amdgcn_cvt_pk_fp8_f32(pv[2], pv[3], p0, true);
                    int32_t p1 = __builtin_amdgcn_cvt_pk_fp8_f32(pv[4], pv[5], 0, false);
                    p1         = __builtin_amdgcn_cvt_pk_fp8_f32(pv[6], pv[7], p1, true);
                    *ptr_as<int2>(&sm.p[i][n][w * kPage + h * 8]) = make_int2(p0, p1);
                    if(h == 0)
                        sm.red[1][w][i * kRows + n] = ps;
                }
                if(t + 1 < steps)
                    land(t + 1);
                __syncthreads();
                float alpha[kMTiles];
                v8i pf[kMTiles];
#pragma unroll
                for(index_t i = 0; i < kMTiles; ++i)
                {
                    float psum = 0.f;
                    for(index_t k = 0; k < kNumWaves; ++k)
                        psum += sm.red[1][k][i * kRows + n];
                    alpha[i] = exp2(m_run[i] - m_use[i]);
                    l_run[i] = l_run[i] * alpha[i] + psum;
                    m_run[i] = m_new[i];
                    // tokens of pages h and h + 2 on both operands
                    const v4i px = *ptr_as<v4i>(&sm.p[i][n][h * kPage]);
                    const v4i py = *ptr_as<v4i>(&sm.p[i][n][(h + 2) * kPage]);
                    pf[i]        = v8i{px[0], px[1], px[2], px[3], py[0], py[1], py[2], py[3]};
                }
                // O^T += V^T P^T for head-dim blocks w, w + 4, ...
                for(index_t bk = 0; bk < kAcc; ++bk)
                {
                    const index_t d0 = (bk * kNumWaves + w) * 16;
                    v8i vf;
                    if constexpr(kRowMajor)
                    {
                        // operand byte 8q + j: token (q % 2) * 8 + j of page h + 2 * (q / 2)
                        vf = v_tr8(n, [&](index_t q, index_t j) {
                            return &sm.v[st][h + 2 * (q / 2)][((q % 2) * 8 + j) * kRowPitch + d0];
                        });
                    }
                    else
                    {
                        const v4i x = *ptr_as<v4i>(&sm.v[st][h][(d0 + n) * 16]);
                        const v4i y = *ptr_as<v4i>(&sm.v[st][h + 2][(d0 + n) * 16]);
                        vf          = v8i{x[0], x[1], x[2], x[3], y[0], y[1], y[2], y[3]};
                    }
#pragma unroll
                    for(index_t i = 0; i < kMTiles; ++i)
                        acc[i][bk] = __builtin_amdgcn_wmma_f32_16x16x64_fp8_fp8(
                            vf, pf[i], 0, acc[i][bk] * alpha[i], false, false);
                }
            }
        }

        // (batch, kv head, row block) of this workgroup, and its partials of this group
        const long_index_t head =
            (static_cast<long_index_t>(b) * a.nhead_k + g) * a.row_blocks + rb;
        const long_index_t part0 =
            ((head * (kGroups ? a.num_groups : 1) + gi) * kNumSplits) * kMRows;
        // O in the output type; partials stay bf16
        const auto pack_out = [&](float x, float y) {
            return kQoFp16 ? pack_fp16(x, y) : pack_bf16(x, y);
        };
        const auto pack_part = [](float x, float y) { return pack_bf16(x, y); };

#pragma unroll
        for(index_t i = 0; i < kMTiles; ++i)
        {
            // tile i's result in registers of thread (row, kHdim / 8 or 16-wide blocks of head
            // dims): o[] (wave split) or acc[i], m_row, l_row; stored as O / l. The tile loops
            // are unrolled: acc[i] with a variable i would live in scratch.
            index_t row;
            index_t d0;
            // wave split: 8 threads per row, kHdim / 8 head dims each
            [[maybe_unused]] float o[kWaveSplit ? kHdim / 8 : 1];
            float m_row, l_row;
            if constexpr(kWaveSplit)
            {
                // merge the waves; lane (n, h) holds O[row n][d = db*16 + h*8 + r]
                if(i > 0)
                    __syncthreads(); // the previous tile's merge reads are done
                if(h == 0)
                {
                    sm.ml[0][w][n] = m_run[i];
                    sm.ml[1][w][n] = l_run[i];
                }
                __syncthreads(); // also: every wave is done with the stages the merge buffer reuses
                float M = -numeric<float>::infinity();
                for(index_t k = 0; k < kNumWaves; ++k)
                    M = fmaxf(M, sm.ml[0][k][n]);
                const float f = exp2(m_run[i] - (is_neg_inf(M) ? 0.f : M));
                for(index_t db = 0; db < kAcc; ++db)
                {
                    const v8f x = acc[i][db] * f;
                    float4* dst = ptr_as<float4>(&sm.o[w][n][db * 16 + h * 8]);
                    dst[0]      = make_float4(x[0], x[1], x[2], x[3]);
                    dst[1]      = make_float4(x[4], x[5], x[6], x[7]);
                }
                __syncthreads();
                row   = threadIdx.x / 8;
                d0    = (threadIdx.x % 8) * (kHdim / 8);
                m_row = -numeric<float>::infinity();
                for(index_t k = 0; k < kNumWaves; ++k)
                    m_row = fmaxf(m_row, sm.ml[0][k][row]);
                const float mu = is_neg_inf(m_row) ? 0.f : m_row;
                l_row          = 0.f;
                for(index_t k = 0; k < kNumWaves; ++k)
                    l_row += sm.ml[1][k][row] * exp2(sm.ml[0][k][row] - mu);
                for(index_t e = 0; e < kHdim / 8; ++e)
                    o[e] = 0.f;
                for(index_t k = 0; k < kNumWaves; ++k)
                    for(index_t e4 = 0; e4 < kHdim / 32; ++e4)
                    {
                        const float4 x = *ptr_as<float4>(&sm.o[k][row][d0 + 4 * e4]);
                        o[4 * e4] += x.x;
                        o[4 * e4 + 1] += x.y;
                        o[4 * e4 + 2] += x.z;
                        o[4 * e4 + 3] += x.w;
                    }
            }
            else
            {
                row   = n;
                m_row = m_run[i];
                l_row = l_run[i];
                d0    = 0; // unused: the lane holds d = (bk*4 + w)*16 + h*8 + r
            }
            // row_ptr: this thread's row, hdim elements
            const auto store_row = [&](uint16_t* row_ptr, float f, auto pack) {
                if constexpr(kWaveSplit)
                {
                    uint32_t pk[kHdim / 16];
                    for(index_t e = 0; e < kHdim / 16; ++e)
                        pk[e] = pack(o[2 * e] * f, o[2 * e + 1] * f);
                    v4i* dst = ptr_as<v4i>(row_ptr + d0);
                    for(index_t e = 0; e < kHdim / 64; ++e)
                        dst[e] = v4i{int32_t(pk[4 * e]),
                                     int32_t(pk[4 * e + 1]),
                                     int32_t(pk[4 * e + 2]),
                                     int32_t(pk[4 * e + 3])};
                }
                else
                {
                    for(index_t bk = 0; bk < kAcc; ++bk)
                    {
                        uint32_t pk[4];
                        for(index_t e = 0; e < 4; ++e)
                            pk[e] = pack(acc[i][bk][2 * e] * f, acc[i][bk][2 * e + 1] * f);
                        *ptr_as<v4i>(row_ptr + (bk * kNumWaves + w) * 16 + h * 8) =
                            v4i{int32_t(pk[0]), int32_t(pk[1]), int32_t(pk[2]), int32_t(pk[3])};
                    }
                }
            };
            const index_t lr = i * kRows + row; // row within the block
            if constexpr(kNumSplits == 1)
            {
                const index_t r   = rb * kMRows + lr;
                const bool writer = kWaveSplit ? (threadIdx.x % 8 == 0) : (w == 0 && h == 0);
                if(r < a.rows)
                    store_row(a.o + RowOffset(a, b, g, r),
                              Finish(a, b, g, r, m_row, l_row, vd, writer),
                              pack_out);
            }
            else
            {
                store_row(a.ws_o + (part0 + s * kMRows + lr) * kHdim,
                          l_row > 0.f ? 1.f / l_row : 0.f,
                          pack_part);
                const bool ml_writer = kWaveSplit ? (threadIdx.x % 8 == 0) : (w == 0 && h == 0);
                if(ml_writer)
                    *ptr_as<float2>(a.ws_ml + (part0 + s * kMRows + lr) * 2) =
                        make_float2(m_row, l_row);
            }
        }

        if constexpr(kNumSplits > 1)
        {
            asm volatile("s_wait_storecnt 0x0" ::: "memory");
            __builtin_amdgcn_s_cluster_barrier();
            asm volatile("global_inv scope:SCOPE_SE" ::: "memory");

            // merge the splits: split s takes rows [s*kRowsPer, +kRowsPer); thread -> (row, 2 dims)
            constexpr index_t kRowsPer = (kMRows + kNumSplits - 1) / kNumSplits;
            for(index_t rr = threadIdx.x / kMergeTpr; rr < kRowsPer; rr += kBlockSize / kMergeTpr)
            {
                const index_t mrow = s * kRowsPer + rr;
                if(mrow >= kMRows || rb * kMRows + mrow >= a.rows)
                    break;
                const index_t d2 = (threadIdx.x % kMergeTpr) * 2;
                float2 mlv[kNumSplits];
                uint32_t oi[kNumSplits];
                for(index_t k = 0; k < kNumSplits; ++k)
                {
                    mlv[k] = *ptr_as<float2>(a.ws_ml + (part0 + k * kMRows + mrow) * 2);
                    oi[k]  = *ptr_as<uint32_t>(a.ws_o + (part0 + k * kMRows + mrow) * kHdim + d2);
                }
                float mm = -numeric<float>::infinity();
                for(index_t k = 0; k < kNumSplits; ++k)
                    mm = fmaxf(mm, mlv[k].x);
                float ll = 0.f, o0 = 0.f, o1 = 0.f;
                for(index_t k = 0; k < kNumSplits; ++k)
                {
                    const float wgt = mlv[k].y > 0.f ? mlv[k].y * exp2(mlv[k].x - mm) : 0.f;
                    ll += wgt;
                    o0 += wgt * bf16_to_f32(oi[k] & 0xffff);
                    o1 += wgt * bf16_to_f32(oi[k] >> 16);
                }
                if constexpr(!kGroups)
                {
                    const index_t r = rb * kMRows + mrow;
                    const float f   = Finish(a, b, g, r, mm, ll, vd, d2 == 0);
                    *ptr_as<uint32_t>(a.o + RowOffset(a, b, g, r) + d2) = pack_out(o0 * f, o1 * f);
                }
                else
                {
                    // this group's rows: O / L and (M, L), at device scope for the other groups
                    const long_index_t grow = (head * a.num_groups + gi) * kMRows + mrow;
                    const float f           = ll > 0.f ? 1.f / ll : 0.f;
                    __scoped_atomic_store_n(ptr_as<uint32_t>(a.gws_o + grow * kHdim + d2),
                                            pack_bf16(o0 * f, o1 * f),
                                            __ATOMIC_RELAXED,
                                            __MEMORY_SCOPE_DEVICE);
                    if(d2 == 0)
                    {
                        __scoped_atomic_store_n(ptr_as<uint32_t>(a.gws_ml + grow * 2),
                                                __float_as_uint(mm),
                                                __ATOMIC_RELAXED,
                                                __MEMORY_SCOPE_DEVICE);
                        __scoped_atomic_store_n(ptr_as<uint32_t>(a.gws_ml + grow * 2 + 1),
                                                __float_as_uint(ll),
                                                __ATOMIC_RELAXED,
                                                __MEMORY_SCOPE_DEVICE);
                    }
                }
            }
            if constexpr(kGroups)
                MergeGroups(a, head, b, g, rb, s, (span - 1) / (kNumSplits * pps) + 1, vd);
        }
    }

    // Split s of the last group to finish row block s merges that block over the groups. The
    // groups' rows were stored at device scope; once every wave's stores are acknowledged a
    // relaxed count hands them over, and the loads are device scope as well.
    CK_TILE_DEVICE void MergeGroups(const Kargs& a,
                                    long_index_t head,
                                    index_t b,
                                    index_t g,
                                    index_t rb,
                                    index_t s,
                                    index_t groups,
                                    float vd) const
    {
        constexpr index_t kRowsPer = (kMRows + kNumSplits - 1) / kNumSplits;
        __shared__ int32_t last;
        asm volatile("s_wait_storecnt 0x0" ::: "memory");
        __syncthreads();
        if(threadIdx.x == 0)
        {
            int32_t* counter = a.counters + head * kNumSplits + s;
            const bool is_last =
                __scoped_atomic_fetch_add(counter, 1, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) ==
                groups - 1;
            if(is_last) // zeroed again for the next launch; every group has counted
                __scoped_atomic_store_n(counter, 0, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
            last = is_last ? 1 : 0;
        }
        __syncthreads();
        if(last == 0)
            return;
        for(index_t rr = threadIdx.x / kMergeTpr; rr < kRowsPer; rr += kBlockSize / kMergeTpr)
        {
            const index_t mrow = s * kRowsPer + rr;
            if(mrow >= kMRows || rb * kMRows + mrow >= a.rows)
                break;
            const index_t d2 = (threadIdx.x % kMergeTpr) * 2;
            float mm = -numeric<float>::infinity(), ll = 0.f, o0 = 0.f, o1 = 0.f;
            // kChunk groups per round of loads; the device-scope buffer loads of a round go out
            // together (atomic loads would be issued one round trip at a time)
            constexpr index_t kChunk = 8;
            const long_index_t row0  = head * a.num_groups * kMRows + mrow;
            const auto ml_res        = make_wave_buffer_resource(a.gws_ml, 0x7ffffffe);
            const auto o_res         = make_wave_buffer_resource(a.gws_o, 0x7ffffffe);
            for(index_t i0 = 0; i0 < groups; i0 += kChunk)
            {
                float2 mli[kChunk];
                uint32_t oi[kChunk];
                for(index_t j = 0; j < kChunk; ++j)
                {
                    const long_index_t grow = row0 + min(i0 + j, groups - 1) * kMRows;
                    const auto ml2 =
                        amd_buffer_load_impl<float, 2, amd_buffer_coherence_enum::DEVICE>(
                            ml_res, static_cast<index_t>(grow * 2 * sizeof(float)), 0);
                    mli[j] = make_float2(ml2[number<0>{}], ml2[number<1>{}]);
                    oi[j]  = __float_as_uint(
                        amd_buffer_load_impl<float, 1, amd_buffer_coherence_enum::DEVICE>(
                            o_res, static_cast<index_t>((grow * kHdim + d2) * 2), 0)[number<0>{}]);
                }
                for(index_t j = 0; j < kChunk; ++j)
                {
                    const float mi = mli[j].x, li = mli[j].y;
                    if(i0 + j >= groups || !(li > 0.f))
                        continue;
                    // online merge: the running sums follow the largest max seen so far
                    const float mn = fmaxf(mm, mi);
                    const float c0 = exp2(mm - mn), c1 = exp2(mi - mn);
                    ll = ll * c0 + li * c1;
                    o0 = o0 * c0 + li * c1 * bf16_to_f32(oi[j] & 0xffff);
                    o1 = o1 * c0 + li * c1 * bf16_to_f32(oi[j] >> 16);
                    mm = mn;
                }
            }
            const index_t r = rb * kMRows + mrow;
            const float f   = Finish(a, b, g, r, mm, ll, vd, d2 == 0);
            *ptr_as<uint32_t>(a.o + RowOffset(a, b, g, r) + d2) =
                kQoFp16 ? pack_fp16(o0 * f, o1 * f) : pack_bf16(o0 * f, o1 * f);
        }
    }
#else
    CK_TILE_DEVICE void operator()(const Kargs&) const {}
#endif
};

// The cluster size is a compile-time attribute: HIP stream capture drops a cluster size given only
// as a launch attribute, so a captured graph would run without clusters. Device passes for other
// architectures reject the attribute; their kernel is empty anyway.
#if !defined(__HIP_DEVICE_COMPILE__) || defined(__gfx125__)
#define CK_TILE_FMHA_DECODE_PAGED5D_CLUSTER(n) __attribute__((cluster_dims(n, 1, 1)))
#else
#define CK_TILE_FMHA_DECODE_PAGED5D_CLUSTER(n)
#endif

template <typename Kernel>
__global__ __launch_bounds__(Kernel::kBlockSize) CK_TILE_FMHA_DECODE_PAGED5D_CLUSTER(
    Kernel::kNumSplits) void fmha_decode_paged5d_entry(typename Kernel::Kargs kargs)
{
    Kernel{}(kargs);
}

#undef CK_TILE_FMHA_DECODE_PAGED5D_CLUSTER

} // namespace ck_tile
