// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Host side of FmhaDecodePaged5dKernel for the example and the tests: launch dispatch over the
// compile-time split count and cache page size, random inputs in the 5D layout, a CPU reference
// and timing.

#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/fmha/kernel/fmha_fwd_decode_paged5d_kernel.hpp"

namespace fmha_decode_paged5d {

using ck_tile::index_t;
constexpr index_t kPage = 16;

using Layout = ck_tile::FmhaDecodePagedLayout;

template <bool kWaveSplit,
          index_t kNumSplits,
          bool kGroups,
          index_t kPageSize,
          bool kQoFp16,
          Layout kLayout,
          index_t kHdim,
          index_t kMTiles,
          bool kPerTokenScale>
hipError_t launch_one(const ck_tile::FmhaDecodePaged5dHostArgs& h, hipStream_t stream)
{
    using K    = ck_tile::FmhaDecodePaged5dKernel<kWaveSplit,
                                                  kNumSplits,
                                                  kGroups,
                                                  kPageSize,
                                                  kQoFp16,
                                                  kLayout,
                                                  kHdim,
                                                  kMTiles,
                                                  kPerTokenScale>;
    auto kargs = K::MakeKargs(h);
    if constexpr(kNumSplits > 15)
    {
        // see fmha_decode_paged5d_entry_launch_cluster
        static const hipFunction_t fn = [] {
            hipFunction_t f = nullptr;
            HIP_CHECK_ERROR(
                hipGetFuncBySymbol(&f,
                                   reinterpret_cast<const void*>(
                                       &ck_tile::fmha_decode_paged5d_entry_launch_cluster<K>)));
            return f;
        }();
        const dim3 grid = K::GridSize(h);
        hipLaunchAttribute attr[1];
        attr[0].id               = hipLaunchAttributeClusterDimension;
        attr[0].val.clusterDim.x = kNumSplits;
        attr[0].val.clusterDim.y = 1;
        attr[0].val.clusterDim.z = 1;
        HIP_LAUNCH_CONFIG cfg{};
        cfg.gridDimX    = grid.x;
        cfg.gridDimY    = grid.y;
        cfg.gridDimZ    = grid.z;
        cfg.blockDimX   = K::kBlockSize;
        cfg.blockDimY   = 1;
        cfg.blockDimZ   = 1;
        cfg.hStream     = stream;
        cfg.attrs       = attr;
        cfg.numAttrs    = 1;
        void* params[1] = {&kargs};
        return hipDrvLaunchKernelEx(&cfg, fn, params, nullptr);
    }
    else
    {
        hipLaunchConfig_t cfg{};
        cfg.gridDim  = K::GridSize(h);
        cfg.blockDim = K::BlockSize();
        cfg.stream   = stream;
        // must match the cluster_dims of the entry
        hipLaunchAttribute attr[1];
        attr[0].id               = hipLaunchAttributeClusterDimension;
        attr[0].val.clusterDim.x = kNumSplits;
        attr[0].val.clusterDim.y = 1;
        attr[0].val.clusterDim.z = 1;
        cfg.attrs                = attr;
        cfg.numAttrs             = 1;
        return hipLaunchKernelEx(&cfg, ck_tile::fmha_decode_paged5d_entry<K>, kargs);
    }
}

// the kernel's compile-time parameters other than the split count and the layout of the splits
template <index_t kPageSize_,
          bool kQoFp16_,
          Layout kLayout_,
          index_t kHdim_,
          index_t kMTiles_,
          bool kPerTokenScale_>
struct Instance
{
    static constexpr index_t kPageSize   = kPageSize_;
    static constexpr bool kQoFp16        = kQoFp16_;
    static constexpr Layout kLayout      = kLayout_;
    static constexpr index_t kHdim       = kHdim_;
    static constexpr index_t kMTiles     = kMTiles_;
    static constexpr bool kPerTokenScale = kPerTokenScale_;
};

template <bool kWaveSplit, bool kGroups, typename I, index_t... Ns>
hipError_t dispatch(const ck_tile::FmhaDecodePaged5dHostArgs& h,
                    index_t splits,
                    hipStream_t stream,
                    std::integer_sequence<index_t, Ns...>)
{
    hipError_t e       = hipErrorInvalidValue;
    const auto try_one = [&](auto ns) {
        if(splits == decltype(ns)::value)
            e = launch_one<kWaveSplit,
                           decltype(ns)::value,
                           kGroups,
                           I::kPageSize,
                           I::kQoFp16,
                           I::kLayout,
                           I::kHdim,
                           I::kMTiles,
                           I::kPerTokenScale>(h, stream);
    };
    (try_one(ck_tile::number<Ns + 1>{}), ...);
    return e;
}

template <typename I>
hipError_t launch_instance(const ck_tile::FmhaDecodePaged5dHostArgs& h,
                           const ck_tile::FmhaDecodePaged5dConfig& c,
                           hipStream_t stream)
{
    const auto seq = std::make_integer_sequence<index_t, ck_tile::kMaxClusterSplits>{};
    if(c.wave_split)
        return c.num_groups > 1 ? dispatch<true, true, I>(h, c.num_splits, stream, seq)
                                : dispatch<true, false, I>(h, c.num_splits, stream, seq);
    return c.num_groups > 1 ? dispatch<false, true, I>(h, c.num_splits, stream, seq)
                            : dispatch<false, false, I>(h, c.num_splits, stream, seq);
}

// launch_instance is instantiated in the translation units of fmha_decode_paged5d_instances.cmake
#define FMHA_DECODE_PAGED5D_INSTANCE(prefix, page_size, qo_fp16, layout, hdim, m_tiles, per_token) \
    prefix hipError_t                                                                              \
    launch_instance<Instance<page_size, qo_fp16, Layout::layout, hdim, m_tiles, per_token>>(       \
        const ck_tile::FmhaDecodePaged5dHostArgs&,                                                 \
        const ck_tile::FmhaDecodePaged5dConfig&,                                                   \
        hipStream_t);
#define FMHA_DECODE_PAGED5D_EXTERN_QO(layout, hdim, page_size, m_tiles, per_token) \
    FMHA_DECODE_PAGED5D_INSTANCE(                                                  \
        extern template, page_size, false, layout, hdim, m_tiles, per_token)       \
    FMHA_DECODE_PAGED5D_INSTANCE(extern template, page_size, true, layout, hdim, m_tiles, per_token)
#define FMHA_DECODE_PAGED5D_EXTERN_MT(layout, hdim, page_size)       \
    FMHA_DECODE_PAGED5D_EXTERN_QO(layout, hdim, page_size, 1, false) \
    FMHA_DECODE_PAGED5D_EXTERN_QO(layout, hdim, page_size, 2, false) \
    FMHA_DECODE_PAGED5D_EXTERN_QO(layout, hdim, page_size, 1, true)  \
    FMHA_DECODE_PAGED5D_EXTERN_QO(layout, hdim, page_size, 2, true)
#define FMHA_DECODE_PAGED5D_EXTERN_PAGE(layout, hdim) \
    FMHA_DECODE_PAGED5D_EXTERN_MT(layout, hdim, 16)   \
    FMHA_DECODE_PAGED5D_EXTERN_MT(layout, hdim, 64)   \
    FMHA_DECODE_PAGED5D_EXTERN_MT(layout, hdim, 128)
#define FMHA_DECODE_PAGED5D_EXTERN_LAYOUT(layout) \
    FMHA_DECODE_PAGED5D_EXTERN_PAGE(layout, 64)   \
    FMHA_DECODE_PAGED5D_EXTERN_PAGE(layout, 128)  \
    FMHA_DECODE_PAGED5D_EXTERN_PAGE(layout, 256)
FMHA_DECODE_PAGED5D_EXTERN_LAYOUT(shuffle_5d)
FMHA_DECODE_PAGED5D_EXTERN_LAYOUT(hnd)
FMHA_DECODE_PAGED5D_EXTERN_LAYOUT(nhd)
#undef FMHA_DECODE_PAGED5D_EXTERN_LAYOUT
#undef FMHA_DECODE_PAGED5D_EXTERN_PAGE
#undef FMHA_DECODE_PAGED5D_EXTERN_MT
#undef FMHA_DECODE_PAGED5D_EXTERN_QO

template <index_t kPageSize, bool kQoFp16, Layout kLayout, index_t kHdim, index_t kMTiles>
hipError_t launch_scale(const ck_tile::FmhaDecodePaged5dHostArgs& h,
                        const ck_tile::FmhaDecodePaged5dConfig& c,
                        hipStream_t stream)
{
    return h.per_token_scale
               ? launch_instance<Instance<kPageSize, kQoFp16, kLayout, kHdim, kMTiles, true>>(
                     h, c, stream)
               : launch_instance<Instance<kPageSize, kQoFp16, kLayout, kHdim, kMTiles, false>>(
                     h, c, stream);
}

template <index_t kPageSize, bool kQoFp16, Layout kLayout, index_t kHdim>
hipError_t launch_mt(const ck_tile::FmhaDecodePaged5dHostArgs& h,
                     const ck_tile::FmhaDecodePaged5dConfig& c,
                     hipStream_t stream)
{
    switch(c.m_tiles)
    {
    case 1: return launch_scale<kPageSize, kQoFp16, kLayout, kHdim, 1>(h, c, stream);
    case 2: return launch_scale<kPageSize, kQoFp16, kLayout, kHdim, 2>(h, c, stream);
    default: return hipErrorInvalidValue;
    }
}

template <bool kQoFp16, Layout kLayout, index_t kHdim>
hipError_t launch_hdim(const ck_tile::FmhaDecodePaged5dHostArgs& h,
                       const ck_tile::FmhaDecodePaged5dConfig& c,
                       hipStream_t stream)
{
    switch(h.page_size)
    {
    case 16: return launch_mt<16, kQoFp16, kLayout, kHdim>(h, c, stream);
    case 64: return launch_mt<64, kQoFp16, kLayout, kHdim>(h, c, stream);
    case 128: return launch_mt<128, kQoFp16, kLayout, kHdim>(h, c, stream);
    default: return hipErrorInvalidValue;
    }
}

template <bool kQoFp16, Layout kLayout>
hipError_t launch_type(const ck_tile::FmhaDecodePaged5dHostArgs& h,
                       const ck_tile::FmhaDecodePaged5dConfig& c,
                       hipStream_t stream)
{
    switch(h.hdim)
    {
    case 64: return launch_hdim<kQoFp16, kLayout, 64>(h, c, stream);
    case 128: return launch_hdim<kQoFp16, kLayout, 128>(h, c, stream);
    case 256: return launch_hdim<kQoFp16, kLayout, 256>(h, c, stream);
    default: return hipErrorInvalidValue;
    }
}

inline hipError_t launch(const ck_tile::FmhaDecodePaged5dHostArgs& h,
                         const ck_tile::FmhaDecodePaged5dConfig& c,
                         hipStream_t stream)
{
    // larger cache pages need whole 4-page groups per split (see FmhaDecodePaged5dKernel::Run)
    if(h.page_size != 16 && c.pages_per_split % 4 != 0)
        return hipErrorInvalidValue;
    // the row blocks of c must cover the query rows
    if(h.qlen < 1 || c.row_blocks * c.m_tiles * 16 < h.nhead_q / h.nhead_k * h.qlen)
        return hipErrorInvalidValue;
    const auto with_layout = [&](auto layout) {
        constexpr Layout kLayout = decltype(layout)::value;
        return h.qo_fp16 ? launch_type<true, kLayout>(h, c, stream)
                         : launch_type<false, kLayout>(h, c, stream);
    };
    switch(h.layout)
    {
    case Layout::shuffle_5d:
        return with_layout(std::integral_constant<Layout, Layout::shuffle_5d>{});
    case Layout::hnd: return with_layout(std::integral_constant<Layout, Layout::hnd>{});
    case Layout::nhd: return with_layout(std::integral_constant<Layout, Layout::nhd>{});
    }
    return hipErrorInvalidValue;
}

inline float bf16_to_float(uint16_t x)
{
    return ck_tile::type_convert<float>(ck_tile::bit_cast<ck_tile::bf16_t>(x));
}
inline float fp16_to_float(uint16_t x)
{
    return ck_tile::type_convert<float>(ck_tile::bit_cast<ck_tile::fp16_t>(x));
}
inline float fp8_to_float(uint8_t x)
{
    return ck_tile::type_convert<float>(ck_tile::bit_cast<ck_tile::fp8_t>(x));
}

struct Case
{
    index_t batch   = 1;
    index_t nhead   = 16;
    index_t nhead_k = 1;
    std::vector<index_t> seqlen_k{8192}; // one per batch, or one for all
    index_t window          = 2048;      // 0 = no window
    index_t splits          = 0;         // 0 = heuristic
    index_t pages_per_split = 0;         // 0 = heuristic
    std::string layout      = "auto";    // auto, coop, wave
    index_t max_seqlen_k    = 0;         // the bound the grid is sized for; 0 = longest seqlen_k
    index_t page_size       = 16;        // tokens per cache page: 16, 64 or 128
    bool fp16               = false;     // fp16 Q and output instead of bf16
    Layout kv_layout        = Layout::shuffle_5d;
    index_t hdim            = 128;   // 64, 128 or 256
    index_t qlen            = 1;     // query tokens per sequence (MTP)
    index_t m_tiles         = 0;     // 16-row tiles per workgroup: 1 or 2; 0 = heuristic
    bool sink               = false; // random sinks per q head (one -inf, one +inf with >= 4 heads)
    bool lse                = false; // store and check the log-sum-exp
    // per-token K/V scales instead of one each, K 2^-1 .. 2^1, V 2^-2 .. 2^2: larger K scales
    // sharpen these scores (fp8 Q error amplified), wider V scales let a few tokens carry the
    // output with the fp8 P error of each, past the checks
    bool per_token = false;
    unsigned seed  = 1;
};

struct Result
{
    bool valid     = false; // set only when validated
    double max_err = 0, max_ref = 0, cosine = 0;
    double max_lse_err = 0; // set only when the case stores the log-sum-exp
    ck_tile::FmhaDecodePaged5dConfig config{};
    float us_stream = 0, us_graph = 0; // set only when timed
    bool graph_same = true;            // graph replay wrote the output of a stream launch
    std::string error;
};

// Runs one case: random Q (bf16 or fp16), K/V (finite fp8 in [-1, 1], pages in a random order, the
// unused tail of each sequence's last page filled with NaN codes), a workspace filled with NaN
// codes, then validates against a CPU reference and/or times stream and HIP-graph launches.
inline Result run(const Case& c, bool validate, int warmup = 0, int repeat = 0)
{
    Result res;
    const index_t batch = c.batch;
    std::vector<index_t> seqlens(batch);
    for(index_t b = 0; b < batch; ++b)
        seqlens[b] = c.seqlen_k[std::min<size_t>(b, c.seqlen_k.size() - 1)];
    const index_t max_seqlen =
        std::max(c.max_seqlen_k, *std::max_element(seqlens.begin(), seqlens.end()));
    if(c.nhead % c.nhead_k != 0 || c.qlen < 1)
    {
        res.error = "h must be a multiple of h_k, q_len at least 1";
        return res;
    }
    const index_t gqa  = c.nhead / c.nhead_k;
    const index_t qlen = c.qlen;

    const index_t ps = c.page_size;
    if(ps != 16 && ps != 64 && ps != 128)
    {
        res.error = "page_size must be 16, 64 or 128";
        return res;
    }
    auto cfg = ck_tile::fmha_decode_paged5d_config(
        batch, c.nhead_k, max_seqlen, c.window, ps, c.hdim, gqa * qlen, qlen, c.m_tiles);
    if(c.layout == "coop" || c.layout == "wave")
        cfg.wave_split = c.layout == "wave";
    const index_t lo_align = ps > kPage ? 4 : 1;
    const index_t span     = ck_tile::fmha_decode_paged5d_max_span(max_seqlen, c.window, ps, qlen);
    if(c.pages_per_split > 0)
        cfg.pages_per_split = c.pages_per_split;
    if(c.splits > 0)
        cfg.pages_per_split = (span + c.splits - 1) / c.splits;
    if(c.pages_per_split > 0 || c.splits > 0)
    {
        cfg.pages_per_split = (cfg.pages_per_split + lo_align - 1) / lo_align * lo_align;
        const index_t total = (span + cfg.pages_per_split - 1) / cfg.pages_per_split;
        cfg.num_groups      = (total + ck_tile::kMaxClusterSplits - 1) / ck_tile::kMaxClusterSplits;
        cfg.num_splits      = (total + cfg.num_groups - 1) / cfg.num_groups;
    }
    res.config = cfg;

    std::mt19937 rng(c.seed);
    std::uniform_real_distribution<float> uni(-1.f, 1.f);
    const index_t max_blocks = (max_seqlen + ps - 1) / ps;
    index_t num_blocks       = 1;
    for(index_t b = 0; b < batch; ++b)
        num_blocks += (seqlens[b] + ps - 1) / ps;
    const index_t kHdim = c.hdim;
    if(kHdim != 64 && kHdim != 128 && kHdim != 256)
    {
        res.error = "hdim must be 64, 128 or 256";
        return res;
    }
    // one cache page of one kv head, K and V alike
    const index_t page_bytes = ps * kHdim;
    // byte of (block, kv head, token in the page, head dim) in the K or V cache
    const auto addr = [&](index_t blk, index_t hk, index_t tok, index_t d, bool is_v) {
        const size_t blk_head = static_cast<size_t>(blk) * c.nhead_k + hk;
        if(c.kv_layout == Layout::hnd)
            return (blk_head * ps + tok) * kHdim + d;
        if(c.kv_layout == Layout::nhd)
            return ((static_cast<size_t>(blk) * ps + tok) * c.nhead_k + hk) * kHdim + d;
        return blk_head * page_bytes + (is_v ? (tok / 16) * kHdim * 16 + d * 16 + tok % 16
                                             : (d / 16) * ps * 16 + tok * 16 + d % 16);
    };
    // [batch][qlen][nhead][hdim]
    std::vector<uint16_t> q(static_cast<size_t>(batch) * qlen * c.nhead * kHdim);
    for(auto& x : q)
        x = c.fp16 ? ck_tile::bit_cast<uint16_t>(
                         ck_tile::type_convert<ck_tile::fp16_t>(2.f * uni(rng)))
                   : ck_tile::bit_cast<uint16_t>(
                         ck_tile::type_convert<ck_tile::bf16_t>(2.f * uni(rng)));
    const auto qo_to_float = [&](uint16_t x) {
        return c.fp16 ? fp16_to_float(x) : bf16_to_float(x);
    };
    std::vector<uint8_t> finite;
    for(int i = 0; i < 256; ++i)
        if((i & 0x7f) != 0x7f && std::fabs(fp8_to_float(static_cast<uint8_t>(i))) <= 1.f)
            finite.push_back(static_cast<uint8_t>(i));
    std::uniform_int_distribution<size_t> pick(0, finite.size() - 1);
    std::vector<uint8_t> k(static_cast<size_t>(num_blocks) * c.nhead_k * page_bytes);
    std::vector<uint8_t> v(k.size());
    for(auto& x : k)
        x = finite[pick(rng)];
    for(auto& x : v)
        x = finite[pick(rng)];
    std::vector<int32_t> perm(num_blocks - 1);
    std::iota(perm.begin(), perm.end(), 1);
    std::shuffle(perm.begin(), perm.end(), rng);
    std::vector<int32_t> block_table(static_cast<size_t>(batch) * max_blocks, 0);
    std::vector<int32_t> seqlens32(seqlens.begin(), seqlens.end());
    {
        size_t next = 0;
        for(index_t b = 0; b < batch; ++b)
        {
            const index_t pages = (seqlens[b] + ps - 1) / ps;
            for(index_t p = 0; p < pages; ++p)
                block_table[b * max_blocks + p] = perm[next++];
            if(pages == 0)
                continue;
            const int32_t last = block_table[b * max_blocks + pages - 1];
            for(index_t hk = 0; hk < c.nhead_k; ++hk)
                for(index_t tok = seqlens[b] - (pages - 1) * ps; tok < ps; ++tok)
                    for(index_t d = 0; d < kHdim; ++d)
                    {
                        k[addr(last, hk, tok, d, false)] = 0x7f;
                        v[addr(last, hk, tok, d, true)]  = 0x7f;
                    }
        }
    }
    const float k_descale = 0.75f, v_descale = 1.25f;
    // per-token scales [num_blocks][nhead_k][page], NaN past each sequence's last token
    std::vector<float> ks(c.per_token ? k.size() / kHdim : 0), vs(ks.size());
    for(size_t i = 0; i < ks.size(); ++i)
    {
        ks[i] = std::exp2(uni(rng));
        vs[i] = std::exp2(2.f * uni(rng));
    }
    if(c.per_token)
        for(index_t b = 0; b < batch; ++b)
        {
            const index_t pages = (seqlens[b] + ps - 1) / ps;
            if(pages == 0)
                continue;
            const size_t last = static_cast<size_t>(block_table[b * max_blocks + pages - 1]);
            for(index_t hk = 0; hk < c.nhead_k; ++hk)
                for(index_t tok = seqlens[b] - (pages - 1) * ps; tok < ps; ++tok)
                    ks[(last * c.nhead_k + hk) * ps + tok] =
                        vs[(last * c.nhead_k + hk) * ps + tok] =
                            std::numeric_limits<float>::quiet_NaN();
        }
    // K or V scale of token t of block blk, kv head hk
    const auto tok_scale = [&](bool is_v, int32_t blk, index_t hk, index_t t) {
        return c.per_token ? (is_v ? vs : ks)[(static_cast<size_t>(blk) * c.nhead_k + hk) * ps + t]
                           : (is_v ? v_descale : k_descale);
    };
    // sinks around the scale of the scores; -inf (none) and +inf (no output) once each
    std::vector<float> sinks(c.nhead, -std::numeric_limits<float>::infinity());
    if(c.sink)
    {
        for(auto& x : sinks)
            x = 3.f * uni(rng);
        if(c.nhead >= 4)
        {
            sinks[1] = -std::numeric_limits<float>::infinity();
            sinks[2] = std::numeric_limits<float>::infinity();
        }
    }
    const float scale_s = 1.f / std::sqrt(static_cast<float>(kHdim));

    ck_tile::DeviceMem q_buf(q.size() * 2), k_buf(k.size()), v_buf(v.size());
    ck_tile::DeviceMem bt_buf(block_table.size() * 4), sl_buf(seqlens32.size() * 4);
    ck_tile::DeviceMem o_buf(q.size() * 2), scale_buf(8);
    ck_tile::DeviceMem ks_buf(ks.size() * 4), vs_buf(vs.size() * 4);
    ck_tile::DeviceMem sink_buf(sinks.size() * 4), lse_buf(q.size() / kHdim * 4);
    ck_tile::DeviceMem ws_buf(
        ck_tile::fmha_decode_paged5d_workspace_bytes(batch,
                                                     c.nhead_k,
                                                     cfg.num_splits,
                                                     cfg.num_groups,
                                                     kHdim,
                                                     cfg.row_blocks * cfg.m_tiles * 16));
    ck_tile::DeviceMem cnt_buf(
        ck_tile::fmha_decode_paged5d_num_counters(batch, c.nhead_k, cfg.row_blocks) * 4);
    HIP_CHECK_ERROR(hipMemset(ws_buf.GetDeviceBuffer(), 0xff, ws_buf.GetBufferSize()));
    cnt_buf.SetZero();
    q_buf.ToDevice(q.data());
    k_buf.ToDevice(k.data());
    v_buf.ToDevice(v.data());
    bt_buf.ToDevice(block_table.data());
    sl_buf.ToDevice(seqlens32.data());
    const float scales[2] = {k_descale, v_descale};
    scale_buf.ToDevice(scales);
    ks_buf.ToDevice(ks.data());
    vs_buf.ToDevice(vs.data());
    sink_buf.ToDevice(sinks.data());
    HIP_CHECK_ERROR(hipMemset(lse_buf.GetDeviceBuffer(), 0xff, lse_buf.GetBufferSize()));

    const ck_tile::FmhaDecodePaged5dHostArgs args{
        q_buf.GetDeviceBuffer(),
        k_buf.GetDeviceBuffer(),
        v_buf.GetDeviceBuffer(),
        static_cast<const int32_t*>(bt_buf.GetDeviceBuffer()),
        static_cast<const int32_t*>(sl_buf.GetDeviceBuffer()),
        static_cast<const float*>(c.per_token ? ks_buf.GetDeviceBuffer()
                                              : scale_buf.GetDeviceBuffer()),
        static_cast<const float*>(c.per_token ? vs_buf.GetDeviceBuffer()
                                              : scale_buf.GetDeviceBuffer()) +
            (c.per_token ? 0 : 1),
        o_buf.GetDeviceBuffer(),
        ws_buf.GetDeviceBuffer(),
        static_cast<int32_t*>(cnt_buf.GetDeviceBuffer()),
        batch,
        c.nhead,
        c.nhead_k,
        max_blocks,
        ps,
        c.window,
        cfg.pages_per_split,
        cfg.num_groups,
        scale_s,
        c.fp16,
        c.kv_layout,
        kHdim,
        qlen,
        c.per_token,
        c.sink ? sink_buf.GetDeviceBuffer() : nullptr,
        0,
        c.lse ? static_cast<float*>(lse_buf.GetDeviceBuffer()) : nullptr};

    hipStream_t stream;
    HIP_CHECK_ERROR(hipDeviceSynchronize()); // the fills above, before a non-blocking stream
    HIP_CHECK_ERROR(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    HIP_CHECK_ERROR(launch(args, cfg, stream));
    HIP_CHECK_ERROR(hipStreamSynchronize(stream));

    if(validate)
    {
        std::vector<uint16_t> out(q.size());
        o_buf.FromDevice(out.data());
        std::vector<float> lse(q.size() / kHdim);
        lse_buf.FromDevice(lse.data());
        double dot = 0, nr = 0, ng = 0;
        for(index_t b = 0; b < batch; ++b)
            for(index_t qi = 0; qi < qlen; ++qi)
                for(index_t hq = 0; hq < c.nhead; ++hq)
                {
                    // query token qi is the context's (qlen - qi)-th last
                    const index_t sk  = std::max(seqlens[b] - qlen + 1 + qi, 0);
                    const index_t lo  = c.window > 0 ? std::max(0, sk - c.window) : 0;
                    const size_t qrow = (static_cast<size_t>(b) * qlen + qi) * c.nhead + hq;
                    const index_t hk  = hq / gqa;
                    const auto at     = [&](index_t t, index_t d, bool is_v) {
                        return addr(block_table[b * max_blocks + t / ps], hk, t % ps, d, is_v);
                    };
                    std::vector<double> sc(std::max(0, sk - lo));
                    double mx = -1e300;
                    for(index_t t = lo; t < sk; ++t)
                    {
                        double acc = 0;
                        for(index_t d = 0; d < kHdim; ++d)
                            acc +=
                                qo_to_float(q[qrow * kHdim + d]) * fp8_to_float(k[at(t, d, false)]);
                        sc[t - lo] =
                            acc *
                            tok_scale(false, block_table[b * max_blocks + t / ps], hk, t % ps) *
                            scale_s;
                        mx = std::max(mx, sc[t - lo]);
                    }
                    double l = 0;
                    std::vector<double> o(kHdim, 0.0);
                    for(index_t t = lo; t < sk; ++t)
                    {
                        const double p = std::exp(sc[t - lo] - mx);
                        l += p;
                        const double pv =
                            p * tok_scale(true, block_table[b * max_blocks + t / ps], hk, t % ps);
                        for(index_t d = 0; d < kHdim; ++d)
                            o[d] += pv * fp8_to_float(v[at(t, d, true)]);
                    }
                    // the sink joins the denominator; no keys: the max is the sink's
                    if(sk <= lo)
                        mx = 0;
                    if(c.sink)
                        l += std::exp(static_cast<double>(sinks[hq]) - mx);
                    if(c.lse)
                    {
                        const double ref = l > 0 ? mx + std::log(l) : -INFINITY;
                        const double got = lse[qrow];
                        const double err = std::isinf(ref) && std::isinf(got) &&
                                                   std::signbit(ref) == std::signbit(got)
                                               ? 0.0
                                               : std::fabs(ref - got);
                        res.max_lse_err  = std::max(res.max_lse_err, err);
                    }
                    for(index_t d = 0; d < kHdim; ++d)
                    {
                        const double ref = l > 0 && std::isfinite(l) ? o[d] / l : 0.0;
                        const double got = qo_to_float(out[qrow * kHdim + d]);
                        res.max_err      = std::max(res.max_err, std::fabs(ref - got));
                        res.max_ref      = std::max(res.max_ref, std::fabs(ref));
                        dot += ref * got;
                        nr += ref * ref;
                        ng += got * got;
                    }
                }
        res.cosine = nr + ng <= 0.0 ? 1.0 : dot / std::sqrt(nr * ng + 1e-300);
        // fp8 Q and P: the output follows the reference direction tightly, element errors stay
        // a few percent of the largest output
        res.valid = std::isfinite(res.max_err) && res.cosine > 0.999 &&
                    res.max_err < 0.05 * std::max(res.max_ref, 1e-3);
        // scores from fp8 Q and K: a few hundredths in the log
        if(c.lse)
            res.valid = res.valid && res.max_lse_err < 0.05;
    }

    if(repeat > 0)
    {
        for(int i = 0; i < warmup; ++i)
            HIP_CHECK_ERROR(launch(args, cfg, stream));
        hipEvent_t e0, e1;
        HIP_CHECK_ERROR(hipEventCreate(&e0));
        HIP_CHECK_ERROR(hipEventCreate(&e1));
        HIP_CHECK_ERROR(hipStreamSynchronize(stream));
        HIP_CHECK_ERROR(hipEventRecord(e0, stream));
        for(int i = 0; i < repeat; ++i)
            HIP_CHECK_ERROR(launch(args, cfg, stream));
        HIP_CHECK_ERROR(hipEventRecord(e1, stream));
        HIP_CHECK_ERROR(hipEventSynchronize(e1));
        float ms = 0;
        HIP_CHECK_ERROR(hipEventElapsedTime(&ms, e0, e1));
        res.us_stream = ms * 1e3f / repeat;

        // HIP graph replay, how a serving engine runs decode
        hipGraph_t graph;
        hipGraphExec_t graph_exec;
        HIP_CHECK_ERROR(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal));
        for(int i = 0; i < repeat; ++i)
            HIP_CHECK_ERROR(launch(args, cfg, stream));
        HIP_CHECK_ERROR(hipStreamEndCapture(stream, &graph));
        HIP_CHECK_ERROR(hipGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
        {
            std::vector<uint16_t> ref_out(q.size()), graph_out(q.size());
            o_buf.FromDevice(ref_out.data());
            // stale partials of the stream launches must not be able to stand in
            HIP_CHECK_ERROR(
                hipMemsetAsync(o_buf.GetDeviceBuffer(), 0xff, o_buf.GetBufferSize(), stream));
            HIP_CHECK_ERROR(
                hipMemsetAsync(ws_buf.GetDeviceBuffer(), 0xff, ws_buf.GetBufferSize(), stream));
            HIP_CHECK_ERROR(hipGraphLaunch(graph_exec, stream));
            HIP_CHECK_ERROR(hipStreamSynchronize(stream));
            o_buf.FromDevice(graph_out.data());
            res.graph_same = ref_out == graph_out;
        }
        float best = 1e30f;
        for(int r = 0; r < 5; ++r)
        {
            HIP_CHECK_ERROR(hipEventRecord(e0, stream));
            HIP_CHECK_ERROR(hipGraphLaunch(graph_exec, stream));
            HIP_CHECK_ERROR(hipEventRecord(e1, stream));
            HIP_CHECK_ERROR(hipEventSynchronize(e1));
            HIP_CHECK_ERROR(hipEventElapsedTime(&ms, e0, e1));
            best = std::min(best, ms);
        }
        res.us_graph = best * 1e3f / repeat;
        HIP_CHECK_ERROR(hipGraphExecDestroy(graph_exec));
        HIP_CHECK_ERROR(hipGraphDestroy(graph));
        HIP_CHECK_ERROR(hipEventDestroy(e0));
        HIP_CHECK_ERROR(hipEventDestroy(e1));
    }
    HIP_CHECK_ERROR(hipStreamDestroy(stream));
    return res;
}

} // namespace fmha_decode_paged5d
