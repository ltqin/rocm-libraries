// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha/block/page_block_navigator.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm_policy.hpp"

namespace ck_tile {

namespace detail {

template <typename Problem>
inline constexpr bool is_qr_tdm_pingpong_qkv_v =
    (std::is_same_v<typename Problem::QDataType, bf16_t> &&
     std::is_same_v<typename Problem::KDataType, bf16_t> &&
     std::is_same_v<typename Problem::VDataType, bf16_t>) ||
    (std::is_same_v<typename Problem::QDataType, half_t> &&
     std::is_same_v<typename Problem::KDataType, half_t> &&
     std::is_same_v<typename Problem::VDataType, half_t>) ||
    (std::is_same_v<typename Problem::QDataType, fp8_t> &&
     std::is_same_v<typename Problem::KDataType, fp8_t> &&
     std::is_same_v<typename Problem::VDataType, fp8_t>);

// A split-KV problem with kDynamicQ reads Q as QInputDataType and quantizes it to QDataType in
// the pipeline (see BlockFmhaFwdSplitKVPipelineProblem); every other problem reads QDataType.
template <typename Problem, typename = void>
inline constexpr bool problem_dynamic_q_v = false;

template <typename Problem>
inline constexpr bool problem_dynamic_q_v<Problem, std::void_t<decltype(Problem::kDynamicQ)>> =
    Problem::kDynamicQ;

template <typename Problem, bool = problem_dynamic_q_v<Problem>>
struct q_input_data
{
    using type = typename Problem::QDataType;
};

template <typename Problem>
struct q_input_data<Problem, true>
{
    using type = typename Problem::QInputDataType;
};

template <typename Problem>
using q_input_data_t = typename q_input_data<Problem>::type;

// One pad per row, expressed as a row stride: the layout the TDM writer produces with a
// padding interval of one row, with affine reader address math (no merge/unmerge div/mod).
template <typename DataType, index_t Rows, index_t Cols, typename PaddingConfig>
CK_TILE_HOST_DEVICE constexpr auto make_qr_tdm_pingpong_lds_descriptor()
{
    static_assert(numeric_traits<DataType>::PackedSize == 1);
    constexpr index_t kAccessElements = kQrTdmLdsAccessBytes / sizeof(DataType);
    constexpr index_t kPadElements    = PaddingConfig::kPadBytes / sizeof(DataType);

    static_assert(!PaddingConfig::kEnabled ||
                      (PaddingConfig::kIntervalBytes == Cols * sizeof(DataType) &&
                       PaddingConfig::kPadBytes % sizeof(DataType) == 0),
                  "qr_tdm_pingpong pads each LDS row once");
    return make_naive_tensor_descriptor(make_tuple(number<Rows>{}, number<Cols>{}),
                                        make_tuple(number<Cols + kPadElements>{}, number<1>{}),
                                        number<kAccessElements>{},
                                        number<1>{});
}

// Rows padded once need a power-of-2 row width, like the TDM padding interval.
template <typename Problem>
inline constexpr bool is_qr_tdm_pingpong_padding_enabled_v =
    is_qr_tdm_padding_supported_arch_v && Problem::BlockFmhaShape::IsVLayoutRowMajor &&
    is_power_of_two_integer(Problem::BlockFmhaShape::kSubQKHeaddim *
                            sizeof(typename Problem::KDataType)) &&
    is_power_of_two_integer(Problem::BlockFmhaShape::kN1 * sizeof(typename Problem::VDataType)) &&
    Problem::BlockFmhaShape::kSubQKHeaddim * sizeof(typename Problem::KDataType) >= 16 &&
    Problem::BlockFmhaShape::kSubQKHeaddim * sizeof(typename Problem::KDataType) <= 1024 &&
    Problem::BlockFmhaShape::kN1 * sizeof(typename Problem::VDataType) >= 16 &&
    Problem::BlockFmhaShape::kN1 * sizeof(typename Problem::VDataType) <= 1024;

// 16 B per Q and K row (row-strided ds_load_b128), 32 B per V row (transposed ds_load_tr).
template <typename Problem, bool Enabled = is_qr_tdm_pingpong_padding_enabled_v<Problem>>
struct QrTdmPingPongPadding
{
    using Q = LdsPaddingConfig<false, 0, 0>;
    using K = LdsPaddingConfig<false, 0, 0>;
    using V = LdsPaddingConfig<false, 0, 0>;
};

template <typename Problem>
struct QrTdmPingPongPadding<Problem, true>
{
    using Shape = typename Problem::BlockFmhaShape;

    using Q = LdsPaddingConfig<true,
                               Shape::kSubQKHeaddim * sizeof(q_input_data_t<Problem>),
                               kQrTdmLdsAccessBytes>;
    using K = LdsPaddingConfig<true,
                               Shape::kSubQKHeaddim * sizeof(typename Problem::KDataType),
                               kQrTdmLdsAccessBytes>;
    using V = LdsPaddingConfig<true,
                               Shape::kN1 * sizeof(typename Problem::VDataType),
                               2 * kQrTdmLdsAccessBytes>;
};

// [K buffer 0][K buffer 1][S][V, aliased by Q]. Q is read once before the main loop and
// V's first write follows the loop-head barrier, so Q sits on V and K's first transfer can
// be issued right behind Q's. S only takes space with the gemm_0 warps on N. d128
// fp16/bf16, padded: K0=0, K1=17408, Q/V0=34816, end=53248 (warps on M, M=64); the
// 16-row short-query tile: S=34800, Q/V0=39168, end=57600.
template <typename Problem, typename QPadding, typename KPadding, typename VPadding>
struct QrTdmPingPongLdsArenaLayout
{
    using Shape = typename Problem::BlockFmhaShape;

    static constexpr index_t kArenaAlignment  = 256;
    static constexpr index_t kRegionAlignment = 256;

    static constexpr index_t kQBytes = make_qr_tdm_pingpong_lds_descriptor<q_input_data_t<Problem>,
                                                                           Shape::kM0,
                                                                           Shape::kSubQKHeaddim,
                                                                           QPadding>()
                                           .get_element_space_size() *
                                       sizeof(q_input_data_t<Problem>);
    static constexpr index_t kKBytes =
        make_qr_tdm_pingpong_lds_descriptor<typename Problem::KDataType,
                                            Shape::kN0,
                                            Shape::kSubQKHeaddim,
                                            KPadding>()
            .get_element_space_size() *
        sizeof(typename Problem::KDataType);
    static constexpr index_t kVBytes =
        make_qr_tdm_pingpong_lds_descriptor<typename Problem::VDataType,
                                            Shape::kN0,
                                            Shape::kN1,
                                            VPadding>()
            .get_element_space_size() *
        sizeof(typename Problem::VDataType);

    static constexpr index_t kSBytes =
        BlockFmhaPipelineQRKSVSTdmDefaultPolicy::template GetSmemSizeS<Problem>();
    static constexpr index_t kSAlignment =
        BlockFmhaPipelineQRKSVSTdmDefaultPolicy::template GetSmemNPackS<Problem>() *
        sizeof(typename Problem::SaccDataType);

    static constexpr index_t kK0Offset = 0;
    static constexpr index_t kK1Offset =
        integer_least_multiple(kK0Offset + kKBytes, kRegionAlignment);
    static constexpr index_t kSOffset = integer_least_multiple(kK1Offset + kKBytes, kSAlignment);
    static constexpr index_t kV0Offset =
        integer_least_multiple(kSOffset + kSBytes, kRegionAlignment);
    static constexpr index_t kQOffset = kV0Offset;
    static constexpr index_t kArenaBytes =
        integer_least_multiple(max(kQOffset + kQBytes, kV0Offset + kVBytes), kArenaAlignment);

    static_assert(!KPadding::kEnabled || (kK1Offset - kK0Offset) % KPadding::kIntervalBytes == 0);
    static_assert(kArenaBytes <= 128 * 1024);
};

} // namespace detail

// qr_tdm_pingpong: the qr_tdm policy with a single V buffer, whole-tile K staged in two
// buffers, and Q on V (see QrTdmPingPongLdsArenaLayout).
struct BlockFmhaPipelineQRKSVSTdmPingPongPolicy : BlockFmhaPipelineQRKSVSTdmDefaultPolicy
{
    template <typename Problem>
    using LdsPaddingConfigQ = typename detail::QrTdmPingPongPadding<Problem>::Q;

    template <typename Problem>
    using LdsPaddingConfigK = typename detail::QrTdmPingPongPadding<Problem>::K;

    template <typename Problem>
    using LdsPaddingConfigV = typename detail::QrTdmPingPongPadding<Problem>::V;

    template <typename Problem,
              typename QPadding = LdsPaddingConfigQ<Problem>,
              typename KPadding = LdsPaddingConfigK<Problem>,
              typename VPadding = LdsPaddingConfigV<Problem>>
    using LdsArenaLayout =
        detail::QrTdmPingPongLdsArenaLayout<Problem, QPadding, KPadding, VPadding>;

    // the qr_tdm alignment, of the Q that is read (see detail::q_input_data_t)
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetAlignmentQ()
    {
        constexpr index_t kElemPerThread = Problem::BlockFmhaShape::kM0 *
                                           Problem::BlockFmhaShape::kSubQKHeaddim /
                                           Problem::kBlockSize;
        static_assert(0 < kElemPerThread);
        return min(kElemPerThread,
                   static_cast<index_t>(16 / sizeof(detail::q_input_data_t<Problem>)));
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeQLdsBlockDescriptor()
    {
        return detail::make_qr_tdm_pingpong_lds_descriptor<detail::q_input_data_t<Problem>,
                                                           Problem::BlockFmhaShape::kM0,
                                                           Problem::BlockFmhaShape::kSubQKHeaddim,
                                                           LdsPaddingConfigQ<Problem>>();
    }

    // The whole K tile; the pipeline walks its kK0 slices with the read window.
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeKLdsBlockDescriptor()
    {
        return detail::make_qr_tdm_pingpong_lds_descriptor<typename Problem::KDataType,
                                                           Problem::BlockFmhaShape::kN0,
                                                           Problem::BlockFmhaShape::kSubQKHeaddim,
                                                           LdsPaddingConfigK<Problem>>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeVLdsBlockDescriptor()
    {
        return detail::make_qr_tdm_pingpong_lds_descriptor<typename Problem::VDataType,
                                                           Problem::BlockFmhaShape::kN0,
                                                           Problem::BlockFmhaShape::kN1,
                                                           LdsPaddingConfigV<Problem>>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return LdsArenaLayout<Problem>::kArenaBytes;
    }

    // S comes back out of LDS (gemm_0 warps on N) to be softmaxed and fed into gemm_1
    // as its A operand, so it takes P's distribution. An encoding built from the QK
    // gemm's warp layout would permute P's columns within a row once the warps sit on
    // N: the row sums (and LSE) stay right while each key meets the wrong value row.
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeSRegTileDistribution()
    {
        return MakePRegTileDistribution<Problem>();
    }

    // Row sum of P on the matrix unit: P (M x kN0) times ones (kN0 x one warp tile), with
    // the PV gemm's types and warp gemm, so the sum lands replicated in o_acc's layout.
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetRowSumBlockGemm()
    {
        using PVBlockGemm = remove_cvref_t<decltype(GetPVBlockGemm<Problem>())>;
        using WarpGemm    = typename PVBlockGemm::WarpGemm;

        static_assert(Problem::BlockFmhaShape::Gemm1BlockWarps::at(number<1>{}) == 1,
                      "one warp tile of columns covers the row sum only with the gemm_1 "
                      "warps all on M");

        using GemmProblem = BlockGemmProblem<
            typename Problem::PDataType,
            typename Problem::VDataType,
            typename Problem::OaccDataType,
            Problem::kBlockSize,
            TileGemmShape<
                sequence<Problem::BlockFmhaShape::kM0, WarpGemm::kN, Problem::BlockFmhaShape::kK1>,
                typename Problem::BlockFmhaShape::Gemm1BlockWarps,
                typename Problem::BlockFmhaShape::Gemm1WarpTile>>;

        using BlockGemmPolicy =
            BlockGemmARegBRegCRegV2CustomPolicy<typename Problem::PDataType,
                                                typename Problem::VDataType,
                                                typename Problem::OaccDataType,
                                                typename Problem::BlockFmhaShape::Gemm1BlockWarps,
                                                WarpGemm,
                                                GemmLoopOrder::KMN>;

        return BlockGemmARegBRegCRegV2<GemmProblem, BlockGemmPolicy>{};
    }
};

// qr_tdm_pingpong on 5D paged KV (see Paged5DTileNavigator), one page per K tile: each warp's
// TDM box copies whole rows of a page into unpadded LDS, which keeps the page layout, and the
// reads go through that layout. V needs no transposed read: a page holds V token-minor.
struct BlockFmhaPipelineQRKSVSTdmPingPongPaged5DPolicy : BlockFmhaPipelineQRKSVSTdmPingPongPolicy
{
    template <typename Problem>
    using Page = Paged5DTileNavigator<typename Problem::KDataType,
                                      typename Problem::VDataType,
                                      Problem::BlockFmhaShape::kN0,
                                      Problem::BlockFmhaShape::kQKHeaddim>;

    template <typename Problem>
    using LdsPaddingConfigK = detail::LdsPaddingConfig<false, 0, 0>;

    template <typename Problem>
    using LdsPaddingConfigV = detail::LdsPaddingConfig<false, 0, 0>;

    template <typename Problem,
              typename QPadding = LdsPaddingConfigQ<Problem>,
              typename KPadding = LdsPaddingConfigK<Problem>,
              typename VPadding = LdsPaddingConfigV<Problem>>
    using LdsArenaLayout =
        detail::QrTdmPingPongLdsArenaLayout<Problem, QPadding, KPadding, VPadding>;

    // LDS of BlockFmhaPipelineQRKSVSTdmDecodePagedKV, whose two main loops share one arena:
    // - cooperative: [K x kStages][V x kStages]. A tile's K and V are staged kStages - 1 trips
    //   ahead of the trip that reads them, into the buffers of stage (tile % kStages).
    // - wave split: a block per warp, [K x kWaveStages][V x kWaveStages], which holds the
    //   warp's O for the merge at the end.
    // Behind them S's region (P and the row partials) and Q, which both loops read once
    // before they stage a tile over anything.
    template <typename Problem>
    struct DecodeLdsLayout
    {
        using Base = LdsArenaLayout<Problem>;

        static constexpr index_t kStages     = 3;
        static constexpr index_t kWaveStages = 2;
        static constexpr index_t kNumWarps   = Problem::kBlockSize / get_warp_size();

        static constexpr index_t kQBytes = Base::kQBytes;
        static constexpr index_t kKBytes = Base::kKBytes;
        static constexpr index_t kVBytes = Base::kVBytes;
        static constexpr index_t kSBytes = Base::kSBytes;

        static constexpr index_t kKStride = integer_least_multiple(kKBytes, Base::kRegionAlignment);
        static constexpr index_t kVStride = integer_least_multiple(kVBytes, Base::kRegionAlignment);

        static constexpr index_t kK0Offset = 0;
        static constexpr index_t kV0Offset = kK0Offset + kStages * kKStride;

        static constexpr index_t kWaveBlockBytes = kWaveStages * (kKStride + kVStride);
        static constexpr index_t kWaveOffset     = 0;

        static constexpr index_t kSOffset = integer_least_multiple(
            max(kV0Offset + kStages * kVStride, kWaveOffset + kNumWarps * kWaveBlockBytes),
            max(Base::kSAlignment, Base::kRegionAlignment));
        static constexpr index_t kQOffset =
            integer_least_multiple(kSOffset + kSBytes, Base::kRegionAlignment);
        static constexpr index_t kArenaBytes =
            integer_least_multiple(kQOffset + kQBytes, Base::kArenaAlignment);
    };

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize()
    {
        return DecodeLdsLayout<Problem>::kArenaBytes;
    }

    // The keys of a page a gemm_0 warp reads, (d / X, key * X): every warp copies its own with
    // one TDM box (see Paged5DTileNavigator::make_k_keys_view) into a block of its own, so
    // K takes no barrier.
    template <typename Problem>
    static constexpr index_t kKeysPerWarp =
        Page<Problem>::kPageSize / (Problem::kBlockSize / get_warp_size());

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeKWarpKeysDramTileDistribution()
    {
        using P                   = Page<Problem>;
        constexpr index_t warpNum = Problem::kBlockSize / get_warp_size();
        static_assert(Problem::BlockFmhaShape::Gemm0BlockWarps::at(number<1>{}) == warpNum);

        // the same box in every warp; the warps differ in the view they copy
        return make_static_tile_distribution(
            tile_distribution_encoding<
                sequence<warpNum>,
                tuple<sequence<P::kKRows>, sequence<kKeysPerWarp<Problem> * P::kX>>,
                tuple<sequence<0>>,
                tuple<sequence<0>>,
                sequence<1, 2>,
                sequence<0, 0>>{},
            bool_constant<true>{});
    }

    // a warp's block
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeKWarpKeysLdsBlockDescriptor()
    {
        using P = Page<Problem>;
        return MakeDenseLdsDescriptor<typename Problem::KDataType,
                                      P::kKRows,
                                      kKeysPerWarp<Problem> * P::kX>();
    }

    // the blocks of all warps, [warp][d / X][key of the warp][X], as K (key, d)
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeKWarpKeysLdsReadDescriptor()
    {
        using P                   = Page<Problem>;
        constexpr index_t warpNum = Problem::kBlockSize / get_warp_size();
        constexpr index_t kKeys   = P::kPageSize / warpNum;

        constexpr auto desc = make_naive_tensor_descriptor(
            make_tuple(number<warpNum>{}, number<P::kKRows>{}, number<kKeys>{}, number<P::kX>{}),
            make_tuple(number<P::kKRows * kKeys * P::kX>{},
                       number<kKeys * P::kX>{},
                       number<P::kX>{},
                       number<1>{}),
            number<P::kX>{},
            number<1>{});
        return transform_tensor_descriptor(
            desc,
            make_tuple(make_merge_transform_v3_division_mod(
                           make_tuple(number<warpNum>{}, number<kKeys>{})),
                       make_merge_transform_v3_division_mod(
                           make_tuple(number<P::kKRows>{}, number<P::kX>{}))),
            make_tuple(sequence<0, 2>{}, sequence<1, 3>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));
    }

    // Rows x Cols, each warp taking whole rows: one TDM box per warp
    template <typename Problem, index_t Rows, index_t Cols>
    CK_TILE_HOST_DEVICE static constexpr auto MakeWarpRowsDramTileDistribution()
    {
        constexpr index_t warpNum = Problem::kBlockSize / get_warp_size();
        static_assert(Rows % warpNum == 0);

        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<>,
                                       tuple<sequence<warpNum, Rows / warpNum>, sequence<Cols>>,
                                       tuple<sequence<1>>,
                                       tuple<sequence<0>>,
                                       sequence<1, 2>,
                                       sequence<1, 0>>{},
            bool_constant<true>{});
    }

    template <typename Problem, bool LoadOnce = false>
    CK_TILE_HOST_DEVICE static constexpr auto MakeKDramTileDistribution()
    {
        using P = Page<Problem>;
        return MakeWarpRowsDramTileDistribution<Problem, P::kKRows, P::kPageSize * P::kX>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeVDramTileDistribution()
    {
        using P = Page<Problem>;
        return MakeWarpRowsDramTileDistribution<Problem, P::kVRows, P::kHeadDim * P::kX>();
    }

    // see Paged5DTileNavigator::make_v_tail_view: warp w copies the X tokens of rows
    // [w * kHeadDim, (w + 1) * kHeadDim), the same bytes as its row of the full V box
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeVTailDramTileDistribution()
    {
        using P = Page<Problem>;
        static_assert(Problem::kBlockSize / get_warp_size() == P::kVRows);
        return MakeWarpRowsDramTileDistribution<Problem, P::kVRows * P::kHeadDim, P::kX>();
    }

    template <typename DataType, index_t Rows, index_t Cols>
    CK_TILE_HOST_DEVICE static constexpr auto MakeDenseLdsDescriptor()
    {
        constexpr index_t kX = 16 / sizeof(DataType);
        return make_naive_tensor_descriptor(make_tuple(number<Rows>{}, number<Cols>{}),
                                            make_tuple(number<Cols>{}, number<1>{}),
                                            number<kX>{},
                                            number<1>{});
    }

    // the TDM-written K / V buffers, a page each
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeKLdsBlockDescriptor()
    {
        using P = Page<Problem>;
        return MakeDenseLdsDescriptor<typename Problem::KDataType,
                                      P::kKRows,
                                      P::kPageSize * P::kX>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeVLdsBlockDescriptor()
    {
        using P = Page<Problem>;
        return MakeDenseLdsDescriptor<typename Problem::VDataType,
                                      P::kVRows,
                                      P::kHeadDim * P::kX>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeVTailLdsBlockDescriptor()
    {
        using P = Page<Problem>;
        return MakeDenseLdsDescriptor<typename Problem::VDataType,
                                      P::kVRows * P::kHeadDim,
                                      P::kX>();
    }

    // (Inner, Outer * X) over [Outer][Inner][X]. A WMMA operand lane holds 8 consecutive k, which
    // stay inside one X block, so the reads keep the vector.
    template <index_t Outer, index_t Inner, index_t X>
    CK_TILE_HOST_DEVICE static constexpr auto MakePageReadDescriptor()
    {
        constexpr auto page =
            make_naive_tensor_descriptor(make_tuple(number<Outer>{}, number<Inner>{}, number<X>{}),
                                         make_tuple(number<Inner * X>{}, number<X>{}, number<1>{}),
                                         number<X>{},
                                         number<1>{});

        return transform_tensor_descriptor(
            page,
            make_tuple(
                make_pass_through_transform(number<Inner>{}),
                make_merge_transform_v3_division_mod(make_tuple(number<Outer>{}, number<X>{}))),
            make_tuple(sequence<1>{}, sequence<0, 2>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));
    }

    // K as (token, d)
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeKLdsReadDescriptor()
    {
        using P = Page<Problem>;
        return MakePageReadDescriptor<P::kKRows, P::kPageSize, P::kX>();
    }

    // V as (d, token)
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeVLdsReadDescriptor()
    {
        using P = Page<Problem>;
        return MakePageReadDescriptor<P::kVRows, P::kHeadDim, P::kX>();
    }

    // The gemms of the wave-split loop: every warp runs the whole kM0 x kN0 x kK0 QK gemm and
    // the whole kM0 x kN1 x kK1 PV gemm on a tile of its own, so their tensors are warp tiles
    // (the same in every warp).
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetQKWarpBlockGemm()
    {
        using Shape      = typename Problem::BlockFmhaShape;
        using BlockWarps = sequence<1, 1, 1>;
        using GemmProblem =
            BlockGemmProblem<typename Problem::QDataType,
                             typename Problem::KDataType,
                             typename Problem::SaccDataType,
                             Problem::kBlockSize,
                             TileGemmShape<sequence<Shape::kM0, Shape::kN0, Shape::kK0>,
                                           BlockWarps,
                                           typename Shape::Gemm0WarpTile>,
                             Problem::kBlockSize / get_warp_size()>;
        using WarpGemm = typename remove_cvref_t<decltype(GetQKBlockGemm<Problem>())>::WarpGemm;
        using BlockGemmPolicy = BlockGemmARegBRegCRegV2CustomPolicy<typename Problem::QDataType,
                                                                    typename Problem::KDataType,
                                                                    typename Problem::SaccDataType,
                                                                    BlockWarps,
                                                                    WarpGemm,
                                                                    GemmLoopOrder::MNK>;
        return BlockGemmARegBRegCRegV2<GemmProblem, BlockGemmPolicy>{};
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetPVWarpBlockGemm()
    {
        using Shape      = typename Problem::BlockFmhaShape;
        using BlockWarps = sequence<1, 1, 1>;
        using GemmProblem =
            BlockGemmProblem<typename Problem::PDataType,
                             typename Problem::VDataType,
                             typename Problem::OaccDataType,
                             Problem::kBlockSize,
                             TileGemmShape<sequence<Shape::kM0, Shape::kN1, Shape::kK1>,
                                           BlockWarps,
                                           typename Shape::Gemm1WarpTile>,
                             Problem::kBlockSize / get_warp_size()>;
        using WarpGemm = typename remove_cvref_t<decltype(GetPVBlockGemm<Problem>())>::WarpGemm;
        using BlockGemmPolicy = BlockGemmARegBRegCRegV2CustomPolicy<typename Problem::PDataType,
                                                                    typename Problem::VDataType,
                                                                    typename Problem::OaccDataType,
                                                                    BlockWarps,
                                                                    WarpGemm,
                                                                    GemmLoopOrder::KMN>;
        return BlockGemmARegBRegCRegV2<GemmProblem, BlockGemmPolicy>{};
    }

    // Rows x Cols as one TDM box, the same in every warp: the warps differ in the page they
    // copy and in where it lands.
    template <typename Problem, index_t Rows, index_t Cols>
    CK_TILE_HOST_DEVICE static constexpr auto MakeWarpWholeDramTileDistribution()
    {
        constexpr index_t warpNum = Problem::kBlockSize / get_warp_size();
        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<warpNum>,
                                       tuple<sequence<Rows>, sequence<Cols>>,
                                       tuple<sequence<0>>,
                                       tuple<sequence<0>>,
                                       sequence<1, 2>,
                                       sequence<0, 0>>{},
            bool_constant<true>{});
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeKWarpPageDramTileDistribution()
    {
        using P = Page<Problem>;
        return MakeWarpWholeDramTileDistribution<Problem, P::kKRows, P::kPageSize * P::kX>();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeVWarpPageDramTileDistribution()
    {
        using P = Page<Problem>;
        return MakeWarpWholeDramTileDistribution<Problem, P::kVRows, P::kHeadDim * P::kX>();
    }

    // A V page in LDS as rows of the X tokens of a (token / X, d), every lane of a warp with
    // its share of the rows: to clear the tokens past the end of a tile.
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeVTokensRegTileDistribution()
    {
        using P                 = Page<Problem>;
        constexpr index_t kRows = P::kVRows * P::kHeadDim;
        static_assert(kRows % get_warp_size() == 0);
        return make_static_tile_distribution(
            tile_distribution_encoding<
                sequence<>,
                tuple<sequence<kRows / get_warp_size(), get_warp_size()>, sequence<P::kX>>,
                tuple<sequence<1>>,
                tuple<sequence<1>>,
                sequence<1, 2>,
                sequence<0, 0>>{});
    }

    // A warp's O (kM0 x kN1) for the merge of the wave-split loop. A lane holds 16 bytes of a
    // row: a row is padded by 16 bytes, so that the rows a wave accesses at once spread over the
    // LDS banks (an unpadded row is a whole number of bank spans, and every row hits the same).
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeOaccLdsBlockDescriptor()
    {
        using Shape            = typename Problem::BlockFmhaShape;
        constexpr index_t kVec = 16 / sizeof(typename Problem::OaccDataType);
        return make_naive_tensor_descriptor(make_tuple(number<Shape::kM0>{}, number<Shape::kN1>{}),
                                            make_tuple(number<Shape::kN1 + kVec>{}, number<1>{}),
                                            number<kVec>{},
                                            number<1>{});
    }

    // the PV gemm's B operand, read untransposed
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeVRegTileDistribution()
    {
        using PVBlockGemm = remove_cvref_t<decltype(GetPVBlockGemm<Problem>())>;
        return make_static_tile_distribution(
            decltype(PVBlockGemm::MakeBBlockDistributionEncode()){});
    }

    // P (kM0 x kN0) for BlockFmhaPipelineQRKSVSTdmDecodePagedKV: 16-byte packs of a row, a pack of
    // every row after another with one row of padding, so that the 16 rows a wave reads at
    // once fall on different LDS banks.
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakePLdsBlockDescriptor()
    {
        constexpr index_t kMPerBlock = Problem::BlockFmhaShape::kM0;
        constexpr index_t kNPerBlock = Problem::BlockFmhaShape::kN0;
        constexpr index_t kNPack     = 16 / sizeof(typename Problem::PDataType);

        constexpr auto p_lds_block_desc_0 = make_naive_tensor_descriptor(
            make_tuple(number<kNPerBlock / kNPack>{}, number<kMPerBlock>{}, number<kNPack>{}),
            make_tuple(number<(kMPerBlock + 1) * kNPack>{}, number<kNPack>{}, number<1>{}),
            number<kNPack>{},
            number<1>{});

        return transform_tensor_descriptor(
            p_lds_block_desc_0,
            make_tuple(
                make_pass_through_transform(number<kMPerBlock>{}),
                make_merge_transform(make_tuple(number<kNPerBlock / kNPack>{}, number<kNPack>{}))),
            make_tuple(sequence<1>{}, sequence<0, 2>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));
    }

    // The row statistics the gemm_0 warps exchange: (warp, row).
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeRowPartialLdsDescriptor()
    {
        constexpr index_t kNWarp = Problem::BlockFmhaShape::Gemm0BlockWarps::at(number<1>{});
        constexpr index_t kM0    = Problem::BlockFmhaShape::kM0;
        return make_naive_tensor_descriptor(make_tuple(number<kNWarp>{}, number<kM0>{}),
                                            make_tuple(number<kM0>{}, number<1>{}),
                                            number<1>{},
                                            number<1>{});
    }

    // A warp's own row of MakeRowPartialLdsDescriptor: lane = row, as in gemm_0's result, and
    // the two halves of a wave hold the same value.
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeRowPartialRegDistribution()
    {
        constexpr index_t kNWarp = Problem::BlockFmhaShape::Gemm0BlockWarps::at(number<1>{});
        constexpr index_t kM0    = Problem::BlockFmhaShape::kM0;
        static_assert(get_warp_size() == 2 * kM0);

        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<2>,
                                       tuple<sequence<kNWarp>, sequence<kM0, 1>>,
                                       tuple<sequence<1>, sequence<0, 2>>,
                                       tuple<sequence<0>, sequence<0, 0>>,
                                       sequence<2>,
                                       sequence<1>>{});
    }

    // Every warp's value of a lane's row.
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeRowPartialsRegDistribution()
    {
        constexpr index_t kNWarp = Problem::BlockFmhaShape::Gemm0BlockWarps::at(number<1>{});
        constexpr index_t kM0    = Problem::BlockFmhaShape::kM0;
        static_assert(get_warp_size() == 2 * kM0);

        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<kNWarp, 2>,
                                       tuple<sequence<kNWarp>, sequence<kM0>>,
                                       tuple<sequence<0>, sequence<0, 2>>,
                                       tuple<sequence<0>, sequence<1, 0>>,
                                       sequence<1>,
                                       sequence<0>>{});
    }
};

} // namespace ck_tile
