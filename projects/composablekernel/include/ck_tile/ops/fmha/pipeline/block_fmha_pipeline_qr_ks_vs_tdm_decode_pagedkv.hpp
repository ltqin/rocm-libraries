// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_tdm_pingpong.hpp"

namespace ck_tile {

// qr_tdm_pingpong's gemms and register layouts on the vectorized-layout paged fp8 KV cache (see
// Paged5DTileNavigator), with a main loop of two barriers a tile:
// - The softmax runs on gemm_0's result as it is: a warp holds 16 keys of every row, so it
//   computes P for its own keys only. The warps exchange the row max and the row sum of the
//   tile through LDS, and P reaches gemm_1 as fp8 (the scores themselves never go to LDS).
// - A warp copies the K of its own keys, so K takes no barrier. K and V have three LDS
//   buffers each: a tile's transfers are issued two trips ahead of the gemms that read it,
//   which covers their latency.
template <typename Problem_, typename Policy_ = BlockFmhaPipelineQRKSVSTdmPingPongPaged5DPolicy>
struct BlockFmhaPipelineQRKSVSTdmDecodePagedKV
    : BlockFmhaPipelineQRKSVSTdmPingPong<Problem_, Policy_>
{
    using Base = BlockFmhaPipelineQRKSVSTdmPingPong<Problem_, Policy_>;

    using typename Base::BlockFmhaShape;
    using typename Base::FmhaMask;
    using typename Base::KDataType;
    using typename Base::LSEDataType;
    using typename Base::PDataType;
    using typename Base::Policy;
    using typename Base::Problem;
    using typename Base::QDataType;
    using typename Base::QInputDataType;
    using typename Base::SaccDataType;
    using typename Base::SMPLComputeDataType;
    using typename Base::VDataType;

    using Base::BiasEnum;
    using Base::I0;
    using Base::I1;
    using Base::kDynamicQ;
    using Base::kFp8P;
    using Base::kHasSink;
    using Base::kHasUnevenSplits;
    using Base::kK0;
    using Base::kK1;
    using Base::kM0;
    using Base::kN0;
    using Base::kN1;
    using Base::kNWarp;
    using Base::kPaged5D;
    using Base::kPScale;
    using Base::kQKHeaddim;
    using Base::kStoreLSE;
    using Base::kSubQKHeaddim;

    static_assert(
        kPaged5D && kFp8P && kNWarp > 1 && kK0 == kQKHeaddim && kK1 == kN0,
        "qr_tdm decode pagedkv: fp8 P, the gemm_0 warps on N, one WMMA per warp and gemm");

    using Layout = typename Policy::template DecodeLdsLayout<Problem>;

    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize()
    {
        return Layout::kArenaBytes;
    }

    // P and the row partials share S's region
    static constexpr index_t kPBytes =
        Policy::template MakePLdsBlockDescriptor<Problem>().get_element_space_size() *
        sizeof(PDataType);
    static constexpr index_t kRowPartialBytes =
        Policy::template MakeRowPartialLdsDescriptor<Problem>().get_element_space_size() *
        sizeof(SMPLComputeDataType);
    static constexpr index_t kRowMaxOffset = integer_least_multiple(kPBytes, 64);
    static constexpr index_t kRowSumOffset =
        kRowMaxOffset + integer_least_multiple(kRowPartialBytes, 64);
    static_assert(kRowSumOffset + kRowPartialBytes <= Layout::kSBytes);

    // ranges of this many tiles and more take the wave-split loop: two tiles a warp
    static constexpr index_t kNumWarps          = Layout::kNumWarps;
    static constexpr index_t kWaveSplitMinTiles = 2 * kNumWarps;

    // A row statistic of the block, one partial per warp, at `offset` of S's region: the
    // window a warp stores its own partial through, and the one it loads all of them through
    // behind a barrier.
    CK_TILE_DEVICE static auto MakeRowPartialWindows(void* smem_arena, index_t offset)
    {
        auto view = make_tensor_view<address_space_enum::lds>(
            reinterpret_cast<SMPLComputeDataType*>(static_cast<char*>(smem_arena) +
                                                   Layout::kSOffset + offset),
            Policy::template MakeRowPartialLdsDescriptor<Problem>());
        constexpr auto lengths =
            Policy::template MakeRowPartialLdsDescriptor<Problem>().get_lengths();
        return make_tuple(
            make_tile_window(view, lengths, {0, 0}),
            make_tile_window(
                view, lengths, {0, 0}, Policy::template MakeRowPartialsRegDistribution<Problem>()));
    }

    // gemm_0's Q operand. Dynamic Q quantizes it and sets row_scale to scale_s times each row's
    // dequant scale.
    template <typename QInputTile, typename RowScaleTile>
    CK_TILE_DEVICE static auto
    MakeQTile(const QInputTile& q_in, float scale_s, RowScaleTile& row_scale)
    {
        if constexpr(kDynamicQ)
        {
            auto [q, q_scale, q_inv] = Base::QuantizeQRows(q_in);
            ignore                   = q_inv;
            // a thread holds the same rows of Q and of the scores
            using QRow = typename decltype(q_scale)::StaticTileDistribution::DstrEncode;
            using SRow = typename RowScaleTile::StaticTileDistribution::DstrEncode;
            static_assert(decltype(q_scale)::get_thread_buffer_size() ==
                              RowScaleTile::get_thread_buffer_size(),
                          "qr_tdm decode pagedkv dynamic Q: the Q and S row layouts differ");
            Base::template AssertSameRowOrder<QRow, SRow>();
            static_for<0, RowScaleTile::get_thread_buffer_size(), 1>{}([&](auto i) {
                row_scale.get_thread_buffer()(i) = scale_s * q_scale.get_thread_buffer()[i];
            });
            return q;
        }
        else
        {
            ignore = scale_s;
            ignore = row_scale;
            return q_in;
        }
    }

    // LSE of the rows from the row max m (log2 domain, of the scaled scores) and the row sum l,
    // in single precision throughout.
    template <typename LSEaccDramBlockWindowTmp, typename RowTile>
    CK_TILE_DEVICE static void
    StoreLse(LSEaccDramBlockWindowTmp& lse_acc_dram_window, const RowTile& m, const RowTile& l)
    {
        constexpr LSEDataType kLn2 = static_cast<LSEDataType>(1.0 / C_LOG2E);
        auto lse_acc = make_static_distributed_tensor<LSEDataType>(m.get_tile_distribution());
        static_for<0, RowTile::get_thread_buffer_size(), 1>{}([&](auto i) {
            lse_acc.get_thread_buffer()(i) =
                m.get_thread_buffer()[i] * kLn2 + log(l.get_thread_buffer()[i]);
        });
        store_tile(lse_acc_dram_window, lse_acc);
    }

    // The wave-split main loop, for long ranges: every warp runs whole kN0-key tiles on its
    // own, tile (kNumWarps * j + warp) of the range in its trip j, with its own softmax state
    // and its own K / V buffers, so the loop has no barrier. At the end the warps exchange
    // their O and row statistics through LDS and combine them into the block's o_acc,
    // m_base and l.
    template <typename QLdsView,
              typename BiasDramWindow,
              typename QOrigin,
              typename PositionEncoding,
              typename PageNavigator,
              typename OaccTile,
              typename RowTile>
    CK_TILE_DEVICE void RunWaveSplit(const QLdsView& q_lds_view,
                                     const BiasDramWindow& bias_dram_window,
                                     const FmhaMask& mask,
                                     const QOrigin& q_origin,
                                     PositionEncoding position_encoding,
                                     float scale_s,
                                     void* smem_arena,
                                     const PageNavigator& page_nav,
                                     index_t i_page_start,
                                     index_t seqlen_k_start,
                                     index_t seqlen_k_end,
                                     index_t num_total_loop,
                                     OaccTile& o_acc,
                                     RowTile& m_base,
                                     RowTile& l) const
    {
        using Nav = remove_cvref_t<PageNavigator>;

        constexpr auto gemm_0 = Policy::template GetQKWarpBlockGemm<Problem>();
        constexpr auto gemm_1 = Policy::template GetPVWarpBlockGemm<Problem>();
        using Gemm0           = remove_cvref_t<decltype(gemm_0)>;
        using Gemm1           = remove_cvref_t<decltype(gemm_1)>;

        const auto f_max = [](auto e0, auto e1) { return max(e0, e1); };
        const auto f_sum = [](auto e0, auto e1) { return e0 + e1; };

        using SaccTile = decltype(gemm_0.MakeCBlockTile());
        using SRowTile = decltype(block_tile_reduce<SMPLComputeDataType>(
            SaccTile{}, sequence<1>{}, f_max, SaccDataType{0}));
        static_assert(SRowTile::get_thread_buffer_size() == 1 &&
                          RowTile::get_thread_buffer_size() == 1,
                      "qr_tdm decode pagedkv: a lane holds one row of the scores and of O");

        const index_t i_warp = get_warp_id();
        // tiles i_warp, i_warp + kNumWarps, ... of the range; one at least (kWaveSplitMinTiles)
        const index_t num_trips = (num_total_loop - i_warp + kNumWarps - 1) / kNumWarps;

        // the warp's block: K stages, then V stages
        char* const block =
            static_cast<char*>(smem_arena) + Layout::kWaveOffset + i_warp * Layout::kWaveBlockBytes;
        KDataType* const k_buf0 = reinterpret_cast<KDataType*>(block);
        VDataType* const v_buf0 =
            reinterpret_cast<VDataType*>(block + Layout::kWaveStages * Layout::kKStride);
        constexpr index_t kKStride = Layout::kKStride / sizeof(KDataType);
        constexpr index_t kVStride = Layout::kVStride / sizeof(VDataType);
        constexpr index_t kStages  = Layout::kWaveStages;
        // waits until at most `count` of the warp's transfers are in flight
        const auto wait_transfers = [](index_t count) {
            static_for<0, 2 * Layout::kWaveStages, 1>{}([&](auto i) {
                if(count == i)
                    s_wait_tensorcnt<i>();
            });
        };

        TDMConfig tdm_config_k =
            Base::template MakeTdmConfig<typename Policy::template LdsPaddingConfigK<Problem>>();
        TDMConfig tdm_config_v =
            Base::template MakeTdmConfig<typename Policy::template LdsPaddingConfigV<Problem>>();

        // a whole page of K and of V per tile, one TDM box each
        auto k_dram_window =
            make_tile_window(page_nav.make_k_page_view(0),
                             Nav::k_page_lengths(),
                             {0, 0},
                             Policy::template MakeKWarpPageDramTileDistribution<Problem>());
        auto k_lds_write_window =
            make_tile_window(make_tensor_view<address_space_enum::lds>(
                                 k_buf0, Policy::template MakeKLdsBlockDescriptor<Problem>()),
                             Policy::template MakeKLdsBlockDescriptor<Problem>().get_lengths(),
                             {0, 0});
        auto k_lds_read_window =
            make_tile_window(make_tensor_view<address_space_enum::lds>(
                                 k_buf0, Policy::template MakeKLdsReadDescriptor<Problem>()),
                             make_tuple(number<kN0>{}, number<kK0>{}),
                             {0, 0},
                             make_static_tile_distribution(Gemm0::MakeBBlockDistributionEncode()));

        auto v_dram_window =
            make_tile_window(page_nav.make_v_page_view(0),
                             Nav::v_page_lengths(),
                             {0, 0},
                             Policy::template MakeVWarpPageDramTileDistribution<Problem>());
        auto v_lds_write_window =
            make_tile_window(make_tensor_view<address_space_enum::lds>(
                                 v_buf0, Policy::template MakeVLdsBlockDescriptor<Problem>()),
                             Policy::template MakeVLdsBlockDescriptor<Problem>().get_lengths(),
                             {0, 0});
        auto v_lds_read_window =
            make_tile_window(make_tensor_view<address_space_enum::lds>(
                                 v_buf0, Policy::template MakeVLdsReadDescriptor<Problem>()),
                             make_tuple(number<kN1>{}, number<kK1>{}),
                             {0, 0},
                             make_static_tile_distribution(Gemm1::MakeBBlockDistributionEncode()));
        // a token chunk of V as words, to clear the tokens past the range
        constexpr auto v_words_desc = Policy::template MakeVChunkWordsLdsBlockDescriptor<Problem>();
        constexpr index_t kWords    = Nav::kX * sizeof(VDataType) / sizeof(uint32_t);
        auto v_lds_words_window =
            make_tile_window(make_tensor_view<address_space_enum::lds>(
                                 reinterpret_cast<uint32_t*>(v_buf0), v_words_desc),
                             make_tuple(number<Nav::kHeadDim>{}, number<kWords>{}),
                             {0, 0},
                             Policy::template MakeVChunkWordsRegTileDistribution<Problem>());

        // K, then V, of the tile on page `page`, into the buffers of `stage`
        const auto issue_tile = [&](index_t page, index_t stage) {
            const long_index_t page_off = page_nav.page_offset(page);
            k_dram_window.set_bottom_tensor_view_data_ptr(page_nav.k_page_at(page_off));
            k_lds_write_window.set_bottom_tensor_view_data_ptr(k_buf0 + stage * kKStride);
            load_tile_tdm(tdm_config_k, k_lds_write_window, k_dram_window);
            v_dram_window.set_bottom_tensor_view_data_ptr(page_nav.v_page_at(page_off));
            v_lds_write_window.set_bottom_tensor_view_data_ptr(v_buf0 + stage * kVStride);
            load_tile_tdm(tdm_config_v, v_lds_write_window, v_dram_window);
        };

        // A tile is staged kStages - 1 trips ahead of the trip that reads it, into the buffers
        // of stage (trip % kStages). The pages of the warp's first kStages tiles: all but the
        // last are issued here, K then V of each as in every trip, and the last in the first
        // trip. Their users stay behind the Q transfer.
        index_t page_ids[kStages];
        for(index_t i = 0; i < kStages; ++i)
            page_ids[i] = page_nav.page_id(i_page_start + i_warp + i * kNumWarps);
        __builtin_amdgcn_sched_barrier(0);
        for(index_t i = 0; i < kStages; ++i)
            asm volatile("" : "+s"(page_ids[i]));
        for(index_t i = 0; i + 1 < kStages; ++i)
            if(i < num_trips)
                issue_tile(page_ids[i], i);
        index_t page_id_ahead = page_ids[kStages - 1];

        // retire Q only; the tiles, issued after it, stay in flight
        wait_transfers(2 * min(num_trips, kStages - 1));
        block_sync_lds();
        auto q_lds_read_window =
            make_tile_window(q_lds_view,
                             Policy::template MakeQLdsBlockDescriptor<Problem>().get_lengths(),
                             {0, 0},
                             make_static_tile_distribution(Gemm0::MakeABlockDistributionEncode()));
        [[maybe_unused]] auto row_scale = SRowTile{};
        const auto q_tile = MakeQTile(load_tile(q_lds_read_window), scale_s, row_scale);

        // the warp's softmax state, of a lane's row: the row max, the row sum of p, and the
        // row max o_w and l_w are normalised to (see the cooperative loop)
        auto s_acc = SaccTile{};
        auto o_w   = gemm_1.MakeCBlockTile();
        clear_tile(o_w);
        SMPLComputeDataType m_w      = -numeric<SMPLComputeDataType>::infinity();
        SMPLComputeDataType m_base_w = -numeric<SMPLComputeDataType>::infinity();
        SMPLComputeDataType l_w      = 0;

        index_t stage = 0;
        for(index_t i_trip = 0; i_trip < num_trips; ++i_trip)
        {
            const index_t stage_ahead = stage == 0 ? kStages - 1 : stage - 1;
            const index_t i_tile      = i_trip * kNumWarps + i_warp;
            // this trip's tile and the ones staged behind it
            const index_t tiles_in_flight = min(num_trips - i_trip, kStages);

            // The tile kStages - 1 ahead goes into the buffers the last trip read. This
            // trip's K is the first of the transfers in flight.
            if(i_trip + kStages - 1 < num_trips)
                issue_tile(page_id_ahead, stage_ahead);
            wait_transfers(2 * tiles_in_flight - 1);
            // the page of the tile the next trip stages, read under the gemm
            page_id_ahead = page_nav.page_id(i_page_start + i_tile + kStages * kNumWarps);

            const index_t kv_tile_start = seqlen_k_start + i_tile * kN0;

            // STAGE 1, QK gemm
            clear_tile(s_acc);
            k_lds_read_window.set_bottom_tensor_view_data_ptr(k_buf0 + stage * kKStride);
            gemm_0(s_acc, q_tile, load_tile(k_lds_read_window));

            // STAGE 2: scale_s and the mask
            Base::ApplyScaleAndBias(s_acc,
                                    bias_dram_window,
                                    q_origin,
                                    kv_tile_start,
                                    scale_s,
                                    position_encoding,
                                    row_scale);
            Base::MaskScores(
                s_acc, mask, q_origin, kv_tile_start, i_tile == num_total_loop - 1, seqlen_k_end);

            // STAGE 3, softmax
            const auto s = cast_tile<SMPLComputeDataType>(s_acc);
            {
                auto m_tile = block_tile_reduce<SMPLComputeDataType>(
                    s, sequence<1>{}, f_max, -numeric<SMPLComputeDataType>::infinity());
                Base::ReduceRowSync(m_tile, f_max);
                m_w = max(m_w, m_tile.get_thread_buffer()[0]);
            }
            const SMPLComputeDataType gap = m_w - m_base_w;
            if(gap > 0)
            {
                const auto tmp = exp2(-gap);
                l_w *= tmp;
                m_base_w = m_w;
                static_for<0, decltype(o_w)::get_thread_buffer_size(), 1>{}(
                    [&](auto i) { o_w.get_thread_buffer()(i) *= tmp; });
            }
            const SMPLComputeDataType row_exp = [&]() {
                if constexpr(FmhaMask::IsMasking)
                    return m_base_w == -numeric<SMPLComputeDataType>::infinity()
                               ? type_convert<SMPLComputeDataType>(0.f)
                               : m_base_w;
                else
                    return m_base_w;
            }();

            auto p_compute =
                make_static_distributed_tensor<SMPLComputeDataType>(s.get_tile_distribution());
            static_for<0, decltype(p_compute)::get_thread_buffer_size(), 1>{}([&](auto i) {
                p_compute.get_thread_buffer()(i) = exp2(s.get_thread_buffer()[i] - row_exp);
            });
            {
                auto l_tile = block_tile_reduce<SMPLComputeDataType>(
                    p_compute, sequence<1>{}, f_sum, SMPLComputeDataType{0});
                Base::ReduceRowSync(l_tile, f_sum);
                l_w += l_tile.get_thread_buffer()[0];
            }

            // P as fp8 (kPScale * p, rounded once): gemm_0's result layout of a warp is
            // gemm_1's A layout
            tile_elementwise_inout([](auto& x) { x *= kPScale; }, p_compute);
            const auto p_fp8 = cast_tile<PDataType>(p_compute);
            auto p_tile      = make_static_distributed_tensor<PDataType>(
                make_static_tile_distribution(Gemm1::MakeABlockDistributionEncode()));
            static_assert(decltype(p_tile)::get_thread_buffer_size() ==
                              decltype(p_fp8)::get_thread_buffer_size() &&
                          Gemm1::MIterPerWarp == 1 && Gemm1::KIterPerWarp == 1);
            static_for<0, decltype(p_tile)::get_thread_buffer_size(), 1>{}(
                [&](auto i) { p_tile.get_thread_buffer()(i) = p_fp8.get_thread_buffer()[i]; });

            // retire this trip's V
            wait_transfers(2 * tiles_in_flight - 2);

            // The tokens of the page past the range may hold anything (P is 0 there, but
            // 0 * NaN is not): clear them on the tile the range ends in.
            const index_t tile_valid = min(max(seqlen_k_end - kv_tile_start, 0), kN0);
            if(tile_valid < kN0)
            {
                // The chunk the range ends in keeps its first tokens, the ones behind it none.
                // The masks are scalars, set up once: per-word conditions would become
                // branches.
                v_lds_words_window.set_bottom_tensor_view_data_ptr(
                    reinterpret_cast<uint32_t*>(v_buf0 + stage * kVStride));
                index_t i_chunk    = tile_valid / Nav::kX;
                const index_t keep = (tile_valid - i_chunk * Nav::kX) * sizeof(VDataType);
                if(keep > 0)
                {
                    uint32_t kept[kWords];
                    for(index_t w = 0; w < kWords; ++w)
                    {
                        // bytes of word w that stay, 0 to 4
                        const index_t n = min(max(keep - w * 4, 0), 4);
                        kept[w]         = static_cast<uint32_t>((uint64_t{1} << (8 * n)) - 1);
                    }
                    v_lds_words_window.set_window_origin({i_chunk * Nav::kHeadDim, 0});
                    auto words = load_tile(v_lds_words_window);
                    static_for<0, decltype(words)::get_thread_buffer_size(), 1>{}(
                        [&](auto i) { words.get_thread_buffer()(i) &= kept[i % kWords]; });
                    store_tile(v_lds_words_window, words);
                    ++i_chunk;
                }
                auto zeros = make_static_distributed_tensor<uint32_t>(
                    v_lds_words_window.get_tile_distribution());
                clear_tile(zeros);
                for(; i_chunk < kN0 / Nav::kX; ++i_chunk)
                {
                    v_lds_words_window.set_window_origin({i_chunk * Nav::kHeadDim, 0});
                    store_tile(v_lds_words_window, zeros);
                }
            }

            // STAGE 4, PV gemm
            v_lds_read_window.set_bottom_tensor_view_data_ptr(v_buf0 + stage * kVStride);
            gemm_1(o_w, p_tile, load_tile(v_lds_read_window));
            stage = stage + 1 == kStages ? 0 : stage + 1;
        }

        // The merge. A warp's O goes to the head of its block, whose transfers and reads are
        // all done, and its row max and row sum to the partials.
        constexpr auto o_lds_desc = Policy::template MakeOaccLdsBlockDescriptor<Problem>();
        static_assert(o_lds_desc.get_element_space_size() * sizeof(typename OaccTile::DataType) <=
                      Layout::kWaveBlockBytes);
        using OaccDataType = typename OaccTile::DataType;
        auto o_lds_write_window =
            make_tile_window(make_tensor_view<address_space_enum::lds>(
                                 reinterpret_cast<OaccDataType*>(block), o_lds_desc),
                             o_lds_desc.get_lengths(),
                             {0, 0});
        store_tile(o_lds_write_window, o_w);

        auto [max_lds_write_window, max_lds_read_window] =
            MakeRowPartialWindows(smem_arena, kRowMaxOffset);
        auto [sum_lds_write_window, sum_lds_read_window] =
            MakeRowPartialWindows(smem_arena, kRowSumOffset);
        auto row_partial = make_static_distributed_tensor<SMPLComputeDataType>(
            Policy::template MakeRowPartialRegDistribution<Problem>());
        row_partial.get_thread_buffer()(0) = m_base_w;
        store_tile(max_lds_write_window, row_partial);
        row_partial.get_thread_buffer()(0) = l_w;
        store_tile(sum_lds_write_window, row_partial);

        block_sync_lds();

        // every warp's O is relative to its own row max: weigh them to the block's
        const auto m_warps          = load_tile(max_lds_read_window);
        const auto l_warps          = load_tile(sum_lds_read_window);
        SMPLComputeDataType m_block = m_warps.get_thread_buffer()[0];
        static_for<1, kNumWarps, 1>{}(
            [&](auto i) { m_block = max(m_block, m_warps.get_thread_buffer()[i]); });

        auto o_lds_read_window =
            make_tile_window(make_tensor_view<address_space_enum::lds>(
                                 reinterpret_cast<OaccDataType*>(static_cast<char*>(smem_arena) +
                                                                 Layout::kWaveOffset),
                                 o_lds_desc),
                             o_lds_desc.get_lengths(),
                             {0, 0},
                             o_acc.get_tile_distribution());
        SMPLComputeDataType l_block = 0;
        static_for<0, kNumWarps, 1>{}([&](auto i) {
            // a warp without keys (row max -inf) weighs nothing
            const SMPLComputeDataType m_i = m_warps.get_thread_buffer()[i];
            const SMPLComputeDataType w_i = m_i == -numeric<SMPLComputeDataType>::infinity()
                                                ? SMPLComputeDataType{0}
                                                : exp2(m_i - m_block);
            l_block += w_i * l_warps.get_thread_buffer()[i];
            o_lds_read_window.set_bottom_tensor_view_data_ptr(
                reinterpret_cast<OaccDataType*>(static_cast<char*>(smem_arena) +
                                                Layout::kWaveOffset + i * Layout::kWaveBlockBytes));
            const auto o_i = load_tile(o_lds_read_window);
            static_for<0, OaccTile::get_thread_buffer_size(), 1>{}(
                [&](auto j) { o_acc.get_thread_buffer()(j) += w_i * o_i.get_thread_buffer()[j]; });
        });
        m_base.get_thread_buffer()(0) = m_block;
        l.get_thread_buffer()(0)      = l_block;
    }

    template <typename QDramBlockWindowTmp,
              typename KDramBlockWindowTmp,
              typename VDramBlockWindowTmp,
              typename BiasDramBlockWindowTmp,
              typename LSEaccDramBlockWindowTmp,
              typename PositionEncoding,
              typename PageNavigator>
    CK_TILE_HOST_DEVICE auto operator()(const QDramBlockWindowTmp& q_dram_block_window_tmp,
                                        const KDramBlockWindowTmp& k_dram_block_window_tmp,
                                        const VDramBlockWindowTmp& v_dram_block_window_tmp,
                                        const BiasDramBlockWindowTmp& bias_dram_block_window_tmp,
                                        LSEaccDramBlockWindowTmp& lse_acc_dram_window_tmp,
                                        FmhaMask mask,
                                        PositionEncoding position_encoding,
                                        float scale_s,
                                        void* smem_arena,
                                        float sink_v,
                                        const PageNavigator& page_nav) const
    {
        using Nav = remove_cvref_t<PageNavigator>;

        static_assert(
            std::is_same_v<QInputDataType,
                           remove_cvref_t<typename QDramBlockWindowTmp::DataType>> &&
                std::is_same_v<KDataType, remove_cvref_t<typename KDramBlockWindowTmp::DataType>> &&
                std::is_same_v<VDataType, remove_cvref_t<typename VDramBlockWindowTmp::DataType>>,
            "wrong!");
        static_assert(kM0 == QDramBlockWindowTmp{}.get_window_lengths()[I0] &&
                          kSubQKHeaddim == QDramBlockWindowTmp{}.get_window_lengths()[I1],
                      "wrong!");
        // the K / V windows are a page (see Paged5DTileNavigator)
        static_assert(Nav::kPageSize == kN0 && Nav::kHeadDim == kQKHeaddim &&
                      Nav::kKRows == KDramBlockWindowTmp{}.get_window_lengths()[I0] &&
                      Nav::kPageSize * Nav::kX == KDramBlockWindowTmp{}.get_window_lengths()[I1] &&
                      Nav::kVRows == VDramBlockWindowTmp{}.get_window_lengths()[I0] &&
                      Nav::kHeadDim * Nav::kX == VDramBlockWindowTmp{}.get_window_lengths()[I1] &&
                      Layout::kKBytes == kN0 * kQKHeaddim * sizeof(KDataType) &&
                      Layout::kVBytes == kN0 * kN1 * sizeof(VDataType));
        static_assert(!kHasSink && BiasEnum == BlockAttentionBiasEnum::NO_BIAS);

        constexpr auto gemm_0 = Policy::template GetQKBlockGemm<Problem>();
        constexpr auto gemm_1 = Policy::template GetPVBlockGemm<Problem>();

        using SaccBlockTileType = decltype(gemm_0.MakeCBlockTile());
        auto s_acc              = SaccBlockTileType{};

        const auto f_max = [](auto e0, auto e1) { return max(e0, e1); };
        const auto f_sum = [](auto e0, auto e1) { return e0 + e1; };

        using OaccBlockTileType = decltype(gemm_1.MakeCBlockTile());
        auto o_acc              = OaccBlockTileType{};

        using SBlockTileType  = decltype(cast_tile<SMPLComputeDataType>(o_acc));
        using MLBlockTileType = decltype(block_tile_reduce<SMPLComputeDataType>(
            SBlockTileType{}, sequence<1>{}, f_max, SMPLComputeDataType{0}));

        // row max m, row sum l of p itself (O is divided by kPScale * l), and the row max
        // o_acc and l are normalised to: p must not exceed 1, so it follows m (see kFp8P)
        auto m      = MLBlockTileType{};
        auto l      = MLBlockTileType{};
        auto m_base = MLBlockTileType{};

        clear_tile(o_acc);
        Base::InitRowMaxSum(m, l, sink_v, scale_s);
        m_base = m;

        const auto q_origin = q_dram_block_window_tmp.get_window_origin();

        const auto tile_range_result      = Base::GetKTileRange(mask, q_origin);
        const auto logical_seqlen_k_start = tile_range_result.get(ck_tile::number<1>{});
        const auto logical_seqlen_k_end   = tile_range_result.get(ck_tile::number<2>{});

        // nothing to do: an empty range, or a seqlen_k of 0
        const index_t num_total_loop =
            integer_divide_ceil(logical_seqlen_k_end - logical_seqlen_k_start, kN0);
        if(num_total_loop <= 0)
        {
            if constexpr(kStoreLSE)
                Base::StoreLseOfEmptyRows(lse_acc_dram_window_tmp, m, sink_v, scale_s);
            return o_acc;
        }

        // Tile i is the page row[i_page_start + i] of the block table, as a split starts on a
        // multiple of kN0 (SplitKVAttentionMask, through SplitTileRange). The ids are read
        // behind the empty-range exit (an empty sequence has no page).
        const index_t i_page_start = logical_seqlen_k_start / kN0;

        TDMConfig tdm_config_k =
            Base::template MakeTdmConfig<typename Policy::template LdsPaddingConfigK<Problem>>();
        TDMConfig tdm_config_v =
            Base::template MakeTdmConfig<typename Policy::template LdsPaddingConfigV<Problem>>();

        // Q through LDS. Issued next to the page ids of the main loop: neither depends on the
        // other.
        TDMConfig tdm_config_q =
            Base::template MakeTdmConfig<typename Policy::template LdsPaddingConfigQ<Problem>>();
        auto q_dram_window = make_tile_window(
            q_dram_block_window_tmp, Policy::template MakeQDramTileDistribution<Problem>());
        auto q_lds_view = make_tensor_view<address_space_enum::lds>(
            reinterpret_cast<QInputDataType*>(static_cast<char*>(smem_arena) + Layout::kQOffset),
            Policy::template MakeQLdsBlockDescriptor<Problem>());
        auto q_lds_store_window = make_tile_window(
            q_lds_view, Policy::template MakeQLdsBlockDescriptor<Problem>().get_lengths(), {0, 0});
        auto q_lds_read_window =
            make_tile_window(q_lds_view,
                             Policy::template MakeQLdsBlockDescriptor<Problem>().get_lengths(),
                             {0, 0},
                             Policy::template MakeQRegTileDistribution<Problem>());
        load_tile_tdm(tdm_config_q, q_lds_store_window, q_dram_window);

        if(kWaveSplitMinTiles <= num_total_loop)
        {
            RunWaveSplit(q_lds_view,
                         bias_dram_block_window_tmp,
                         mask,
                         q_origin,
                         position_encoding,
                         scale_s,
                         smem_arena,
                         page_nav,
                         i_page_start,
                         logical_seqlen_k_start,
                         logical_seqlen_k_end,
                         num_total_loop,
                         o_acc,
                         m_base,
                         l);
        }
        else
        {
            // The cooperative main loop: the warps share every tile, 16 keys each.
            index_t page_id = page_nav.page_id(i_page_start);
            // a tile is staged Layout::kStages - 1 = 2 trips ahead: the second tile's id too
            static_assert(Layout::kStages == 3);
            index_t page_id_1 = page_nav.page_id(i_page_start + 1);

            // the page ids' users, and so their wait, stay behind the Q transfer
            __builtin_amdgcn_sched_barrier(0);
            asm volatile("" : "+s"(page_id));
            asm volatile("" : "+s"(page_id_1));
            // page id * page stride of the first tile, shared by its K and V
            const long_index_t page_off = page_nav.page_offset(page_id);

            const index_t physical_seqlen_k_start = logical_seqlen_k_start;
            const index_t physical_seqlen_k_end   = logical_seqlen_k_end;

            const auto bias_origin = bias_dram_block_window_tmp.get_window_origin();
            auto bias_dram_window =
                make_tile_window(bias_dram_block_window_tmp.get_bottom_tensor_view(),
                                 bias_dram_block_window_tmp.get_window_lengths(),
                                 {bias_origin.at(number<0>{}), 0},
                                 gemm_0.MakeCBlockTile().get_tile_distribution());

            // K: a warp copies the keys it reads itself, into a block of its own, so K takes no
            // barrier.
            ignore                         = k_dram_block_window_tmp; // its page is split here
            constexpr index_t kKeysPerWarp = Policy::template kKeysPerWarp<Problem>;
            const index_t i_warp_key       = get_warp_id() * kKeysPerWarp;
            constexpr auto k_keys_lengths =
                make_tuple(number<Nav::kKRows>{}, number<kKeysPerWarp * Nav::kX>{});
            auto k_dram_window = make_tile_window(
                page_nav.make_k_keys_view(page_off, i_warp_key, number<kKeysPerWarp>{}),
                k_keys_lengths,
                {0, 0},
                Policy::template MakeKWarpKeysDramTileDistribution<Problem>());

            KDataType* const k_smem_buf0 =
                reinterpret_cast<KDataType*>(static_cast<char*>(smem_arena) + Layout::kK0Offset);
            constexpr index_t kKStride = Layout::kKStride / sizeof(KDataType);
            // the warp's block of a buffer
            const index_t k_block_offset = get_warp_id() * (Nav::kKRows * kKeysPerWarp * Nav::kX);

            auto k_lds_write_window =
                make_tile_window(make_tensor_view<address_space_enum::lds>(
                                     k_smem_buf0 + k_block_offset,
                                     Policy::template MakeKWarpKeysLdsBlockDescriptor<Problem>()),
                                 k_keys_lengths,
                                 {0, 0});
            // read back as (key, d)
            auto k_lds_read_window = make_tile_window(
                make_tensor_view<address_space_enum::lds>(
                    k_smem_buf0, Policy::template MakeKWarpKeysLdsReadDescriptor<Problem>()),
                make_tuple(number<kN0>{}, number<kK0>{}),
                {0, 0},
                Policy::template MakeKRegTileDistribution<Problem>());

            // the K transfer of the tile whose page is at page_off_, into the warp's block of k_buf
            const auto issue_k = [&](long_index_t page_off_, KDataType* k_buf) {
                k_dram_window.set_bottom_tensor_view_data_ptr(page_nav.k_page_at(page_off_) +
                                                              i_warp_key * Nav::kX);
                k_lds_write_window.set_bottom_tensor_view_data_ptr(k_buf + k_block_offset);
                load_tile_tdm(tdm_config_k, k_lds_write_window, k_dram_window);
            };

            // P in LDS: written in gemm_0's result layout, read as gemm_1's A operand
            PDataType* const p_smem =
                reinterpret_cast<PDataType*>(static_cast<char*>(smem_arena) + Layout::kSOffset);
            auto p_lds_view = make_tensor_view<address_space_enum::lds>(
                p_smem, Policy::template MakePLdsBlockDescriptor<Problem>());
            auto p_lds_write_window =
                make_tile_window(p_lds_view,
                                 Policy::template MakePLdsBlockDescriptor<Problem>().get_lengths(),
                                 {0, 0});
            auto p_lds_read_window =
                make_tile_window(p_lds_view,
                                 Policy::template MakePLdsBlockDescriptor<Problem>().get_lengths(),
                                 {0, 0},
                                 Policy::template MakePRegTileDistribution<Problem>());

            // the row max and row sum of a tile
            auto [max_lds_write_window, max_lds_read_window] =
                MakeRowPartialWindows(smem_arena, kRowMaxOffset);
            auto [sum_lds_write_window, sum_lds_read_window] =
                MakeRowPartialWindows(smem_arena, kRowSumOffset);

            // a warp's partial, one row per lane: gemm_0's rows
            auto row_partial = make_static_distributed_tensor<SMPLComputeDataType>(
                Policy::template MakeRowPartialRegDistribution<Problem>());
            using RowPartialTile  = decltype(row_partial);
            using RowPartialsTile = decltype(load_tile(max_lds_read_window));
            static_assert(RowPartialTile::get_thread_buffer_size() == 1 &&
                          RowPartialsTile::get_thread_buffer_size() == kNWarp);

            // V in LDS: written by the TDM, each warp whole rows of the page, and read back
            // through the page layout behind the barrier ahead of gemm_1
            auto v_dram_window =
                make_tile_window(v_dram_block_window_tmp,
                                 {0, 0},
                                 Policy::template MakeVDramTileDistribution<Problem>());
            VDataType* const v_smem_buf0 =
                reinterpret_cast<VDataType*>(static_cast<char*>(smem_arena) + Layout::kV0Offset);
            constexpr index_t kVStride = Layout::kVStride / sizeof(VDataType);
            auto v_lds_view            = make_tensor_view<address_space_enum::lds>(
                v_smem_buf0, Policy::template MakeVLdsBlockDescriptor<Problem>());
            auto v_lds_write_window =
                make_tile_window(v_lds_view,
                                 Policy::template MakeVLdsBlockDescriptor<Problem>().get_lengths(),
                                 {0, 0});
            auto v_lds_read_window = make_tile_window(
                make_tensor_view<address_space_enum::lds>(
                    v_smem_buf0, Policy::template MakeVLdsReadDescriptor<Problem>()),
                make_tuple(number<kN1>{}, number<kK1>{}),
                {0, 0},
                Policy::template MakeVRegTileDistribution<Problem>());

            // dynamic Q: scale_s times the Q dequant scale of each row, in s_acc's row layout
            using SaccRowTileType           = decltype(block_tile_reduce<SMPLComputeDataType>(
                SaccBlockTileType{}, sequence<1>{}, f_max, SaccDataType{0}));
            [[maybe_unused]] auto row_scale = SaccRowTileType{};
            static_assert(SaccRowTileType::get_thread_buffer_size() == 1 &&
                              MLBlockTileType::get_thread_buffer_size() == 1,
                          "qr_tdm decode pagedkv: a lane holds one row of the scores and of O");

            // The V transfer of tile i_tile, from its page at page_off_, into v_buf. On a tile cut
            // short by the split end each warp copies its kX-token chunk through the tail view,
            // which zero-fills the tokens past the end (P is 0 there, but 0 * NaN is not).
            const auto issue_v = [&](index_t i_tile, long_index_t page_off_, VDataType* v_buf) {
                const index_t tile_start = physical_seqlen_k_start + i_tile * kN0;
                const index_t tile_valid = min(max(physical_seqlen_k_end - tile_start, 0), kN0);
                if(tile_valid < kN0)
                {
                    const index_t v_tail_len =
                        min(max(tile_valid - Nav::kX * get_warp_id(), 0), Nav::kX);
                    constexpr auto tail_desc =
                        Policy::template MakeVTailLdsBlockDescriptor<Problem>();
                    auto v_tail_lds_window = make_tile_window(
                        make_tensor_view<address_space_enum::lds>(v_buf, tail_desc),
                        tail_desc.get_lengths(),
                        {0, 0});
                    auto v_tail_dram_window =
                        make_tile_window(page_nav.make_v_tail_view(page_off_, v_tail_len),
                                         tail_desc.get_lengths(),
                                         {0, 0},
                                         Policy::template MakeVTailDramTileDistribution<Problem>());
                    load_tile_tdm(tdm_config_v, v_tail_lds_window, v_tail_dram_window);
                }
                else
                {
                    v_dram_window.set_bottom_tensor_view_data_ptr(page_nav.v_page_at(page_off_));
                    v_lds_write_window.set_bottom_tensor_view_data_ptr(v_buf);
                    load_tile_tdm(tdm_config_v, v_lds_write_window, v_dram_window);
                }
            };
            // The first two tiles, K then V of each as in every trip: the waits count on that
            // order. They are issued while Q is in flight, and stay in flight when Q is retired.
            issue_k(page_off, k_smem_buf0);
            issue_v(0, page_off, v_smem_buf0);
            if(1 < num_total_loop)
            {
                const long_index_t page_off_1 = page_nav.page_offset(page_id_1);
                issue_k(page_off_1, k_smem_buf0 + kKStride);
                issue_v(1, page_off_1, v_smem_buf0 + kVStride);
                s_wait_tensorcnt<4>();
            }
            else
                s_wait_tensorcnt<2>();
            block_sync_lds();
            const auto q_tile = MakeQTile(load_tile(q_lds_read_window), scale_s, row_scale);

            index_t i_total_loops = 0;
            index_t stage         = 0;

            do
            {
                // trip i reads the buffers of stage (i % kStages) and stages tile i + 2 into those
                // of stage ((i + 2) % kStages), which trip i - 1 read
                const index_t stage_ahead    = stage == 0 ? Layout::kStages - 1 : stage - 1;
                KDataType* const k_read_ptr  = k_smem_buf0 + stage * kKStride;
                KDataType* const k_write_ptr = k_smem_buf0 + stage_ahead * kKStride;
                VDataType* const v_read_ptr  = v_smem_buf0 + stage * kVStride;
                VDataType* const v_write_ptr = v_smem_buf0 + stage_ahead * kVStride;

                // tiles past the range are not staged
                const bool has_next         = i_total_loops + 1 < num_total_loop;
                const bool stage_ahead_tile = i_total_loops + 2 < num_total_loop;

                // In flight: this trip's K and V, then the next tile's. The warp reads only the
                // keys it copied, so K needs the wait and no barrier. The id of the page to stage
                // is read under the gemm.
                if(has_next)
                    s_wait_tensorcnt<3>();
                else
                    s_wait_tensorcnt<1>();
                const index_t page_id_ahead = page_nav.page_id(i_page_start + i_total_loops + 2);
                const index_t kv_tile_start = physical_seqlen_k_start + i_total_loops * kN0;

                // STAGE 1, QK gemm
                clear_tile(s_acc);
                k_lds_read_window.set_bottom_tensor_view_data_ptr(k_read_ptr);
                gemm_0(s_acc, q_tile, load_tile(k_lds_read_window));

                // STAGE 2: scale_s and the mask
                Base::ApplyScaleAndBias(s_acc,
                                        bias_dram_window,
                                        q_origin,
                                        kv_tile_start,
                                        scale_s,
                                        position_encoding,
                                        row_scale);
                Base::MaskScores(s_acc,
                                 mask,
                                 q_origin,
                                 kv_tile_start,
                                 i_total_loops == num_total_loop - 1,
                                 physical_seqlen_k_end);

                // K two tiles ahead
                const long_index_t page_off_ahead = page_nav.page_offset(page_id_ahead);
                if(stage_ahead_tile)
                    issue_k(page_off_ahead, k_write_ptr);

                // STAGE 3, softmax on the warp's own keys. The row max of the tile first.
                const auto s = cast_tile<SMPLComputeDataType>(s_acc);
                {
                    auto m_warp = block_tile_reduce<SMPLComputeDataType>(
                        s, sequence<1>{}, f_max, -numeric<SMPLComputeDataType>::infinity());
                    Base::ReduceRowSync(m_warp, f_max);
                    row_partial.get_thread_buffer()(0) = m_warp.get_thread_buffer()[0];
                }
                store_tile(max_lds_write_window, row_partial);
                block_sync_lds();
                // V two tiles ahead, into the buffer the last trip's gemm_1 read: every warp is
                // past those reads here.
                if(stage_ahead_tile)
                    issue_v(i_total_loops + 2, page_off_ahead, v_write_ptr);
                {
                    const auto partials        = load_tile(max_lds_read_window);
                    SMPLComputeDataType m_tile = partials.get_thread_buffer()[0];
                    static_for<1, kNWarp, 1>{}(
                        [&](auto i) { m_tile = max(m_tile, partials.get_thread_buffer()[i]); });
                    m.get_thread_buffer()(0) = max(m.get_thread_buffer()[0], m_tile);
                }

                // o_acc and l follow the row max: p of the row max is exactly 1 (see kFp8P). A
                // fully masked row keeps its -inf and takes 0 as the exponent's reference.
                const SMPLComputeDataType m_row = m.get_thread_buffer()[0];
                const SMPLComputeDataType gap   = m_row - m_base.get_thread_buffer()[0];
                if(gap > 0)
                {
                    const auto tmp = exp2(-gap);
                    l.get_thread_buffer()(0) *= tmp;
                    m_base.get_thread_buffer()(0) = m_row;
                    static_for<0, OaccBlockTileType::get_thread_buffer_size(), 1>{}(
                        [&](auto i) { o_acc.get_thread_buffer()(i) *= tmp; });
                }
                const SMPLComputeDataType row_exp = [&]() {
                    const auto raw = m_base.get_thread_buffer()[0];
                    if constexpr(FmhaMask::IsMasking)
                        return raw == -numeric<SMPLComputeDataType>::infinity()
                                   ? type_convert<SMPLComputeDataType>(0.f)
                                   : raw;
                    else
                        return raw;
                }();

                auto p_compute =
                    make_static_distributed_tensor<SMPLComputeDataType>(s.get_tile_distribution());
                static_for<0, decltype(p_compute)::get_thread_buffer_size(), 1>{}([&](auto i) {
                    p_compute.get_thread_buffer()(i) = exp2(s.get_thread_buffer()[i] - row_exp);
                });

                // the row sum of the warp's keys, and P as fp8 (kPScale * p, rounded once)
                {
                    auto l_warp = block_tile_reduce<SMPLComputeDataType>(
                        p_compute, sequence<1>{}, f_sum, SMPLComputeDataType{0});
                    Base::ReduceRowSync(l_warp, f_sum);
                    row_partial.get_thread_buffer()(0) = l_warp.get_thread_buffer()[0];
                }
                store_tile(sum_lds_write_window, row_partial);
                tile_elementwise_inout([](auto& x) { x *= kPScale; }, p_compute);
                store_tile(p_lds_write_window, cast_tile<PDataType>(p_compute));

                // In flight: V, then the K and V of the tiles ahead. This retires V and leaves
                // the rest running under the PV gemm; the barrier publishes V, P and the row sums.
                if(stage_ahead_tile)
                    s_wait_tensorcnt<4>();
                else if(has_next)
                    s_wait_tensorcnt<2>();
                else
                    s_wait_tensorcnt<0>();
                block_sync_lds();

                {
                    const auto partials        = load_tile(sum_lds_read_window);
                    SMPLComputeDataType l_tile = partials.get_thread_buffer()[0];
                    static_for<1, kNWarp, 1>{}(
                        [&](auto i) { l_tile += partials.get_thread_buffer()[i]; });
                    l.get_thread_buffer()(0) += l_tile;
                }

                // STAGE 4, PV gemm
                v_lds_read_window.set_bottom_tensor_view_data_ptr(v_read_ptr);
                gemm_1(o_acc, load_tile(p_lds_read_window), load_tile(v_lds_read_window));
                stage = stage + 1 == Layout::kStages ? 0 : stage + 1;
            } while(++i_total_loops < num_total_loop);
        }

        if constexpr(kStoreLSE)
        {
            // a fully masked row keeps m_base at -inf
            auto m_lse = m_base;
            if constexpr(FmhaMask::IsMasking)
            {
                tile_elementwise_inout(
                    [](auto& x) {
                        if(x == -numeric<SMPLComputeDataType>::infinity())
                            x = type_convert<SMPLComputeDataType>(0.f);
                    },
                    m_lse);
            }
            StoreLse(lse_acc_dram_window_tmp, m_lse, l);
        }

        tile_elementwise_inout([](auto& x) { x *= kPScale; }, l);
        Base::NormalizeO(o_acc, l);
        return o_acc;
    }
};

} // namespace ck_tile
