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

    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize()
    {
        return Policy::template DecodeLdsLayout<Problem>::kArenaBytes;
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
        using Layout = typename Policy::template DecodeLdsLayout<Problem>;
        using Nav    = remove_cvref_t<PageNavigator>;

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

        // the row max / row sum exchange and P share S's region
        constexpr index_t kPBytes =
            Policy::template MakePLdsBlockDescriptor<Problem>().get_element_space_size() *
            sizeof(PDataType);
        constexpr index_t kStatBytes =
            Policy::template MakeRowPartialLdsDescriptor<Problem>().get_element_space_size() *
            sizeof(SMPLComputeDataType);
        constexpr index_t kStatMaxOffset = integer_least_multiple(kPBytes, 64);
        constexpr index_t kStatSumOffset = kStatMaxOffset + integer_least_multiple(kStatBytes, 64);
        static_assert(kStatSumOffset + kStatBytes <= Layout::kSBytes);

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
        // multiple of kN0 (SplitKVAttentionMask, through SplitTileRange). The first id is read
        // behind the empty-range exit (an empty sequence has no page).
        const index_t i_page_start = logical_seqlen_k_start / kN0;
        index_t page_id            = page_nav.page_id(i_page_start);
        // a tile is staged Layout::kStages - 1 = 2 trips ahead: the second tile's id too
        static_assert(Layout::kStages == 3);
        index_t page_id_1 = page_nav.page_id(i_page_start + 1);

        TDMConfig tdm_config_k =
            Base::template MakeTdmConfig<typename Policy::template LdsPaddingConfigK<Problem>>();
        TDMConfig tdm_config_v =
            Base::template MakeTdmConfig<typename Policy::template LdsPaddingConfigV<Problem>>();

        // Q through LDS, on V's last buffer: its reads are done before the first trip stages
        // a V tile there. Issued next to the page ids: neither depends on the other.
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
        static_assert(Layout::kQBytes <= Layout::kVBytes);
        load_tile_tdm(tdm_config_q, q_lds_store_window, q_dram_window);

        // the page id's users, and so its wait, stay behind the Q transfer
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
        auto p_lds_write_window = make_tile_window(
            p_lds_view, Policy::template MakePLdsBlockDescriptor<Problem>().get_lengths(), {0, 0});
        auto p_lds_read_window =
            make_tile_window(p_lds_view,
                             Policy::template MakePLdsBlockDescriptor<Problem>().get_lengths(),
                             {0, 0},
                             Policy::template MakePRegTileDistribution<Problem>());

        // The row max and row sum of a tile, one partial per gemm_0 warp: a warp stores its
        // own and, behind a barrier, loads all of them.
        const auto make_partial_windows = [&](index_t offset) {
            auto view = make_tensor_view<address_space_enum::lds>(
                reinterpret_cast<SMPLComputeDataType*>(static_cast<char*>(smem_arena) +
                                                       Layout::kSOffset + offset),
                Policy::template MakeRowPartialLdsDescriptor<Problem>());
            constexpr auto lengths =
                Policy::template MakeRowPartialLdsDescriptor<Problem>().get_lengths();
            return make_tuple(
                make_tile_window(view, lengths, {0, 0}),
                make_tile_window(view,
                                 lengths,
                                 {0, 0},
                                 Policy::template MakeRowPartialsRegDistribution<Problem>()));
        };
        auto [max_lds_write_window, max_lds_read_window] = make_partial_windows(kStatMaxOffset);
        auto [sum_lds_write_window, sum_lds_read_window] = make_partial_windows(kStatSumOffset);

        // a warp's partial, one row per lane: gemm_0's rows
        auto row_partial = make_static_distributed_tensor<SMPLComputeDataType>(
            Policy::template MakeRowPartialRegDistribution<Problem>());
        using RowPartialTile  = decltype(row_partial);
        using RowPartialsTile = decltype(load_tile(max_lds_read_window));
        static_assert(RowPartialTile::get_thread_buffer_size() == 1 &&
                      RowPartialsTile::get_thread_buffer_size() == kNWarp);

        // V in LDS: written by the TDM, each warp whole rows of the page, and read back
        // through the page layout behind the barrier ahead of gemm_1
        auto v_dram_window = make_tile_window(
            v_dram_block_window_tmp, {0, 0}, Policy::template MakeVDramTileDistribution<Problem>());
        VDataType* const v_smem_buf0 =
            reinterpret_cast<VDataType*>(static_cast<char*>(smem_arena) + Layout::kV0Offset);
        constexpr index_t kVStride = Layout::kVStride / sizeof(VDataType);
        auto v_lds_view            = make_tensor_view<address_space_enum::lds>(
            v_smem_buf0, Policy::template MakeVLdsBlockDescriptor<Problem>());
        auto v_lds_write_window = make_tile_window(
            v_lds_view, Policy::template MakeVLdsBlockDescriptor<Problem>().get_lengths(), {0, 0});
        auto v_lds_read_window =
            make_tile_window(make_tensor_view<address_space_enum::lds>(
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
                constexpr auto tail_desc = Policy::template MakeVTailLdsBlockDescriptor<Problem>();
                auto v_tail_lds_window =
                    make_tile_window(make_tensor_view<address_space_enum::lds>(v_buf, tail_desc),
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
            Base::StoreLse(lse_acc_dram_window_tmp, m_lse, l, scale_s);
        }

        tile_elementwise_inout([](auto& x) { x *= kPScale; }, l);
        Base::NormalizeO(o_acc, l);
        return o_acc;
    }
};

} // namespace ck_tile
