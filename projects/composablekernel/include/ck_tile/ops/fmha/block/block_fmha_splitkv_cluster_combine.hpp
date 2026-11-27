// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

// How the workgroups of a split-KV tile combine the partial results of its splits themselves,
// for split kernels launched in workgroup clusters (see FmhaFwdSplitKVKernel). seqlen_q is 1, so
// a row is a Q head of the tile's head group.
//
// A split leaves its O, divided by the row sum and in the output's type FinalODataType, in the
// leading bytes of its rows of the O workspace, and the LSE of its rows in the LSE workspace.
// The workgroups share the rows of the tile: workgroup i_share of num_shares combines rows
// [i_share * rows_per, ...) with rows_per = ceil(kM0 / num_shares).
//
// The partials are read with plain loads whose offsets are all set up ahead of the first one,
// not through tile windows: the split stride is a run-time value, and loads that wait for an
// address, or bounds-checked window loads, cost more than the rest of the combine.
template <typename OaccDataType_,
          typename FinalODataType_,
          typename LSEDataType_,
          index_t kM0_,
          index_t kNumThreads_>
struct BlockFmhaSplitKVClusterCombine
{
    using OaccDataType   = remove_cvref_t<OaccDataType_>;
    using FinalODataType = remove_cvref_t<FinalODataType_>;
    using LSEDataType    = remove_cvref_t<LSEDataType_>;

    static constexpr index_t kM0         = kM0_;
    static constexpr index_t kNumThreads = kNumThreads_;
    // workgroups of a cluster, and so shares of a tile's rows, at most
    static constexpr index_t kMaxShares = 16;

    // The workspace of the tile's head group: row 0 of split 0. Strides in elements of
    // OaccDataType and of LSEDataType.
    struct Partials
    {
        OaccDataType* o;
        LSEDataType* lse;
        index_t row_stride_o;
        index_t row_stride_lse;
        index_t split_stride_o;
        index_t split_stride_lse;
    };

    // The outputs of the tile's head group: row 0. lse is null without an LSE output.
    struct Output
    {
        FinalODataType* o;
        LSEDataType* lse;
        index_t row_stride_o;
        index_t row_stride_lse;
    };

#if defined(__gfx125__)
    // One cluster a tile: combines the num_splits (<= kMaxShares) splits of the tile whose first
    // row is i_m0 into the output. The workgroups of the splits are the cluster; i_split is this
    // one's. num_group_rows: rows of the head group.
    CK_TILE_DEVICE static void Combine(const Partials& parts,
                                       const Output& out,
                                       index_t i_m0,
                                       index_t i_split,
                                       index_t num_splits,
                                       index_t num_group_rows,
                                       index_t hdim_v)
    {
        if(num_splits > 1)
            cluster_sync_memory();
        // the splits padded to a power of two, so that a thread's loads are a constant count
        const auto combine = [&](auto n) {
            constexpr index_t kParts = decltype(n)::value;
            CombineRows<kParts, kParts, false, true>(
                parts, out, i_m0, 0, 1, num_splits, i_split, num_splits, num_group_rows, hdim_v);
        };
        if(num_splits <= 2)
            combine(number<2>{});
        else if(num_splits <= 4)
            combine(number<4>{});
        else if(num_splits <= 8)
            combine(number<8>{});
        else
            combine(number<16>{});
    }

    // Several clusters of group_splits splits a tile (num_splits is a multiple of it). A cluster
    // combines its splits into the workspace slot of its first split, at device scope. Split s
    // of every cluster then counts on counters[s], and the one of the last cluster to arrive
    // combines the clusters' rows of share s into the output.
    // counters: kMaxShares int32 of the tile, zero before the launch, which leaves them zero.
    // smem: LDS no longer in use, one int32.
    CK_TILE_DEVICE static void CombineClusters(const Partials& parts,
                                               const Output& out,
                                               index_t i_m0,
                                               index_t i_split,
                                               index_t group_splits,
                                               index_t num_splits,
                                               index_t num_group_rows,
                                               index_t hdim_v,
                                               int32_t* counters,
                                               void* smem)
    {
        constexpr index_t kShares = kMaxShares;
        const index_t count       = group_splits;
        const index_t i_local     = i_split % count;
        const index_t i_first     = i_split - i_local;
        cluster_sync_memory();
        CombineRows<kShares, kShares, false, false>(
            parts, out, i_m0, i_first, 1, count, i_local, count, num_group_rows, hdim_v);

        // every wave's stores are acknowledged before the workgroup counts
        asm volatile("s_wait_storecnt 0x0" ::: "memory");
        int32_t* const is_last = static_cast<int32_t*>(smem);
        block_sync_lds();
        const index_t clusters = num_splits / count;
        if(threadIdx.x == 0)
        {
            int32_t* const counter = counters + i_local;
            const bool last =
                __scoped_atomic_fetch_add(counter, 1, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE) ==
                clusters - 1;
            if(last) // every cluster has counted: zero for the next launch
                __scoped_atomic_store_n(counter, 0, __ATOMIC_RELAXED, __MEMORY_SCOPE_DEVICE);
            *is_last = last ? 1 : 0;
        }
        block_sync_lds();
        if(*is_last == 0)
            return;
        const auto combine = [&](auto n) {
            constexpr index_t kParts = decltype(n)::value;
            CombineRows<kParts, kShares, true, true>(
                parts, out, i_m0, 0, count, clusters, i_local, count, num_group_rows, hdim_v);
        };
        if(clusters <= 2)
            combine(number<2>{});
        else if(clusters <= 4)
            combine(number<4>{});
        else if(clusters <= 8)
            combine(number<8>{});
        else
            combine(number<16>{});
    }

    // Combines num_parts <= kParts partials of a tile: the workspace slots i_first + i * step.
    // This workgroup takes share i_share of the rows, split over num_shares (in (kShares / 2,
    // kShares]) workgroups: kThreads threads per row with kDims head dims each (hdim_v is a
    // multiple of kDims), a 16-byte store of the output. A thread issues all its loads before it
    // uses one; partials past num_parts read the last one again and weigh nothing.
    // The partials are of the output's type, in the leading bytes of the workspace's rows.
    // kDeviceScope: the partials were stored at device scope by other clusters.
    // kToOutput: into the final output, else into slot i_first at device scope.
    template <index_t kParts, index_t kShares, bool kDeviceScope, bool kToOutput>
    CK_TILE_DEVICE static void CombineRows(const Partials& parts,
                                           const Output& out,
                                           index_t i_m0,
                                           index_t i_first,
                                           index_t step,
                                           index_t num_parts,
                                           index_t i_share,
                                           index_t num_shares,
                                           index_t num_group_rows,
                                           index_t hdim_v)
    {
        constexpr index_t kDims    = 16 / sizeof(FinalODataType);
        constexpr index_t kThreads = 16; // per row
        static_assert(kM0 <= kMaxShares && kNumThreads % kThreads == 0);
        static_assert(std::is_same_v<OaccDataType, float> && std::is_same_v<LSEDataType, float>);
        // rows the block takes at once, and the most a share has
        constexpr index_t kRowSlots = kNumThreads / kThreads;
        constexpr index_t kMaxRowsPer =
            kShares == 2 ? kM0 : (kM0 + kShares / 2) / (kShares / 2 + 1);
        constexpr index_t kRows = (kMaxRowsPer + kRowSlots - 1) / kRowSlots;

        // the slot of split i_first; the workspace's strides are in OaccDataType elements
        FinalODataType* o_acc = reinterpret_cast<FinalODataType*>(
            parts.o + static_cast<long_index_t>(i_first) * parts.split_stride_o);
        constexpr index_t kOaccBytes = sizeof(OaccDataType);
        LSEDataType* lse_acc =
            parts.lse + static_cast<long_index_t>(i_first) * parts.split_stride_lse;
        const index_t o_step   = step * parts.split_stride_o;
        const index_t lse_step = step * parts.split_stride_lse;

        // rows [row0, row_end) of the tile are this share's
        const index_t num_rows = min(num_group_rows - i_m0, kM0);
        const index_t rows_per = (kM0 + num_shares - 1) / num_shares;
        const index_t row0     = i_share * rows_per;
        const index_t row_end  = min(row0 + rows_per, num_rows);
        const index_t i_slot   = threadIdx.x / kThreads;
        const index_t i_d      = (threadIdx.x % kThreads) * kDims;
        if(i_d >= hdim_v || row0 + i_slot >= row_end)
            return;

        using OaccVec = ext_vector_t<OaccDataType, kDims>;
        using PartVec = thread_buffer<FinalODataType, kDims>;
        LSEDataType lse[kRows][kParts];
        PartVec part[kRows][kParts];
        // The byte offset of every load, all set up ahead of the first one: a load that had to
        // wait for the address register of the one before it would follow it only then.
        uint32_t lse_at[kRows][kParts];
        uint32_t o_at[kRows][kParts];
        for(index_t j = 0; j < kRows; ++j)
        {
            // rows past the share read its last row again and are not stored
            const index_t i_row = i_m0 + min(row0 + j * kRowSlots + i_slot, row_end - 1);
            for(index_t s = 0; s < kParts; ++s)
            {
                const index_t i_s = min(s, num_parts - 1);
                lse_at[j][s] =
                    (i_s * lse_step + i_row * parts.row_stride_lse) * sizeof(LSEDataType);
                // the thread's dims in the row of partial i_s
                o_at[j][s] = (i_s * o_step + i_row * parts.row_stride_o) * kOaccBytes +
                             i_d * sizeof(FinalODataType);
            }
        }
        __builtin_amdgcn_sched_barrier(0);
        for(index_t j = 0; j < kRows; ++j)
        {
            for(index_t s = 0; s < kParts; ++s)
            {
                if constexpr(kDeviceScope)
                {
                    constexpr auto kScope  = amd_buffer_coherence_enum::DEVICE;
                    const auto o_acc_res   = make_wave_buffer_resource(o_acc, 0x7ffffffe);
                    const auto lse_acc_res = make_wave_buffer_resource(lse_acc, 0x7ffffffe);
                    lse[j][s]              = amd_buffer_load_impl<LSEDataType, 1, kScope>(
                        lse_acc_res, lse_at[j][s], 0)[number<0>{}];
                    part[j][s] = amd_buffer_load_impl<FinalODataType, kDims, kScope>(
                        o_acc_res, o_at[j][s], 0);
                }
                else
                {
                    __builtin_memcpy(&lse[j][s],
                                     reinterpret_cast<const char*>(lse_acc) + lse_at[j][s],
                                     sizeof(LSEDataType));
                    __builtin_memcpy(&part[j][s],
                                     reinterpret_cast<const char*>(o_acc) + o_at[j][s],
                                     sizeof(PartVec));
                }
            }
        }
        __builtin_amdgcn_sched_barrier(0);

        for(index_t j = 0; j < kRows; ++j)
        {
            const index_t row = row0 + j * kRowSlots + i_slot;
            if(row >= row_end)
                break;
            const index_t i_row = i_m0 + row;

            LSEDataType lse_max = -numeric<LSEDataType>::infinity();
            for(index_t s = 0; s < kParts; ++s)
            {
                lse[j][s] = s < num_parts ? lse[j][s] : -numeric<LSEDataType>::infinity();
                lse_max   = max(lse_max, lse[j][s]);
            }
            // a row without keys in any partial is all zeros
            const bool empty = lse_max == -numeric<LSEDataType>::infinity();
            LSEDataType sum  = 0;
            OaccVec acc      = {};
            for(index_t s = 0; s < kParts; ++s)
            {
                const LSEDataType w =
                    empty ? LSEDataType(0)
                          : ck_tile::exp2((lse[j][s] - lse_max) * log2e_v<LSEDataType>);
                sum += w;
                OaccVec p = {};
                for(index_t d = 0; d < kDims; ++d)
                    p[d] = type_convert<OaccDataType>(part[j][s][d]);
                acc += w * p;
            }
            const LSEDataType rcp = empty ? LSEDataType(0) : LSEDataType(1) / sum;
            // taken only where the lse is stored
            const auto lse_row = [&]() { return empty ? lse_max : ck_tile::log(sum) + lse_max; };

            if constexpr(kToOutput)
            {
                FinalODataType o_row[kDims];
                for(index_t d = 0; d < kDims; ++d)
                    o_row[d] = type_convert<FinalODataType>(acc[d] * rcp);
                __builtin_memcpy(out.o + static_cast<long_index_t>(i_row) * out.row_stride_o + i_d,
                                 o_row,
                                 sizeof(o_row));

                if(out.lse != nullptr && i_d == 0)
                    out.lse[static_cast<long_index_t>(i_row) * out.row_stride_lse] = lse_row();
            }
            else
            {
                // Only this workgroup reads the rows it writes to the slot, and it has issued
                // those loads.
                const auto o_acc_res   = make_wave_buffer_resource(o_acc, 0x7ffffffe);
                const auto lse_acc_res = make_wave_buffer_resource(lse_acc, 0x7ffffffe);
                PartVec v;
                for(index_t d = 0; d < kDims; ++d)
                    v[d] = type_convert<FinalODataType>(acc[d] * rcp);
                amd_buffer_store_impl<FinalODataType, kDims, amd_buffer_coherence_enum::DEVICE>(
                    v,
                    o_acc_res,
                    i_row * parts.row_stride_o * kOaccBytes + i_d * sizeof(FinalODataType),
                    0);
                if(i_d == 0)
                {
                    thread_buffer<LSEDataType, 1> l;
                    l[number<0>{}] = lse_row();
                    amd_buffer_store_impl<LSEDataType, 1, amd_buffer_coherence_enum::DEVICE>(
                        l, lse_acc_res, i_row * parts.row_stride_lse * sizeof(LSEDataType), 0);
                }
            }
        }
    }
#endif
};

} // namespace ck_tile
