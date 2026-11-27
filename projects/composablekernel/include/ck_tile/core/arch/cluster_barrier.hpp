// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/config.hpp"

// Cluster barrier (gfx1250): barrier id -3 spans every workgroup of the cluster, as id -1 spans
// the waves of one workgroup. One wave per workgroup joins it, so a full cluster sync is a
// workgroup barrier and then the cluster barrier on one wave.
//
// These are inline asm, in the sequence llvm.amdgcn.s.cluster.barrier lowers to. The asm keeps
// the code where the caller put it: no LDS drain is added and nothing is scheduled into the
// sequence. A caller that needs a wait (s_wait_dscnt, s_wait_tensorcnt) issues it first.
//
// cluster_barrier_arrive() and cluster_barrier_wait() are cluster_barrier() split in two, so a
// workgroup can work between its arrival and the cluster's. Each arrive needs its wait before the
// next arrive or cluster_barrier().
//
// On other targets these are no-ops.

namespace ck_tile {

/// Every wave of every workgroup in the cluster: the workgroup barrier, then the workgroup's
/// first wave signals and waits on the cluster barrier.
CK_TILE_DEVICE void cluster_barrier()
{
#if defined(__gfx125__)
    asm volatile("s_barrier_signal_isfirst -1\n"
                 "s_barrier_wait -1\n"
                 "s_cbranch_scc0 1\n"
                 "s_barrier_signal -3\n"
                 "s_barrier_wait -3" ::
                     : "scc", "memory");
#endif
}

/// Publishes the workgroup's stores to its cluster and takes in the stores of the cluster's other
/// workgroups: a release fence, the cluster barrier and an acquire fence, all of cluster scope.
CK_TILE_DEVICE void cluster_sync_memory()
{
#if defined(__gfx125__)
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "cluster");
    cluster_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "cluster");
#endif
}

/// The first half of cluster_barrier(): the workgroup barrier, then the workgroup's first wave
/// signals the cluster barrier.
CK_TILE_DEVICE void cluster_barrier_arrive()
{
#if defined(__gfx125__)
    asm volatile("s_barrier_signal_isfirst -1\n"
                 "s_barrier_wait -1\n"
                 "s_cbranch_scc0 1\n"
                 "s_barrier_signal -3" ::
                     : "scc", "memory");
#endif
}

/// The second half of cluster_barrier(): every workgroup of the cluster has arrived.
CK_TILE_DEVICE void cluster_barrier_wait()
{
#if defined(__gfx125__)
    asm volatile("s_barrier_wait -3" ::: "memory");
#endif
}

} // namespace ck_tile
