// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// C entry points of FmhaDecodePaged5dKernel for serving frameworks (see
// fmha_decode_paged5d.py for the PyTorch wrapper). The launch configuration depends only on the
// batch, the heads, the query length, the window and an upper bound of the context lengths, so one
// call sequence can be captured in a HIP graph and replayed while the context lengths (read on the
// device) change below that bound.

#include "fmha_decode_paged5d_runner.hpp"

namespace {
ck_tile::FmhaDecodePaged5dConfig config(int batch,
                                        int nhead_q,
                                        int nhead_k,
                                        int max_seqlen_k,
                                        int window,
                                        int page_size,
                                        int hdim,
                                        int qlen)
{
    return ck_tile::fmha_decode_paged5d_config(
        batch, nhead_k, max_seqlen_k, window, page_size, hdim, nhead_q / nhead_k * qlen, qlen);
}
} // namespace

extern "C" {

// Bytes of scratch workspace for these sizes.
size_t ck_fmha_decode_paged5d_workspace_bytes(int batch,
                                              int nhead_q,
                                              int nhead_k,
                                              int max_seqlen_k,
                                              int window,
                                              int page_size,
                                              int hdim,
                                              int qlen)
{
    if(batch <= 0 || nhead_k <= 0 || nhead_q % nhead_k != 0 || qlen < 1)
        return 0;
    const auto c = config(batch, nhead_q, nhead_k, max_seqlen_k, window, page_size, hdim, qlen);
    return ck_tile::fmha_decode_paged5d_workspace_bytes(
        batch, nhead_k, c.num_splits, c.num_groups, hdim, c.row_blocks * c.m_tiles * 16);
}

// int32 counters for these heads and query length; zero them once, calls of any shape with at most
// as many query rows per kv head (q heads per kv head x qlen) share them (not concurrently).
size_t ck_fmha_decode_paged5d_num_counters(int batch, int nhead_q, int nhead_k, int qlen)
{
    if(nhead_k <= 0 || qlen < 1)
        return 0;
    // at most one row block per 16 rows
    const int rows = nhead_q / nhead_k * qlen;
    return ck_tile::fmha_decode_paged5d_num_counters(batch, nhead_k, std::max(1, (rows + 15) / 16));
}

// q, o: [batch][qlen][nhead_q][hdim] bf16, or fp16 with qo_fp16 != 0 (query token i of a sequence
// is its context's position seqlen_k - qlen + i and sees the keys up to it); k, v: fp8 e4m3 caches
// of page_size (16, 64 or 128) tokens per page in kv_layout (ck_tile::FmhaDecodePagedLayout:
// 0 shuffle_5d, 1 hnd, 2 nhd); hdim: 64, 128 or 256; block_table: int32 rows of
// block_table_stride entries; seqlen_k: int32 [batch], each <= max_seqlen_k; k/v descale: one
// float each; sinks: optional [nhead_q] logits joining every denominator (-inf: none), fp32,
// bf16 or fp16 by sink_dtype 0, 1, 2;
// lse: optional fp32 [batch][qlen][nhead_q] log-sum-exp output. Returns a hipError_t.
int ck_fmha_decode_paged5d(const void* q,
                           const void* k,
                           const void* v,
                           const int32_t* block_table,
                           int block_table_stride,
                           const int32_t* seqlen_k,
                           const float* k_descale,
                           const float* v_descale,
                           const void* sinks,
                           int sink_dtype,
                           void* o,
                           float* lse,
                           void* workspace,
                           int32_t* counters,
                           int batch,
                           int nhead_q,
                           int nhead_k,
                           int qlen,
                           int max_seqlen_k,
                           int window,
                           int page_size,
                           int hdim,
                           int kv_layout,
                           int qo_fp16,
                           float scale_s,
                           hipStream_t stream)
{
    if(batch <= 0 || nhead_k <= 0 || nhead_q % nhead_k != 0 || qlen < 1)
        return hipErrorInvalidValue;
    if(workspace == nullptr || counters == nullptr || kv_layout < 0 || kv_layout > 2)
        return hipErrorInvalidValue;
    if(hdim != 64 && hdim != 128 && hdim != 256)
        return hipErrorInvalidValue;
    if(sink_dtype < 0 || sink_dtype > 2)
        return hipErrorInvalidValue;
    const auto c = config(batch, nhead_q, nhead_k, max_seqlen_k, window, page_size, hdim, qlen);
    const ck_tile::FmhaDecodePaged5dHostArgs args{
        q,
        k,
        v,
        block_table,
        seqlen_k,
        k_descale,
        v_descale,
        o,
        workspace,
        counters,
        batch,
        nhead_q,
        nhead_k,
        block_table_stride,
        page_size,
        window,
        c.pages_per_split,
        c.num_groups,
        scale_s,
        qo_fp16 != 0,
        static_cast<ck_tile::FmhaDecodePagedLayout>(kv_layout),
        hdim,
        qlen,
        sinks,
        sink_dtype,
        lse};
    return fmha_decode_paged5d::launch(args, c, stream);
}
}
