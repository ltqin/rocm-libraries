// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// FmhaDecodePaged5dKernel (gfx125) against the CPU reference of fmha_decode_paged5d_runner.

#include <gtest/gtest.h>

#include "fmha_decode_paged5d_runner.hpp"

namespace {

using fmha_decode_paged5d::Case;

struct Param
{
    std::string name;
    Case c;
    // layout the case must run with: "coop" or "wave", "+groups" for several clusters per
    // (batch, kv head); empty: not checked
    const char* layout = "";
    // 16-row tiles per workgroup and row blocks the case must run with; 0: not checked
    ck_tile::index_t m_tiles    = 0;
    ck_tile::index_t row_blocks = 0;
};

std::vector<Param> params()
{
    std::vector<Param> p;
    const auto add = [&](const std::string& name, Case c, const char* layout = "") {
        p.push_back({name, c, layout});
    };
    // query_length qlen, tiles per workgroup (0: heuristic), and the tiles x row blocks the
    // query rows must take
    const auto add_rows = [&](const std::string& name,
                              Case c,
                              ck_tile::index_t qlen,
                              ck_tile::index_t force_tiles,
                              ck_tile::index_t m_tiles,
                              ck_tile::index_t row_blocks,
                              const char* layout = "") {
        c.qlen    = qlen;
        c.m_tiles = force_tiles;
        p.push_back({name, c, layout, m_tiles, row_blocks});
    };
    // window 2048 at short, medium and long contexts
    add("swa_b1_s1024", {1, 16, 1, {1024}, 2048}, "coop");
    add("swa_b8_s8192", {8, 16, 1, {8192}, 2048}, "coop");
    add("swa_b2_s73728", {2, 16, 1, {73728}, 2048}, "coop");
    // partial last page, sequences shorter than one step, one page
    add("swa_b3_s1000", {3, 16, 1, {1000}, 2048});
    add("swa_b2_s77", {2, 16, 1, {77}, 2048});
    add("swa_b1_s5", {1, 16, 1, {5}, 2048});
    // different lengths in one batch: some splits of the shorter sequences are empty
    add("swa_mixed", {4, 16, 1, {9000, 100, 2048, 3333}, 2048});
    // no window: one cluster of either layout, several clusters
    add("full_b2_s3000", {2, 16, 1, {3000}, 0}, "coop");
    add("full_b8_s8192", {8, 16, 1, {8192}, 0}, "wave");
    add("full_b1_s20000", {1, 16, 1, {20000}, 0}, "coop+groups");
    add("groups_full_b1_s32768", {1, 16, 1, {32768}, 0}, "coop+groups");
    add("groups_mixed", {3, 16, 1, {40000, 5000, 70}, 0}, "coop+groups");
    add("groups_forced", {2, 16, 2, {6000}, 2048, 30}, "coop+groups");
    // forced layouts
    add("full_coop", {2, 16, 1, {5000}, 0, 0, 0, "coop"}, "coop");
    add("swa_wave", {2, 16, 1, {5000}, 2048, 0, 0, "wave"}, "wave");
    // one split (direct output), the most splits a cluster takes
    add("one_split", {2, 16, 1, {700}, 2048, 1});
    add("fifteen_splits", {2, 16, 1, {960}, 2048, 15});
    // other GQA shapes: fewer than 16 rows, several kv heads
    add("gqa_8_1", {2, 8, 1, {2500}, 2048});
    add("gqa_16_2", {2, 16, 2, {4000}, 2048});
    add("gqa_32_4_wave", {2, 32, 4, {999}, 2048, 0, 0, "wave"}, "wave");
    add("gqa_6_3", {2, 6, 3, {1500}, 512});
    // launches sized for a longer bound than the contexts, as in graph replay: the ranges spread
    // over the grid, clusters without pages exit
    add("bound_full_b8", {8, 16, 1, {8192}, 0, 0, 0, "auto", 131072}, "wave+groups");
    add("bound_groups_mixed",
        {3, 16, 1, {40000, 5000, 70}, 0, 0, 0, "auto", 131072},
        "wave+groups");
    add("bound_coop_groups", {2, 16, 1, {2000}, 0, 0, 0, "auto", 30000}, "coop+groups");
    add("bound_one_cluster", {2, 16, 1, {300}, 0, 0, 0, "auto", 3840}, "coop");
    add("bound_swa_gqa", {4, 16, 2, {100, 7000}, 2048, 0, 0, "auto", 65536}, "coop");
    // a window over a context just below the bound covers one page (64-token pages: four) more
    // than the bound's own range; splits sized for the bound alone would drop the newest keys
    add("bound_swa_b64_s8191", {64, 16, 1, {8191}, 2048, 0, 0, "auto", 8192}, "coop");
    add("bound_groups_s29999", {1, 16, 1, {29999}, 23040, 0, 0, "auto", 30000}, "coop+groups");
    for(const ck_tile::index_t ps : {16, 64, 128})
        for(const ck_tile::index_t w : {1920, 2048})
            for(const ck_tile::index_t ctx : {8191, 8180})
                add("bound_p" + std::to_string(ps) + "_w" + std::to_string(w) + "_s" +
                        std::to_string(ctx),
                    {1, 16, 1, {ctx}, w, 0, 0, "auto", 8192, ps});
    // cache pages of 64 and 128 tokens: windows starting inside a cache page, partial last
    // cache pages, both layouts, several clusters, a longer launch bound
    for(const ck_tile::index_t ps : {64, 128})
    {
        const auto name = [&](const char* n) { return "p" + std::to_string(ps) + "_" + n; };
        add(name("swa_b2_s8192"), {2, 16, 1, {8192}, 2048, 0, 0, "auto", 0, ps}, "coop");
        add(name("swa_b3_s1000_w300"), {3, 16, 1, {1000}, 300, 0, 0, "auto", 0, ps});
        add(name("swa_b1_s5"), {1, 16, 1, {5}, 2048, 0, 0, "auto", 0, ps});
        add(name("swa_mixed"), {4, 16, 1, {9000, 100, 2048, 3333}, 2048, 0, 0, "auto", 0, ps});
        add(name("swa_wave"), {2, 16, 1, {5000}, 2048, 0, 0, "wave", 0, ps}, "wave");
        add(name("full_b8_s8192"), {8, 16, 1, {8192}, 0, 0, 0, "auto", 0, ps}, "wave");
        add(name("full_b1_s20000"), {1, 16, 1, {20000}, 0, 0, 0, "auto", 0, ps}, "coop+groups");
        add(name("groups_forced"), {2, 16, 2, {6000}, 2048, 30, 0, "auto", 0, ps}, "coop+groups");
        add(name("gqa_16_2"), {2, 16, 2, {4000}, 2048, 0, 0, "auto", 0, ps});
        add(name("gqa_32_4_wave"), {2, 32, 4, {999}, 2048, 0, 0, "wave", 0, ps}, "wave");
        add(name("bound_full_b8"), {8, 16, 1, {8192}, 0, 0, 0, "auto", 131072, ps}, "wave+groups");
    }
    // fp16 Q and output: direct output, both layouts within a cluster, several clusters
    add("fp16_one_split", {2, 16, 1, {700}, 2048, 1, 0, "auto", 0, 16, true});
    add("fp16_swa_b8_s8192", {8, 16, 1, {8192}, 2048, 0, 0, "auto", 0, 16, true}, "coop");
    add("fp16_full_b8_s8192", {8, 16, 1, {8192}, 0, 0, 0, "auto", 0, 16, true}, "wave");
    add("fp16_full_b1_s20000", {1, 16, 1, {20000}, 0, 0, 0, "auto", 0, 16, true}, "coop+groups");
    add("fp16_p64_bound_full_b8",
        {8, 16, 1, {8192}, 0, 0, 0, "auto", 131072, 64, true},
        "wave+groups");
    // head dims 64 and 256: both layouts, several clusters, every page size, GQA and fp16
    for(const ck_tile::index_t d : {64, 256})
    {
        const auto add_d = [&](const std::string& n, Case c, const char* layout = "") {
            c.hdim = d;
            add("d" + std::to_string(d) + "_" + n, c, layout);
        };
        add_d("one_split", {2, 16, 1, {700}, 2048, 1, 0, "auto"});
        add_d("swa_coop", {2, 16, 1, {5000}, 2048, 0, 0, "coop"}, "coop");
        add_d("swa_wave", {2, 16, 1, {5000}, 2048, 0, 0, "wave"}, "wave");
        add_d("swa_mixed", {4, 16, 1, {9000, 100, 2048, 3333}, 2048, 0, 0, "auto"});
        add_d("full_b1_s20000", {1, 16, 1, {20000}, 0, 0, 0, "auto"}, "wave+groups");
        add_d("coop_groups", {1, 16, 1, {20000}, 0, 0, 0, "coop"}, "coop+groups");
        add_d("gqa_32_4_wave", {2, 32, 4, {999}, 2048, 0, 0, "wave"}, "wave");
        for(const ck_tile::index_t ps : {64, 128})
        {
            const auto pn = "p" + std::to_string(ps) + "_";
            add_d(pn + "swa_b3_s1000_w300", {3, 16, 1, {1000}, 300, 0, 0, "auto", 0, ps});
            add_d(pn + "full_b8_s8192", {8, 16, 1, {8192}, 0, 0, 0, "auto", 0, ps}, "wave");
            add_d(pn + "bound_full_b8",
                  {8, 16, 1, {8192}, 0, 0, 0, "auto", 131072, ps},
                  "wave+groups");
        }
        add_d("fp16_full_b8_s8192", {8, 16, 1, {8192}, 0, 0, 0, "auto", 0, 16, true}, "wave");
        add_d("fp16_p64_swa", {2, 16, 2, {4000}, 2048, 0, 0, "coop", 0, 64, true}, "coop");
    }
    // more than 16 query rows per kv head: row blocks of one tile, or of two tiles sharing each
    // K/V page (forced here; the heuristic takes two only for large launches), rows not a multiple
    // of 16
    add_rows("gqa32_swa", {2, 32, 1, {5000}, 2048}, 1, 0, 1, 2);
    add_rows("gqa24_h48_2", {3, 48, 2, {3000}, 0}, 1, 0, 1, 2);
    add_rows("gqa64_swa", {2, 64, 1, {9000}, 2048}, 1, 0, 1, 4, "coop");
    add_rows("gqa40_short", {2, 40, 1, {77}, 2048}, 1, 0, 1, 3);
    add_rows("gqa128_blocks", {2, 128, 1, {4000}, 0}, 1, 0, 1, 8);
    add_rows("gqa96_h192_2", {1, 192, 2, {2500}, 1000}, 1, 0, 1, 6);
    add_rows("gqa32_fp16_p64", {2, 32, 1, {7000}, 0, 0, 0, "auto", 0, 64, true}, 1, 0, 1, 2);
    add_rows("m2_gqa32_swa", {2, 32, 1, {5000}, 2048}, 1, 2, 2, 1);
    add_rows("m2_gqa32_wave", {2, 32, 1, {5000}, 2048, 0, 0, "wave"}, 1, 2, 2, 1, "wave");
    add_rows(
        "m2_gqa32_wave_groups", {1, 32, 1, {20000}, 0, 0, 0, "wave"}, 1, 2, 2, 1, "wave+groups");
    add_rows(
        "m2_gqa64_coop_groups", {1, 64, 1, {30000}, 0, 0, 0, "coop"}, 1, 2, 2, 2, "coop+groups");
    add_rows("m2_gqa40_short", {2, 40, 1, {77}, 2048}, 1, 2, 2, 2);
    add_rows("m2_gqa96_h192_2", {1, 192, 2, {2500}, 1000}, 1, 2, 2, 3);
    add_rows("m2_gqa64_p128", {2, 64, 1, {7000}, 2048, 0, 0, "auto", 0, 128}, 1, 2, 2, 2);
    add_rows("m2_gqa32_fp16_p64", {2, 32, 1, {7000}, 0, 0, 0, "auto", 0, 64, true}, 1, 2, 2, 1);
    // MTP: query_length tokens per sequence, each seeing the keys up to its own position
    add_rows("mtp2_gqa8_swa", {2, 8, 1, {5000}, 2048}, 2, 0, 1, 1);
    add_rows("mtp2_gqa8_full", {2, 8, 1, {3000}, 0}, 2, 0, 1, 1);
    add_rows("mtp4_gqa1", {3, 4, 4, {900}, 0}, 4, 0, 1, 1);
    add_rows("mtp3_gqa16", {2, 16, 1, {6000}, 2048}, 3, 0, 1, 3);
    add_rows("mtp3_window5", {2, 8, 1, {300}, 5}, 3, 0, 1, 2);
    add_rows("mtp4_short_ctx", {2, 8, 1, {2, 3}, 0}, 4, 0, 1, 2);
    add_rows("mtp3_p64_w100", {3, 8, 1, {1000, 64, 2500}, 100, 0, 0, "auto", 0, 64}, 3, 0, 1, 2);
    add_rows("mtp2_bound_groups", {4, 8, 1, {8192}, 0, 0, 0, "auto", 131072}, 2, 0, 1, 1);
    add_rows("mtp2_fp16_p128", {2, 16, 2, {4000}, 0, 0, 0, "auto", 0, 128, true}, 2, 0, 1, 1);
    add_rows("m2_mtp2_gqa16_wave", {2, 16, 1, {5000}, 0, 0, 0, "wave"}, 2, 2, 2, 1, "wave");
    add_rows("m2_mtp5_gqa16", {1, 16, 1, {12000}, 4096}, 5, 2, 2, 3);
    add_rows("m2_mtp3_window5", {2, 8, 1, {300}, 5}, 3, 2, 2, 1);
    add_rows("m2_mtp4_short_ctx", {2, 8, 1, {2, 3}, 0}, 4, 2, 2, 1);
    for(const ck_tile::index_t d : {64, 256})
    {
        const auto dn = "d" + std::to_string(d) + "_";
        Case c{2, 32, 1, {5000}, 2048, 0, 0, "wave"};
        c.hdim = d;
        add_rows(dn + "m2_gqa32_wave", c, 1, 2, 2, 1, "wave");
        c      = Case{2, 16, 1, {6000}, 0};
        c.hdim = d;
        add_rows(dn + "mtp4_gqa16", c, 4, 0, 1, 4);
        add_rows(dn + "m2_mtp4_gqa16", c, 4, 2, 2, 2);
    }
    // sinks (one -inf, one +inf) and the log-sum-exp, through each way the output is finished:
    // one split, a cluster merge, a group merge; also rows without keys (MTP, short contexts)
    const auto sink_case = [](Case c, ck_tile::index_t qlen = 1, ck_tile::index_t tiles = 0) {
        c.sink    = true;
        c.lse     = true;
        c.qlen    = qlen;
        c.m_tiles = tiles;
        return c;
    };
    add("sink_one_split", sink_case({2, 16, 1, {700}, 2048, 1}));
    add("sink_coop", sink_case({2, 16, 1, {3000}, 2048}), "coop");
    add("sink_wave", sink_case({2, 16, 1, {5000}, 0, 0, 0, "wave"}), "wave");
    add("sink_coop_groups", sink_case({1, 16, 1, {20000}, 0}), "coop+groups");
    add("sink_wave_groups", sink_case({8, 16, 1, {8192}, 0, 0, 0, "auto", 131072}), "wave+groups");
    add("sink_short", sink_case({2, 16, 1, {1, 5}, 2048}));
    add("sink_gqa8_h_k2", sink_case({2, 16, 2, {4000}, 0, 0, 0, "auto", 0, 64}));
    add("sink_m2_gqa32", sink_case({2, 32, 1, {5000}, 2048}, 1, 2));
    add("sink_mtp3_window5", sink_case({2, 8, 1, {300}, 5}, 3));
    add("sink_mtp4_short_ctx", sink_case({2, 8, 1, {2, 3}, 0}, 4));
    add("sink_fp16_p128", sink_case({2, 16, 1, {9000}, 0, 0, 0, "auto", 0, 128, true}));
    {
        Case c{2, 16, 1, {3000}, 0};
        c.lse = true;
        add("lse_only", c);
    }
    for(const ck_tile::index_t d : {64, 256})
    {
        Case c{2, 16, 1, {5000}, 0, 0, 0, "wave"};
        c.hdim = d;
        add("sink_d" + std::to_string(d) + "_wave", sink_case(c), "wave");
    }
    // per-token K/V scales (NaN past each sequence's end) through both layouts, every finishing
    // path, cache page sizes, two tiles, MTP with sinks, head dims
    const auto pt_case = [](Case c, ck_tile::index_t qlen = 1, ck_tile::index_t tiles = 0) {
        c.per_token = true;
        c.qlen      = qlen;
        c.m_tiles   = tiles;
        return c;
    };
    add("pt_one_split", pt_case({2, 16, 1, {700}, 2048, 1}));
    add("pt_coop", pt_case({2, 16, 1, {3000}, 2048}), "coop");
    add("pt_wave", pt_case({2, 16, 1, {5000}, 0, 0, 0, "wave"}), "wave");
    add("pt_coop_groups", pt_case({1, 16, 1, {20000}, 0}), "coop+groups");
    add("pt_wave_groups", pt_case({8, 16, 1, {8192}, 0, 0, 0, "auto", 131072}), "wave+groups");
    add("pt_short", pt_case({3, 16, 1, {1, 5, 77}, 2048}));
    add("pt_mixed", pt_case({4, 16, 1, {9000, 100, 2048, 3333}, 2048}));
    add("pt_p64_wave", pt_case({2, 16, 1, {5000}, 2048, 0, 0, "wave", 0, 64}), "wave");
    add("pt_p128", pt_case({3, 16, 1, {1000}, 300, 0, 0, "auto", 0, 128}));
    add("pt_gqa8_h_k2", pt_case({2, 16, 2, {4000}, 0}));
    add("pt_m2_gqa32", pt_case({2, 32, 1, {5000}, 2048}, 1, 2));
    add("pt_fp16", pt_case({8, 16, 1, {8192}, 0, 0, 0, "auto", 0, 16, true}));
    {
        Case c = pt_case({2, 8, 1, {300, 2}, 5}, 3);
        c.sink = true;
        c.lse  = true;
        add("pt_mtp3_sink_lse", c);
    }
    for(const ck_tile::index_t d : {64, 256})
    {
        Case c{2, 16, 1, {5000}, 0, 0, 0, "wave"};
        c.hdim = d;
        add("pt_d" + std::to_string(d) + "_wave", pt_case(c), "wave");
        c      = Case{2, 16, 1, {3000}, 2048, 0, 0, "coop"};
        c.hdim = d;
        add("pt_d" + std::to_string(d) + "_coop", pt_case(c), "coop");
    }
    // every case again over hnd and nhd caches (token rows in LDS, transposing V reads)
    const size_t n5d = p.size();
    for(const auto& [prefix, layout] : {std::pair{"hnd_", fmha_decode_paged5d::Layout::hnd},
                                        std::pair{"nhd_", fmha_decode_paged5d::Layout::nhd}})
        for(size_t i = 0; i < n5d; ++i)
        {
            Param q       = p[i];
            q.name        = prefix + q.name;
            q.c.kv_layout = layout;
            p.push_back(q);
        }
    return p;
}

class DecodePaged5d : public ::testing::TestWithParam<Param>
{
};

TEST_P(DecodePaged5d, MatchesReference)
{
    if(ck_tile::get_device_name().rfind("gfx125", 0) != 0)
        GTEST_SKIP() << "gfx125 only";
    // one timed launch: also replays a captured graph and compares its output
    const auto r = fmha_decode_paged5d::run(GetParam().c, true, 0, 1);
    ASSERT_TRUE(r.error.empty()) << r.error;
    EXPECT_TRUE(r.graph_same) << "graph replay output differs from a stream launch";
    const std::string layout = std::string(r.config.wave_split ? "wave" : "coop") +
                               (r.config.num_groups > 1 ? "+groups" : "");
    if(*GetParam().layout != '\0')
        EXPECT_EQ(layout, GetParam().layout);
    if(GetParam().m_tiles > 0)
    {
        EXPECT_EQ(r.config.m_tiles, GetParam().m_tiles);
        EXPECT_EQ(r.config.row_blocks, GetParam().row_blocks);
    }
    EXPECT_TRUE(r.valid) << "max abs err " << r.max_err << " (max |ref| " << r.max_ref << "), cos "
                         << r.cosine << ", splits " << r.config.num_splits << ", pages/split "
                         << r.config.pages_per_split << (r.config.wave_split ? ", wave" : ", coop");
}

// The tiles heuristic (host only): one tile per workgroup for small launches, two once the
// (tile, page) tasks or the grid of single tiles are large
TEST(DecodePaged5dConfig, Tiles)
{
    const auto tiles = [](ck_tile::index_t batch,
                          ck_tile::index_t rows,
                          ck_tile::index_t seqlen,
                          ck_tile::index_t window) {
        const auto c =
            ck_tile::fmha_decode_paged5d_config(batch, 1, seqlen, window, 16, 128, rows, 1);
        return std::pair{c.m_tiles, c.row_blocks};
    };
    EXPECT_EQ(tiles(8, 16, 73728, 0), std::pair(1, 1));
    EXPECT_EQ(tiles(8, 32, 8192, 0), std::pair(1, 2));
    EXPECT_EQ(tiles(1, 64, 73728, 0), std::pair(1, 4));
    EXPECT_EQ(tiles(8, 32, 73728, 0), std::pair(2, 1));
    EXPECT_EQ(tiles(32, 64, 8192, 0), std::pair(2, 2));
    EXPECT_EQ(tiles(64, 64, 2048, 0), std::pair(2, 2));
    EXPECT_EQ(tiles(64, 32, 8192, 2048), std::pair(1, 2));
}

INSTANTIATE_TEST_SUITE_P(Gfx125,
                         DecodePaged5d,
                         ::testing::ValuesIn(params()),
                         [](const auto& test_info) { return test_info.param.name; });

} // namespace
