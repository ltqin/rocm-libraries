// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include "ck_tile/core.hpp"
#include "ck_tile/ops/fmha.hpp"
#include "fmha_test_common.hpp"

namespace {

using namespace ck_tile;

// Instantiate both pipelines on the same problem so their names can be
// compared. The geometry is the only one this release supports:
// hdim=128, hdim_v=128, kM0=128, kN0=128, kK0=32, kN1=128, kK1=32.
using TestProblem = ck_tile::test::Model<128, false, false, false, bf16_t, 128, 128>::Problem;
using TestPolicy  = ck_tile::test::Policy<TestProblem>;

using DensePipelineForTest = BlockFmhaPipelineQRKSVSTdmSched<TestProblem, TestPolicy>;
using PagedPipelineForTest = BlockFmhaFwdPagedKVPipelineQRKSVSTdmSched<TestProblem, TestPolicy>;

// If the paged pipeline ever ends up sharing the dense pipeline's name, the
// codegen blobs collide and one silently overwrites the other. Assert the
// property that actually matters -- that the two differ -- rather than
// restating the literal, which would pass no matter what either side is
// renamed to.
TEST(QrTdmSchedPaged, PipelineNameDiffersFromDense)
{
    constexpr const char* paged = PagedPipelineForTest::name;
    constexpr const char* dense = DensePipelineForTest::name;

    ASSERT_NE(paged, nullptr);
    ASSERT_NE(dense, nullptr);
    EXPECT_STRNE(paged, dense)
        << "paged and dense qr_tdm_sched share a name; codegen blobs will collide";
}

} // namespace

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
