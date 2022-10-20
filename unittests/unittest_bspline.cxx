/**
   @file unittests/unittest_bspline.cxx

   @brief B-Spline unittests

   @author Matthias Moller

   @copyright This file is part of the IgaNet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include <iganet.hpp>
#include <filesystem>
#include <iostream>

#include <Sources/Splines/b_spline.hpp>
#include <gtest/gtest.h>

TEST(BSpline, UniformBSpline_parDim1_geoDim1_degrees1_double)
{
  iganet::UniformBSpline<double, 1, 1> bspline({0});
  EXPECT_EQ(bspline.parDim(),   1);
  EXPECT_EQ(bspline.geoDim(),   1);
  EXPECT_EQ(bspline.degree(0),  1);
  EXPECT_EQ(bspline.nknots(0),  2);
  EXPECT_EQ(bspline.ncoeffs(0), 0);
  EXPECT_EQ(bspline.ncoeffs(),  0);
}

TEST(BSpline, UniformBSpline_parDim1_geoDim1_degrees2_double)
{
  EXPECT_THROW( (iganet::UniformBSpline<double, 1, 2>({0})), std::runtime_error);
  iganet::UniformBSpline<double, 1, 2> bspline({1});
  EXPECT_EQ(bspline.parDim(),   1);
  EXPECT_EQ(bspline.geoDim(),   1);
  EXPECT_EQ(bspline.degree(0),  2);
  EXPECT_EQ(bspline.nknots(0),  4);
  EXPECT_EQ(bspline.ncoeffs(0), 1);
  EXPECT_EQ(bspline.ncoeffs(),  1);
}

TEST(BSpline, UniformBSpline_parDim1_geoDim1_degrees3_double)
{
  EXPECT_THROW( (iganet::UniformBSpline<double, 1, 3>({0})), std::runtime_error);
  EXPECT_THROW( (iganet::UniformBSpline<double, 1, 3>({1})), std::runtime_error);
  iganet::UniformBSpline<double, 1, 3> bspline({2});
  EXPECT_EQ(bspline.parDim(),   1);
  EXPECT_EQ(bspline.geoDim(),   1);
  EXPECT_EQ(bspline.degree(0),  3);
  EXPECT_EQ(bspline.nknots(0),  6);
  EXPECT_EQ(bspline.ncoeffs(0), 2);
  EXPECT_EQ(bspline.ncoeffs(),  2);
}

TEST(BSpline, UniformBSpline_parDim1_geoDim2_degrees4_double)
{
  EXPECT_THROW( (iganet::UniformBSpline<double, 2, 4>({0})), std::runtime_error);
  EXPECT_THROW( (iganet::UniformBSpline<double, 2, 4>({1})), std::runtime_error);
  EXPECT_THROW( (iganet::UniformBSpline<double, 2, 4>({2})), std::runtime_error);
  iganet::UniformBSpline<double, 2, 4> bspline({3});
  EXPECT_EQ(bspline.parDim(),   1);
  EXPECT_EQ(bspline.geoDim(),   2);
  EXPECT_EQ(bspline.degree(0),  4);
  EXPECT_EQ(bspline.nknots(0),  8);
  EXPECT_EQ(bspline.ncoeffs(0), 3);
  EXPECT_EQ(bspline.ncoeffs(),  3);
}

TEST(BSpline, UniformBSpline_parDim1_geoDim3_degrees5_double)
{
  EXPECT_THROW( (iganet::UniformBSpline<double, 3, 5>({0})), std::runtime_error);
  EXPECT_THROW( (iganet::UniformBSpline<double, 3, 5>({1})), std::runtime_error);
  EXPECT_THROW( (iganet::UniformBSpline<double, 3, 5>({2})), std::runtime_error);
  EXPECT_THROW( (iganet::UniformBSpline<double, 3, 5>({3})), std::runtime_error);
  iganet::UniformBSpline<double, 3, 5> bspline({4});
  EXPECT_EQ(bspline.parDim(),   1);
  EXPECT_EQ(bspline.geoDim(),   3);
  EXPECT_EQ(bspline.degree(0),  5);
  EXPECT_EQ(bspline.nknots(0), 10);
  EXPECT_EQ(bspline.ncoeffs(0), 4);
  EXPECT_EQ(bspline.ncoeffs(),  4);
}

TEST(BSpline, UniformBSpline_parDim1_geoDim4_degrees6_double)
{
  EXPECT_THROW( (iganet::UniformBSpline<double, 4, 6>({0})), std::runtime_error);
  EXPECT_THROW( (iganet::UniformBSpline<double, 4, 6>({1})), std::runtime_error);
  EXPECT_THROW( (iganet::UniformBSpline<double, 4, 6>({2})), std::runtime_error);
  EXPECT_THROW( (iganet::UniformBSpline<double, 4, 6>({3})), std::runtime_error);
  EXPECT_THROW( (iganet::UniformBSpline<double, 4, 6>({4})), std::runtime_error);
  iganet::UniformBSpline<double, 4, 6> bspline({5});
  EXPECT_EQ(bspline.parDim(),   1);
  EXPECT_EQ(bspline.geoDim(),   4);
  EXPECT_EQ(bspline.degree(0),  6);
  EXPECT_EQ(bspline.nknots(0), 12);
  EXPECT_EQ(bspline.ncoeffs(0), 5);
  EXPECT_EQ(bspline.ncoeffs(),  5);
}

TEST(BSpline, UniformBSpline_parDim2_geoDim1_degrees34_double)
{
  EXPECT_THROW( (iganet::UniformBSpline<double, 1, 3, 4>({0, 0})), std::runtime_error);
  EXPECT_THROW( (iganet::UniformBSpline<double, 1, 3, 4>({1, 0})), std::runtime_error);
  EXPECT_THROW( (iganet::UniformBSpline<double, 1, 3, 4>({0, 1})), std::runtime_error);
  EXPECT_THROW( (iganet::UniformBSpline<double, 1, 3, 4>({1, 1})), std::runtime_error);
  EXPECT_THROW( (iganet::UniformBSpline<double, 1, 3, 4>({2, 1})), std::runtime_error);
  EXPECT_THROW( (iganet::UniformBSpline<double, 1, 3, 4>({1, 2})), std::runtime_error);
  EXPECT_THROW( (iganet::UniformBSpline<double, 1, 3, 4>({2, 2})), std::runtime_error);
  iganet::UniformBSpline<double, 1, 3, 4> bspline({2,3});
  EXPECT_EQ(bspline.parDim(),   2);
  EXPECT_EQ(bspline.geoDim(),   1);
  EXPECT_EQ(bspline.degree(0),  3);
  EXPECT_EQ(bspline.degree(1),  4);
  EXPECT_EQ(bspline.nknots(0),  6);
  EXPECT_EQ(bspline.nknots(1),  8);
  EXPECT_EQ(bspline.ncoeffs(0), 2);
  EXPECT_EQ(bspline.ncoeffs(1), 3);
  EXPECT_EQ(bspline.ncoeffs(),  6);
}

TEST(BSpline, UniformBSpline_parDim2_geoDim2_degrees34_double)
{
  EXPECT_THROW( (iganet::UniformBSpline<double, 2, 3, 4>({0, 0})), std::runtime_error);
  EXPECT_THROW( (iganet::UniformBSpline<double, 2, 3, 4>({1, 0})), std::runtime_error);
  EXPECT_THROW( (iganet::UniformBSpline<double, 2, 3, 4>({0, 1})), std::runtime_error);
  EXPECT_THROW( (iganet::UniformBSpline<double, 2, 3, 4>({1, 1})), std::runtime_error);
  EXPECT_THROW( (iganet::UniformBSpline<double, 2, 3, 4>({2, 1})), std::runtime_error);
  EXPECT_THROW( (iganet::UniformBSpline<double, 2, 3, 4>({1, 2})), std::runtime_error);
  EXPECT_THROW( (iganet::UniformBSpline<double, 2, 3, 4>({2, 2})), std::runtime_error);
  iganet::UniformBSpline<double, 2, 3, 4> bspline({2,3});
  EXPECT_EQ(bspline.parDim(),   2);
  EXPECT_EQ(bspline.geoDim(),   2);
  EXPECT_EQ(bspline.degree(0),  3);
  EXPECT_EQ(bspline.degree(1),  4);
  EXPECT_EQ(bspline.nknots(0),  6);
  EXPECT_EQ(bspline.nknots(1),  8);
  EXPECT_EQ(bspline.ncoeffs(0), 2);
  EXPECT_EQ(bspline.ncoeffs(1), 3);
  EXPECT_EQ(bspline.ncoeffs(),  6);
}

TEST(BSpline, UniformBSpline_parDim2_geoDim3_degrees34_double)
{
  EXPECT_THROW( (iganet::UniformBSpline<double, 3, 3, 4>({0, 0})), std::runtime_error);
  EXPECT_THROW( (iganet::UniformBSpline<double, 3, 3, 4>({1, 0})), std::runtime_error);
  EXPECT_THROW( (iganet::UniformBSpline<double, 3, 3, 4>({0, 1})), std::runtime_error);
  EXPECT_THROW( (iganet::UniformBSpline<double, 3, 3, 4>({1, 1})), std::runtime_error);
  EXPECT_THROW( (iganet::UniformBSpline<double, 3, 3, 4>({2, 1})), std::runtime_error);
  EXPECT_THROW( (iganet::UniformBSpline<double, 3, 3, 4>({1, 2})), std::runtime_error);
  EXPECT_THROW( (iganet::UniformBSpline<double, 3, 3, 4>({2, 2})), std::runtime_error);
  iganet::UniformBSpline<double, 3, 3, 4> bspline({2,3});
  EXPECT_EQ(bspline.parDim(),   2);
  EXPECT_EQ(bspline.geoDim(),   3);
  EXPECT_EQ(bspline.degree(0),  3);
  EXPECT_EQ(bspline.degree(1),  4);
  EXPECT_EQ(bspline.nknots(0),  6);
  EXPECT_EQ(bspline.nknots(1),  8);
  EXPECT_EQ(bspline.ncoeffs(0), 2);
  EXPECT_EQ(bspline.ncoeffs(1), 3);
  EXPECT_EQ(bspline.ncoeffs(),  6);
}

TEST(BSpline, UniformBSpline_parDim2_geoDim4_degrees34_double)
{
  EXPECT_THROW( (iganet::UniformBSpline<double, 4, 3, 4>({0, 0})), std::runtime_error);
  EXPECT_THROW( (iganet::UniformBSpline<double, 4, 3, 4>({1, 0})), std::runtime_error);
  EXPECT_THROW( (iganet::UniformBSpline<double, 4, 3, 4>({0, 1})), std::runtime_error);
  EXPECT_THROW( (iganet::UniformBSpline<double, 4, 3, 4>({1, 1})), std::runtime_error);
  EXPECT_THROW( (iganet::UniformBSpline<double, 4, 3, 4>({2, 1})), std::runtime_error);
  EXPECT_THROW( (iganet::UniformBSpline<double, 4, 3, 4>({1, 2})), std::runtime_error);
  EXPECT_THROW( (iganet::UniformBSpline<double, 4, 3, 4>({2, 2})), std::runtime_error);
  iganet::UniformBSpline<double, 4, 3, 4> bspline({2,3});
  EXPECT_EQ(bspline.parDim(),   2);
  EXPECT_EQ(bspline.geoDim(),   4);
  EXPECT_EQ(bspline.degree(0),  3);
  EXPECT_EQ(bspline.degree(1),  4);
  EXPECT_EQ(bspline.nknots(0),  6);
  EXPECT_EQ(bspline.nknots(1),  8);
  EXPECT_EQ(bspline.ncoeffs(0), 2);
  EXPECT_EQ(bspline.ncoeffs(1), 3);
  EXPECT_EQ(bspline.ncoeffs(),  6);
}

TEST(BSpline, UniformBSpline_parDim3_geoDim1_degrees342_double)
{
  EXPECT_THROW( (iganet::UniformBSpline<double, 1, 3, 4, 2>({0, 0, 0})), std::runtime_error);
  EXPECT_THROW( (iganet::UniformBSpline<double, 1, 3, 4, 2>({1, 0, 0})), std::runtime_error);
  EXPECT_THROW( (iganet::UniformBSpline<double, 1, 3, 4, 2>({0, 1, 0})), std::runtime_error);
  EXPECT_THROW( (iganet::UniformBSpline<double, 1, 3, 4, 2>({1, 1, 0})), std::runtime_error);
  EXPECT_THROW( (iganet::UniformBSpline<double, 1, 3, 4, 2>({2, 1, 0})), std::runtime_error);
  EXPECT_THROW( (iganet::UniformBSpline<double, 1, 3, 4, 2>({1, 2, 0})), std::runtime_error);
  EXPECT_THROW( (iganet::UniformBSpline<double, 1, 3, 4, 2>({2, 2, 0})), std::runtime_error);
  iganet::UniformBSpline<double, 1, 3, 4, 2> bspline({2,3,4});
  EXPECT_EQ(bspline.parDim(),   3);
  EXPECT_EQ(bspline.geoDim(),   1);
  EXPECT_EQ(bspline.degree(0),  3);
  EXPECT_EQ(bspline.degree(1),  4);
  EXPECT_EQ(bspline.degree(2),  2);
  EXPECT_EQ(bspline.nknots(0),  6);
  EXPECT_EQ(bspline.nknots(1),  8);
  EXPECT_EQ(bspline.nknots(2),  7);
  EXPECT_EQ(bspline.ncoeffs(0), 2);
  EXPECT_EQ(bspline.ncoeffs(1), 3);
  EXPECT_EQ(bspline.ncoeffs(2), 4);
  EXPECT_EQ(bspline.ncoeffs(), 24);
}

TEST(BSpline, UniformBSpline_parDim2_geoDim3_degrees342_double)
{
  EXPECT_THROW( (iganet::UniformBSpline<double, 2, 3, 4, 2>({0, 0, 0})), std::runtime_error);
  EXPECT_THROW( (iganet::UniformBSpline<double, 2, 3, 4, 2>({1, 0, 0})), std::runtime_error);
  EXPECT_THROW( (iganet::UniformBSpline<double, 2, 3, 4, 2>({0, 1, 0})), std::runtime_error);
  EXPECT_THROW( (iganet::UniformBSpline<double, 2, 3, 4, 2>({1, 1, 0})), std::runtime_error);
  EXPECT_THROW( (iganet::UniformBSpline<double, 2, 3, 4, 2>({2, 1, 0})), std::runtime_error);
  EXPECT_THROW( (iganet::UniformBSpline<double, 2, 3, 4, 2>({1, 2, 0})), std::runtime_error);
  EXPECT_THROW( (iganet::UniformBSpline<double, 2, 3, 4, 2>({2, 2, 0})), std::runtime_error);
  iganet::UniformBSpline<double, 2, 3, 4, 2> bspline({2,3,4});
  EXPECT_EQ(bspline.parDim(),   3);
  EXPECT_EQ(bspline.geoDim(),   2);
  EXPECT_EQ(bspline.degree(0),  3);
  EXPECT_EQ(bspline.degree(1),  4);
  EXPECT_EQ(bspline.degree(2),  2);
  EXPECT_EQ(bspline.nknots(0),  6);
  EXPECT_EQ(bspline.nknots(1),  8);
  EXPECT_EQ(bspline.nknots(2),  7);
  EXPECT_EQ(bspline.ncoeffs(0), 2);
  EXPECT_EQ(bspline.ncoeffs(1), 3);
  EXPECT_EQ(bspline.ncoeffs(2), 4);
  EXPECT_EQ(bspline.ncoeffs(), 24);
}

TEST(BSpline, UniformBSpline_parDim3_geoDim3_degrees342_double)
{
  EXPECT_THROW( (iganet::UniformBSpline<double, 3, 3, 4, 2>({0, 0, 0})), std::runtime_error);
  EXPECT_THROW( (iganet::UniformBSpline<double, 3, 3, 4, 2>({1, 0, 0})), std::runtime_error);
  EXPECT_THROW( (iganet::UniformBSpline<double, 3, 3, 4, 2>({0, 1, 0})), std::runtime_error);
  EXPECT_THROW( (iganet::UniformBSpline<double, 3, 3, 4, 2>({1, 1, 0})), std::runtime_error);
  EXPECT_THROW( (iganet::UniformBSpline<double, 3, 3, 4, 2>({2, 1, 0})), std::runtime_error);
  EXPECT_THROW( (iganet::UniformBSpline<double, 3, 3, 4, 2>({1, 2, 0})), std::runtime_error);
  EXPECT_THROW( (iganet::UniformBSpline<double, 3, 3, 4, 2>({2, 2, 0})), std::runtime_error);
  iganet::UniformBSpline<double, 3, 3, 4, 2> bspline({2,3,4});
  EXPECT_EQ(bspline.parDim(),   3);
  EXPECT_EQ(bspline.geoDim(),   3);
  EXPECT_EQ(bspline.degree(0),  3);
  EXPECT_EQ(bspline.degree(1),  4);
  EXPECT_EQ(bspline.degree(2),  2);
  EXPECT_EQ(bspline.nknots(0),  6);
  EXPECT_EQ(bspline.nknots(1),  8);
  EXPECT_EQ(bspline.nknots(2),  7);
  EXPECT_EQ(bspline.ncoeffs(0), 2);
  EXPECT_EQ(bspline.ncoeffs(1), 3);
  EXPECT_EQ(bspline.ncoeffs(2), 4);
  EXPECT_EQ(bspline.ncoeffs(), 24);
}

TEST(BSpline, UniformBSpline_parDim3_geoDim4_degrees342_double)
{
  EXPECT_THROW( (iganet::UniformBSpline<double, 4, 3, 4, 2>({0, 0, 0})), std::runtime_error);
  EXPECT_THROW( (iganet::UniformBSpline<double, 4, 3, 4, 2>({1, 0, 0})), std::runtime_error);
  EXPECT_THROW( (iganet::UniformBSpline<double, 4, 3, 4, 2>({0, 1, 0})), std::runtime_error);
  EXPECT_THROW( (iganet::UniformBSpline<double, 4, 3, 4, 2>({1, 1, 0})), std::runtime_error);
  EXPECT_THROW( (iganet::UniformBSpline<double, 4, 3, 4, 2>({2, 1, 0})), std::runtime_error);
  EXPECT_THROW( (iganet::UniformBSpline<double, 4, 3, 4, 2>({1, 2, 0})), std::runtime_error);
  EXPECT_THROW( (iganet::UniformBSpline<double, 4, 3, 4, 2>({2, 2, 0})), std::runtime_error);
  iganet::UniformBSpline<double, 4, 3, 4, 2> bspline({2,3,4});
  EXPECT_EQ(bspline.parDim(),   3);
  EXPECT_EQ(bspline.geoDim(),   4);
  EXPECT_EQ(bspline.degree(0),  3);
  EXPECT_EQ(bspline.degree(1),  4);
  EXPECT_EQ(bspline.degree(2),  2);
  EXPECT_EQ(bspline.nknots(0),  6);
  EXPECT_EQ(bspline.nknots(1),  8);
  EXPECT_EQ(bspline.nknots(2),  7);
  EXPECT_EQ(bspline.ncoeffs(0), 2);
  EXPECT_EQ(bspline.ncoeffs(1), 3);
  EXPECT_EQ(bspline.ncoeffs(2), 4);
  EXPECT_EQ(bspline.ncoeffs(), 24);
}

TEST(BSpline, UniformBSpline_parDim4_geoDim1_degrees3421_double)
{
  EXPECT_THROW( (iganet::UniformBSpline<double, 1, 3, 4, 2, 1>({0, 0, 0, 0})), std::runtime_error);
  EXPECT_THROW( (iganet::UniformBSpline<double, 1, 3, 4, 2, 1>({1, 0, 0, 0})), std::runtime_error);
  EXPECT_THROW( (iganet::UniformBSpline<double, 1, 3, 4, 2, 1>({0, 1, 0, 0})), std::runtime_error);
  EXPECT_THROW( (iganet::UniformBSpline<double, 1, 3, 4, 2, 1>({1, 1, 0, 0})), std::runtime_error);
  EXPECT_THROW( (iganet::UniformBSpline<double, 1, 3, 4, 2, 1>({2, 1, 0, 0})), std::runtime_error);
  EXPECT_THROW( (iganet::UniformBSpline<double, 1, 3, 4, 2, 1>({1, 2, 0, 0})), std::runtime_error);
  EXPECT_THROW( (iganet::UniformBSpline<double, 1, 3, 4, 2, 1>({2, 2, 0, 0})), std::runtime_error);
  iganet::UniformBSpline<double, 1, 3, 4, 2, 1> bspline({2,3,4,5});
  EXPECT_EQ(bspline.parDim(),   4);
  EXPECT_EQ(bspline.geoDim(),   1);
  EXPECT_EQ(bspline.degree(0),  3);
  EXPECT_EQ(bspline.degree(1),  4);
  EXPECT_EQ(bspline.degree(2),  2);
  EXPECT_EQ(bspline.degree(3),  1);
  EXPECT_EQ(bspline.nknots(0),  6);
  EXPECT_EQ(bspline.nknots(1),  8);
  EXPECT_EQ(bspline.nknots(2),  7);
  EXPECT_EQ(bspline.nknots(3),  7);
  EXPECT_EQ(bspline.ncoeffs(0), 2);
  EXPECT_EQ(bspline.ncoeffs(1), 3);
  EXPECT_EQ(bspline.ncoeffs(2), 4);
  EXPECT_EQ(bspline.ncoeffs(3), 5);
  EXPECT_EQ(bspline.ncoeffs(),120);
}

TEST(BSpline, UniformBSpline_parDim4_geoDim2_degrees3421_double)
{
  EXPECT_THROW( (iganet::UniformBSpline<double, 2, 3, 4, 2, 1>({0, 0, 0, 0})), std::runtime_error);
  EXPECT_THROW( (iganet::UniformBSpline<double, 2, 3, 4, 2, 1>({1, 0, 0, 0})), std::runtime_error);
  EXPECT_THROW( (iganet::UniformBSpline<double, 2, 3, 4, 2, 1>({0, 1, 0, 0})), std::runtime_error);
  EXPECT_THROW( (iganet::UniformBSpline<double, 2, 3, 4, 2, 1>({1, 1, 0, 0})), std::runtime_error);
  EXPECT_THROW( (iganet::UniformBSpline<double, 2, 3, 4, 2, 1>({2, 1, 0, 0})), std::runtime_error);
  EXPECT_THROW( (iganet::UniformBSpline<double, 2, 3, 4, 2, 1>({1, 2, 0, 0})), std::runtime_error);
  EXPECT_THROW( (iganet::UniformBSpline<double, 2, 3, 4, 2, 1>({2, 2, 0, 0})), std::runtime_error);
  iganet::UniformBSpline<double, 2, 3, 4, 2, 1> bspline({2,3,4,5});
  EXPECT_EQ(bspline.parDim(),   4);
  EXPECT_EQ(bspline.geoDim(),   2);
  EXPECT_EQ(bspline.degree(0),  3);
  EXPECT_EQ(bspline.degree(1),  4);
  EXPECT_EQ(bspline.degree(2),  2);
  EXPECT_EQ(bspline.degree(3),  1);
  EXPECT_EQ(bspline.nknots(0),  6);
  EXPECT_EQ(bspline.nknots(1),  8);
  EXPECT_EQ(bspline.nknots(2),  7);
  EXPECT_EQ(bspline.nknots(3),  7);
  EXPECT_EQ(bspline.ncoeffs(0), 2);
  EXPECT_EQ(bspline.ncoeffs(1), 3);
  EXPECT_EQ(bspline.ncoeffs(2), 4);
  EXPECT_EQ(bspline.ncoeffs(3), 5);
  EXPECT_EQ(bspline.ncoeffs(),120);
}

TEST(BSpline, UniformBSpline_parDim4_geoDim3_degrees3421_double)
{
  EXPECT_THROW( (iganet::UniformBSpline<double, 3, 3, 4, 2, 1>({0, 0, 0, 0})), std::runtime_error);
  EXPECT_THROW( (iganet::UniformBSpline<double, 3, 3, 4, 2, 1>({1, 0, 0, 0})), std::runtime_error);
  EXPECT_THROW( (iganet::UniformBSpline<double, 3, 3, 4, 2, 1>({0, 1, 0, 0})), std::runtime_error);
  EXPECT_THROW( (iganet::UniformBSpline<double, 3, 3, 4, 2, 1>({1, 1, 0, 0})), std::runtime_error);
  EXPECT_THROW( (iganet::UniformBSpline<double, 3, 3, 4, 2, 1>({2, 1, 0, 0})), std::runtime_error);
  EXPECT_THROW( (iganet::UniformBSpline<double, 3, 3, 4, 2, 1>({1, 2, 0, 0})), std::runtime_error);
  EXPECT_THROW( (iganet::UniformBSpline<double, 3, 3, 4, 2, 1>({2, 2, 0, 0})), std::runtime_error);
  iganet::UniformBSpline<double, 3, 3, 4, 2, 1> bspline({2,3,4,5});
  EXPECT_EQ(bspline.parDim(),   4);
  EXPECT_EQ(bspline.geoDim(),   3);
  EXPECT_EQ(bspline.degree(0),  3);
  EXPECT_EQ(bspline.degree(1),  4);
  EXPECT_EQ(bspline.degree(2),  2);
  EXPECT_EQ(bspline.degree(3),  1);
  EXPECT_EQ(bspline.nknots(0),  6);
  EXPECT_EQ(bspline.nknots(1),  8);
  EXPECT_EQ(bspline.nknots(2),  7);
  EXPECT_EQ(bspline.nknots(3),  7);
  EXPECT_EQ(bspline.ncoeffs(0), 2);
  EXPECT_EQ(bspline.ncoeffs(1), 3);
  EXPECT_EQ(bspline.ncoeffs(2), 4);
  EXPECT_EQ(bspline.ncoeffs(3), 5);
  EXPECT_EQ(bspline.ncoeffs(),120);
}

TEST(BSpline, UniformBSpline_parDim4_geoDim4_degrees3421_double)
{
  EXPECT_THROW( (iganet::UniformBSpline<double, 4, 3, 4, 2, 1>({0, 0, 0, 0})), std::runtime_error);
  EXPECT_THROW( (iganet::UniformBSpline<double, 4, 3, 4, 2, 1>({1, 0, 0, 0})), std::runtime_error);
  EXPECT_THROW( (iganet::UniformBSpline<double, 4, 3, 4, 2, 1>({0, 1, 0, 0})), std::runtime_error);
  EXPECT_THROW( (iganet::UniformBSpline<double, 4, 3, 4, 2, 1>({1, 1, 0, 0})), std::runtime_error);
  EXPECT_THROW( (iganet::UniformBSpline<double, 4, 3, 4, 2, 1>({2, 1, 0, 0})), std::runtime_error);
  EXPECT_THROW( (iganet::UniformBSpline<double, 4, 3, 4, 2, 1>({1, 2, 0, 0})), std::runtime_error);
  EXPECT_THROW( (iganet::UniformBSpline<double, 4, 3, 4, 2, 1>({2, 2, 0, 0})), std::runtime_error);
  iganet::UniformBSpline<double, 4, 3, 4, 2, 1> bspline({2,3,4,5});
  EXPECT_EQ(bspline.parDim(),   4);
  EXPECT_EQ(bspline.geoDim(),   4);
  EXPECT_EQ(bspline.degree(0),  3);
  EXPECT_EQ(bspline.degree(1),  4);
  EXPECT_EQ(bspline.degree(2),  2);
  EXPECT_EQ(bspline.degree(3),  1);
  EXPECT_EQ(bspline.nknots(0),  6);
  EXPECT_EQ(bspline.nknots(1),  8);
  EXPECT_EQ(bspline.nknots(2),  7);
  EXPECT_EQ(bspline.nknots(3),  7);
  EXPECT_EQ(bspline.ncoeffs(0), 2);
  EXPECT_EQ(bspline.ncoeffs(1), 3);
  EXPECT_EQ(bspline.ncoeffs(2), 4);
  EXPECT_EQ(bspline.ncoeffs(3), 5);
  EXPECT_EQ(bspline.ncoeffs(),120);
}

TEST(BSpline, NonUniformBSpline_parDim1_geoDim1_degrees1_double)
{
  EXPECT_THROW( (iganet::NonUniformBSpline<double, 1, 1>( {{{0.0, 0.0, 1.0}}} )), std::runtime_error);
  iganet::NonUniformBSpline<double, 1, 1> bspline( {{{0.0, 0.0, 0.5, 1.0, 1.0}}} );
  EXPECT_EQ(bspline.parDim(),   1);
  EXPECT_EQ(bspline.geoDim(),   1);
  EXPECT_EQ(bspline.degree(0),  1);
  EXPECT_EQ(bspline.nknots(0),  5);
  EXPECT_EQ(bspline.ncoeffs(0), 3);
  EXPECT_EQ(bspline.ncoeffs(),  3);
}

TEST(BSpline, NonUniformBSpline_parDim1_geoDim2_degrees1_double)
{
  EXPECT_THROW( (iganet::NonUniformBSpline<double, 2, 1>( {{{0.0, 0.0, 1.0}}} )), std::runtime_error);
  iganet::NonUniformBSpline<double, 2, 1> bspline( {{{0.0, 0.0, 0.5, 1.0, 1.0}}} );
  EXPECT_EQ(bspline.parDim(),   1);
  EXPECT_EQ(bspline.geoDim(),   2);
  EXPECT_EQ(bspline.degree(0),  1);
  EXPECT_EQ(bspline.nknots(0),  5);
  EXPECT_EQ(bspline.ncoeffs(0), 3);
  EXPECT_EQ(bspline.ncoeffs(),  3);
}

TEST(BSpline, NonUniformBSpline_parDim1_geoDim3_degrees1_double)
{
  EXPECT_THROW( (iganet::NonUniformBSpline<double, 3, 1>( {{{0.0, 0.0, 1.0}}} )), std::runtime_error);
  iganet::NonUniformBSpline<double, 3, 1> bspline( {{{0.0, 0.0, 0.5, 1.0, 1.0}}} );
  EXPECT_EQ(bspline.parDim(),   1);
  EXPECT_EQ(bspline.geoDim(),   3);
  EXPECT_EQ(bspline.degree(0),  1);
  EXPECT_EQ(bspline.nknots(0),  5);
  EXPECT_EQ(bspline.ncoeffs(0), 3);
  EXPECT_EQ(bspline.ncoeffs(),  3);
}

TEST(BSpline, NonUniformBSpline_parDim1_geoDim4_degrees1_double)
{
  EXPECT_THROW( (iganet::NonUniformBSpline<double, 4, 1>( {{{0.0, 0.0, 1.0}}} )), std::runtime_error);
  iganet::NonUniformBSpline<double, 4, 1> bspline( {{{0.0, 0.0, 0.5, 1.0, 1.0}}} );
  EXPECT_EQ(bspline.parDim(),   1);
  EXPECT_EQ(bspline.geoDim(),   4);
  EXPECT_EQ(bspline.degree(0),  1);
  EXPECT_EQ(bspline.nknots(0),  5);
  EXPECT_EQ(bspline.ncoeffs(0), 3);
  EXPECT_EQ(bspline.ncoeffs(),  3);
}

TEST(BSpline, NonUniformBSpline_parDim2_geoDim1_degrees12_double)
{
  iganet::NonUniformBSpline<double, 1, 1, 2> bspline( {{{0.0, 0.0, 0.5, 1.0, 1.0},
                                                        {0.0, 0.0, 0.0, 1.0, 1.0, 1.0}}} );
  EXPECT_EQ(bspline.parDim(),   2);
  EXPECT_EQ(bspline.geoDim(),   1);
  EXPECT_EQ(bspline.degree(0),  1);
  EXPECT_EQ(bspline.degree(1),  2);
  EXPECT_EQ(bspline.nknots(0),  5);
  EXPECT_EQ(bspline.nknots(1),  6);
  EXPECT_EQ(bspline.ncoeffs(0), 3);
  EXPECT_EQ(bspline.ncoeffs(1), 3);
  EXPECT_EQ(bspline.ncoeffs(),  9);
}

TEST(BSpline, NonUniformBSpline_parDim2_geoDim2_degrees12_double)
{
  iganet::NonUniformBSpline<double, 2, 1, 2> bspline( {{{0.0, 0.0, 0.5, 1.0, 1.0},
                                                        {0.0, 0.0, 0.0, 1.0, 1.0, 1.0}}} );
  EXPECT_EQ(bspline.parDim(),   2);
  EXPECT_EQ(bspline.geoDim(),   2);
  EXPECT_EQ(bspline.degree(0),  1);
  EXPECT_EQ(bspline.degree(1),  2);
  EXPECT_EQ(bspline.nknots(0),  5);
  EXPECT_EQ(bspline.nknots(1),  6);
  EXPECT_EQ(bspline.ncoeffs(0), 3);
  EXPECT_EQ(bspline.ncoeffs(1), 3);
  EXPECT_EQ(bspline.ncoeffs(),  9);
}

TEST(BSpline, NonUniformBSpline_parDim2_geoDim3_degrees12_double)
{
  iganet::NonUniformBSpline<double, 3, 1, 2> bspline( {{{0.0, 0.0, 0.5, 1.0, 1.0},
                                                        {0.0, 0.0, 0.0, 1.0, 1.0, 1.0}}} );
  EXPECT_EQ(bspline.parDim(),   2);
  EXPECT_EQ(bspline.geoDim(),   3);
  EXPECT_EQ(bspline.degree(0),  1);
  EXPECT_EQ(bspline.degree(1),  2);
  EXPECT_EQ(bspline.nknots(0),  5);
  EXPECT_EQ(bspline.nknots(1),  6);
  EXPECT_EQ(bspline.ncoeffs(0), 3);
  EXPECT_EQ(bspline.ncoeffs(1), 3);
  EXPECT_EQ(bspline.ncoeffs(),  9);
}

TEST(BSpline, NonUniformBSpline_parDim2_geoDim4_degrees12_double)
{
  iganet::NonUniformBSpline<double, 4, 1, 2> bspline( {{{0.0, 0.0, 0.5, 1.0, 1.0},
                                                        {0.0, 0.0, 0.0, 1.0, 1.0, 1.0}}} );
  EXPECT_EQ(bspline.parDim(),   2);
  EXPECT_EQ(bspline.geoDim(),   4);
  EXPECT_EQ(bspline.degree(0),  1);
  EXPECT_EQ(bspline.degree(1),  2);
  EXPECT_EQ(bspline.nknots(0),  5);
  EXPECT_EQ(bspline.nknots(1),  6);
  EXPECT_EQ(bspline.ncoeffs(0), 3);
  EXPECT_EQ(bspline.ncoeffs(1), 3);
  EXPECT_EQ(bspline.ncoeffs(),  9);
}

TEST(BSpline, NonUniformBSpline_parDim3_geoDim1_degrees123_double)
{
  iganet::NonUniformBSpline<double, 1, 1, 2, 3> bspline( {{{0.0, 0.0, 0.5, 1.0, 1.0},
                                                           {0.0, 0.0, 0.0, 1.0, 1.0, 1.0},
                                                           {0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0}}} );
  EXPECT_EQ(bspline.parDim(),   3);
  EXPECT_EQ(bspline.geoDim(),   1);
  EXPECT_EQ(bspline.degree(0),  1);
  EXPECT_EQ(bspline.degree(1),  2);
  EXPECT_EQ(bspline.degree(2),  3);
  EXPECT_EQ(bspline.nknots(0),  5);
  EXPECT_EQ(bspline.nknots(1),  6);
  EXPECT_EQ(bspline.nknots(2),  9);
  EXPECT_EQ(bspline.ncoeffs(0), 3);
  EXPECT_EQ(bspline.ncoeffs(1), 3);
  EXPECT_EQ(bspline.ncoeffs(2), 5);
  EXPECT_EQ(bspline.ncoeffs(), 45);
}

TEST(BSpline, NonUniformBSpline_parDim3_geoDim2_degrees123_double)
{
  iganet::NonUniformBSpline<double, 2, 1, 2, 3> bspline( {{{0.0, 0.0, 0.5, 1.0, 1.0},
                                                           {0.0, 0.0, 0.0, 1.0, 1.0, 1.0},
                                                           {0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0}}} );
  EXPECT_EQ(bspline.parDim(),   3);
  EXPECT_EQ(bspline.geoDim(),   2);
  EXPECT_EQ(bspline.degree(0),  1);
  EXPECT_EQ(bspline.degree(1),  2);
  EXPECT_EQ(bspline.degree(2),  3);
  EXPECT_EQ(bspline.nknots(0),  5);
  EXPECT_EQ(bspline.nknots(1),  6);
  EXPECT_EQ(bspline.nknots(2),  9);
  EXPECT_EQ(bspline.ncoeffs(0), 3);
  EXPECT_EQ(bspline.ncoeffs(1), 3);
  EXPECT_EQ(bspline.ncoeffs(2), 5);
  EXPECT_EQ(bspline.ncoeffs(), 45);
}

TEST(BSpline, NonUniformBSpline_parDim3_geoDim3_degrees123_double)
{
  iganet::NonUniformBSpline<double, 3, 1, 2, 3> bspline( {{{0.0, 0.0, 0.5, 1.0, 1.0},
                                                           {0.0, 0.0, 0.0, 1.0, 1.0, 1.0},
                                                           {0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0}}} );
  EXPECT_EQ(bspline.parDim(),   3);
  EXPECT_EQ(bspline.geoDim(),   3);
  EXPECT_EQ(bspline.degree(0),  1);
  EXPECT_EQ(bspline.degree(1),  2);
  EXPECT_EQ(bspline.degree(2),  3);
  EXPECT_EQ(bspline.nknots(0),  5);
  EXPECT_EQ(bspline.nknots(1),  6);
  EXPECT_EQ(bspline.nknots(2),  9);
  EXPECT_EQ(bspline.ncoeffs(0), 3);
  EXPECT_EQ(bspline.ncoeffs(1), 3);
  EXPECT_EQ(bspline.ncoeffs(2), 5);
  EXPECT_EQ(bspline.ncoeffs(), 45);
}

TEST(BSpline, NonUniformBSpline_parDim3_geoDim4_degrees123_double)
{
  iganet::NonUniformBSpline<double, 4, 1, 2, 3> bspline( {{{0.0, 0.0, 0.5, 1.0, 1.0},
                                                           {0.0, 0.0, 0.0, 1.0, 1.0, 1.0},
                                                           {0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0}}} );
  EXPECT_EQ(bspline.parDim(),   3);
  EXPECT_EQ(bspline.geoDim(),   4);
  EXPECT_EQ(bspline.degree(0),  1);
  EXPECT_EQ(bspline.degree(1),  2);
  EXPECT_EQ(bspline.degree(2),  3);
  EXPECT_EQ(bspline.nknots(0),  5);
  EXPECT_EQ(bspline.nknots(1),  6);
  EXPECT_EQ(bspline.nknots(2),  9);
  EXPECT_EQ(bspline.ncoeffs(0), 3);
  EXPECT_EQ(bspline.ncoeffs(1), 3);
  EXPECT_EQ(bspline.ncoeffs(2), 5);
  EXPECT_EQ(bspline.ncoeffs(), 45);
}

TEST(BSpline, NonUniformBSpline_parDim4_geoDim1_degrees1234_double)
{
  iganet::NonUniformBSpline<double, 1, 1, 2, 3, 4> bspline( {{{0.0, 0.0, 0.5, 1.0, 1.0},
                                                              {0.0, 0.0, 0.0, 1.0, 1.0, 1.0},
                                                              {0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0},
                                                              {0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0}}} );
  EXPECT_EQ(bspline.parDim(),   4);
  EXPECT_EQ(bspline.geoDim(),   1);
  EXPECT_EQ(bspline.degree(0),  1);
  EXPECT_EQ(bspline.degree(1),  2);
  EXPECT_EQ(bspline.degree(2),  3);
  EXPECT_EQ(bspline.degree(3),  4);
  EXPECT_EQ(bspline.nknots(0),  5);
  EXPECT_EQ(bspline.nknots(1),  6);
  EXPECT_EQ(bspline.nknots(2),  9);
  EXPECT_EQ(bspline.nknots(3), 11);
  EXPECT_EQ(bspline.ncoeffs(0), 3);
  EXPECT_EQ(bspline.ncoeffs(1), 3);
  EXPECT_EQ(bspline.ncoeffs(2), 5);
  EXPECT_EQ(bspline.ncoeffs(3), 6);
  EXPECT_EQ(bspline.ncoeffs(),270);
}

TEST(BSpline, NonUniformBSpline_parDim4_geoDim2_degrees1234_double)
{
  iganet::NonUniformBSpline<double, 2, 1, 2, 3, 4> bspline( {{{0.0, 0.0, 0.5, 1.0, 1.0},
                                                              {0.0, 0.0, 0.0, 1.0, 1.0, 1.0},
                                                              {0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0},
                                                              {0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0}}} );
  EXPECT_EQ(bspline.parDim(),   4);
  EXPECT_EQ(bspline.geoDim(),   2);
  EXPECT_EQ(bspline.degree(0),  1);
  EXPECT_EQ(bspline.degree(1),  2);
  EXPECT_EQ(bspline.degree(2),  3);
  EXPECT_EQ(bspline.degree(3),  4);
  EXPECT_EQ(bspline.nknots(0),  5);
  EXPECT_EQ(bspline.nknots(1),  6);
  EXPECT_EQ(bspline.nknots(2),  9);
  EXPECT_EQ(bspline.nknots(3), 11);
  EXPECT_EQ(bspline.ncoeffs(0), 3);
  EXPECT_EQ(bspline.ncoeffs(1), 3);
  EXPECT_EQ(bspline.ncoeffs(2), 5);
  EXPECT_EQ(bspline.ncoeffs(3), 6);
  EXPECT_EQ(bspline.ncoeffs(),270);
}

TEST(BSpline, NonUniformBSpline_parDim4_geoDim3_degrees1234_double)
{
  iganet::NonUniformBSpline<double, 3, 1, 2, 3, 4> bspline( {{{0.0, 0.0, 0.5, 1.0, 1.0},
                                                              {0.0, 0.0, 0.0, 1.0, 1.0, 1.0},
                                                              {0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0},
                                                              {0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0}}} );
  EXPECT_EQ(bspline.parDim(),   4);
  EXPECT_EQ(bspline.geoDim(),   3);
  EXPECT_EQ(bspline.degree(0),  1);
  EXPECT_EQ(bspline.degree(1),  2);
  EXPECT_EQ(bspline.degree(2),  3);
  EXPECT_EQ(bspline.degree(3),  4);
  EXPECT_EQ(bspline.nknots(0),  5);
  EXPECT_EQ(bspline.nknots(1),  6);
  EXPECT_EQ(bspline.nknots(2),  9);
  EXPECT_EQ(bspline.nknots(3), 11);
  EXPECT_EQ(bspline.ncoeffs(0), 3);
  EXPECT_EQ(bspline.ncoeffs(1), 3);
  EXPECT_EQ(bspline.ncoeffs(2), 5);
  EXPECT_EQ(bspline.ncoeffs(3), 6);
  EXPECT_EQ(bspline.ncoeffs(),270);
}

TEST(BSpline, NonUniformBSpline_parDim4_geoDim4_degrees1234_double)
{
  iganet::NonUniformBSpline<double, 4, 1, 2, 3, 4> bspline( {{{0.0, 0.0, 0.5, 1.0, 1.0},
                                                              {0.0, 0.0, 0.0, 1.0, 1.0, 1.0},
                                                              {0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0},
                                                              {0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0}}} );
  EXPECT_EQ(bspline.parDim(),   4);
  EXPECT_EQ(bspline.geoDim(),   4);
  EXPECT_EQ(bspline.degree(0),  1);
  EXPECT_EQ(bspline.degree(1),  2);
  EXPECT_EQ(bspline.degree(2),  3);
  EXPECT_EQ(bspline.degree(3),  4);
  EXPECT_EQ(bspline.nknots(0),  5);
  EXPECT_EQ(bspline.nknots(1),  6);
  EXPECT_EQ(bspline.nknots(2),  9);
  EXPECT_EQ(bspline.nknots(3), 11);
  EXPECT_EQ(bspline.ncoeffs(0), 3);
  EXPECT_EQ(bspline.ncoeffs(1), 3);
  EXPECT_EQ(bspline.ncoeffs(2), 5);
  EXPECT_EQ(bspline.ncoeffs(3), 6);
  EXPECT_EQ(bspline.ncoeffs(),270);
}

TEST(BSpline, UniformBSpline_init_double)
{
  {
    iganet::UniformBSpline<double, 1, 1> bspline({5}, iganet::BSplineInit::zeros);
    EXPECT_TRUE(torch::allclose(bspline.coeffs(0), torch::zeros(5, bspline.options())));
  }

  {
    iganet::UniformBSpline<double, 1, 1> bspline({5}, iganet::BSplineInit::ones);
    EXPECT_TRUE(torch::allclose(bspline.coeffs(0), torch::ones(5, bspline.options())));
  }
    
  {
    iganet::UniformBSpline<double, 1, 1> bspline({5}, iganet::BSplineInit::linear);
    EXPECT_TRUE(torch::allclose(bspline.coeffs(0), torch::linspace(0, 1, 5, bspline.options())));
  }

  {
    iganet::UniformBSpline<double, 1, 1> bspline({5}, iganet::BSplineInit::greville);
    EXPECT_TRUE(torch::allclose(bspline.coeffs(0), torch::linspace(0, 1, 5, bspline.options())));
  }

  {
    iganet::UniformBSpline<double, 1, 2> bspline({5}, iganet::BSplineInit::greville);
    EXPECT_FALSE(torch::allclose(bspline.coeffs(0), torch::linspace(0, 1, 5, bspline.options())));
  }

  {
    iganet::UniformBSpline<double, 2, 1> bspline({5}, iganet::BSplineInit::zeros);
    EXPECT_TRUE(torch::allclose(bspline.coeffs(0), torch::zeros(5, bspline.options())));
    EXPECT_TRUE(torch::allclose(bspline.coeffs(1), torch::zeros(5, bspline.options())));
  }

  {
    iganet::UniformBSpline<double, 2, 1> bspline({5}, iganet::BSplineInit::ones);
    EXPECT_TRUE(torch::allclose(bspline.coeffs(0), torch::ones(5, bspline.options())));
    EXPECT_TRUE(torch::allclose(bspline.coeffs(1), torch::ones(5, bspline.options())));
  }

  {
    iganet::UniformBSpline<double, 2, 1> bspline({5}, iganet::BSplineInit::linear);
    EXPECT_TRUE(torch::allclose(bspline.coeffs(0), torch::linspace(0, 1, 5, bspline.options())));
    EXPECT_TRUE(torch::allclose(bspline.coeffs(1), torch::ones(5, bspline.options())));
  }

  {
    iganet::UniformBSpline<double, 2, 1> bspline({5}, iganet::BSplineInit::greville);
    EXPECT_TRUE(torch::allclose(bspline.coeffs(0), torch::linspace(0, 1, 5, bspline.options())));
    EXPECT_TRUE(torch::allclose(bspline.coeffs(1), torch::ones(5, bspline.options())));
  }

  {
    iganet::UniformBSpline<double, 2, 2> bspline({5}, iganet::BSplineInit::greville);
    EXPECT_FALSE(torch::allclose(bspline.coeffs(0), torch::linspace(0, 1, 5, bspline.options())));
    EXPECT_TRUE(torch::allclose(bspline.coeffs(1), torch::ones(5, bspline.options())));
  }
  
  {
    iganet::UniformBSpline<double, 2, 2, 2> bspline({5, 8}, iganet::BSplineInit::zeros);
    EXPECT_TRUE(torch::allclose(bspline.coeffs(0), torch::zeros(40, bspline.options())));
    EXPECT_TRUE(torch::allclose(bspline.coeffs(1), torch::zeros(40, bspline.options())));
  }

  {
    iganet::UniformBSpline<double, 2, 2, 2> bspline({5, 8}, iganet::BSplineInit::ones);
    EXPECT_TRUE(torch::allclose(bspline.coeffs(0), torch::ones(40, bspline.options())));
    EXPECT_TRUE(torch::allclose(bspline.coeffs(1), torch::ones(40, bspline.options())));
  }

  {
    iganet::UniformBSpline<double, 2, 2, 2> bspline({5, 8}, iganet::BSplineInit::linear);
    EXPECT_TRUE(torch::allclose(bspline.coeffs(0),
                                torch::linspace(0, 1, 5, bspline.options()).repeat(8)));
    EXPECT_TRUE(torch::allclose(bspline.coeffs(1),
                                torch::linspace(0, 1, 8, bspline.options()).repeat_interleave(5)));
  }
  
  {
    iganet::UniformBSpline<double, 2, 1, 1> bspline({5, 8}, iganet::BSplineInit::greville);
    EXPECT_TRUE(torch::allclose(bspline.coeffs(0),
                                torch::linspace(0, 1, 5, bspline.options()).repeat(8)));
    EXPECT_TRUE(torch::allclose(bspline.coeffs(1),
                                torch::linspace(0, 1, 8, bspline.options()).repeat_interleave(5)));
  }
  
  {
    iganet::UniformBSpline<double, 3, 2, 2> bspline({5, 8}, iganet::BSplineInit::zeros);
    EXPECT_TRUE(torch::allclose(bspline.coeffs(0), torch::zeros(40, bspline.options())));
    EXPECT_TRUE(torch::allclose(bspline.coeffs(1), torch::zeros(40, bspline.options())));
    EXPECT_TRUE(torch::allclose(bspline.coeffs(2), torch::zeros(40, bspline.options())));
  }

  {
    iganet::UniformBSpline<double, 3, 2, 2> bspline({5, 8}, iganet::BSplineInit::ones);
    EXPECT_TRUE(torch::allclose(bspline.coeffs(0), torch::ones(40, bspline.options())));
    EXPECT_TRUE(torch::allclose(bspline.coeffs(1), torch::ones(40, bspline.options())));
    EXPECT_TRUE(torch::allclose(bspline.coeffs(2), torch::ones(40, bspline.options())));
  }

  {
    iganet::UniformBSpline<double, 3, 2, 2> bspline({5, 8}, iganet::BSplineInit::linear);
    EXPECT_TRUE(torch::allclose(bspline.coeffs(0),
                                torch::linspace(0, 1, 5, bspline.options()).repeat(8)));
    EXPECT_TRUE(torch::allclose(bspline.coeffs(1),
                                torch::linspace(0, 1, 8, bspline.options()).repeat_interleave(5)));
    EXPECT_TRUE(torch::allclose(bspline.coeffs(2), torch::ones(40, bspline.options())));
  }
  
  {
    iganet::UniformBSpline<double, 3, 1, 1> bspline({5, 8}, iganet::BSplineInit::greville);
    EXPECT_TRUE(torch::allclose(bspline.coeffs(0),
                                torch::linspace(0, 1, 5, bspline.options()).repeat(8)));
    EXPECT_TRUE(torch::allclose(bspline.coeffs(1),
                                torch::linspace(0, 1, 8, bspline.options()).repeat_interleave(5)));
    EXPECT_TRUE(torch::allclose(bspline.coeffs(2), torch::ones(40, bspline.options())));
  }

  {
    iganet::UniformBSpline<double, 4, 2, 2> bspline({5, 8}, iganet::BSplineInit::zeros);
    EXPECT_TRUE(torch::allclose(bspline.coeffs(0), torch::zeros(40, bspline.options())));
    EXPECT_TRUE(torch::allclose(bspline.coeffs(1), torch::zeros(40, bspline.options())));
    EXPECT_TRUE(torch::allclose(bspline.coeffs(2), torch::zeros(40, bspline.options())));
    EXPECT_TRUE(torch::allclose(bspline.coeffs(3), torch::zeros(40, bspline.options())));
  }

  {
    iganet::UniformBSpline<double, 4, 2, 2> bspline({5, 8}, iganet::BSplineInit::ones);
    EXPECT_TRUE(torch::allclose(bspline.coeffs(0), torch::ones(40, bspline.options())));
    EXPECT_TRUE(torch::allclose(bspline.coeffs(1), torch::ones(40, bspline.options())));
    EXPECT_TRUE(torch::allclose(bspline.coeffs(2), torch::ones(40, bspline.options())));
    EXPECT_TRUE(torch::allclose(bspline.coeffs(3), torch::ones(40, bspline.options())));
  }

  {
    iganet::UniformBSpline<double, 4, 2, 2> bspline({5, 8}, iganet::BSplineInit::linear);
    EXPECT_TRUE(torch::allclose(bspline.coeffs(0),
                                torch::linspace(0, 1, 5, bspline.options()).repeat(8)));
    EXPECT_TRUE(torch::allclose(bspline.coeffs(1),
                                torch::linspace(0, 1, 8, bspline.options()).repeat_interleave(5)));
    EXPECT_TRUE(torch::allclose(bspline.coeffs(2), torch::ones(40, bspline.options())));
    EXPECT_TRUE(torch::allclose(bspline.coeffs(3), torch::ones(40, bspline.options())));
  }
  
  {
    iganet::UniformBSpline<double, 4, 1, 1> bspline({5, 8}, iganet::BSplineInit::greville);
    EXPECT_TRUE(torch::allclose(bspline.coeffs(0),
                                torch::linspace(0, 1, 5, bspline.options()).repeat(8)));
    EXPECT_TRUE(torch::allclose(bspline.coeffs(1),
                                torch::linspace(0, 1, 8, bspline.options()).repeat_interleave(5)));
    EXPECT_TRUE(torch::allclose(bspline.coeffs(2), torch::ones(40, bspline.options())));
    EXPECT_TRUE(torch::allclose(bspline.coeffs(3), torch::ones(40, bspline.options())));
  } 
}

TEST(BSpline, UniformBSpline_read_write_double)
{
  std::filesystem::path filename = std::filesystem::temp_directory_path() / std::to_string(rand());
  iganet::UniformBSpline<double, 3, 3, 4> bspline_out({2,3});
  bspline_out.save(filename.c_str());
  iganet::UniformBSpline<double, 3, 3, 4> bspline_in;
  bspline_in.load(filename.c_str());
  std::filesystem::remove(filename);

  EXPECT_EQ( (bspline_in == bspline_out), true);
  EXPECT_EQ( (bspline_in != bspline_out), false);
}

TEST(BSpline, NonUniformBSpline_read_write_double)
{
  std::filesystem::path filename = std::filesystem::temp_directory_path() / std::to_string(rand());
  iganet::NonUniformBSpline<double, 3, 1, 2, 3> bspline_out( {{{0.0, 0.0, 0.5, 1.0, 1.0},
                                                               {0.0, 0.0, 0.0, 1.0, 1.0, 1.0},
                                                               {0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0}}} );
  bspline_out.save(filename.c_str());
  iganet::NonUniformBSpline<double, 3, 1, 2, 3> bspline_in;
  bspline_in.load(filename.c_str());
  std::filesystem::remove(filename);

  EXPECT_EQ( (bspline_in == bspline_out), true);
  EXPECT_EQ( (bspline_in != bspline_out), false);
}

template<iganet::BSplineDeriv deriv,
         typename BSpline_t, typename SplineLibBSpline_t, typename TensorArray_t>
void test_bspline_eval_(BSpline_t& bspline, SplineLibBSpline_t splinelib_bspline,
                        TensorArray_t& xi, typename BSpline_t::value_type tol = 1e-12)
{
  // B-spline evaluation
  using ParametricCoordinate       = typename SplineLibBSpline_t::ParametricCoordinate_;
  using ScalarParametricCoordinate = typename ParametricCoordinate::value_type;
  using Derivative                 = typename SplineLibBSpline_t::ParameterSpace_::Derivative_;
  using ScalarDerivative           = typename Derivative::value_type;
  
  auto bspline_val = bspline.template eval_<deriv>(xi);
  
  for (int64_t i=0; i<xi[0].size(0); ++i) {
    if constexpr (BSpline_t::parDim() == 1 &&
                  BSpline_t::geoDim() == 1)
      EXPECT_NEAR(bspline_val[i].template item<typename BSpline_t::value_type>(),
                  splinelib_bspline(ParametricCoordinate
                                    {
                                      ScalarParametricCoordinate{(xi[0])[i].template item<typename BSpline_t::value_type>()}
                                    },
                                    Derivative
                                    {
                                      ScalarDerivative{(short_t) deriv % 10}
                                    }
                                    )[0], tol);
    else if constexpr (BSpline_t::parDim() == 1 &&
                       BSpline_t::geoDim() >  1)
      for (short_t k=0; k<bspline.geoDim(); ++k)
        EXPECT_NEAR(bspline_val[k][i].template item<typename BSpline_t::value_type>(),
                    splinelib_bspline(ParametricCoordinate
                                      {
                                        ScalarParametricCoordinate{(xi[0])[i].template item<typename BSpline_t::value_type>()}                                           
                                      },
                                      Derivative
                                      {
                                        ScalarDerivative{(short_t) deriv %10}
                                      }
                                      )[k], tol);
    else if constexpr (BSpline_t::parDim() == 2 &&
                       BSpline_t::geoDim() == 1)
      EXPECT_NEAR(bspline_val[i].template item<typename BSpline_t::value_type>(),
                  splinelib_bspline(ParametricCoordinate
                                    {
                                      ScalarParametricCoordinate{(xi[0])[i].template item<typename BSpline_t::value_type>()},
                                      ScalarParametricCoordinate{(xi[1])[i].template item<typename BSpline_t::value_type>()}
                                    },
                                    Derivative
                                    {
                                      ScalarDerivative{ (short_t)deriv    %10},
                                      ScalarDerivative{((short_t)deriv/10)%10}
                                    }
                                    )[0], tol);
    else if constexpr (BSpline_t::parDim() == 2 &&
                       BSpline_t::geoDim() >  1)
      for (short_t k=0; k<bspline.geoDim(); ++k)
        EXPECT_NEAR(bspline_val[k][i].template item<typename BSpline_t::value_type>(),
                    splinelib_bspline(ParametricCoordinate
                                      {
                                        ScalarParametricCoordinate{(xi[0])[i].template item<typename BSpline_t::value_type>()},
                                        ScalarParametricCoordinate{(xi[1])[i].template item<typename BSpline_t::value_type>()}                                           
                                      },
                                      Derivative
                                      {
                                        ScalarDerivative{ (short_t)deriv    %10},
                                        ScalarDerivative{((short_t)deriv/10)%10}
                                      }
                                      )[k], tol);
    else if constexpr (BSpline_t::parDim() == 3 &&
                       BSpline_t::geoDim() == 1)
      EXPECT_NEAR(bspline_val[i].template item<typename BSpline_t::value_type>(),
                  splinelib_bspline(ParametricCoordinate
                                    {
                                      ScalarParametricCoordinate{(xi[0])[i].template item<typename BSpline_t::value_type>()},
                                      ScalarParametricCoordinate{(xi[1])[i].template item<typename BSpline_t::value_type>()},
                                      ScalarParametricCoordinate{(xi[2])[i].template item<typename BSpline_t::value_type>()}
                                    },
                                    Derivative
                                    {
                                      ScalarDerivative{ (short_t)deriv     %10},
                                      ScalarDerivative{((short_t)deriv/ 10)%10},
                                      ScalarDerivative{((short_t)deriv/100)%10}
                                    }
                                    )[0], tol);
    else if constexpr (BSpline_t::parDim() == 3 &&
                       BSpline_t::geoDim() >  1)
      for (short_t k=0; k<bspline.geoDim(); ++k)
        EXPECT_NEAR(bspline_val[k][i].template item<typename BSpline_t::value_type>(),
                    splinelib_bspline(ParametricCoordinate
                                      {
                                        ScalarParametricCoordinate{(xi[0])[i].template item<typename BSpline_t::value_type>()},
                                        ScalarParametricCoordinate{(xi[1])[i].template item<typename BSpline_t::value_type>()},
                                        ScalarParametricCoordinate{(xi[2])[i].template item<typename BSpline_t::value_type>()}                                                                                     
                                      },
                                      Derivative
                                      {
                                        ScalarDerivative{ (short_t)deriv     %10},
                                        ScalarDerivative{((short_t)deriv/ 10)%10},
                                        ScalarDerivative{((short_t)deriv/100)%10}
                                      }
                                      )[k], tol);
    else if constexpr (BSpline_t::parDim() == 4 &&
                       BSpline_t::geoDim() == 1)
      EXPECT_NEAR(bspline_val[i].template item<typename BSpline_t::value_type>(),
                  splinelib_bspline(ParametricCoordinate
                                    {
                                      ScalarParametricCoordinate{(xi[0])[i].template item<typename BSpline_t::value_type>()},
                                      ScalarParametricCoordinate{(xi[1])[i].template item<typename BSpline_t::value_type>()},
                                      ScalarParametricCoordinate{(xi[2])[i].template item<typename BSpline_t::value_type>()},
                                      ScalarParametricCoordinate{(xi[3])[i].template item<typename BSpline_t::value_type>()}
                                    },
                                    Derivative
                                    {
                                      ScalarDerivative{ (short_t)deriv      %10},
                                      ScalarDerivative{((short_t)deriv/  10)%10},
                                      ScalarDerivative{((short_t)deriv/ 100)%10},
                                      ScalarDerivative{((short_t)deriv/1000)%10}
                                    }
                                    )[0], tol);
    else if constexpr (BSpline_t::parDim() == 4 &&
                       BSpline_t::geoDim() >  1)
      for (short_t k=0; k<bspline.geoDim(); ++k)
        EXPECT_NEAR(bspline_val[k][i].template item<typename BSpline_t::value_type>(),
                    splinelib_bspline(ParametricCoordinate
                                      {
                                        ScalarParametricCoordinate{(xi[0])[i].template item<typename BSpline_t::value_type>()},
                                        ScalarParametricCoordinate{(xi[1])[i].template item<typename BSpline_t::value_type>()},
                                        ScalarParametricCoordinate{(xi[2])[i].template item<typename BSpline_t::value_type>()},
                                        ScalarParametricCoordinate{(xi[3])[i].template item<typename BSpline_t::value_type>()}                                                                                     
                                      },
                                      Derivative
                                      {
                                        ScalarDerivative{ (short_t)deriv      %10},
                                        ScalarDerivative{((short_t)deriv/  10)%10},
                                        ScalarDerivative{((short_t)deriv/ 100)%10},
                                        ScalarDerivative{((short_t)deriv/1000)%10}
                                      }
                                      )[k], tol);
    else
      throw std::runtime_error("Unsupported parametric/ geometric dimension");
  }
}

template<typename BSpline_t, typename TensorArray_t>
void test_bspline_eval(BSpline_t& bspline, TensorArray_t& xi, typename BSpline_t::value_type tol = 1e-12)
{  
  // B-spline construction
  using BSpline          = splinelib::sources::splines::BSpline<BSpline_t::parDim(), BSpline_t::geoDim()>;
  using ParameterSpace   = typename BSpline::ParameterSpace_;
  using VectorSpace      = typename BSpline::VectorSpace_;
  using Coordinates      = typename VectorSpace::Coordinates_;
  using Coordinate       = typename Coordinates::value_type;
  using ScalarCoordinate = typename Coordinate::value_type;
  using Degrees          = typename ParameterSpace::Degrees_;
  using Degree           = typename Degrees::value_type;
  using KnotVectors      = typename ParameterSpace::KnotVectors_;
  using KnotVector       = typename KnotVectors::value_type::element_type;
  using Knots            = typename KnotVector::Knots_;
  using Knot             = typename Knots::value_type;

  // B-spline evaluation
  using ParametricCoordinate       = typename BSpline::ParametricCoordinate_;
  using ScalarParametricCoordinate = typename ParametricCoordinate::value_type;
  using Derivative                 = typename ParameterSpace::Derivative_;
  using ScalarDerivative           = typename Derivative::value_type;
  
  // Create degress structure
  Degrees degrees;
  for (short_t k=0; k<bspline.parDim(); ++k)
    degrees[k] = Degree{ bspline.degree(k) };
  
  // Create knot vectors
  KnotVectors knot_vectors;
  for (short_t k=0; k<bspline.parDim(); ++k) {
    Knots knots_;
    for (int64_t i=0; i<bspline.nknots(k); ++i)
      knots_.emplace_back( Knot{bspline.knots(k)[i].template item<typename BSpline_t::value_type>()} );
    splinelib::SharedPointer<KnotVector> knot_vector{std::make_shared<KnotVector>(knots_)};
    knot_vectors[k] = std::move(knot_vector);
  }

  // Create parametric space
  splinelib::SharedPointer<ParameterSpace> parameter_space{std::make_shared<ParameterSpace>(knot_vectors, degrees)};
  
  // Create coordinate vector(s)
  Coordinates coordinates;
  for (int64_t i=0; i<bspline.ncoeffs(); ++i)
    if constexpr (BSpline_t::geoDim() == 1)
      coordinates.emplace_back(Coordinate{ScalarCoordinate{bspline.coeffs(0)[i].template item<typename BSpline_t::value_type>()}} );
    else if constexpr (BSpline_t::geoDim() == 2)
      coordinates.emplace_back(Coordinate{ScalarCoordinate{bspline.coeffs(0)[i].template item<typename BSpline_t::value_type>()},
                                          ScalarCoordinate{bspline.coeffs(1)[i].template item<typename BSpline_t::value_type>()}} );
    else if constexpr (BSpline_t::geoDim() == 3)
      coordinates.emplace_back(Coordinate{ScalarCoordinate{bspline.coeffs(0)[i].template item<typename BSpline_t::value_type>()},
                                          ScalarCoordinate{bspline.coeffs(1)[i].template item<typename BSpline_t::value_type>()},
                                          ScalarCoordinate{bspline.coeffs(2)[i].template item<typename BSpline_t::value_type>()}} );
    else if constexpr (BSpline_t::geoDim() == 4)
      coordinates.emplace_back(Coordinate{ScalarCoordinate{bspline.coeffs(0)[i].template item<typename BSpline_t::value_type>()},
                                          ScalarCoordinate{bspline.coeffs(1)[i].template item<typename BSpline_t::value_type>()},
                                          ScalarCoordinate{bspline.coeffs(2)[i].template item<typename BSpline_t::value_type>()},
                                          ScalarCoordinate{bspline.coeffs(3)[i].template item<typename BSpline_t::value_type>()}} );
    else
      throw std::runtime_error("Unsupported geometric dimension");

  // Create vector space
  splinelib::SharedPointer<VectorSpace> vector_space{std:: make_shared<VectorSpace>(coordinates)};

  // Create B-Spline
  BSpline splinelib_bspline{parameter_space, vector_space};
  
  // Evaluate function and derivatives
  test_bspline_eval_<iganet::BSplineDeriv::func>(bspline, splinelib_bspline, xi, tol);
  test_bspline_eval_<iganet::BSplineDeriv::dx1 >(bspline, splinelib_bspline, xi, tol);
  test_bspline_eval_<iganet::BSplineDeriv::dx2 >(bspline, splinelib_bspline, xi, tol);
  test_bspline_eval_<iganet::BSplineDeriv::dx3 >(bspline, splinelib_bspline, xi, tol);
  test_bspline_eval_<iganet::BSplineDeriv::dx4 >(bspline, splinelib_bspline, xi, tol);    
}

TEST(BSpline, UniformBSpline_eval_parDim1_geoDim1_degrees1_double)
{
  iganet::UniformBSpline<double, 1, 1> bspline({11}, iganet::BSplineInit::linear);
  auto xi  = iganet::to_tensorArray({0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0}, bspline.options());
  test_bspline_eval(bspline, xi, 1e-12);
}

TEST(BSpline, UniformBSpline_eval_parDim1_geoDim1_degrees2_double)
{
  iganet::UniformBSpline<double, 1, 2> bspline({11}, iganet::BSplineInit::linear);
  auto xi  = iganet::to_tensorArray({0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0}, bspline.options());
  test_bspline_eval(bspline, xi, 1e-12); 
}

TEST(BSpline, UniformBSpline_eval_parDim1_geoDim1_degrees3_double)
{
  iganet::UniformBSpline<double, 1, 3> bspline({11}, iganet::BSplineInit::linear);
  auto xi  = iganet::to_tensorArray({0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0}, bspline.options());
  test_bspline_eval(bspline, xi, 1e-12);  
}

TEST(BSpline, UniformBSpline_eval_parDim1_geoDim1_degrees4_double)
{
  iganet::UniformBSpline<double, 1, 4> bspline({11}, iganet::BSplineInit::linear);
  auto xi  = iganet::to_tensorArray({0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0}, bspline.options());
  test_bspline_eval(bspline, xi, 1e-10); 
}

TEST(BSpline, UniformBSpline_eval_parDim1_geoDim1_degrees5_double)
{
  iganet::UniformBSpline<double, 1, 5> bspline({11}, iganet::BSplineInit::linear);
  auto xi  = iganet::to_tensorArray({0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0}, bspline.options());
  test_bspline_eval(bspline, xi, 1e-10); 
}

TEST(BSpline, UniformBSpline_eval_parDim1_geoDim1_degrees6_double)
{
  iganet::UniformBSpline<double, 1, 6> bspline({11}, iganet::BSplineInit::linear);
  auto xi  = iganet::to_tensorArray({0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0}, bspline.options());
  test_bspline_eval(bspline, xi, 1e-10); 
}

TEST(BSpline, UniformBSpline_eval_parDim1_geoDim2_degrees1_double)
{
  iganet::UniformBSpline<double, 2, 1> bspline({11}, iganet::BSplineInit::linear);
  auto xi  = iganet::to_tensorArray({0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0}, bspline.options());
  test_bspline_eval(bspline, xi, 1e-12);
}

TEST(BSpline, UniformBSpline_eval_parDim1_geoDim2_degrees2_double)
{
  iganet::UniformBSpline<double, 2, 2> bspline({11}, iganet::BSplineInit::linear);
  auto xi  = iganet::to_tensorArray({0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0}, bspline.options());
  test_bspline_eval(bspline, xi, 1e-12);
}

TEST(BSpline, UniformBSpline_eval_parDim1_geoDim2_degrees3_double)
{
  iganet::UniformBSpline<double, 2, 3> bspline({11}, iganet::BSplineInit::linear);
  auto xi  = iganet::to_tensorArray({0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0}, bspline.options());
  test_bspline_eval(bspline, xi, 1e-12);
}

TEST(BSpline, UniformBSpline_eval_parDim1_geoDim2_degrees4_double)
{
  iganet::UniformBSpline<double, 2, 4> bspline({11}, iganet::BSplineInit::linear);
  auto xi  = iganet::to_tensorArray({0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0}, bspline.options());
  test_bspline_eval(bspline, xi, 1e-10);
}

TEST(BSpline, UniformBSpline_eval_parDim1_geoDim2_degrees5_double)
{
  iganet::UniformBSpline<double, 2, 5> bspline({11}, iganet::BSplineInit::linear);
  auto xi  = iganet::to_tensorArray({0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0}, bspline.options());
  test_bspline_eval(bspline, xi, 1e-10);
}

TEST(BSpline, UniformBSpline_eval_parDim1_geoDim2_degrees6_double)
{
  iganet::UniformBSpline<double, 2, 6> bspline({11}, iganet::BSplineInit::linear);
  auto xi  = iganet::to_tensorArray({0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0}, bspline.options());
  test_bspline_eval(bspline, xi, 1e-10);
}

TEST(BSpline, UniformBSpline_eval_parDim1_geoDim3_degrees1_double)
{
  iganet::UniformBSpline<double, 3, 1> bspline({11}, iganet::BSplineInit::linear);
  auto xi  = iganet::to_tensorArray({0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0}, bspline.options());
  test_bspline_eval(bspline, xi, 1e-12);
}

TEST(BSpline, UniformBSpline_eval_parDim1_geoDim3_degrees2_double)
{
  iganet::UniformBSpline<double, 3, 2> bspline({11}, iganet::BSplineInit::linear);
  auto xi  = iganet::to_tensorArray({0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0}, bspline.options());
  test_bspline_eval(bspline, xi, 1e-12);
}

TEST(BSpline, UniformBSpline_eval_parDim1_geoDim3_degrees3_double)
{
  iganet::UniformBSpline<double, 3, 3> bspline({11}, iganet::BSplineInit::linear);
  auto xi  = iganet::to_tensorArray({0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0}, bspline.options());
  test_bspline_eval(bspline, xi, 1e-12);
}

TEST(BSpline, UniformBSpline_eval_parDim1_geoDim3_degrees4_double)
{
  iganet::UniformBSpline<double, 3, 4> bspline({11}, iganet::BSplineInit::linear);
  auto xi  = iganet::to_tensorArray({0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0}, bspline.options());
  test_bspline_eval(bspline, xi, 1e-10);
}

TEST(BSpline, UniformBSpline_eval_parDim1_geoDim3_degrees5_double)
{
  iganet::UniformBSpline<double, 3, 5> bspline({11}, iganet::BSplineInit::linear);
  auto xi  = iganet::to_tensorArray({0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0}, bspline.options());
  test_bspline_eval(bspline, xi, 1e-10);
}

TEST(BSpline, UniformBSpline_eval_parDim1_geoDim3_degrees6_double)
{
  iganet::UniformBSpline<double, 3, 6> bspline({11}, iganet::BSplineInit::linear);
  auto xi  = iganet::to_tensorArray({0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0}, bspline.options());
  test_bspline_eval(bspline, xi, 1e-10);
}

TEST(BSpline, UniformBSpline_eval_parDim1_geoDim4_degrees1_double)
{
  iganet::UniformBSpline<double, 4, 1> bspline({11}, iganet::BSplineInit::linear);
  auto xi  = iganet::to_tensorArray({0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0}, bspline.options());
  test_bspline_eval(bspline, xi, 1e-12);
}

TEST(BSpline, UniformBSpline_eval_parDim1_geoDim4_degrees2_double)
{
  iganet::UniformBSpline<double, 4, 2> bspline({11}, iganet::BSplineInit::linear);
  auto xi  = iganet::to_tensorArray({0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0}, bspline.options());
  test_bspline_eval(bspline, xi, 1e-12);
}

TEST(BSpline, UniformBSpline_eval_parDim1_geoDim4_degrees3_double)
{
  iganet::UniformBSpline<double, 4, 3> bspline({11}, iganet::BSplineInit::linear);
  auto xi  = iganet::to_tensorArray({0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0}, bspline.options());
  test_bspline_eval(bspline, xi, 1e-12);
}

TEST(BSpline, UniformBSpline_eval_parDim1_geoDim4_degrees4_double)
{
  iganet::UniformBSpline<double, 4, 4> bspline({11}, iganet::BSplineInit::linear);
  auto xi  = iganet::to_tensorArray({0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0}, bspline.options());
  test_bspline_eval(bspline, xi, 1e-10);
}

TEST(BSpline, UniformBSpline_eval_parDim1_geoDim4_degrees5_double)
{
  iganet::UniformBSpline<double, 4, 5> bspline({11}, iganet::BSplineInit::linear);
  auto xi  = iganet::to_tensorArray({0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0}, bspline.options());
  test_bspline_eval(bspline, xi, 1e-10);
}

TEST(BSpline, UniformBSpline_eval_parDim1_geoDim4_degrees6_double)
{
  iganet::UniformBSpline<double, 4, 6> bspline({11}, iganet::BSplineInit::linear);
  auto xi  = iganet::to_tensorArray({0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0}, bspline.options());
  test_bspline_eval(bspline, xi, 1e-10);
}

TEST(BSpline, UniformBSpline_eval_parDim2_geoDim2_degrees22_double)
{
  iganet::UniformBSpline<double, 2, 1, 1> bspline({6, 5}, iganet::BSplineInit::linear);
  auto xi  = iganet::to_tensorArray({0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0},
                                    {0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0}, bspline.options());
  test_bspline_eval(bspline, xi, 1e-12);
}

TEST(BSpline, UniformBSpline_eval_parDim2_geoDim2_degrees46_double)
{
  iganet::UniformBSpline<double, 2, 4, 6> bspline({5, 11}, iganet::BSplineInit::linear);
  auto xi  = iganet::to_tensorArray({0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0},
                                    {0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0}, bspline.options());
  test_bspline_eval(bspline, xi, 1e-10);
}

TEST(BSpline, UniformBSpline_eval_parDim2_geoDim2_degrees64_double)
{
  iganet::UniformBSpline<double, 2, 6, 4> bspline({11, 5}, iganet::BSplineInit::linear);
  auto xi  = iganet::to_tensorArray({0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0},
                                    {0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0}, bspline.options());
  test_bspline_eval(bspline, xi, 1e-10);
}

TEST(BSpline, UniformBSpline_eval_parDim2_geoDim3_degrees64_double)
{
  iganet::UniformBSpline<double, 3, 6, 4> bspline({11, 5}, iganet::BSplineInit::linear);
  auto xi  = iganet::to_tensorArray({0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0},
                                    {0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0}, bspline.options());
  test_bspline_eval(bspline, xi, 1e-10);
}

TEST(BSpline, UniformBSpline_eval_parDim2_geoDim4_degrees64_double)
{
  iganet::UniformBSpline<double, 4, 6, 4> bspline({11, 5}, iganet::BSplineInit::linear);
  auto xi  = iganet::to_tensorArray({0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0},
                                    {0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0}, bspline.options());
  test_bspline_eval(bspline, xi, 1e-10);
}




TEST(BSpline, UniformBSpline_eval_parDim3_geoDim3_degrees641_double)
{
  iganet::UniformBSpline<double, 3, 6, 4, 2> bspline({11, 5, 3}, iganet::BSplineInit::linear);
  auto xi  = iganet::to_tensorArray({0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0},
                                    {0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0},
                                    {0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0}, bspline.options());
  test_bspline_eval(bspline, xi, 1e-10);
}


int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  iganet::init();
  return RUN_ALL_TESTS();
}
