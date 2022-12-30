/**
   @file unittests/unittest_uniform_bspline.cxx

   @brief B-Spline unittests

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include <iganet.hpp>
#include <filesystem>
#include <iostream>

#include "unittest_splinelib.hpp"
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

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  iganet::init();
  return RUN_ALL_TESTS();
}
