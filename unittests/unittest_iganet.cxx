/**
   @file unittests/unittest_iganet.cxx

   @brief IgANet unittests

   @author Matthias Moller

   @copyright This file is part of the IgaNet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include <iganet.hpp>
#include <iostream>

#include <gtest/gtest.h>

TEST(BSpline, IgaNet_UniformBSpline_1d)
{
  using real_t      = double;
  using optimizer_t = torch::optim::Adam;

  iganet::IgANet<real_t, optimizer_t,
                 iganet::UniformBSpline, 
                 5> net({50,30,70}, // Number of neurons per layers
                        {6});       // Number of B-spline coefficients

  EXPECT_EQ(net.geo().parDim(), 1);
  EXPECT_EQ(net.rhs().parDim(), 1);
  EXPECT_EQ(net.sol().parDim(), 1);

  EXPECT_EQ(net.geo().geoDim(), 1);
  EXPECT_EQ(net.rhs().geoDim(), 1);
  EXPECT_EQ(net.sol().geoDim(), 1);

  EXPECT_EQ(net.geo().degree(0), 5);
  EXPECT_EQ(net.rhs().degree(0), 5);
  EXPECT_EQ(net.sol().degree(0), 5);

  EXPECT_EQ(net.geo().ncoeffs(0), 6);
  EXPECT_EQ(net.rhs().ncoeffs(0), 6);
  EXPECT_EQ(net.sol().ncoeffs(0), 6);
}

TEST(BSpline, IgaNet_UniformBSpline_2d)
{
  using real_t      = double;
  using optimizer_t = torch::optim::Adam;

  iganet::IgANet<real_t, optimizer_t,
                 iganet::UniformBSpline, 
                 3, 5> net({50,30,70}, // Number of neurons per layers
                           {4,6});     // Number of B-spline coefficients

  EXPECT_EQ(net.geo().parDim(), 2);
  EXPECT_EQ(net.rhs().parDim(), 2);
  EXPECT_EQ(net.sol().parDim(), 2);

  EXPECT_EQ(net.geo().geoDim(), 2);
  EXPECT_EQ(net.rhs().geoDim(), 1);
  EXPECT_EQ(net.sol().geoDim(), 1);

  EXPECT_EQ(net.geo().degree(0), 3);
  EXPECT_EQ(net.rhs().degree(0), 3);
  EXPECT_EQ(net.sol().degree(0), 3);

  EXPECT_EQ(net.geo().degree(1), 5);
  EXPECT_EQ(net.rhs().degree(1), 5);
  EXPECT_EQ(net.sol().degree(1), 5);

  EXPECT_EQ(net.geo().ncoeffs(0), 4);
  EXPECT_EQ(net.rhs().ncoeffs(0), 4);
  EXPECT_EQ(net.sol().ncoeffs(0), 4);

  EXPECT_EQ(net.geo().ncoeffs(1), 6);
  EXPECT_EQ(net.rhs().ncoeffs(1), 6);
  EXPECT_EQ(net.sol().ncoeffs(1), 6);
}

TEST(BSpline, IgaNet_UniformBSpline_3d)
{
  using real_t      = double;
  using optimizer_t = torch::optim::Adam;

  iganet::IgANet<real_t, optimizer_t,
                 iganet::UniformBSpline,
                 3, 5, 1> net({50,30,70}, // Number of neurons per layers
                              {4,6,3});   // Number of B-spline coefficients

  EXPECT_EQ(net.geo().parDim(), 3);
  EXPECT_EQ(net.rhs().parDim(), 3);
  EXPECT_EQ(net.sol().parDim(), 3);

  EXPECT_EQ(net.geo().geoDim(), 3);
  EXPECT_EQ(net.rhs().geoDim(), 1);
  EXPECT_EQ(net.sol().geoDim(), 1);

  EXPECT_EQ(net.geo().degree(0), 3);
  EXPECT_EQ(net.rhs().degree(0), 3);
  EXPECT_EQ(net.sol().degree(0), 3);

  EXPECT_EQ(net.geo().degree(1), 5);
  EXPECT_EQ(net.rhs().degree(1), 5);
  EXPECT_EQ(net.sol().degree(1), 5);

  EXPECT_EQ(net.geo().degree(2), 1);
  EXPECT_EQ(net.rhs().degree(2), 1);
  EXPECT_EQ(net.sol().degree(2), 1);

  EXPECT_EQ(net.geo().ncoeffs(0), 4);
  EXPECT_EQ(net.rhs().ncoeffs(0), 4);
  EXPECT_EQ(net.sol().ncoeffs(0), 4);

  EXPECT_EQ(net.geo().ncoeffs(1), 6);
  EXPECT_EQ(net.rhs().ncoeffs(1), 6);
  EXPECT_EQ(net.sol().ncoeffs(1), 6);

  EXPECT_EQ(net.geo().ncoeffs(2), 3);
  EXPECT_EQ(net.rhs().ncoeffs(2), 3);
  EXPECT_EQ(net.sol().ncoeffs(2), 3);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
