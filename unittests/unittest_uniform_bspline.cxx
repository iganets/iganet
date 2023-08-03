/**
   @file unittests/unittest_uniform_bspline.cxx

   @brief B-Spline unittests

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include <iganet.h>
#include <filesystem>
#include <iostream>

#include <gtest/gtest.h>

class BSplineTest
  : public ::testing::Test
{
protected:
  using real_t = double;
  iganet::Options<real_t> options;
};

TEST_F(BSplineTest, UniformBSpline_parDim1_geoDim1_degrees1)
{
  for (iganet::short_t n0=0; n0<2; n0++)
    EXPECT_THROW( (iganet::UniformBSpline<real_t, 1, 1>({n0})), std::runtime_error);

  iganet::UniformBSpline<real_t, 1, 1> bspline({2});
  EXPECT_EQ(bspline.parDim(),     1);
  EXPECT_EQ(bspline.geoDim(),     1);
  EXPECT_EQ(bspline.degree(0),    1);
  EXPECT_EQ(bspline.nknots(0),    4);
  EXPECT_EQ(bspline.ncoeffs(0),   2);
  EXPECT_EQ(bspline.ncumcoeffs(), 2);
}

TEST_F(BSplineTest, UniformBSpline_parDim1_geoDim1_degrees2)
{
  for (iganet::short_t n0=0; n0<3; n0++)
    EXPECT_THROW( (iganet::UniformBSpline<real_t, 1, 2>({n0})), std::runtime_error);

  iganet::UniformBSpline<real_t, 1, 2> bspline({3});
  EXPECT_EQ(bspline.parDim(),     1);
  EXPECT_EQ(bspline.geoDim(),     1);
  EXPECT_EQ(bspline.degree(0),    2);
  EXPECT_EQ(bspline.nknots(0),    6);
  EXPECT_EQ(bspline.ncoeffs(0),   3);
  EXPECT_EQ(bspline.ncumcoeffs(), 3);
}

TEST_F(BSplineTest, UniformBSpline_parDim1_geoDim1_degrees3)
{
  for (iganet::short_t n0=0; n0<4; n0++)
    EXPECT_THROW( (iganet::UniformBSpline<real_t, 1, 3>({n0})), std::runtime_error);

  iganet::UniformBSpline<real_t, 1, 3> bspline({4});
  EXPECT_EQ(bspline.parDim(),     1);
  EXPECT_EQ(bspline.geoDim(),     1);
  EXPECT_EQ(bspline.degree(0),    3);
  EXPECT_EQ(bspline.nknots(0),    8);
  EXPECT_EQ(bspline.ncoeffs(0),   4);
  EXPECT_EQ(bspline.ncumcoeffs(), 4);
}

TEST_F(BSplineTest, UniformBSpline_parDim1_geoDim2_degrees4)
{
  for (iganet::short_t n0=0; n0<5; n0++)
    EXPECT_THROW( (iganet::UniformBSpline<real_t, 2, 4>({n0})), std::runtime_error);

  iganet::UniformBSpline<real_t, 2, 4> bspline({5});
  EXPECT_EQ(bspline.parDim(),     1);
  EXPECT_EQ(bspline.geoDim(),     2);
  EXPECT_EQ(bspline.degree(0),    4);
  EXPECT_EQ(bspline.nknots(0),   10);
  EXPECT_EQ(bspline.ncoeffs(0),   5);
  EXPECT_EQ(bspline.ncumcoeffs(), 5);
}

TEST_F(BSplineTest, UniformBSpline_parDim1_geoDim3_degrees5)
{
  for (iganet::short_t n0=0; n0<6; n0++)
    EXPECT_THROW( (iganet::UniformBSpline<real_t, 3, 5>({n0})), std::runtime_error);

  iganet::UniformBSpline<real_t, 3, 5> bspline({6});
  EXPECT_EQ(bspline.parDim(),     1);
  EXPECT_EQ(bspline.geoDim(),     3);
  EXPECT_EQ(bspline.degree(0),    5);
  EXPECT_EQ(bspline.nknots(0),   12);
  EXPECT_EQ(bspline.ncoeffs(0),   6);
  EXPECT_EQ(bspline.ncumcoeffs(), 6);
}

TEST_F(BSplineTest, UniformBSpline_parDim1_geoDim4_degrees6)
{
  for (iganet::short_t n0=0; n0<7; n0++)
    EXPECT_THROW( (iganet::UniformBSpline<real_t, 4, 6>({n0})), std::runtime_error);

  iganet::UniformBSpline<real_t, 4, 6> bspline({7});
  EXPECT_EQ(bspline.parDim(),     1);
  EXPECT_EQ(bspline.geoDim(),     4);
  EXPECT_EQ(bspline.degree(0),    6);
  EXPECT_EQ(bspline.nknots(0),   14);
  EXPECT_EQ(bspline.ncoeffs(0),   7);
  EXPECT_EQ(bspline.ncumcoeffs(), 7);
}

TEST_F(BSplineTest, UniformBSpline_parDim2_geoDim1_degrees34)
{
  for (iganet::short_t n0=0; n0<4; n0++)
    for (iganet::short_t n1=0; n1<5; n1++)
      EXPECT_THROW( (iganet::UniformBSpline<real_t, 1, 3, 4>({n0, n1})), std::runtime_error);

  iganet::UniformBSpline<real_t, 1, 3, 4> bspline({4,5});
  EXPECT_EQ(bspline.parDim(),      2);
  EXPECT_EQ(bspline.geoDim(),      1);
  EXPECT_EQ(bspline.degree(0),     3);
  EXPECT_EQ(bspline.degree(1),     4);
  EXPECT_EQ(bspline.nknots(0),     8);
  EXPECT_EQ(bspline.nknots(1),    10);
  EXPECT_EQ(bspline.ncoeffs(0),    4);
  EXPECT_EQ(bspline.ncoeffs(1),    5);
  EXPECT_EQ(bspline.ncumcoeffs(), 20);
}

TEST_F(BSplineTest, UniformBSpline_parDim2_geoDim2_degrees34)
{
  for (iganet::short_t n0=0; n0<4; n0++)
    for (iganet::short_t n1=0; n1<5; n1++)
      EXPECT_THROW( (iganet::UniformBSpline<real_t, 2, 3, 4>({n0, n1})), std::runtime_error);

  iganet::UniformBSpline<real_t, 2, 3, 4> bspline({4,5});
  EXPECT_EQ(bspline.parDim(),      2);
  EXPECT_EQ(bspline.geoDim(),      2);
  EXPECT_EQ(bspline.degree(0),     3);
  EXPECT_EQ(bspline.degree(1),     4);
  EXPECT_EQ(bspline.nknots(0),     8);
  EXPECT_EQ(bspline.nknots(1),    10);
  EXPECT_EQ(bspline.ncoeffs(0),    4);
  EXPECT_EQ(bspline.ncoeffs(1),    5);
  EXPECT_EQ(bspline.ncumcoeffs(), 20);
}

TEST_F(BSplineTest, UniformBSpline_parDim2_geoDim3_degrees34)
{
  for (iganet::short_t n0=0; n0<4; n0++)
    for (iganet::short_t n1=0; n1<5; n1++)
      EXPECT_THROW( (iganet::UniformBSpline<real_t, 3, 3, 4>({n0, n1})), std::runtime_error);

  iganet::UniformBSpline<real_t, 3, 3, 4> bspline({4,5});
  EXPECT_EQ(bspline.parDim(),      2);
  EXPECT_EQ(bspline.geoDim(),      3);
  EXPECT_EQ(bspline.degree(0),     3);
  EXPECT_EQ(bspline.degree(1),     4);
  EXPECT_EQ(bspline.nknots(0),     8);
  EXPECT_EQ(bspline.nknots(1),    10);
  EXPECT_EQ(bspline.ncoeffs(0),    4);
  EXPECT_EQ(bspline.ncoeffs(1),    5);
  EXPECT_EQ(bspline.ncumcoeffs(), 20);
}

TEST_F(BSplineTest, UniformBSpline_parDim2_geoDim4_degrees34)
{
  for (iganet::short_t n0=0; n0<4; n0++)
    for (iganet::short_t n1=0; n1<5; n1++)
      EXPECT_THROW( (iganet::UniformBSpline<real_t, 4, 3, 4>({n0, n1})), std::runtime_error);

  iganet::UniformBSpline<real_t, 4, 3, 4> bspline({4,5});
  EXPECT_EQ(bspline.parDim(),      2);
  EXPECT_EQ(bspline.geoDim(),      4);
  EXPECT_EQ(bspline.degree(0),     3);
  EXPECT_EQ(bspline.degree(1),     4);
  EXPECT_EQ(bspline.nknots(0),     8);
  EXPECT_EQ(bspline.nknots(1),    10);
  EXPECT_EQ(bspline.ncoeffs(0),    4);
  EXPECT_EQ(bspline.ncoeffs(1),    5);
  EXPECT_EQ(bspline.ncumcoeffs(), 20);
}

TEST_F(BSplineTest, UniformBSpline_parDim3_geoDim1_degrees342)
{
  for (iganet::short_t n0=0; n0<4; n0++)
    for (iganet::short_t n1=0; n1<5; n1++)
      for (iganet::short_t n2=0; n2<3; n2++)
        EXPECT_THROW( (iganet::UniformBSpline<real_t, 1, 3, 4, 2>({n0, n1, n2})), std::runtime_error);

  iganet::UniformBSpline<real_t, 1, 3, 4, 2> bspline({4,5,3});
  EXPECT_EQ(bspline.parDim(),      3);
  EXPECT_EQ(bspline.geoDim(),      1);
  EXPECT_EQ(bspline.degree(0),     3);
  EXPECT_EQ(bspline.degree(1),     4);
  EXPECT_EQ(bspline.degree(2),     2);
  EXPECT_EQ(bspline.nknots(0),     8);
  EXPECT_EQ(bspline.nknots(1),    10);
  EXPECT_EQ(bspline.nknots(2),     6);
  EXPECT_EQ(bspline.ncoeffs(0),    4);
  EXPECT_EQ(bspline.ncoeffs(1),    5);
  EXPECT_EQ(bspline.ncoeffs(2),    3);
  EXPECT_EQ(bspline.ncumcoeffs(), 60);
}

TEST_F(BSplineTest, UniformBSpline_parDim2_geoDim3_degrees342)
{
  for (iganet::short_t n0=0; n0<4; n0++)
    for (iganet::short_t n1=0; n1<5; n1++)
      for (iganet::short_t n2=0; n2<3; n2++)
        EXPECT_THROW( (iganet::UniformBSpline<real_t, 2, 3, 4, 2>({n0, n1, n2})), std::runtime_error);

  iganet::UniformBSpline<real_t, 2, 3, 4, 2> bspline({4,5,3});
  EXPECT_EQ(bspline.parDim(),      3);
  EXPECT_EQ(bspline.geoDim(),      2);
  EXPECT_EQ(bspline.degree(0),     3);
  EXPECT_EQ(bspline.degree(1),     4);
  EXPECT_EQ(bspline.degree(2),     2);
  EXPECT_EQ(bspline.nknots(0),     8);
  EXPECT_EQ(bspline.nknots(1),    10);
  EXPECT_EQ(bspline.nknots(2),     6);
  EXPECT_EQ(bspline.ncoeffs(0),    4);
  EXPECT_EQ(bspline.ncoeffs(1),    5);
  EXPECT_EQ(bspline.ncoeffs(2),    3);
  EXPECT_EQ(bspline.ncumcoeffs(), 60);
}

TEST_F(BSplineTest, UniformBSpline_parDim3_geoDim3_degrees342)
{
  for (iganet::short_t n0=0; n0<4; n0++)
    for (iganet::short_t n1=0; n1<5; n1++)
      for (iganet::short_t n2=0; n2<3; n2++)
        EXPECT_THROW( (iganet::UniformBSpline<real_t, 3, 3, 4, 2>({n0, n1, n2})), std::runtime_error);

  iganet::UniformBSpline<real_t, 3, 3, 4, 2> bspline({4,5,3});
  EXPECT_EQ(bspline.parDim(),      3);
  EXPECT_EQ(bspline.geoDim(),      3);
  EXPECT_EQ(bspline.degree(0),     3);
  EXPECT_EQ(bspline.degree(1),     4);
  EXPECT_EQ(bspline.degree(2),     2);
  EXPECT_EQ(bspline.nknots(0),     8);
  EXPECT_EQ(bspline.nknots(1),    10);
  EXPECT_EQ(bspline.nknots(2),     6);
  EXPECT_EQ(bspline.ncoeffs(0),    4);
  EXPECT_EQ(bspline.ncoeffs(1),    5);
  EXPECT_EQ(bspline.ncoeffs(2),    3);
  EXPECT_EQ(bspline.ncumcoeffs(), 60);
}

TEST_F(BSplineTest, UniformBSpline_parDim3_geoDim4_degrees342)
{
  for (iganet::short_t n0=0; n0<4; n0++)
    for (iganet::short_t n1=0; n1<5; n1++)
      for (iganet::short_t n2=0; n2<3; n2++)
        EXPECT_THROW( (iganet::UniformBSpline<real_t, 4, 3, 4, 2>({n0, n1, n2})), std::runtime_error);

  iganet::UniformBSpline<real_t, 4, 3, 4, 2> bspline({4,5,3});
  EXPECT_EQ(bspline.parDim(),      3);
  EXPECT_EQ(bspline.geoDim(),      4);
  EXPECT_EQ(bspline.degree(0),     3);
  EXPECT_EQ(bspline.degree(1),     4);
  EXPECT_EQ(bspline.degree(2),     2);
  EXPECT_EQ(bspline.nknots(0),     8);
  EXPECT_EQ(bspline.nknots(1),    10);
  EXPECT_EQ(bspline.nknots(2),     6);
  EXPECT_EQ(bspline.ncoeffs(0),    4);
  EXPECT_EQ(bspline.ncoeffs(1),    5);
  EXPECT_EQ(bspline.ncoeffs(2),    3);
  EXPECT_EQ(bspline.ncumcoeffs(), 60);
}

TEST_F(BSplineTest, UniformBSpline_parDim4_geoDim1_degrees3421)
{
  for (iganet::short_t n0=0; n0<4; n0++)
    for (iganet::short_t n1=0; n1<5; n1++)
      for (iganet::short_t n2=0; n2<3; n2++)
        for (iganet::short_t n3=0; n3<2; n3++)
          EXPECT_THROW( (iganet::UniformBSpline<real_t, 1, 3, 4, 2, 1>({n0, n1, n2, n3})), std::runtime_error);

  iganet::UniformBSpline<real_t, 1, 3, 4, 2, 1> bspline({4,5,3,2});
  EXPECT_EQ(bspline.parDim(),       4);
  EXPECT_EQ(bspline.geoDim(),       1);
  EXPECT_EQ(bspline.degree(0),      3);
  EXPECT_EQ(bspline.degree(1),      4);
  EXPECT_EQ(bspline.degree(2),      2);
  EXPECT_EQ(bspline.degree(3),      1);
  EXPECT_EQ(bspline.nknots(0),      8);
  EXPECT_EQ(bspline.nknots(1),     10);
  EXPECT_EQ(bspline.nknots(2),      6);
  EXPECT_EQ(bspline.nknots(3),      4);
  EXPECT_EQ(bspline.ncoeffs(0),     4);
  EXPECT_EQ(bspline.ncoeffs(1),     5);
  EXPECT_EQ(bspline.ncoeffs(2),     3);
  EXPECT_EQ(bspline.ncoeffs(3),     2);
  EXPECT_EQ(bspline.ncumcoeffs(), 120);
}

TEST_F(BSplineTest, UniformBSpline_parDim4_geoDim2_degrees3421)
{
  for (iganet::short_t n0=0; n0<4; n0++)
    for (iganet::short_t n1=0; n1<5; n1++)
      for (iganet::short_t n2=0; n2<3; n2++)
        for (iganet::short_t n3=0; n3<2; n3++)
          EXPECT_THROW( (iganet::UniformBSpline<real_t, 2, 3, 4, 2, 1>({n0, n1, n2, n3})), std::runtime_error);

  iganet::UniformBSpline<real_t, 2, 3, 4, 2, 1> bspline({4,5,3,2});
  EXPECT_EQ(bspline.parDim(),       4);
  EXPECT_EQ(bspline.geoDim(),       2);
  EXPECT_EQ(bspline.degree(0),      3);
  EXPECT_EQ(bspline.degree(1),      4);
  EXPECT_EQ(bspline.degree(2),      2);
  EXPECT_EQ(bspline.degree(3),      1);
  EXPECT_EQ(bspline.nknots(0),      8);
  EXPECT_EQ(bspline.nknots(1),     10);
  EXPECT_EQ(bspline.nknots(2),      6);
  EXPECT_EQ(bspline.nknots(3),      4);
  EXPECT_EQ(bspline.ncoeffs(0),     4);
  EXPECT_EQ(bspline.ncoeffs(1),     5);
  EXPECT_EQ(bspline.ncoeffs(2),     3);
  EXPECT_EQ(bspline.ncoeffs(3),     2);
  EXPECT_EQ(bspline.ncumcoeffs(), 120);
}

TEST_F(BSplineTest, UniformBSpline_parDim4_geoDim3_degrees3421)
{
  for (iganet::short_t n0=0; n0<4; n0++)
    for (iganet::short_t n1=0; n1<5; n1++)
      for (iganet::short_t n2=0; n2<3; n2++)
        for (iganet::short_t n3=0; n3<2; n3++)
          EXPECT_THROW( (iganet::UniformBSpline<real_t, 3, 3, 4, 2, 1>({n0, n1, n2, n3})), std::runtime_error);

  iganet::UniformBSpline<real_t, 3, 3, 4, 2, 1> bspline({4,5,3,2});
  EXPECT_EQ(bspline.parDim(),       4);
  EXPECT_EQ(bspline.geoDim(),       3);
  EXPECT_EQ(bspline.degree(0),      3);
  EXPECT_EQ(bspline.degree(1),      4);
  EXPECT_EQ(bspline.degree(2),      2);
  EXPECT_EQ(bspline.degree(3),      1);
  EXPECT_EQ(bspline.nknots(0),      8);
  EXPECT_EQ(bspline.nknots(1),     10);
  EXPECT_EQ(bspline.nknots(2),      6);
  EXPECT_EQ(bspline.nknots(3),      4);
  EXPECT_EQ(bspline.ncoeffs(0),     4);
  EXPECT_EQ(bspline.ncoeffs(1),     5);
  EXPECT_EQ(bspline.ncoeffs(2),     3);
  EXPECT_EQ(bspline.ncoeffs(3),     2);
  EXPECT_EQ(bspline.ncumcoeffs(), 120);
}

TEST_F(BSplineTest, UniformBSpline_parDim4_geoDim4_degrees3421)
{
  for (iganet::short_t n0=0; n0<4; n0++)
    for (iganet::short_t n1=0; n1<5; n1++)
      for (iganet::short_t n2=0; n2<3; n2++)
        for (iganet::short_t n3=0; n3<2; n3++)
          EXPECT_THROW( (iganet::UniformBSpline<real_t, 4, 3, 4, 2, 1>({n0, n1, n2, n3})), std::runtime_error);

  iganet::UniformBSpline<real_t, 4, 3, 4, 2, 1> bspline({4,5,3,2});
  EXPECT_EQ(bspline.parDim(),       4);
  EXPECT_EQ(bspline.geoDim(),       4);
  EXPECT_EQ(bspline.degree(0),      3);
  EXPECT_EQ(bspline.degree(1),      4);
  EXPECT_EQ(bspline.degree(2),      2);
  EXPECT_EQ(bspline.degree(3),      1);
  EXPECT_EQ(bspline.nknots(0),      8);
  EXPECT_EQ(bspline.nknots(1),     10);
  EXPECT_EQ(bspline.nknots(2),      6);
  EXPECT_EQ(bspline.nknots(3),      4);
  EXPECT_EQ(bspline.ncoeffs(0),     4);
  EXPECT_EQ(bspline.ncoeffs(1),     5);
  EXPECT_EQ(bspline.ncoeffs(2),     3);
  EXPECT_EQ(bspline.ncoeffs(3),     2);
  EXPECT_EQ(bspline.ncumcoeffs(), 120);
}

TEST_F(BSplineTest, UniformBSpline_init)
{
  {
    iganet::UniformBSpline<real_t, 1, 1> bspline({5}, iganet::init::zeros, options);
    EXPECT_TRUE(torch::allclose(bspline.coeffs(0), torch::zeros(5, options)));
  }

  {
    iganet::UniformBSpline<real_t, 1, 1> bspline({5}, iganet::init::ones, options);
    EXPECT_TRUE(torch::allclose(bspline.coeffs(0), torch::ones(5, options)));
  }

  {
    iganet::UniformBSpline<real_t, 1, 1> bspline({5}, iganet::init::linear, options);
    EXPECT_TRUE(torch::allclose(bspline.coeffs(0), torch::linspace(0, 1, 5, options)));
  }

  {
    iganet::UniformBSpline<real_t, 1, 1> bspline({5}, iganet::init::greville, options);
    EXPECT_TRUE(torch::allclose(bspline.coeffs(0), torch::linspace(0, 1, 5, options)));
  }

  {
    iganet::UniformBSpline<real_t, 2, 1> bspline({5}, iganet::init::zeros, options);
    EXPECT_TRUE(torch::allclose(bspline.coeffs(0), torch::zeros(5, options)));
    EXPECT_TRUE(torch::allclose(bspline.coeffs(1), torch::zeros(5, options)));
  }

  {
    iganet::UniformBSpline<real_t, 2, 1> bspline({5}, iganet::init::ones, options);
    EXPECT_TRUE(torch::allclose(bspline.coeffs(0), torch::ones(5, options)));
    EXPECT_TRUE(torch::allclose(bspline.coeffs(1), torch::ones(5, options)));
  }

  {
    iganet::UniformBSpline<real_t, 2, 1> bspline({5}, iganet::init::linear, options);
    EXPECT_TRUE(torch::allclose(bspline.coeffs(0), torch::linspace(0, 1, 5, options)));
    EXPECT_TRUE(torch::allclose(bspline.coeffs(1), torch::ones(5, options)));
  }

  {
    iganet::UniformBSpline<real_t, 2, 1> bspline({5}, iganet::init::greville, options);
    EXPECT_TRUE(torch::allclose(bspline.coeffs(0), torch::linspace(0, 1, 5, options)));
    EXPECT_TRUE(torch::allclose(bspline.coeffs(1), torch::ones(5, options)));
  }

  {
    iganet::UniformBSpline<real_t, 2, 2, 2> bspline({5, 8}, iganet::init::zeros, options);
    EXPECT_TRUE(torch::allclose(bspline.coeffs(0), torch::zeros(40, options)));
    EXPECT_TRUE(torch::allclose(bspline.coeffs(1), torch::zeros(40, options)));
  }

  {
    iganet::UniformBSpline<real_t, 2, 2, 2> bspline({5, 8}, iganet::init::ones, options);
    EXPECT_TRUE(torch::allclose(bspline.coeffs(0), torch::ones(40, options)));
    EXPECT_TRUE(torch::allclose(bspline.coeffs(1), torch::ones(40, options)));
  }

  {
    iganet::UniformBSpline<real_t, 2, 2, 2> bspline({5, 8}, iganet::init::linear, options);
    EXPECT_TRUE(torch::allclose(bspline.coeffs(0),
                                torch::linspace(0, 1, 5, options).repeat(8)));
    EXPECT_TRUE(torch::allclose(bspline.coeffs(1),
                                torch::linspace(0, 1, 8, options).repeat_interleave(5)));
  }

  {
    iganet::UniformBSpline<real_t, 2, 1, 1> bspline({5, 8}, iganet::init::greville, options);
    EXPECT_TRUE(torch::allclose(bspline.coeffs(0),
                                torch::linspace(0, 1, 5, options).repeat(8)));
    EXPECT_TRUE(torch::allclose(bspline.coeffs(1),
                                torch::linspace(0, 1, 8, options).repeat_interleave(5)));
  }

  {
    iganet::UniformBSpline<real_t, 3, 2, 2> bspline({5, 8}, iganet::init::zeros, options);
    EXPECT_TRUE(torch::allclose(bspline.coeffs(0), torch::zeros(40, options)));
    EXPECT_TRUE(torch::allclose(bspline.coeffs(1), torch::zeros(40, options)));
    EXPECT_TRUE(torch::allclose(bspline.coeffs(2), torch::zeros(40, options)));
  }

  {
    iganet::UniformBSpline<real_t, 3, 2, 2> bspline({5, 8}, iganet::init::ones, options);
    EXPECT_TRUE(torch::allclose(bspline.coeffs(0), torch::ones(40, options)));
    EXPECT_TRUE(torch::allclose(bspline.coeffs(1), torch::ones(40, options)));
    EXPECT_TRUE(torch::allclose(bspline.coeffs(2), torch::ones(40, options)));
  }

  {
    iganet::UniformBSpline<real_t, 3, 2, 2> bspline({5, 8}, iganet::init::linear, options);
    EXPECT_TRUE(torch::allclose(bspline.coeffs(0),
                                torch::linspace(0, 1, 5, options).repeat(8)));
    EXPECT_TRUE(torch::allclose(bspline.coeffs(1),
                                torch::linspace(0, 1, 8, options).repeat_interleave(5)));
    EXPECT_TRUE(torch::allclose(bspline.coeffs(2), torch::ones(40, options)));
  }

  {
    iganet::UniformBSpline<real_t, 3, 1, 1> bspline({5, 8}, iganet::init::greville, options);
    EXPECT_TRUE(torch::allclose(bspline.coeffs(0),
                                torch::linspace(0, 1, 5, options).repeat(8)));
    EXPECT_TRUE(torch::allclose(bspline.coeffs(1),
                                torch::linspace(0, 1, 8, options).repeat_interleave(5)));
    EXPECT_TRUE(torch::allclose(bspline.coeffs(2), torch::ones(40, options)));
  }

  {
    iganet::UniformBSpline<real_t, 4, 2, 2> bspline({5, 8}, iganet::init::zeros, options);
    EXPECT_TRUE(torch::allclose(bspline.coeffs(0), torch::zeros(40, options)));
    EXPECT_TRUE(torch::allclose(bspline.coeffs(1), torch::zeros(40, options)));
    EXPECT_TRUE(torch::allclose(bspline.coeffs(2), torch::zeros(40, options)));
    EXPECT_TRUE(torch::allclose(bspline.coeffs(3), torch::zeros(40, options)));
  }

  {
    iganet::UniformBSpline<real_t, 4, 2, 2> bspline({5, 8}, iganet::init::ones, options);
    EXPECT_TRUE(torch::allclose(bspline.coeffs(0), torch::ones(40, options)));
    EXPECT_TRUE(torch::allclose(bspline.coeffs(1), torch::ones(40, options)));
    EXPECT_TRUE(torch::allclose(bspline.coeffs(2), torch::ones(40, options)));
    EXPECT_TRUE(torch::allclose(bspline.coeffs(3), torch::ones(40, options)));
  }

  {
    iganet::UniformBSpline<real_t, 4, 2, 2> bspline({5, 8}, iganet::init::linear, options);
    EXPECT_TRUE(torch::allclose(bspline.coeffs(0),
                                torch::linspace(0, 1, 5, options).repeat(8)));
    EXPECT_TRUE(torch::allclose(bspline.coeffs(1),
                                torch::linspace(0, 1, 8, options).repeat_interleave(5)));
    EXPECT_TRUE(torch::allclose(bspline.coeffs(2), torch::ones(40, options)));
    EXPECT_TRUE(torch::allclose(bspline.coeffs(3), torch::ones(40, options)));
  }

  {
    iganet::UniformBSpline<real_t, 4, 1, 1> bspline({5, 8}, iganet::init::greville, options);
    EXPECT_TRUE(torch::allclose(bspline.coeffs(0),
                                torch::linspace(0, 1, 5, options).repeat(8)));
    EXPECT_TRUE(torch::allclose(bspline.coeffs(1),
                                torch::linspace(0, 1, 8, options).repeat_interleave(5)));
    EXPECT_TRUE(torch::allclose(bspline.coeffs(2), torch::ones(40, options)));
    EXPECT_TRUE(torch::allclose(bspline.coeffs(3), torch::ones(40, options)));
  }
}

TEST_F(BSplineTest, UniformBSpline_read_write)
{
  std::filesystem::path filename = std::filesystem::temp_directory_path() / std::to_string(rand());
  iganet::UniformBSpline<real_t, 3, 3, 4> bspline_out({4,5});
  bspline_out.save(filename.c_str());
  iganet::UniformBSpline<real_t, 3, 3, 4> bspline_in;
  bspline_in.load(filename.c_str());
  std::filesystem::remove(filename);

  EXPECT_EQ( (bspline_in == bspline_out), true);
  EXPECT_EQ( (bspline_in != bspline_out), false);
}

TEST_F(BSplineTest, UniformBSpline_uniform_refine)
{
  {
    iganet::UniformBSpline<real_t, 3, 3, 4> bspline({4,5});
    iganet::UniformBSpline<real_t, 3, 3, 4> bspline_ref({5,6});
    bspline.uniform_refine();
    
    EXPECT_EQ( bspline.isclose(bspline_ref), true);
  }

  {
    iganet::UniformBSpline<real_t, 3, 3, 4> bspline({4,5});
    iganet::UniformBSpline<real_t, 3, 3, 4> bspline_ref({7,8});
    bspline.uniform_refine(2);
    
    EXPECT_EQ( bspline.isclose(bspline_ref), true);
  }

  {
    iganet::UniformBSpline<real_t, 3, 3, 4> bspline({4,5});
    iganet::UniformBSpline<real_t, 3, 3, 4> bspline_ref({5,5});
    bspline.uniform_refine(1, 0);
    
    EXPECT_EQ( bspline.isclose(bspline_ref), true);
  }
  
  {
    iganet::UniformBSpline<real_t, 3, 3, 4> bspline({4,5});
    iganet::UniformBSpline<real_t, 3, 3, 4> bspline_ref({5,8});
    bspline.uniform_refine(1, 0).uniform_refine(2, 1);
    
    EXPECT_EQ( bspline.isclose(bspline_ref), true);
  }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  iganet::init();
  return RUN_ALL_TESTS();
}
