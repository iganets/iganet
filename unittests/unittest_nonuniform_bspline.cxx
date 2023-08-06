/**
   @file unittests/unittest_nonuniform_bspline.cxx

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

TEST_F(BSplineTest, NonUniformBSpline_parDim1_geoDim1_degrees1)
{
  EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 1, 1>( {{{0.0, 0.0, 1.0}}} )), std::runtime_error);
  iganet::NonUniformBSpline<real_t, 1, 1> bspline( {{{0.0, 0.0, 0.5, 1.0, 1.0}}} );
  EXPECT_EQ(bspline.parDim(),     1);
  EXPECT_EQ(bspline.geoDim(),     1);
  EXPECT_EQ(bspline.degree(0),    1);
  EXPECT_EQ(bspline.nknots(0),    5);
  EXPECT_EQ(bspline.ncoeffs(0),   3);
  EXPECT_EQ(bspline.ncumcoeffs(), 3);
  EXPECT_TRUE(bspline.is_nonuniform());
  EXPECT_FALSE(bspline.is_uniform());
}

TEST_F(BSplineTest, NonUniformBSpline_parDim1_geoDim2_degrees1)
{
  EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 2, 1>( {{{0.0, 0.0, 1.0}}} )), std::runtime_error);
  iganet::NonUniformBSpline<real_t, 2, 1> bspline( {{{0.0, 0.0, 0.5, 1.0, 1.0}}} );
  EXPECT_EQ(bspline.parDim(),     1);
  EXPECT_EQ(bspline.geoDim(),     2);
  EXPECT_EQ(bspline.degree(0),    1);
  EXPECT_EQ(bspline.nknots(0),    5);
  EXPECT_EQ(bspline.ncoeffs(0),   3);
  EXPECT_EQ(bspline.ncumcoeffs(), 3);
}

TEST_F(BSplineTest, NonUniformBSpline_parDim1_geoDim3_degrees1)
{
  EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 3, 1>( {{{0.0, 0.0, 1.0}}} )), std::runtime_error);
  iganet::NonUniformBSpline<real_t, 3, 1> bspline( {{{0.0, 0.0, 0.5, 1.0, 1.0}}} );
  EXPECT_EQ(bspline.parDim(),     1);
  EXPECT_EQ(bspline.geoDim(),     3);
  EXPECT_EQ(bspline.degree(0),    1);
  EXPECT_EQ(bspline.nknots(0),    5);
  EXPECT_EQ(bspline.ncoeffs(0),   3);
  EXPECT_EQ(bspline.ncumcoeffs(), 3);
}

TEST_F(BSplineTest, NonUniformBSpline_parDim1_geoDim4_degrees1)
{
  EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 4, 1>( {{{0.0, 0.0, 1.0}}} )), std::runtime_error);
  iganet::NonUniformBSpline<real_t, 4, 1> bspline( {{{0.0, 0.0, 0.5, 1.0, 1.0}}} );
  EXPECT_EQ(bspline.parDim(),     1);
  EXPECT_EQ(bspline.geoDim(),     4);
  EXPECT_EQ(bspline.degree(0),    1);
  EXPECT_EQ(bspline.nknots(0),    5);
  EXPECT_EQ(bspline.ncoeffs(0),   3);
  EXPECT_EQ(bspline.ncumcoeffs(), 3);
}

TEST_F(BSplineTest, NonUniformBSpline_parDim2_geoDim1_degrees12)
{
  iganet::NonUniformBSpline<real_t, 1, 1, 2> bspline( {{{0.0, 0.0, 0.5, 1.0, 1.0},
                                                        {0.0, 0.0, 0.0, 1.0, 1.0, 1.0}}} );
  EXPECT_EQ(bspline.parDim(),     2);
  EXPECT_EQ(bspline.geoDim(),     1);
  EXPECT_EQ(bspline.degree(0),    1);
  EXPECT_EQ(bspline.degree(1),    2);
  EXPECT_EQ(bspline.nknots(0),    5);
  EXPECT_EQ(bspline.nknots(1),    6);
  EXPECT_EQ(bspline.ncoeffs(0),   3);
  EXPECT_EQ(bspline.ncoeffs(1),   3);
  EXPECT_EQ(bspline.ncumcoeffs(), 9);
}

TEST_F(BSplineTest, NonUniformBSpline_parDim2_geoDim2_degrees12)
{
  iganet::NonUniformBSpline<real_t, 2, 1, 2> bspline( {{{0.0, 0.0, 0.5, 1.0, 1.0},
                                                        {0.0, 0.0, 0.0, 1.0, 1.0, 1.0}}} );
  EXPECT_EQ(bspline.parDim(),     2);
  EXPECT_EQ(bspline.geoDim(),     2);
  EXPECT_EQ(bspline.degree(0),    1);
  EXPECT_EQ(bspline.degree(1),    2);
  EXPECT_EQ(bspline.nknots(0),    5);
  EXPECT_EQ(bspline.nknots(1),    6);
  EXPECT_EQ(bspline.ncoeffs(0),   3);
  EXPECT_EQ(bspline.ncoeffs(1),   3);
  EXPECT_EQ(bspline.ncumcoeffs(), 9);
}

TEST_F(BSplineTest, NonUniformBSpline_parDim2_geoDim3_degrees12)
{
  iganet::NonUniformBSpline<real_t, 3, 1, 2> bspline( {{{0.0, 0.0, 0.5, 1.0, 1.0},
                                                        {0.0, 0.0, 0.0, 1.0, 1.0, 1.0}}} );
  EXPECT_EQ(bspline.parDim(),     2);
  EXPECT_EQ(bspline.geoDim(),     3);
  EXPECT_EQ(bspline.degree(0),    1);
  EXPECT_EQ(bspline.degree(1),    2);
  EXPECT_EQ(bspline.nknots(0),    5);
  EXPECT_EQ(bspline.nknots(1),    6);
  EXPECT_EQ(bspline.ncoeffs(0),   3);
  EXPECT_EQ(bspline.ncoeffs(1),   3);
  EXPECT_EQ(bspline.ncumcoeffs(), 9);
}

TEST_F(BSplineTest, NonUniformBSpline_parDim2_geoDim4_degrees12)
{
  iganet::NonUniformBSpline<real_t, 4, 1, 2> bspline( {{{0.0, 0.0, 0.5, 1.0, 1.0},
                                                        {0.0, 0.0, 0.0, 1.0, 1.0, 1.0}}} );
  EXPECT_EQ(bspline.parDim(),     2);
  EXPECT_EQ(bspline.geoDim(),     4);
  EXPECT_EQ(bspline.degree(0),    1);
  EXPECT_EQ(bspline.degree(1),    2);
  EXPECT_EQ(bspline.nknots(0),    5);
  EXPECT_EQ(bspline.nknots(1),    6);
  EXPECT_EQ(bspline.ncoeffs(0),   3);
  EXPECT_EQ(bspline.ncoeffs(1),   3);
  EXPECT_EQ(bspline.ncumcoeffs(), 9);
}

TEST_F(BSplineTest, NonUniformBSpline_parDim3_geoDim1_degrees123)
{
  iganet::NonUniformBSpline<real_t, 1, 1, 2, 3> bspline( {{{0.0, 0.0, 0.5, 1.0, 1.0},
                                                           {0.0, 0.0, 0.0, 1.0, 1.0, 1.0},
                                                           {0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0}}} );
  EXPECT_EQ(bspline.parDim(),      3);
  EXPECT_EQ(bspline.geoDim(),      1);
  EXPECT_EQ(bspline.degree(0),     1);
  EXPECT_EQ(bspline.degree(1),     2);
  EXPECT_EQ(bspline.degree(2),     3);
  EXPECT_EQ(bspline.nknots(0),     5);
  EXPECT_EQ(bspline.nknots(1),     6);
  EXPECT_EQ(bspline.nknots(2),     9);
  EXPECT_EQ(bspline.ncoeffs(0),    3);
  EXPECT_EQ(bspline.ncoeffs(1),    3);
  EXPECT_EQ(bspline.ncoeffs(2),    5);
  EXPECT_EQ(bspline.ncumcoeffs(), 45);
}

TEST_F(BSplineTest, NonUniformBSpline_parDim3_geoDim2_degrees123)
{
  iganet::NonUniformBSpline<real_t, 2, 1, 2, 3> bspline( {{{0.0, 0.0, 0.5, 1.0, 1.0},
                                                           {0.0, 0.0, 0.0, 1.0, 1.0, 1.0},
                                                           {0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0}}} );
  EXPECT_EQ(bspline.parDim(),      3);
  EXPECT_EQ(bspline.geoDim(),      2);
  EXPECT_EQ(bspline.degree(0),     1);
  EXPECT_EQ(bspline.degree(1),     2);
  EXPECT_EQ(bspline.degree(2),     3);
  EXPECT_EQ(bspline.nknots(0),     5);
  EXPECT_EQ(bspline.nknots(1),     6);
  EXPECT_EQ(bspline.nknots(2),     9);
  EXPECT_EQ(bspline.ncoeffs(0),    3);
  EXPECT_EQ(bspline.ncoeffs(1),    3);
  EXPECT_EQ(bspline.ncoeffs(2),    5);
  EXPECT_EQ(bspline.ncumcoeffs(), 45);
}

TEST_F(BSplineTest, NonUniformBSpline_parDim3_geoDim3_degrees123)
{
  iganet::NonUniformBSpline<real_t, 3, 1, 2, 3> bspline( {{{0.0, 0.0, 0.5, 1.0, 1.0},
                                                           {0.0, 0.0, 0.0, 1.0, 1.0, 1.0},
                                                           {0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0}}} );
  EXPECT_EQ(bspline.parDim(),      3);
  EXPECT_EQ(bspline.geoDim(),      3);
  EXPECT_EQ(bspline.degree(0),     1);
  EXPECT_EQ(bspline.degree(1),     2);
  EXPECT_EQ(bspline.degree(2),     3);
  EXPECT_EQ(bspline.nknots(0),     5);
  EXPECT_EQ(bspline.nknots(1),     6);
  EXPECT_EQ(bspline.nknots(2),     9);
  EXPECT_EQ(bspline.ncoeffs(0),    3);
  EXPECT_EQ(bspline.ncoeffs(1),    3);
  EXPECT_EQ(bspline.ncoeffs(2),    5);
  EXPECT_EQ(bspline.ncumcoeffs(), 45);
}

TEST_F(BSplineTest, NonUniformBSpline_parDim3_geoDim4_degrees123)
{
  iganet::NonUniformBSpline<real_t, 4, 1, 2, 3> bspline( {{{0.0, 0.0, 0.5, 1.0, 1.0},
                                                           {0.0, 0.0, 0.0, 1.0, 1.0, 1.0},
                                                           {0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0}}} );
  EXPECT_EQ(bspline.parDim(),      3);
  EXPECT_EQ(bspline.geoDim(),      4);
  EXPECT_EQ(bspline.degree(0),     1);
  EXPECT_EQ(bspline.degree(1),     2);
  EXPECT_EQ(bspline.degree(2),     3);
  EXPECT_EQ(bspline.nknots(0),     5);
  EXPECT_EQ(bspline.nknots(1),     6);
  EXPECT_EQ(bspline.nknots(2),     9);
  EXPECT_EQ(bspline.ncoeffs(0),    3);
  EXPECT_EQ(bspline.ncoeffs(1),    3);
  EXPECT_EQ(bspline.ncoeffs(2),    5);
  EXPECT_EQ(bspline.ncumcoeffs(), 45);
}

TEST_F(BSplineTest, NonUniformBSpline_parDim4_geoDim1_degrees1234)
{
  iganet::NonUniformBSpline<real_t, 1, 1, 2, 3, 4> bspline( {{{0.0, 0.0, 0.5, 1.0, 1.0},
                                                              {0.0, 0.0, 0.0, 1.0, 1.0, 1.0},
                                                              {0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0},
                                                              {0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0}}} );
  EXPECT_EQ(bspline.parDim(),       4);
  EXPECT_EQ(bspline.geoDim(),       1);
  EXPECT_EQ(bspline.degree(0),      1);
  EXPECT_EQ(bspline.degree(1),      2);
  EXPECT_EQ(bspline.degree(2),      3);
  EXPECT_EQ(bspline.degree(3),      4);
  EXPECT_EQ(bspline.nknots(0),      5);
  EXPECT_EQ(bspline.nknots(1),      6);
  EXPECT_EQ(bspline.nknots(2),      9);
  EXPECT_EQ(bspline.nknots(3),     11);
  EXPECT_EQ(bspline.ncoeffs(0),     3);
  EXPECT_EQ(bspline.ncoeffs(1),     3);
  EXPECT_EQ(bspline.ncoeffs(2),     5);
  EXPECT_EQ(bspline.ncoeffs(3),     6);
  EXPECT_EQ(bspline.ncumcoeffs(), 270);
}

TEST_F(BSplineTest, NonUniformBSpline_parDim4_geoDim2_degrees1234)
{
  iganet::NonUniformBSpline<real_t, 2, 1, 2, 3, 4> bspline( {{{0.0, 0.0, 0.5, 1.0, 1.0},
                                                              {0.0, 0.0, 0.0, 1.0, 1.0, 1.0},
                                                              {0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0},
                                                              {0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0}}} );
  EXPECT_EQ(bspline.parDim(),       4);
  EXPECT_EQ(bspline.geoDim(),       2);
  EXPECT_EQ(bspline.degree(0),      1);
  EXPECT_EQ(bspline.degree(1),      2);
  EXPECT_EQ(bspline.degree(2),      3);
  EXPECT_EQ(bspline.degree(3),      4);
  EXPECT_EQ(bspline.nknots(0),      5);
  EXPECT_EQ(bspline.nknots(1),      6);
  EXPECT_EQ(bspline.nknots(2),      9);
  EXPECT_EQ(bspline.nknots(3),     11);
  EXPECT_EQ(bspline.ncoeffs(0),     3);
  EXPECT_EQ(bspline.ncoeffs(1),     3);
  EXPECT_EQ(bspline.ncoeffs(2),     5);
  EXPECT_EQ(bspline.ncoeffs(3),     6);
  EXPECT_EQ(bspline.ncumcoeffs(), 270);
}

TEST_F(BSplineTest, NonUniformBSpline_parDim4_geoDim3_degrees1234)
{
  iganet::NonUniformBSpline<real_t, 3, 1, 2, 3, 4> bspline( {{{0.0, 0.0, 0.5, 1.0, 1.0},
                                                              {0.0, 0.0, 0.0, 1.0, 1.0, 1.0},
                                                              {0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0},
                                                              {0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0}}} );
  EXPECT_EQ(bspline.parDim(),       4);
  EXPECT_EQ(bspline.geoDim(),       3);
  EXPECT_EQ(bspline.degree(0),      1);
  EXPECT_EQ(bspline.degree(1),      2);
  EXPECT_EQ(bspline.degree(2),      3);
  EXPECT_EQ(bspline.degree(3),      4);
  EXPECT_EQ(bspline.nknots(0),      5);
  EXPECT_EQ(bspline.nknots(1),      6);
  EXPECT_EQ(bspline.nknots(2),      9);
  EXPECT_EQ(bspline.nknots(3),     11);
  EXPECT_EQ(bspline.ncoeffs(0),     3);
  EXPECT_EQ(bspline.ncoeffs(1),     3);
  EXPECT_EQ(bspline.ncoeffs(2),     5);
  EXPECT_EQ(bspline.ncoeffs(3),     6);
  EXPECT_EQ(bspline.ncumcoeffs(), 270);
}

TEST_F(BSplineTest, NonUniformBSpline_parDim4_geoDim4_degrees1234)
{
  iganet::NonUniformBSpline<real_t, 4, 1, 2, 3, 4> bspline( {{{0.0, 0.0, 0.5, 1.0, 1.0},
                                                              {0.0, 0.0, 0.0, 1.0, 1.0, 1.0},
                                                              {0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0},
                                                              {0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0}}} );
  EXPECT_EQ(bspline.parDim(),       4);
  EXPECT_EQ(bspline.geoDim(),       4);
  EXPECT_EQ(bspline.degree(0),      1);
  EXPECT_EQ(bspline.degree(1),      2);
  EXPECT_EQ(bspline.degree(2),      3);
  EXPECT_EQ(bspline.degree(3),      4);
  EXPECT_EQ(bspline.nknots(0),      5);
  EXPECT_EQ(bspline.nknots(1),      6);
  EXPECT_EQ(bspline.nknots(2),      9);
  EXPECT_EQ(bspline.nknots(3),     11);
  EXPECT_EQ(bspline.ncoeffs(0),     3);
  EXPECT_EQ(bspline.ncoeffs(1),     3);
  EXPECT_EQ(bspline.ncoeffs(2),     5);
  EXPECT_EQ(bspline.ncoeffs(3),     6);
  EXPECT_EQ(bspline.ncumcoeffs(), 270);
}

TEST_F(BSplineTest, NonUniformBSpline_init)
{
  {
    iganet::NonUniformBSpline<real_t, 1, 1> bspline({{{0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0}}},
                                                    iganet::init::zeros, options);
    EXPECT_TRUE(torch::allclose(bspline.coeffs(0), torch::zeros(5, options)));
  }

  {
    iganet::NonUniformBSpline<real_t, 1, 1> bspline({{{0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0}}},
                                                    iganet::init::ones, options);
    EXPECT_TRUE(torch::allclose(bspline.coeffs(0), torch::ones(5, options)));
  }
    
  {
    iganet::NonUniformBSpline<real_t, 1, 1> bspline({{{0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0}}},
                                                    iganet::init::linear, options);
    EXPECT_TRUE(torch::allclose(bspline.coeffs(0), torch::linspace(0, 1, 5, options)));
  }

  {
    iganet::NonUniformBSpline<real_t, 1, 1> bspline({{{0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0}}},
                                                    iganet::init::greville, options);
    EXPECT_TRUE(torch::allclose(bspline.coeffs(0), torch::linspace(0, 1, 5, options)));
  }

  {
    iganet::NonUniformBSpline<real_t, 2, 1> bspline({{{0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0}}},
                                                    iganet::init::zeros, options);
    EXPECT_TRUE(torch::allclose(bspline.coeffs(0), torch::zeros(5, options)));
    EXPECT_TRUE(torch::allclose(bspline.coeffs(1), torch::zeros(5, options)));
  }

  {
    iganet::NonUniformBSpline<real_t, 2, 1> bspline({{{0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0}}},
                                                    iganet::init::ones, options);
    EXPECT_TRUE(torch::allclose(bspline.coeffs(0), torch::ones(5, options)));
    EXPECT_TRUE(torch::allclose(bspline.coeffs(1), torch::ones(5, options)));
  }

  {
    iganet::NonUniformBSpline<real_t, 2, 1> bspline({{{0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0}}},
                                                    iganet::init::linear, options);
    EXPECT_TRUE(torch::allclose(bspline.coeffs(0), torch::linspace(0, 1, 5, options)));
    EXPECT_TRUE(torch::allclose(bspline.coeffs(1), torch::ones(5, options)));
  }

  {
    iganet::NonUniformBSpline<real_t, 2, 1> bspline({{{0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0}}},
                                                    iganet::init::greville, options);
    EXPECT_TRUE(torch::allclose(bspline.coeffs(0), torch::linspace(0, 1, 5, options)));
    EXPECT_TRUE(torch::allclose(bspline.coeffs(1), torch::ones(5, options)));
  }
  
  {
    iganet::NonUniformBSpline<real_t, 2, 2, 2> bspline({{{0.0, 0.0, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.0, 1.0},
                                                         {0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0}}},
      iganet::init::zeros, options);
    EXPECT_TRUE(torch::allclose(bspline.coeffs(0), torch::zeros(28, options)));
    EXPECT_TRUE(torch::allclose(bspline.coeffs(1), torch::zeros(28, options)));
  }

  {
    iganet::NonUniformBSpline<real_t, 2, 2, 2> bspline({{{0.0, 0.0, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.0, 1.0},
                                                         {0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0}}},
      iganet::init::ones, options);
    EXPECT_TRUE(torch::allclose(bspline.coeffs(0), torch::ones(28, options)));
    EXPECT_TRUE(torch::allclose(bspline.coeffs(1), torch::ones(28, options)));
  }

  {
    iganet::NonUniformBSpline<real_t, 2, 2, 2> bspline({{{0.0, 0.0, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.0, 1.0},
                                                         {0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0}}},
      iganet::init::linear, options);
    EXPECT_TRUE(torch::allclose(bspline.coeffs(0),
                                torch::linspace(0, 1, 7, options).repeat(4)));
    EXPECT_TRUE(torch::allclose(bspline.coeffs(1),
                                torch::linspace(0, 1, 4, options).repeat_interleave(7)));
  }
  
  {
    iganet::NonUniformBSpline<real_t, 2, 1, 1> bspline({{{0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0},
                                                         {0.0, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.0}}},
      iganet::init::greville, options);
    EXPECT_TRUE(torch::allclose(bspline.coeffs(0),
                                torch::linspace(0, 1, 5, options).repeat(6)));
    EXPECT_TRUE(torch::allclose(bspline.coeffs(1),
                                torch::linspace(0, 1, 6, options).repeat_interleave(5)));
  }
  
  {
    iganet::NonUniformBSpline<real_t, 3, 2, 2> bspline({{{0.0, 0.0, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.0, 1.0},
                                                         {0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0}}},
      iganet::init::zeros, options);
    EXPECT_TRUE(torch::allclose(bspline.coeffs(0), torch::zeros(28, options)));
    EXPECT_TRUE(torch::allclose(bspline.coeffs(1), torch::zeros(28, options)));
    EXPECT_TRUE(torch::allclose(bspline.coeffs(2), torch::zeros(28, options)));
  }

  {
    iganet::NonUniformBSpline<real_t, 3, 2, 2> bspline({{{0.0, 0.0, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.0, 1.0},
                                                         {0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0}}},
      iganet::init::ones, options);
    EXPECT_TRUE(torch::allclose(bspline.coeffs(0), torch::ones(28, options)));
    EXPECT_TRUE(torch::allclose(bspline.coeffs(1), torch::ones(28, options)));
    EXPECT_TRUE(torch::allclose(bspline.coeffs(2), torch::ones(28, options)));
  }

  {
    iganet::NonUniformBSpline<real_t, 3, 2, 2> bspline({{{0.0, 0.0, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.0, 1.0},
                                                         {0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0}}},
      iganet::init::linear, options);
    EXPECT_TRUE(torch::allclose(bspline.coeffs(0),
                                torch::linspace(0, 1, 7, options).repeat(4)));
    EXPECT_TRUE(torch::allclose(bspline.coeffs(1),
                                torch::linspace(0, 1, 4, options).repeat_interleave(7)));
    EXPECT_TRUE(torch::allclose(bspline.coeffs(2), torch::ones(28, options)));
  }
  
  {
    iganet::NonUniformBSpline<real_t, 3, 1, 1> bspline({{{0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0},
                                                         {0.0, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.0}}},
      iganet::init::greville, options);
    EXPECT_TRUE(torch::allclose(bspline.coeffs(0),
                                torch::linspace(0, 1, 5, options).repeat(6)));
    EXPECT_TRUE(torch::allclose(bspline.coeffs(1),
                                torch::linspace(0, 1, 6, options).repeat_interleave(5)));
    EXPECT_TRUE(torch::allclose(bspline.coeffs(2), torch::ones(30, options)));
  }

  {
    iganet::NonUniformBSpline<real_t, 4, 2, 2> bspline({{{0.0, 0.0, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.0, 1.0},
                                                         {0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0}}},
      iganet::init::zeros, options);
    EXPECT_TRUE(torch::allclose(bspline.coeffs(0), torch::zeros(28, options)));
    EXPECT_TRUE(torch::allclose(bspline.coeffs(1), torch::zeros(28, options)));
    EXPECT_TRUE(torch::allclose(bspline.coeffs(2), torch::zeros(28, options)));
    EXPECT_TRUE(torch::allclose(bspline.coeffs(3), torch::zeros(28, options)));
  }

  {
    iganet::NonUniformBSpline<real_t, 4, 2, 2> bspline({{{0.0, 0.0, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.0, 1.0},
                                                         {0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0}}},
      iganet::init::ones, options);
    EXPECT_TRUE(torch::allclose(bspline.coeffs(0), torch::ones(28, options)));
    EXPECT_TRUE(torch::allclose(bspline.coeffs(1), torch::ones(28, options)));
    EXPECT_TRUE(torch::allclose(bspline.coeffs(2), torch::ones(28, options)));
    EXPECT_TRUE(torch::allclose(bspline.coeffs(3), torch::ones(28, options)));
  }

  {
    iganet::NonUniformBSpline<real_t, 4, 2, 2> bspline({{{0.0, 0.0, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.0, 1.0},
                                                         {0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0}}},
      iganet::init::linear, options);
    EXPECT_TRUE(torch::allclose(bspline.coeffs(0),
                                torch::linspace(0, 1, 7, options).repeat(4)));
    EXPECT_TRUE(torch::allclose(bspline.coeffs(1),
                                torch::linspace(0, 1, 4, options).repeat_interleave(7)));
    EXPECT_TRUE(torch::allclose(bspline.coeffs(2), torch::ones(28, options)));
    EXPECT_TRUE(torch::allclose(bspline.coeffs(3), torch::ones(28, options)));
  }
  
  {
    iganet::NonUniformBSpline<real_t, 4, 1, 1> bspline({{{0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0},
                                                         {0.0, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.0}}},
      iganet::init::greville, options);
    EXPECT_TRUE(torch::allclose(bspline.coeffs(0),
                                torch::linspace(0, 1, 5, options).repeat(6)));
    EXPECT_TRUE(torch::allclose(bspline.coeffs(1),
                                torch::linspace(0, 1, 6, options).repeat_interleave(5)));
    EXPECT_TRUE(torch::allclose(bspline.coeffs(2), torch::ones(30, options)));
    EXPECT_TRUE(torch::allclose(bspline.coeffs(3), torch::ones(30, options)));
  } 
}

TEST_F(BSplineTest, NonUniformBSpline_uniform_refine)
{
  {
    iganet::NonUniformBSpline<real_t, 3, 3, 4> bspline({4,5});
    iganet::NonUniformBSpline<real_t, 3, 3, 4> bspline_ref({5,6});
    bspline.uniform_refine();
    
    EXPECT_EQ( bspline.isclose(bspline_ref), true);
  }

  {
    iganet::NonUniformBSpline<real_t, 3, 3, 4> bspline({4,5});
    iganet::NonUniformBSpline<real_t, 3, 3, 4> bspline_ref({7,8});
    bspline.uniform_refine(2);
    
    EXPECT_EQ( bspline.isclose(bspline_ref), true);
  }

  {
    iganet::NonUniformBSpline<real_t, 3, 3, 4> bspline({4,5});
    iganet::NonUniformBSpline<real_t, 3, 3, 4> bspline_ref({5,5});
    bspline.uniform_refine(1, 0);
    
    EXPECT_EQ( bspline.isclose(bspline_ref), true);
  }
  
  {
    iganet::NonUniformBSpline<real_t, 3, 3, 4> bspline({4,5});
    iganet::NonUniformBSpline<real_t, 3, 3, 4> bspline_ref({5,8});
    bspline.uniform_refine(1, 0).uniform_refine(2, 1);
    
    EXPECT_EQ( bspline.isclose(bspline_ref), true);
  }
}

TEST_F(BSplineTest, NonUniformBSpline_copy_constructor)
{
  iganet::NonUniformBSpline<real_t, 3, 3, 4> bspline_orig({4,5}, iganet::init::greville, options);
  iganet::NonUniformBSpline<real_t, 3, 3, 4> bspline_copy (bspline_orig);
  
  bspline_orig.transform( [](const std::array<real_t,2> xi)
  { return std::array<real_t,3>{0.0, 1.0, 2.0};} );
  
  EXPECT_EQ( (bspline_orig == bspline_copy), true);
}

TEST_F(BSplineTest, NonUniformBSpline_clone_constructor)
{
  iganet::NonUniformBSpline<real_t, 3, 3, 4> bspline_ref ({4,5}, iganet::init::greville, options);
  iganet::NonUniformBSpline<real_t, 3, 3, 4> bspline_orig({4,5}, iganet::init::greville, options);
  iganet::NonUniformBSpline<real_t, 3, 3, 4> bspline_clone(bspline_orig, true);
  
  bspline_orig.transform( [](const std::array<real_t,2> xi)
  { return std::array<real_t,3>{0.0, 1.0, 2.0};} );
  
  EXPECT_EQ( (bspline_ref == bspline_clone), true);    
}

TEST_F(BSplineTest, NonUniformBSpline_move_constructor)
{
  iganet::NonUniformBSpline<real_t, 3, 3, 4> bspline_ref({7,8});
  auto bspline(iganet::NonUniformBSpline<real_t, 3, 3, 4>({4,5}).uniform_refine(2));

  EXPECT_EQ( bspline.isclose(bspline_ref), true);
}

TEST_F(BSplineTest, NonUniformBSpline_copy_assignment)
{
  iganet::NonUniformBSpline<real_t, 3, 3, 4> bspline_orig({4,5}, iganet::init::greville, options);
  auto bspline = bspline_orig;

  bspline_orig.transform( [](const std::array<real_t,2> xi)
  { return std::array<real_t,3>{0.0, 1.0, 2.0};} );

  EXPECT_EQ( bspline.isclose(bspline_orig), true);
}

TEST_F(BSplineTest, NonUniformBSpline_move_assignment)
{
  iganet::NonUniformBSpline<real_t, 3, 3, 4> bspline_ref({7,8}, iganet::init::greville, options);
  auto bspline = iganet::NonUniformBSpline<real_t, 3, 3, 4>({4,5}, iganet::init::greville, options).uniform_refine(2);
  
  EXPECT_EQ( bspline.isclose(bspline_ref), true);
}

TEST_F(BSplineTest, NonUniformBSpline_copy_coeffs_constructor)
{
  iganet::NonUniformBSpline<real_t, 3, 3, 4> bspline_orig({4,5}, iganet::init::greville, options);
  iganet::NonUniformBSpline<real_t, 3, 3, 4> bspline_copy(bspline_orig, bspline_orig.coeffs());
  
  bspline_orig.transform( [](const std::array<real_t,2> xi)
  { return std::array<real_t,3>{0.0, 1.0, 2.0};} );
  
  EXPECT_EQ( (bspline_orig == bspline_copy), true);
}

TEST_F(BSplineTest, NonUniformBSpline_clone_coeffs_constructor)
{
  iganet::NonUniformBSpline<real_t, 3, 3, 4> bspline_ref ({4,5}, iganet::init::greville, options);
  iganet::NonUniformBSpline<real_t, 3, 3, 4> bspline_orig({4,5}, iganet::init::greville, options);
  iganet::NonUniformBSpline<real_t, 3, 3, 4> bspline_clone(bspline_orig, bspline_orig.coeffs(), true);
  
  bspline_orig.transform( [](const std::array<real_t,2> xi)
  { return std::array<real_t,3>{0.0, 1.0, 2.0};} );
  
  EXPECT_EQ( (bspline_ref == bspline_clone), true);    
}

TEST_F(BSplineTest, NonUniformBSpline_read_write)
{
  std::filesystem::path filename = std::filesystem::temp_directory_path() / std::to_string(rand());
  iganet::NonUniformBSpline<real_t, 3, 1, 2, 3> bspline_out( {{{0.0, 0.0, 0.5, 1.0, 1.0},
                                                               {0.0, 0.0, 0.0, 1.0, 1.0, 1.0},
                                                               {0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0}}},
                                                            iganet::init::greville, options);
  bspline_out.save(filename.c_str());
  iganet::NonUniformBSpline<real_t, 3, 1, 2, 3> bspline_in(options);
  bspline_in.load(filename.c_str());
  std::filesystem::remove(filename);

  EXPECT_EQ( (bspline_in == bspline_out), true);
  EXPECT_EQ( (bspline_in != bspline_out), false);
}

TEST_F(BSplineTest, NonUniformBSpline_to_from_xml)
{
  {
    iganet::NonUniformBSpline<real_t, 1, 3> bspline_out({4}, iganet::init::zeros, options);
    
    bspline_out.transform( [](const std::array<real_t,1> xi)
    { return std::array<real_t,1>{static_cast<real_t>(std::rand())}; } );
    
    pugi::xml_document doc = bspline_out.to_xml();

    iganet::NonUniformBSpline<real_t, 1, 3> bspline_in(options);
    bspline_in.from_xml(doc);
    
    EXPECT_EQ( (bspline_in == bspline_out), true);

    // non-matching degree
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 1, 2>{}.from_xml(doc, 0)), std::runtime_error);
    
    // non-matching parametric dimension
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 1, 3, 3>{}.from_xml(doc, 0)), std::runtime_error);
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 1, 3, 3, 3>{}.from_xml(doc, 0)), std::runtime_error);
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 1, 3, 3, 3, 3>{}.from_xml(doc, 0)), std::runtime_error);

    // non-matching geometric dimension
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 2, 3>{}.from_xml(doc, 0)), std::runtime_error);
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 3, 3>{}.from_xml(doc, 0)), std::runtime_error);
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 4, 3>{}.from_xml(doc, 0)), std::runtime_error);
    
    // non-matching id
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 1, 3>{}.from_xml(doc, 1)), std::runtime_error);
  }

  {
    iganet::NonUniformBSpline<real_t, 2, 3> bspline_out({4}, iganet::init::zeros, options);
    
    bspline_out.transform( [](const std::array<real_t,1> xi)
    { return std::array<real_t,2>{static_cast<real_t>(std::rand()), static_cast<real_t>(std::rand())}; } );
    
    pugi::xml_document doc = bspline_out.to_xml();
    
    iganet::NonUniformBSpline<real_t, 2, 3> bspline_in(options);
    bspline_in.from_xml(doc);
    
    EXPECT_EQ( (bspline_in == bspline_out), true);

    // non-matching degree
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 2, 2>{}.from_xml(doc, 0)), std::runtime_error);
    
    // non-matching parametric dimension
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 2, 3, 3>{}.from_xml(doc, 0)), std::runtime_error);
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 2, 3, 3, 3>{}.from_xml(doc, 0)), std::runtime_error);
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 2, 3, 3, 3, 3>{}.from_xml(doc, 0)), std::runtime_error);

    // non-matching geometric dimension
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 1, 3>{}.from_xml(doc, 0)), std::runtime_error);
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 3, 3>{}.from_xml(doc, 0)), std::runtime_error);
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 4, 3>{}.from_xml(doc, 0)), std::runtime_error);
    
    // non-matching id
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 1, 3>{}.from_xml(doc, 1)), std::runtime_error);
  }

  {
    iganet::NonUniformBSpline<real_t, 3, 3> bspline_out({4}, iganet::init::zeros, options);
    
    bspline_out.transform( [](const std::array<real_t,1> xi)
    { return std::array<real_t,3>{static_cast<real_t>(std::rand()),
                                  static_cast<real_t>(std::rand()),
                                  static_cast<real_t>(std::rand())}; } );
    
    pugi::xml_document doc = bspline_out.to_xml();
    
    iganet::NonUniformBSpline<real_t, 3, 3> bspline_in(options);
    bspline_in.from_xml(doc);
    
    EXPECT_EQ( (bspline_in == bspline_out), true);

    // non-matching degree
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 3, 2>{}.from_xml(doc, 0)), std::runtime_error);
    
    // non-matching parametric dimension
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 3, 3, 3>{}.from_xml(doc, 0)), std::runtime_error);
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 3, 3, 3, 3>{}.from_xml(doc, 0)), std::runtime_error);
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 3, 3, 3, 3, 3>{}.from_xml(doc, 0)), std::runtime_error);

    // non-matching geometric dimension
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 1, 3>{}.from_xml(doc, 0)), std::runtime_error);
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 2, 3>{}.from_xml(doc, 0)), std::runtime_error);
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 4, 3>{}.from_xml(doc, 0)), std::runtime_error);
    
    // non-matching id
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 3, 3>{}.from_xml(doc, 1)), std::runtime_error);
  }

  {
    iganet::NonUniformBSpline<real_t, 4, 3> bspline_out({4}, iganet::init::zeros, options);
    
    bspline_out.transform( [](const std::array<real_t,1> xi)
    { return std::array<real_t,4>{static_cast<real_t>(std::rand()),
                                  static_cast<real_t>(std::rand()),
                                  static_cast<real_t>(std::rand()),
                                  static_cast<real_t>(std::rand())}; } );
    
    pugi::xml_document doc = bspline_out.to_xml();
    
    iganet::NonUniformBSpline<real_t, 4, 3> bspline_in(options);
    bspline_in.from_xml(doc);
    
    EXPECT_EQ( (bspline_in == bspline_out), true);

    // non-matching degree
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 4, 2>{}.from_xml(doc, 0)), std::runtime_error);
    
    // non-matching parametric dimension
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 4, 3, 3>{}.from_xml(doc, 0)), std::runtime_error);
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 4, 3, 3, 3>{}.from_xml(doc, 0)), std::runtime_error);
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 4, 3, 3, 3, 3>{}.from_xml(doc, 0)), std::runtime_error);

    // non-matching geometric dimension
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 1, 3>{}.from_xml(doc, 0)), std::runtime_error);
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 2, 3>{}.from_xml(doc, 0)), std::runtime_error);
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 3, 3>{}.from_xml(doc, 0)), std::runtime_error);
    
    // non-matching id
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 4, 3>{}.from_xml(doc, 1)), std::runtime_error);
  }
  
  {
    iganet::NonUniformBSpline<real_t, 1, 3, 4> bspline_out({4,5}, iganet::init::zeros, options);
    
    bspline_out.transform( [](const std::array<real_t,2> xi)
    { return std::array<real_t,1>{static_cast<real_t>(std::rand())}; } );
    
    pugi::xml_document doc = bspline_out.to_xml();

    iganet::NonUniformBSpline<real_t, 1, 3, 4> bspline_in(options);
    bspline_in.from_xml(doc);
    
    EXPECT_EQ( (bspline_in == bspline_out), true);

    // non-matching degree
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 1, 3, 3>{}.from_xml(doc, 0)), std::runtime_error);
    
    // non-matching parametric dimension
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 1, 3>{}.from_xml(doc, 0)), std::runtime_error);
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 1, 3, 4, 3>{}.from_xml(doc, 0)), std::runtime_error);
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 1, 3, 4, 3, 3>{}.from_xml(doc, 0)), std::runtime_error);

    // non-matching geometric dimension
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 2, 3, 4>{}.from_xml(doc, 0)), std::runtime_error);
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 3, 3, 4>{}.from_xml(doc, 0)), std::runtime_error);
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 4, 3, 4>{}.from_xml(doc, 0)), std::runtime_error);
    
    // non-matching id
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 1, 3, 4>{}.from_xml(doc, 1)), std::runtime_error);
  }
    
  {
    iganet::NonUniformBSpline<real_t, 2, 3, 4> bspline_out({4,5}, iganet::init::zeros, options);
    
    bspline_out.transform( [](const std::array<real_t,2> xi)
    { return std::array<real_t,2>{static_cast<real_t>(std::rand()), static_cast<real_t>(std::rand())}; } );
    
    pugi::xml_document doc = bspline_out.to_xml();
    
    iganet::NonUniformBSpline<real_t, 2, 3, 4> bspline_in(options);
    bspline_in.from_xml(doc);
    
    EXPECT_EQ( (bspline_in == bspline_out), true);

    // non-matching degree
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 2, 3, 3>{}.from_xml(doc, 0)), std::runtime_error);
    
    // non-matching parametric dimension
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 2, 3>{}.from_xml(doc, 0)), std::runtime_error);
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 2, 3, 4, 3>{}.from_xml(doc, 0)), std::runtime_error);
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 2, 3, 4, 3, 3>{}.from_xml(doc, 0)), std::runtime_error);

    // non-matching geometric dimension
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 1, 3, 4>{}.from_xml(doc, 0)), std::runtime_error);
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 3, 3, 4>{}.from_xml(doc, 0)), std::runtime_error);
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 4, 3, 4>{}.from_xml(doc, 0)), std::runtime_error);
    
    // non-matching id
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 2, 3, 4>{}.from_xml(doc, 1)), std::runtime_error);
  }
  
  {
    iganet::NonUniformBSpline<real_t, 3, 3, 4> bspline_out({4,5}, iganet::init::zeros, options);
    
    bspline_out.transform( [](const std::array<real_t,2> xi)
    { return std::array<real_t,3>{static_cast<real_t>(std::rand()),
                                  static_cast<real_t>(std::rand()),
                                  static_cast<real_t>(std::rand())}; } );
    
    pugi::xml_document doc = bspline_out.to_xml();
    
    iganet::NonUniformBSpline<real_t, 3, 3, 4> bspline_in(options);
    bspline_in.from_xml(doc);
    
    EXPECT_EQ( (bspline_in == bspline_out), true);

    // non-matching degree
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 3, 3, 3>{}.from_xml(doc, 0)), std::runtime_error);
    
    // non-matching parametric dimension
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 3, 3>{}.from_xml(doc, 0)), std::runtime_error);
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 3, 3, 4, 3>{}.from_xml(doc, 0)), std::runtime_error);
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 3, 3, 4, 3, 3>{}.from_xml(doc, 0)), std::runtime_error);

    // non-matching geometric dimension
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 1, 3, 4>{}.from_xml(doc, 0)), std::runtime_error);
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 2, 3, 4>{}.from_xml(doc, 0)), std::runtime_error);
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 4, 3, 4>{}.from_xml(doc, 0)), std::runtime_error);
    
    // non-matching id
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 3, 3, 4>{}.from_xml(doc, 1)), std::runtime_error);
  }

  {
    iganet::NonUniformBSpline<real_t, 4, 3, 4> bspline_out({4,5}, iganet::init::zeros, options);
    
    bspline_out.transform( [](const std::array<real_t,2> xi)
    { return std::array<real_t,4>{static_cast<real_t>(std::rand()),
                                  static_cast<real_t>(std::rand()),
                                  static_cast<real_t>(std::rand()),
                                  static_cast<real_t>(std::rand())}; } );
    
    pugi::xml_document doc = bspline_out.to_xml();
    
    iganet::NonUniformBSpline<real_t, 4, 3, 4> bspline_in(options);
    bspline_in.from_xml(doc);
    
    EXPECT_EQ( (bspline_in == bspline_out), true);

    // non-matching degree
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 4, 3, 3>{}.from_xml(doc, 0)), std::runtime_error);
    
    // non-matching parametric dimension
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 4, 3>{}.from_xml(doc, 0)), std::runtime_error);
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 4, 3, 4, 3>{}.from_xml(doc, 0)), std::runtime_error);
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 4, 3, 4, 3, 3>{}.from_xml(doc, 0)), std::runtime_error);

    // non-matching geometric dimension
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 1, 3, 4>{}.from_xml(doc, 0)), std::runtime_error);
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 2, 3, 4>{}.from_xml(doc, 0)), std::runtime_error);
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 3, 3, 4>{}.from_xml(doc, 0)), std::runtime_error);
    
    // non-matching id
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 4, 3, 4>{}.from_xml(doc, 1)), std::runtime_error);
  }
  
  {
    iganet::NonUniformBSpline<real_t, 1, 3, 4, 5> bspline_out({4,5,6}, iganet::init::zeros, options);
    
    bspline_out.transform( [](const std::array<real_t,3> xi)
    { return std::array<real_t,1>{static_cast<real_t>(std::rand())}; } );
    
    pugi::xml_document doc = bspline_out.to_xml();

    iganet::NonUniformBSpline<real_t, 1, 3, 4, 5> bspline_in(options);
    bspline_in.from_xml(doc);
    
    EXPECT_EQ( (bspline_in == bspline_out), true);

    // non-matching degree
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 1, 3, 3, 3>{}.from_xml(doc, 0)), std::runtime_error);
    
    // non-matching parametric dimension
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 1, 3>{}.from_xml(doc, 0)), std::runtime_error);
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 1, 3, 4>{}.from_xml(doc, 0)), std::runtime_error);
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 1, 3, 4, 5, 3>{}.from_xml(doc, 0)), std::runtime_error);

    // non-matching geometric dimension
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 2, 3, 4, 5>{}.from_xml(doc, 0)), std::runtime_error);
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 3, 3, 4, 5>{}.from_xml(doc, 0)), std::runtime_error);
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 4, 3, 4, 5>{}.from_xml(doc, 0)), std::runtime_error);
    
    // non-matching id
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 1, 3, 4, 5>{}.from_xml(doc, 1)), std::runtime_error);
  }
    
  {
    iganet::NonUniformBSpline<real_t, 2, 3, 4, 5> bspline_out({4,5,6}, iganet::init::zeros, options);
    
    bspline_out.transform( [](const std::array<real_t,3> xi)
    { return std::array<real_t,2>{static_cast<real_t>(std::rand()), static_cast<real_t>(std::rand())}; } );
    
    pugi::xml_document doc = bspline_out.to_xml();
    
    iganet::NonUniformBSpline<real_t, 2, 3, 4, 5> bspline_in(options);
    bspline_in.from_xml(doc);
    
    EXPECT_EQ( (bspline_in == bspline_out), true);

    // non-matching degree
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 2, 3, 3, 3>{}.from_xml(doc, 0)), std::runtime_error);
    
    // non-matching parametric dimension
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 2, 3>{}.from_xml(doc, 0)), std::runtime_error);
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 2, 3, 4>{}.from_xml(doc, 0)), std::runtime_error);
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 2, 3, 4, 5, 3>{}.from_xml(doc, 0)), std::runtime_error);

    // non-matching geometric dimension
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 1, 3, 4, 5>{}.from_xml(doc, 0)), std::runtime_error);
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 3, 3, 4, 5>{}.from_xml(doc, 0)), std::runtime_error);
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 4, 3, 4, 5>{}.from_xml(doc, 0)), std::runtime_error);
    
    // non-matching id
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 2, 3, 4, 5>{}.from_xml(doc, 1)), std::runtime_error);
  }
  
  {
    iganet::NonUniformBSpline<real_t, 3, 3, 4, 5> bspline_out({4,5,6}, iganet::init::zeros, options);
    
    bspline_out.transform( [](const std::array<real_t,3> xi)
    { return std::array<real_t,3>{static_cast<real_t>(std::rand()),
                                  static_cast<real_t>(std::rand()),
                                  static_cast<real_t>(std::rand())}; } );
    
    pugi::xml_document doc = bspline_out.to_xml();
    
    iganet::NonUniformBSpline<real_t, 3, 3, 4, 5> bspline_in(options);
    bspline_in.from_xml(doc);
    
    EXPECT_EQ( (bspline_in == bspline_out), true);

    // non-matching degree
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 3, 3, 3, 3>{}.from_xml(doc, 0)), std::runtime_error);
    
    // non-matching parametric dimension
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 3, 3>{}.from_xml(doc, 0)), std::runtime_error);
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 3, 3, 4>{}.from_xml(doc, 0)), std::runtime_error);
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 3, 3, 4, 5, 3>{}.from_xml(doc, 0)), std::runtime_error);

    // non-matching geometric dimension
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 1, 3, 4, 5>{}.from_xml(doc, 0)), std::runtime_error);
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 2, 3, 4, 5>{}.from_xml(doc, 0)), std::runtime_error);
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 4, 3, 4, 5>{}.from_xml(doc, 0)), std::runtime_error);
    
    // non-matching id
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 3, 3, 4, 5>{}.from_xml(doc, 1)), std::runtime_error);
  }

  {
    iganet::NonUniformBSpline<real_t, 4, 3, 4, 5> bspline_out({4,5,6}, iganet::init::zeros, options);
    
    bspline_out.transform( [](const std::array<real_t,3> xi)
    { return std::array<real_t,4>{static_cast<real_t>(std::rand()),
                                  static_cast<real_t>(std::rand()),
                                  static_cast<real_t>(std::rand()),
                                  static_cast<real_t>(std::rand())}; } );
    
    pugi::xml_document doc = bspline_out.to_xml();
    
    iganet::NonUniformBSpline<real_t, 4, 3, 4, 5> bspline_in(options);
    bspline_in.from_xml(doc);
    
    EXPECT_EQ( (bspline_in == bspline_out), true);

    // non-matching degree
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 4, 3, 3, 3>{}.from_xml(doc, 0)), std::runtime_error);
    
    // non-matching parametric dimension
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 4, 3>{}.from_xml(doc, 0)), std::runtime_error);
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 4, 3, 4>{}.from_xml(doc, 0)), std::runtime_error);
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 4, 3, 4, 5, 3>{}.from_xml(doc, 0)), std::runtime_error);

    // non-matching geometric dimension
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 1, 3, 4, 5>{}.from_xml(doc, 0)), std::runtime_error);
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 2, 3, 4, 5>{}.from_xml(doc, 0)), std::runtime_error);
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 3, 3, 4, 5>{}.from_xml(doc, 0)), std::runtime_error);
    
    // non-matching id
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 4, 3, 4, 5>{}.from_xml(doc, 1)), std::runtime_error);
  }

  {
    iganet::NonUniformBSpline<real_t, 1, 3, 4, 5, 1> bspline_out({4,5,6,2}, iganet::init::zeros, options);
    
    bspline_out.transform( [](const std::array<real_t,4> xi)
    { return std::array<real_t,1>{static_cast<real_t>(std::rand())}; } );
    
    pugi::xml_document doc = bspline_out.to_xml();

    iganet::NonUniformBSpline<real_t, 1, 3, 4, 5, 1> bspline_in(options);
    bspline_in.from_xml(doc);
    
    EXPECT_EQ( (bspline_in == bspline_out), true);

    // non-matching degree
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 1, 3, 4, 5, 2>{}.from_xml(doc, 0)), std::runtime_error);
    
    // non-matching parametric dimension
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 1, 3>{}.from_xml(doc, 0)), std::runtime_error);
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 1, 3, 4>{}.from_xml(doc, 0)), std::runtime_error);
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 1, 3, 4, 5>{}.from_xml(doc, 0)), std::runtime_error);

    // non-matching geometric dimension
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 2, 3, 4, 5, 1>{}.from_xml(doc, 0)), std::runtime_error);
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 3, 3, 4, 5, 1>{}.from_xml(doc, 0)), std::runtime_error);
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 4, 3, 4, 5, 1>{}.from_xml(doc, 0)), std::runtime_error);
    
    // non-matching id
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 1, 3, 4, 5, 1>{}.from_xml(doc, 1)), std::runtime_error);
  }
    
  {
    iganet::NonUniformBSpline<real_t, 2, 3, 4, 5, 1> bspline_out({4,5,6,2}, iganet::init::zeros, options);
    
    bspline_out.transform( [](const std::array<real_t,4> xi)
    { return std::array<real_t,2>{static_cast<real_t>(std::rand()), static_cast<real_t>(std::rand())}; } );
    
    pugi::xml_document doc = bspline_out.to_xml();
    
    iganet::NonUniformBSpline<real_t, 2, 3, 4, 5, 1> bspline_in(options);
    bspline_in.from_xml(doc);
    
    EXPECT_EQ( (bspline_in == bspline_out), true);

    // non-matching degree
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 2, 3, 4, 5, 2>{}.from_xml(doc, 0)), std::runtime_error);
    
    // non-matching parametric dimension
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 2, 3>{}.from_xml(doc, 0)), std::runtime_error);
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 2, 3, 4>{}.from_xml(doc, 0)), std::runtime_error);
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 2, 3, 4, 5>{}.from_xml(doc, 0)), std::runtime_error);

    // non-matching geometric dimension
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 1, 3, 4, 5, 1>{}.from_xml(doc, 0)), std::runtime_error);
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 3, 3, 4, 5, 1>{}.from_xml(doc, 0)), std::runtime_error);
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 4, 3, 4, 5, 1>{}.from_xml(doc, 0)), std::runtime_error);
    
    // non-matching id
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 2, 3, 4, 5, 1>{}.from_xml(doc, 1)), std::runtime_error);
  }
  
  {
    iganet::NonUniformBSpline<real_t, 3, 3, 4, 5, 1> bspline_out({4,5,6,2}, iganet::init::zeros, options);
    
    bspline_out.transform( [](const std::array<real_t,4> xi)
    { return std::array<real_t,3>{static_cast<real_t>(std::rand()),
                                  static_cast<real_t>(std::rand()),
                                  static_cast<real_t>(std::rand())}; } );
    
    pugi::xml_document doc = bspline_out.to_xml();
    
    iganet::NonUniformBSpline<real_t, 3, 3, 4, 5, 1> bspline_in(options);
    bspline_in.from_xml(doc);
    
    EXPECT_EQ( (bspline_in == bspline_out), true);

    // non-matching degree
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 3, 3, 4, 5, 2>{}.from_xml(doc, 0)), std::runtime_error);
    
    // non-matching parametric dimension
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 3, 3>{}.from_xml(doc, 0)), std::runtime_error);
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 3, 3, 4>{}.from_xml(doc, 0)), std::runtime_error);
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 3, 3, 4, 5>{}.from_xml(doc, 0)), std::runtime_error);

    // non-matching geometric dimension
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 1, 3, 4, 5, 1>{}.from_xml(doc, 0)), std::runtime_error);
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 2, 3, 4, 5, 1>{}.from_xml(doc, 0)), std::runtime_error);
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 4, 3, 4, 5, 1>{}.from_xml(doc, 0)), std::runtime_error);
    
    // non-matching id
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 3, 3, 4, 5, 1>{}.from_xml(doc, 1)), std::runtime_error);
  }

  {
    iganet::NonUniformBSpline<real_t, 4, 3, 4, 5, 1> bspline_out({4,5,6,2}, iganet::init::zeros, options);
    
    bspline_out.transform( [](const std::array<real_t,4> xi)
    { return std::array<real_t,4>{static_cast<real_t>(std::rand()),
                                  static_cast<real_t>(std::rand()),
                                  static_cast<real_t>(std::rand()),
                                  static_cast<real_t>(std::rand())}; } );
    
    pugi::xml_document doc = bspline_out.to_xml();
    
    iganet::NonUniformBSpline<real_t, 4, 3, 4, 5, 1> bspline_in(options);
    bspline_in.from_xml(doc);
    
    EXPECT_EQ( (bspline_in == bspline_out), true);

    // non-matching degree
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 4, 3, 4, 5, 2>{}.from_xml(doc, 0)), std::runtime_error);
    
    // non-matching parametric dimension
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 4, 3>{}.from_xml(doc, 0)), std::runtime_error);
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 4, 3, 4>{}.from_xml(doc, 0)), std::runtime_error);
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 4, 3, 4, 5>{}.from_xml(doc, 0)), std::runtime_error);

    // non-matching geometric dimension
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 1, 3, 4, 5, 1>{}.from_xml(doc, 0)), std::runtime_error);
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 2, 3, 4, 5, 1>{}.from_xml(doc, 0)), std::runtime_error);
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 3, 3, 4, 5, 1>{}.from_xml(doc, 0)), std::runtime_error);
    
    // non-matching id
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 4, 3, 4, 5, 1>{}.from_xml(doc, 1)), std::runtime_error);
  }
}

TEST_F(BSplineTest, NonUniformBSpline_load_from_xml)
{
  {
    pugi::xml_document doc;
    pugi::xml_parse_result result = doc.load_file(IGANET_DATA_DIR "domain1d/line.xml");
    
    iganet::NonUniformBSpline<real_t, 3, 2> bspline_in(options);
    bspline_in.from_xml(doc);
    
    iganet::NonUniformBSpline<real_t, 3, 2> bspline_ref({3}, iganet::init::zeros, options);
    
    bspline_ref.transform( [](const std::array<real_t,1> xi)
    { return std::array<real_t,3>{xi[0], 0.0, 0.0}; } );
    
    EXPECT_EQ( (bspline_in == bspline_ref), true);
  }

  {
    pugi::xml_document doc;
    pugi::xml_parse_result result = doc.load_file(IGANET_DATA_DIR "domain2d/square.xml");
    
    iganet::NonUniformBSpline<real_t, 2, 1, 1> bspline_in(options);
    bspline_in.from_xml(doc, 1);

    iganet::NonUniformBSpline<real_t, 2, 1, 1> bspline_ref({2, 2}, iganet::init::greville, options);
    
    EXPECT_EQ( (bspline_in == bspline_ref), true);
  }

  {
    pugi::xml_document doc;
    pugi::xml_parse_result result = doc.load_file(IGANET_DATA_DIR "domain3d/GshapedVolume.xml");

    iganet::NonUniformBSpline<real_t, 3, 2, 2, 2> bspline_in(options);
    bspline_in.from_xml(doc);
  }

  {
    pugi::xml_document doc;
    pugi::xml_parse_result result = doc.load_file(IGANET_DATA_DIR "surfaces/g_plus_s_surf.xml");

    iganet::NonUniformBSpline<real_t, 3, 3, 3> bspline_in0(options);
    iganet::NonUniformBSpline<real_t, 3, 3, 1> bspline_in1(options);

    for (int i : {0, 1, 4, 5, 8, 9, 12, 13, 18, 19, 22, 23, 27, 31, 32, 33, 36, 37, 39, 44, 45, 49, 50, 51, 52, 53, 56, 57, 58, 59})
      bspline_in0.from_xml(doc, i);
    
    for (int i : {2, 3, 6, 7, 10, 11, 14, 15, 16, 17, 20, 21, 24, 25, 26, 28, 29, 30, 34, 35, 38, 40, 41, 42, 43, 46, 47, 48, 54, 55, 60})
      bspline_in1.from_xml(doc, i);
  }  
}

TEST_F(BSplineTest, NonUniformBSpline_to_from_json)
{
  {
    iganet::NonUniformBSpline<real_t, 1, 3> bspline_out({4}, iganet::init::zeros, options);
    
    bspline_out.transform( [](const std::array<real_t,1> xi)
    { return std::array<real_t,1>{static_cast<real_t>(std::rand())}; } );
    
    nlohmann::json json = bspline_out.to_json();
    
    iganet::NonUniformBSpline<real_t, 1, 3> bspline_in(options);
    bspline_in.from_json(json);
    
    EXPECT_EQ( (bspline_in == bspline_out), true);
    
    // non-matching degree
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 1, 2>{}.from_json(json)), std::runtime_error);
    
    // non-matching parametric dimension
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 1, 3, 3>{}.from_json(json)), std::runtime_error);
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 1, 3, 3, 3>{}.from_json(json)), std::runtime_error);
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 1, 3, 3, 3, 3>{}.from_json(json)), std::runtime_error);
    
    // non-matching geometric dimension
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 2, 3>{}.from_json(json)), std::runtime_error);
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 3, 3>{}.from_json(json)), std::runtime_error);
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 4, 3>{}.from_json(json)), std::runtime_error);  
  }

  {
    iganet::NonUniformBSpline<real_t, 2, 3> bspline_out({4}, iganet::init::zeros, options);
    
    bspline_out.transform( [](const std::array<real_t,1> xi)
    { return std::array<real_t,2>{static_cast<real_t>(std::rand()),
                                  static_cast<real_t>(std::rand())}; } );
    
    nlohmann::json json = bspline_out.to_json();
    
    iganet::NonUniformBSpline<real_t, 2, 3> bspline_in(options);
    bspline_in.from_json(json);
    
    EXPECT_EQ( (bspline_in == bspline_out), true);
    
    // non-matching degree
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 2, 2>{}.from_json(json)), std::runtime_error);
    
    // non-matching parametric dimension
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 2, 3, 3>{}.from_json(json)), std::runtime_error);
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 2, 3, 3, 3>{}.from_json(json)), std::runtime_error);
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 2, 3, 3, 3, 3>{}.from_json(json)), std::runtime_error);
    
    // non-matching geometric dimension
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 1, 3>{}.from_json(json)), std::runtime_error);
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 3, 3>{}.from_json(json)), std::runtime_error);
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 4, 3>{}.from_json(json)), std::runtime_error);  
  }

  {
    iganet::NonUniformBSpline<real_t, 3, 3> bspline_out({4}, iganet::init::zeros, options);
    
    bspline_out.transform( [](const std::array<real_t,1> xi)
    { return std::array<real_t,3>{static_cast<real_t>(std::rand()),
                                  static_cast<real_t>(std::rand()),
                                  static_cast<real_t>(std::rand())}; } );
    
    nlohmann::json json = bspline_out.to_json();
    
    iganet::NonUniformBSpline<real_t, 3, 3> bspline_in(options);
    bspline_in.from_json(json);
    
    EXPECT_EQ( (bspline_in == bspline_out), true);
    
    // non-matching degree
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 3, 2>{}.from_json(json)), std::runtime_error);
    
    // non-matching parametric dimension
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 3, 3, 3>{}.from_json(json)), std::runtime_error);
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 3, 3, 3, 3>{}.from_json(json)), std::runtime_error);
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 3, 3, 3, 3, 3>{}.from_json(json)), std::runtime_error);
    
    // non-matching geometric dimension
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 1, 3>{}.from_json(json)), std::runtime_error);
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 2, 3>{}.from_json(json)), std::runtime_error);
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 4, 3>{}.from_json(json)), std::runtime_error);  
  }

  {
    iganet::NonUniformBSpline<real_t, 4, 3> bspline_out({4}, iganet::init::zeros, options);
    
    bspline_out.transform( [](const std::array<real_t,1> xi)
    { return std::array<real_t,4>{static_cast<real_t>(std::rand()),
                                  static_cast<real_t>(std::rand()),
                                  static_cast<real_t>(std::rand()),
                                  static_cast<real_t>(std::rand())}; } );
    
    nlohmann::json json = bspline_out.to_json();
    
    iganet::NonUniformBSpline<real_t, 4, 3> bspline_in(options);
    bspline_in.from_json(json);
    
    EXPECT_EQ( (bspline_in == bspline_out), true);
    
    // non-matching degree
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 4, 2>{}.from_json(json)), std::runtime_error);
    
    // non-matching parametric dimension
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 4, 3, 3>{}.from_json(json)), std::runtime_error);
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 4, 3, 3, 3>{}.from_json(json)), std::runtime_error);
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 4, 3, 3, 3, 3>{}.from_json(json)), std::runtime_error);
    
    // non-matching geometric dimension
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 1, 3>{}.from_json(json)), std::runtime_error);
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 2, 3>{}.from_json(json)), std::runtime_error);
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 3, 3>{}.from_json(json)), std::runtime_error);  
  }

  {
    iganet::NonUniformBSpline<real_t, 1, 3, 4> bspline_out({4,5}, iganet::init::zeros, options);
    
    bspline_out.transform( [](const std::array<real_t,2> xi)
    { return std::array<real_t,1>{static_cast<real_t>(std::rand())}; } );
    
    nlohmann::json json = bspline_out.to_json();
    
    iganet::NonUniformBSpline<real_t, 1, 3, 4> bspline_in(options);
    bspline_in.from_json(json);
    
    EXPECT_EQ( (bspline_in == bspline_out), true);
    
    // non-matching degree
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 1, 3, 3>{}.from_json(json)), std::runtime_error);
    
    // non-matching parametric dimension
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 1, 3>{}.from_json(json)), std::runtime_error);
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 1, 3, 4, 3>{}.from_json(json)), std::runtime_error);
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 1, 3, 4, 3, 3>{}.from_json(json)), std::runtime_error);
    
    // non-matching geometric dimension
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 2, 3, 4>{}.from_json(json)), std::runtime_error);
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 3, 3, 4>{}.from_json(json)), std::runtime_error);
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 4, 3, 4>{}.from_json(json)), std::runtime_error);  
  }

  {
    iganet::NonUniformBSpline<real_t, 2, 3, 4> bspline_out({4,5}, iganet::init::zeros, options);
    
    bspline_out.transform( [](const std::array<real_t,2> xi)
    { return std::array<real_t,2>{static_cast<real_t>(std::rand()),
                                  static_cast<real_t>(std::rand())}; } );
    
    nlohmann::json json = bspline_out.to_json();
    
    iganet::NonUniformBSpline<real_t, 2, 3, 4> bspline_in(options);
    bspline_in.from_json(json);
    
    EXPECT_EQ( (bspline_in == bspline_out), true);
    
    // non-matching degree
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 2, 3, 3>{}.from_json(json)), std::runtime_error);
    
    // non-matching parametric dimension
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 2, 3>{}.from_json(json)), std::runtime_error);
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 2, 3, 4, 3>{}.from_json(json)), std::runtime_error);
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 2, 3, 4, 3, 3>{}.from_json(json)), std::runtime_error);
    
    // non-matching geometric dimension
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 1, 3, 4>{}.from_json(json)), std::runtime_error);
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 3, 3, 4>{}.from_json(json)), std::runtime_error);
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 4, 3, 4>{}.from_json(json)), std::runtime_error);  
  }

  {
    iganet::NonUniformBSpline<real_t, 3, 3, 4> bspline_out({4,5}, iganet::init::zeros, options);
    
    bspline_out.transform( [](const std::array<real_t,2> xi)
    { return std::array<real_t,3>{static_cast<real_t>(std::rand()),
                                  static_cast<real_t>(std::rand()),
                                  static_cast<real_t>(std::rand())}; } );
    
    nlohmann::json json = bspline_out.to_json();
    
    iganet::NonUniformBSpline<real_t, 3, 3, 4> bspline_in(options);
    bspline_in.from_json(json);
    
    EXPECT_EQ( (bspline_in == bspline_out), true);
    
    // non-matching degree
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 3, 3, 3>{}.from_json(json)), std::runtime_error);
    
    // non-matching parametric dimension
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 3, 3>{}.from_json(json)), std::runtime_error);
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 3, 3, 4, 3>{}.from_json(json)), std::runtime_error);
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 3, 3, 4, 3, 3>{}.from_json(json)), std::runtime_error);
    
    // non-matching geometric dimension
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 1, 3, 4>{}.from_json(json)), std::runtime_error);
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 2, 3, 4>{}.from_json(json)), std::runtime_error);
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 4, 3, 4>{}.from_json(json)), std::runtime_error);  
  }

  {
    iganet::NonUniformBSpline<real_t, 4, 3, 4> bspline_out({4,5}, iganet::init::zeros, options);
    
    bspline_out.transform( [](const std::array<real_t,2> xi)
    { return std::array<real_t,4>{static_cast<real_t>(std::rand()),
                                  static_cast<real_t>(std::rand()),
                                  static_cast<real_t>(std::rand()),
                                  static_cast<real_t>(std::rand())}; } );
    
    nlohmann::json json = bspline_out.to_json();
    
    iganet::NonUniformBSpline<real_t, 4, 3, 4> bspline_in(options);
    bspline_in.from_json(json);
    
    EXPECT_EQ( (bspline_in == bspline_out), true);
    
    // non-matching degree
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 4, 3, 3>{}.from_json(json)), std::runtime_error);
    
    // non-matching parametric dimension
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 4, 3>{}.from_json(json)), std::runtime_error);
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 4, 3, 4, 3>{}.from_json(json)), std::runtime_error);
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 4, 3, 4, 3, 3>{}.from_json(json)), std::runtime_error);
    
    // non-matching geometric dimension
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 1, 3, 4>{}.from_json(json)), std::runtime_error);
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 2, 3, 4>{}.from_json(json)), std::runtime_error);
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 3, 3, 4>{}.from_json(json)), std::runtime_error);  
  }

  {
    iganet::NonUniformBSpline<real_t, 1, 3, 4, 5> bspline_out({4,5,6}, iganet::init::zeros, options);
    
    bspline_out.transform( [](const std::array<real_t,3> xi)
    { return std::array<real_t,1>{static_cast<real_t>(std::rand())}; } );
    
    nlohmann::json json = bspline_out.to_json();
    
    iganet::NonUniformBSpline<real_t, 1, 3, 4, 5> bspline_in(options);
    bspline_in.from_json(json);
    
    EXPECT_EQ( (bspline_in == bspline_out), true);
    
    // non-matching degree
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 1, 3, 3, 3>{}.from_json(json)), std::runtime_error);
    
    // non-matching parametric dimension
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 1, 3>{}.from_json(json)), std::runtime_error);
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 1, 3, 4>{}.from_json(json)), std::runtime_error);
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 1, 3, 4, 5, 3>{}.from_json(json)), std::runtime_error);
    
    // non-matching geometric dimension
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 2, 3, 4, 5>{}.from_json(json)), std::runtime_error);
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 3, 3, 4, 5>{}.from_json(json)), std::runtime_error);
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 4, 3, 4, 5>{}.from_json(json)), std::runtime_error);  
  }

  {
    iganet::NonUniformBSpline<real_t, 2, 3, 4, 5> bspline_out({4,5,6}, iganet::init::zeros, options);
    
    bspline_out.transform( [](const std::array<real_t,3> xi)
    { return std::array<real_t,2>{static_cast<real_t>(std::rand()),
                                  static_cast<real_t>(std::rand())}; } );
    
    nlohmann::json json = bspline_out.to_json();
    
    iganet::NonUniformBSpline<real_t, 2, 3, 4, 5> bspline_in(options);
    bspline_in.from_json(json);
    
    EXPECT_EQ( (bspline_in == bspline_out), true);
    
    // non-matching degree
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 2, 3, 3, 3>{}.from_json(json)), std::runtime_error);
    
    // non-matching parametric dimension
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 2, 3>{}.from_json(json)), std::runtime_error);
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 2, 3, 4>{}.from_json(json)), std::runtime_error);
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 2, 3, 4, 5, 3>{}.from_json(json)), std::runtime_error);
    
    // non-matching geometric dimension
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 1, 3, 4, 5>{}.from_json(json)), std::runtime_error);
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 3, 3, 4, 5>{}.from_json(json)), std::runtime_error);
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 4, 3, 4, 5>{}.from_json(json)), std::runtime_error);  
  }

  {
    iganet::NonUniformBSpline<real_t, 3, 3, 4, 5> bspline_out({4,5,6}, iganet::init::zeros, options);
    
    bspline_out.transform( [](const std::array<real_t,3> xi)
    { return std::array<real_t,3>{static_cast<real_t>(std::rand()),
                                  static_cast<real_t>(std::rand()),
                                  static_cast<real_t>(std::rand())}; } );
    
    nlohmann::json json = bspline_out.to_json();
    
    iganet::NonUniformBSpline<real_t, 3, 3, 4, 5> bspline_in(options);
    bspline_in.from_json(json);
    
    EXPECT_EQ( (bspline_in == bspline_out), true);
    
    // non-matching degree
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 3, 3, 3, 3>{}.from_json(json)), std::runtime_error);
    
    // non-matching parametric dimension
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 3, 3>{}.from_json(json)), std::runtime_error);
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 3, 3, 4>{}.from_json(json)), std::runtime_error);
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 3, 3, 4, 5, 3>{}.from_json(json)), std::runtime_error);
    
    // non-matching geometric dimension
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 1, 3, 4, 5>{}.from_json(json)), std::runtime_error);
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 2, 3, 4, 5>{}.from_json(json)), std::runtime_error);
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 4, 3, 4, 5>{}.from_json(json)), std::runtime_error);  
  }

  {
    iganet::NonUniformBSpline<real_t, 4, 3, 4, 5> bspline_out({4,5,6}, iganet::init::zeros, options);
    
    bspline_out.transform( [](const std::array<real_t,3> xi)
    { return std::array<real_t,4>{static_cast<real_t>(std::rand()),
                                  static_cast<real_t>(std::rand()),
                                  static_cast<real_t>(std::rand()),
                                  static_cast<real_t>(std::rand())}; } );
    
    nlohmann::json json = bspline_out.to_json();
    
    iganet::NonUniformBSpline<real_t, 4, 3, 4, 5> bspline_in(options);
    bspline_in.from_json(json);
    
    EXPECT_EQ( (bspline_in == bspline_out), true);
    
    // non-matching degree
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 4, 3, 3, 3>{}.from_json(json)), std::runtime_error);
    
    // non-matching parametric dimension
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 4, 3>{}.from_json(json)), std::runtime_error);
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 4, 3, 4>{}.from_json(json)), std::runtime_error);
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 4, 3, 4, 5, 3>{}.from_json(json)), std::runtime_error);
    
    // non-matching geometric dimension
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 1, 3, 4, 5>{}.from_json(json)), std::runtime_error);
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 2, 3, 4, 5>{}.from_json(json)), std::runtime_error);
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 3, 3, 4, 5>{}.from_json(json)), std::runtime_error);  
  }

  {
    iganet::NonUniformBSpline<real_t, 1, 3, 4, 5, 1> bspline_out({4,5,6,2}, iganet::init::zeros, options);
    
    bspline_out.transform( [](const std::array<real_t,4> xi)
    { return std::array<real_t,1>{static_cast<real_t>(std::rand())}; } );
    
    nlohmann::json json = bspline_out.to_json();
    
    iganet::NonUniformBSpline<real_t, 1, 3, 4, 5, 1> bspline_in(options);
    bspline_in.from_json(json);
    
    EXPECT_EQ( (bspline_in == bspline_out), true);
    
    // non-matching degree
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 1, 3, 4, 5, 2>{}.from_json(json)), std::runtime_error);
    
    // non-matching parametric dimension
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 1, 3>{}.from_json(json)), std::runtime_error);
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 1, 3, 4>{}.from_json(json)), std::runtime_error);
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 1, 3, 4, 5>{}.from_json(json)), std::runtime_error);
    
    // non-matching geometric dimension
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 2, 3, 4, 5, 1>{}.from_json(json)), std::runtime_error);
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 3, 3, 4, 5, 1>{}.from_json(json)), std::runtime_error);
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 4, 3, 4, 5, 1>{}.from_json(json)), std::runtime_error);  
  }

  {
    iganet::NonUniformBSpline<real_t, 2, 3, 4, 5, 1> bspline_out({4,5,6,2}, iganet::init::zeros, options);
    
    bspline_out.transform( [](const std::array<real_t,4> xi)
    { return std::array<real_t,2>{static_cast<real_t>(std::rand()),
                                  static_cast<real_t>(std::rand())}; } );
    
    nlohmann::json json = bspline_out.to_json();
    
    iganet::NonUniformBSpline<real_t, 2, 3, 4, 5, 1> bspline_in(options);
    bspline_in.from_json(json);
    
    EXPECT_EQ( (bspline_in == bspline_out), true);
    
    // non-matching degree
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 2, 3, 4, 5, 2>{}.from_json(json)), std::runtime_error);
    
    // non-matching parametric dimension
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 2, 3>{}.from_json(json)), std::runtime_error);
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 2, 3, 4>{}.from_json(json)), std::runtime_error);
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 2, 3, 4, 5>{}.from_json(json)), std::runtime_error);
    
    // non-matching geometric dimension
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 1, 3, 4, 5, 1>{}.from_json(json)), std::runtime_error);
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 3, 3, 4, 5, 1>{}.from_json(json)), std::runtime_error);
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 4, 3, 4, 5, 1>{}.from_json(json)), std::runtime_error);  
  }

  {
    iganet::NonUniformBSpline<real_t, 3, 3, 4, 5, 1> bspline_out({4,5,6,2}, iganet::init::zeros, options);
    
    bspline_out.transform( [](const std::array<real_t,4> xi)
    { return std::array<real_t,3>{static_cast<real_t>(std::rand()),
                                  static_cast<real_t>(std::rand()),
                                  static_cast<real_t>(std::rand())}; } );
    
    nlohmann::json json = bspline_out.to_json();
    
    iganet::NonUniformBSpline<real_t, 3, 3, 4, 5, 1> bspline_in(options);
    bspline_in.from_json(json);
    
    EXPECT_EQ( (bspline_in == bspline_out), true);
    
    // non-matching degree
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 3, 3, 4, 5, 2>{}.from_json(json)), std::runtime_error);
    
    // non-matching parametric dimension
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 3, 3>{}.from_json(json)), std::runtime_error);
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 3, 3, 4>{}.from_json(json)), std::runtime_error);
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 3, 3, 4, 5>{}.from_json(json)), std::runtime_error);
    
    // non-matching geometric dimension
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 1, 3, 4, 5, 1>{}.from_json(json)), std::runtime_error);
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 2, 3, 4, 5, 1>{}.from_json(json)), std::runtime_error);
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 4, 3, 4, 5, 1>{}.from_json(json)), std::runtime_error);  
  }

  {
    iganet::NonUniformBSpline<real_t, 4, 3, 4, 5, 1> bspline_out({4,5,6,2}, iganet::init::zeros, options);
    
    bspline_out.transform( [](const std::array<real_t,4> xi)
    { return std::array<real_t,4>{static_cast<real_t>(std::rand()),
                                  static_cast<real_t>(std::rand()),
                                  static_cast<real_t>(std::rand()),
                                  static_cast<real_t>(std::rand())}; } );
    
    nlohmann::json json = bspline_out.to_json();
    
    iganet::NonUniformBSpline<real_t, 4, 3, 4, 5, 1> bspline_in(options);
    bspline_in.from_json(json);
    
    EXPECT_EQ( (bspline_in == bspline_out), true);
    
    // non-matching degree
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 4, 3, 4, 5, 2>{}.from_json(json)), std::runtime_error);
    
    // non-matching parametric dimension
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 4, 3>{}.from_json(json)), std::runtime_error);
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 4, 3, 4>{}.from_json(json)), std::runtime_error);
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 4, 3, 4, 5>{}.from_json(json)), std::runtime_error);
    
    // non-matching geometric dimension
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 1, 3, 4, 5, 1>{}.from_json(json)), std::runtime_error);
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 2, 3, 4, 5, 1>{}.from_json(json)), std::runtime_error);
    EXPECT_THROW( (iganet::NonUniformBSpline<real_t, 3, 3, 4, 5, 1>{}.from_json(json)), std::runtime_error);  
  }
}

TEST_F(BSplineTest, NonUniformBSpline_reduce_continuity)
{
  {
    iganet::NonUniformBSpline<real_t, 3, 3, 4> bspline({5,6});
    iganet::NonUniformBSpline<real_t, 3, 3, 4> bspline_ref(iganet::utils::to_array(iganet::utils::to_vector(0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0),
                                                                                   iganet::utils::to_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0)));
    bspline.reduce_continuity();
    
    EXPECT_EQ( bspline.isclose(bspline_ref), true);
  }

  {
    iganet::NonUniformBSpline<real_t, 3, 3, 4> bspline({5,6});
    iganet::NonUniformBSpline<real_t, 3, 3, 4> bspline_ref(iganet::utils::to_array(iganet::utils::to_vector(0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0),
                                                                                   iganet::utils::to_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0)));
    bspline.reduce_continuity(2);
    
    EXPECT_EQ( bspline.isclose(bspline_ref), true);
  }

  {
    iganet::NonUniformBSpline<real_t, 3, 3, 4> bspline({5,6});
    iganet::NonUniformBSpline<real_t, 3, 3, 4> bspline_ref(iganet::utils::to_array(iganet::utils::to_vector(0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0),
                                                                                   iganet::utils::to_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0)));
    bspline.reduce_continuity(1, 0).reduce_continuity(2, 1);
    
    EXPECT_EQ( bspline.isclose(bspline_ref), true);
  }
}

TEST_F(BSplineTest, NonUniformBSpline_insert_knots)
{
  {
    iganet::NonUniformBSpline<real_t, 3, 3, 4> bspline({5,6});
    iganet::NonUniformBSpline<real_t, 3, 3, 4> bspline_ref(iganet::utils::to_array(iganet::utils::to_vector(0.0, 0.0, 0.0, 0.0, 0.1, 0.3, 0.5, 1.0, 1.0, 1.0, 1.0),
                                                                                   iganet::utils::to_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.4, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0)));
    bspline.insert_knots(iganet::utils::to_tensorArray({0.1, 0.3}, {0.2, 0.4}));
    
    EXPECT_EQ( bspline.isclose(bspline_ref), true);
  }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  iganet::init();
  return RUN_ALL_TESTS();
}
