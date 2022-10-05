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

#include <gtest/gtest.h>

TEST(BSpline, UniformBSpline_parDim1_geoDim1_degrees1_double)
{
  iganet::UniformBSpline<double, 1, 1> bspline({0});
  EXPECT_EQ(bspline.parDim(), 1);
  EXPECT_EQ(bspline.geoDim(), 1);
  EXPECT_EQ(bspline.degree(0), 1);
  EXPECT_EQ(bspline.nknots(0), 2);
  EXPECT_EQ(bspline.ncoeffs(0), 0);
}

TEST(BSpline, UniformBSpline_parDim1_geoDim1_degrees2_double)
{
  EXPECT_THROW( (iganet::UniformBSpline<double, 1, 2>({0})), std::runtime_error);
  iganet::UniformBSpline<double, 1, 2> bspline({1});
  EXPECT_EQ(bspline.parDim(), 1);
  EXPECT_EQ(bspline.geoDim(), 1);
  EXPECT_EQ(bspline.degree(0), 2);
  EXPECT_EQ(bspline.nknots(0), 4);
  EXPECT_EQ(bspline.ncoeffs(0), 1);
}

TEST(BSpline, UniformBSpline_parDim1_geoDim1_degrees3_double)
{
  EXPECT_THROW( (iganet::UniformBSpline<double, 1, 3>({0})), std::runtime_error);
  EXPECT_THROW( (iganet::UniformBSpline<double, 1, 3>({1})), std::runtime_error);
  iganet::UniformBSpline<double, 1, 3> bspline({2});
  EXPECT_EQ(bspline.parDim(), 1);
  EXPECT_EQ(bspline.geoDim(), 1);
  EXPECT_EQ(bspline.degree(0), 3);
  EXPECT_EQ(bspline.nknots(0), 6);
  EXPECT_EQ(bspline.ncoeffs(0), 2);
}

TEST(BSpline, UniformBSpline_parDim1_geoDim2_degrees4_double)
{
  EXPECT_THROW( (iganet::UniformBSpline<double, 2, 4>({0})), std::runtime_error);
  EXPECT_THROW( (iganet::UniformBSpline<double, 2, 4>({1})), std::runtime_error);
  EXPECT_THROW( (iganet::UniformBSpline<double, 2, 4>({2})), std::runtime_error);
  iganet::UniformBSpline<double, 2, 4> bspline({3});
  EXPECT_EQ(bspline.parDim(), 1);
  EXPECT_EQ(bspline.geoDim(), 2);
  EXPECT_EQ(bspline.degree(0), 4);
  EXPECT_EQ(bspline.nknots(0), 8);
  EXPECT_EQ(bspline.ncoeffs(0), 3);
}

TEST(BSpline, UniformBSpline_parDim1_geoDim3_degrees5_double)
{
  EXPECT_THROW( (iganet::UniformBSpline<double, 3, 5>({0})), std::runtime_error);
  EXPECT_THROW( (iganet::UniformBSpline<double, 3, 5>({1})), std::runtime_error);
  EXPECT_THROW( (iganet::UniformBSpline<double, 3, 5>({2})), std::runtime_error);
  EXPECT_THROW( (iganet::UniformBSpline<double, 3, 5>({3})), std::runtime_error);
  iganet::UniformBSpline<double, 3, 5> bspline({4});
  EXPECT_EQ(bspline.parDim(), 1);
  EXPECT_EQ(bspline.geoDim(), 3);
  EXPECT_EQ(bspline.degree(0), 5);
  EXPECT_EQ(bspline.nknots(0), 10);
  EXPECT_EQ(bspline.ncoeffs(0), 4);
}

TEST(BSpline, UniformBSpline_parDim1_geoDim4_degrees6_double)
{
  EXPECT_THROW( (iganet::UniformBSpline<double, 4, 6>({0})), std::runtime_error);
  EXPECT_THROW( (iganet::UniformBSpline<double, 4, 6>({1})), std::runtime_error);
  EXPECT_THROW( (iganet::UniformBSpline<double, 4, 6>({2})), std::runtime_error);
  EXPECT_THROW( (iganet::UniformBSpline<double, 4, 6>({3})), std::runtime_error);
  EXPECT_THROW( (iganet::UniformBSpline<double, 4, 6>({4})), std::runtime_error);
  iganet::UniformBSpline<double, 4, 6> bspline({5});
  EXPECT_EQ(bspline.parDim(), 1);
  EXPECT_EQ(bspline.geoDim(), 4);
  EXPECT_EQ(bspline.degree(0), 6);
  EXPECT_EQ(bspline.nknots(0), 12);
  EXPECT_EQ(bspline.ncoeffs(0), 5);
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
  EXPECT_EQ(bspline.parDim(), 2);
  EXPECT_EQ(bspline.geoDim(), 2);
  EXPECT_EQ(bspline.degree(0), 3);
  EXPECT_EQ(bspline.degree(1), 4);
  EXPECT_EQ(bspline.nknots(0), 6);
  EXPECT_EQ(bspline.nknots(1), 8);
  EXPECT_EQ(bspline.ncoeffs(0), 2);
  EXPECT_EQ(bspline.ncoeffs(1), 3);
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
  EXPECT_EQ(bspline.parDim(), 2);
  EXPECT_EQ(bspline.geoDim(), 3);
  EXPECT_EQ(bspline.degree(0), 3);
  EXPECT_EQ(bspline.degree(1), 4);
  EXPECT_EQ(bspline.nknots(0), 6);
  EXPECT_EQ(bspline.nknots(1), 8);
  EXPECT_EQ(bspline.ncoeffs(0), 2);
  EXPECT_EQ(bspline.ncoeffs(1), 3);
}

TEST(BSpline, NonUniformBSpline_parDim1_geoDim1_degrees1_double)
{
  EXPECT_THROW( (iganet::NonUniformBSpline<double, 1, 1>( {{{0.0, 0.0, 1.0}}} )), std::runtime_error);
  iganet::NonUniformBSpline<double, 1, 1> bspline( {{{0.0, 0.0, 0.5, 1.0, 1.0}}} );
  EXPECT_EQ(bspline.parDim(), 1);
  EXPECT_EQ(bspline.geoDim(), 1);
  EXPECT_EQ(bspline.degree(0), 1);
  EXPECT_EQ(bspline.nknots(0), 5);
  EXPECT_EQ(bspline.ncoeffs(0), 3);
}

TEST(BSpline, NonUniformBSpline_parDim2_geoDim2_degrees12_double)
{
  iganet::NonUniformBSpline<double, 2, 1, 2> bspline( {{{0.0, 0.0, 0.5, 1.0, 1.0},
                                                        {0.0, 0.0, 0.0, 1.0, 1.0, 1.0}}} );
  EXPECT_EQ(bspline.parDim(), 2);
  EXPECT_EQ(bspline.geoDim(), 2);
  EXPECT_EQ(bspline.degree(0), 1);
  EXPECT_EQ(bspline.degree(1), 2);
  EXPECT_EQ(bspline.nknots(0), 5);
  EXPECT_EQ(bspline.nknots(1), 6);
  EXPECT_EQ(bspline.ncoeffs(0), 3);
  EXPECT_EQ(bspline.ncoeffs(1), 3);
}

TEST(BSpline, NonUniformBSpline_parDim3_geoDim3_degrees123_double)
{
  iganet::NonUniformBSpline<double, 3, 1, 2, 3> bspline( {{{0.0, 0.0, 0.5, 1.0, 1.0},
                                                           {0.0, 0.0, 0.0, 1.0, 1.0, 1.0},
                                                           {0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0}}} );
  EXPECT_EQ(bspline.parDim(), 3);
  EXPECT_EQ(bspline.geoDim(), 3);
  EXPECT_EQ(bspline.degree(0), 1);
  EXPECT_EQ(bspline.degree(1), 2);
  EXPECT_EQ(bspline.degree(2), 3);
  EXPECT_EQ(bspline.nknots(0), 5);
  EXPECT_EQ(bspline.nknots(1), 6);
  EXPECT_EQ(bspline.nknots(2), 9);
  EXPECT_EQ(bspline.ncoeffs(0), 3);
  EXPECT_EQ(bspline.ncoeffs(1), 3);
  EXPECT_EQ(bspline.ncoeffs(2), 5);
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

TEST(BSpline, UniformBSpline_eval_degrees1_double)
{
  iganet::UniformBSpline<double, 1, 1> bspline({11}, iganet::BSplineInit::linear);
  
  // Function
  for (auto value : std::vector<std::array<double,2>>{{0.0,  0.0},
                                                      {0.2,  0.2},
                                                      {0.5,  0.5},
                                                      {0.75, 0.75},
                                                      {1.0,  1.0}})
    EXPECT_NEAR(bspline.eval<iganet::BSplineDeriv::func>( torch::ones({1}) * value[0] ).item<double>(), value[1], 1e-8);
  
  // First derivative
  for (auto value : std::vector<std::array<double,2>>{{0.0,  1.0},
                                                      {0.2,  1.0},
                                                      {0.5,  1.0},
                                                      {0.75, 1.0},
                                                      {1.0,  1.0}})
    EXPECT_NEAR(bspline.eval<iganet::BSplineDeriv::dx>( torch::ones({1}) * value[0] ).item<double>(), value[1], 1e-8);

  // Second derivative
  for (auto value : std::vector<std::array<double,2>>{{0.0,  0.0},
                                                      {0.2,  0.0},
                                                      {0.5,  0.0},
                                                      {0.75, 0.0},
                                                      {1.0,  0.0}})
    EXPECT_NEAR(bspline.eval<iganet::BSplineDeriv::dx2>( torch::ones({1}) * value[0] ).item<double>(), value[1], 1e-8);

  // Third derivative
  for (auto value : std::vector<std::array<double,2>>{{0.0,  0.0},
                                                      {0.2,  0.0},
                                                      {0.5,  0.0},
                                                      {0.75, 0.0},
                                                      {1.0,  0.0}})
    EXPECT_NEAR(bspline.eval<iganet::BSplineDeriv::dx3>( torch::ones({1}) * value[0] ).item<double>(), value[1], 1e-8);
  
  // Fourth derivative
  for (auto value : std::vector<std::array<double,2>>{{0.0,  0.0},
                                                      {0.2,  0.0},
                                                      {0.5,  0.0},
                                                      {0.75, 0.0},
                                                      {1.0,  0.0}})
    EXPECT_NEAR(bspline.eval<iganet::BSplineDeriv::dx4>( torch::ones({1}) * value[0] ).item<double>(), value[1], 1e-8);
}

TEST(BSpline, UniformBSpline_eval_degrees2_double)
{
  iganet::UniformBSpline<double, 1, 2> bspline({10}, iganet::BSplineInit::linear);
  
  // Function
  for (auto value : std::vector<std::array<double,2>>{{0.0,  0.0},
                                                      {0.1,  0.14222222222222222},
                                                      {0.2,  0.23333333333333334},
                                                      {0.5,  0.5},
                                                      {0.75, 0.7222222222222222},
                                                      {0.9,  0.8577777777777778},
                                                      {1.0,  1.0}})
    EXPECT_NEAR(bspline.eval<iganet::BSplineDeriv::func>( torch::ones({1}) * value[0] ).item<double>(), value[1], 1e-7);

  // First derivative
  for (auto value : std::vector<std::array<double,2>>{{0.0,  1.7777777777777777},
                                                      {0.1,  1.0666666666666667},
                                                      {0.2,  0.8888888888888888},
                                                      {0.5,  0.8888888888888893},
                                                      {0.75, 0.8888888888888893},
                                                      {0.9,  1.0666666666666664},
                                                      {1.0,  1.7777777777777786}})
    EXPECT_NEAR(bspline.eval<iganet::BSplineDeriv::dx>( torch::ones({1}) * value[0] ).item<double>(), value[1], 1e-6);    
  
  // Second derivative
  for (auto value : std::vector<std::array<double,2>>{{0.0, -7.111111111111111},
                                                      {0.1, -7.111111111111111},
                                                      {0.2,  0.0},
                                                      {0.5,  0.0},
                                                      {0.75, 0.0},
                                                      {0.9,  7.111111111111114},
                                                      {1.0,  7.111111111111114}})
    EXPECT_NEAR(bspline.eval<iganet::BSplineDeriv::dx2>( torch::ones({1}) * value[0] ).item<double>(), value[1], 1e-6);    

  // Third derivative
  for (auto value : std::vector<std::array<double,2>>{{0.0,  0.0},
                                                      {0.1,  0.0},
                                                      {0.2,  0.0},
                                                      {0.5,  0.0},
                                                      {0.75, 0.0},
                                                      {0.9,  0.0},
                                                      {1.0,  0.0}})
    EXPECT_NEAR(bspline.eval<iganet::BSplineDeriv::dx3>( torch::ones({1}) * value[0] ).item<double>(), value[1], 1e-7);

  // Fourth derivative
  for (auto value : std::vector<std::array<double,2>>{{0.0,  0.0},
                                                      {0.1,  0.0},
                                                      {0.2,  0.0},
                                                      {0.5,  0.0},
                                                      {0.75, 0.0},
                                                      {0.9,  0.0},
                                                      {1.0,  0.0}})
    EXPECT_NEAR(bspline.eval<iganet::BSplineDeriv::dx4>( torch::ones({1}) * value[0] ).item<double>(), value[1], 1e-7);
}

TEST(BSpline, UniformBSpline_eval_degrees3_double)
{
  iganet::UniformBSpline<double, 1, 3> bspline({9}, iganet::BSplineInit::linear);
  
  // Function
  for (auto value : std::vector<std::array<double,2>>{{0.0,  0.0},
                                                      {0.1,  0.16875},
                                                      {0.2,  0.26966666666666667},
                                                      {0.5,  0.5},
                                                      {0.75, 0.6888020833333333},
                                                      {0.9,  0.83125},
                                                      {1.0,  1.0}})
    EXPECT_NEAR(bspline.eval<iganet::BSplineDeriv::func>( torch::ones({1}) * value[0] ).item<double>(), value[1], 1e-7);

  // First derivative
  for (auto value : std::vector<std::array<double,2>>{{0.0,  2.25},
                                                      {0.1,  1.2374999999999998},
                                                      {0.2,  0.8699999999999998},
                                                      {0.5,  0.75},
                                                      {0.75, 0.7968749999999996},
                                                      {0.9,  1.2374999999999994},
                                                      {1.0,  2.25}})
    EXPECT_NEAR(bspline.eval<iganet::BSplineDeriv::dx>( torch::ones({1}) * value[0] ).item<double>(), value[1], 1e-6);

  // Second derivative
  for (auto value : std::vector<std::array<double,2>>{{0.0, -13.5},
                                                      {0.1,  -6.75},
                                                      {0.2,  -1.8},
                                                      {0.5,   0.0},
                                                      {0.75,  1.125},
                                                      {0.9,   6.75},
                                                      {1.0,  13.5}})
    EXPECT_NEAR(bspline.eval<iganet::BSplineDeriv::dx2>( torch::ones({1}) * value[0] ).item<double>(), value[1], 1e-5);

  // Third derivative
  for (auto value : std::vector<std::array<double,2>>{{0.0,  67.5},
                                                      {0.1,  67.5},
                                                      {0.2,  13.5},
                                                      {0.5,   0.0},
                                                      {0.75, 13.5},
                                                      {0.9,  67.5},
                                                      {1.0,  67.5}})
    EXPECT_NEAR(bspline.eval<iganet::BSplineDeriv::dx3>( torch::ones({1}) * value[0] ).item<double>(), value[1], 1e-8);

  // Fourth derivative
  for (auto value : std::vector<std::array<double,2>>{{0.0,  0.0},
                                                      {0.1,  0.0},
                                                      {0.2,  0.0},
                                                      {0.5,  0.0},
                                                      {0.75, 0.0},
                                                      {0.9,  0.0},
                                                      {1.0,  0.0}})
    EXPECT_NEAR(bspline.eval<iganet::BSplineDeriv::dx4>( torch::ones({1}) * value[0] ).item<double>(), value[1], 1e-8);
}

TEST(BSpline, UniformBSpline_eval_degrees4_double)
{
  iganet::UniformBSpline<double, 1, 4> bspline({9}, iganet::BSplineInit::linear);

  // Function
  for (auto value : std::vector<std::array<double,2>>{{0.0,  0.0},
                                                      {0.1,  0.17925347222222224},
                                                      {0.2,  0.28472222222222215},
                                                      {0.5,  0.5},
                                                      {0.75, 0.6725328233506945},
                                                      {0.9,  0.8207465277777777},
                                                      {1.0,  1.0}})
    EXPECT_NEAR(bspline.eval<iganet::BSplineDeriv::func>( torch::ones({1}) * value[0] ).item<double>(), value[1], 1e-7);

  // First derivative
  for (auto value : std::vector<std::array<double,2>>{{0.0,  2.5},
                                                      {0.1,  1.284722222222222},
                                                      {0.2,  0.9027777777777777},
                                                      {0.5,  0.6336805555555556},
                                                      {0.75, 0.8110894097222225},
                                                      {0.9,  1.2847222222222223},
                                                      {1.0,  2.5}})
    EXPECT_NEAR(bspline.eval<iganet::BSplineDeriv::dx>( torch::ones({1}) * value[0] ).item<double>(), value[1], 1e-6);

  // Second derivative
  for (auto value : std::vector<std::array<double,2>>{{0.0, -18.75},
                                                      {0.1,  -6.770833333333332},
                                                      {0.2,  -2.083333333333332},
                                                      {0.5,   0.0},
                                                      {0.75,  1.5950520833333357},
                                                      {0.9,   6.770833333333343},
                                                      {1.0,  18.75}})
    EXPECT_NEAR(bspline.eval<iganet::BSplineDeriv::dx2>( torch::ones({1}) * value[0] ).item<double>(), value[1], 1e-5);

  // Third derivative
  for (auto value : std::vector<std::array<double,2>>{{0.0, 156.25},
                                                      {0.1,  83.33333333333339},
                                                      {0.2,  10.4166666666666},
                                                      {0.5,   5.208333333333336},
                                                      {0.75,  9.114583333333371},
                                                      {0.9,  83.33333333333348},
                                                      {1.0, 156.25}})
    EXPECT_NEAR(bspline.eval<iganet::BSplineDeriv::dx3>( torch::ones({1}) * value[0] ).item<double>(), value[1], 1e-4);

  // Fourth derivative
  for (auto value : std::vector<std::array<double,2>>{{0.0, -729.1666666666679},
                                                      {0.1, -729.1666666666679},
                                                      {0.2,  -26.041666666666742},
                                                      {0.5,    0.0},
                                                      {0.75,  26.041666666667197},
                                                      {0.9,  729.1666666666679},
                                                      {1.0,  729.1666666666679}})
    EXPECT_NEAR(bspline.eval<iganet::BSplineDeriv::dx4>( torch::ones({1}) * value[0] ).item<double>(), value[1], 1e-8);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  iganet::init();
  return RUN_ALL_TESTS();
}
