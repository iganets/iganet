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

#include <tinysplinecxx.h>
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

template<typename T, typename BSpline_t, typename TensorArray_t>
void test_bspline_evaluation(BSpline_t& bspline, TensorArray_t& xi, T tol = 1e-12)
{
  tinyspline::BSpline tinybspline(bspline.ncoeffs(0), 1, bspline.degree(0), tinyspline::BSpline::Type::Clamped);
  std::vector<tinyspline::real> knots  = tinybspline.knots();
  std::vector<tinyspline::real> coeffs = tinybspline.controlPoints();
  
  for (int64_t i=0; i<bspline.nknots(0); ++i)
    knots[i] = bspline.knots(0)[i].template item<T>();
  tinybspline.setKnots(knots);
  
  for (int64_t i=0; i<bspline.ncoeffs(0); ++i)
    coeffs[i] = bspline.coeffs(0)[i].template item<T>();
  tinybspline.setControlPoints(coeffs);

  // Function
  auto func = bspline.template eval_<iganet::BSplineDeriv::func>(xi);
  for (int64_t i=0; i<xi[0].size(0); ++i)
    EXPECT_NEAR(func[i].template item<T>(),
                (tinybspline.eval((xi[0])[i].template item<T>()).result())[0], tol);
  
  // First derivative
  auto dx = bspline.template eval_<iganet::BSplineDeriv::dx>(xi);
  for (int64_t i=0; i<xi[0].size(0); ++i)
    EXPECT_NEAR(dx[i].template item<T>(),
                (tinybspline.derive(1, -1).eval((xi[0])[i].template item<T>()).result())[0], tol);

  // Second derivative
  auto dx2 = bspline.template eval_<iganet::BSplineDeriv::dx2>(xi);
  for (int64_t i=0; i<xi[0].size(0); ++i)
    EXPECT_NEAR(dx2[i].template item<T>(),
                (tinybspline.derive(2, -1).eval((xi[0])[i].template item<T>()).result())[0], tol);

  // Third derivative
  auto dx3 = bspline.template eval_<iganet::BSplineDeriv::dx3>(xi);
  for (int64_t i=0; i<xi[0].size(0); ++i)
    EXPECT_NEAR(dx3[i].template item<T>(),
                (tinybspline.derive(3, -1).eval((xi[0])[i].template item<T>()).result())[0], tol);
  
  // Fourth derivative
  auto dx4 = bspline.template eval_<iganet::BSplineDeriv::dx4>(xi);
  for (int64_t i=0; i<xi[0].size(0); ++i)
    EXPECT_NEAR(dx4[i].template item<T>(),
                (tinybspline.derive(4, -1).eval((xi[0])[i].template item<T>()).result())[0], tol);
}

TEST(BSpline, UniformBSpline_eval_degrees1_double)
{
  iganet::UniformBSpline<double, 1, 1> bspline({11}, iganet::BSplineInit::linear);
  auto xi  = iganet::to_tensorArray({0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0}, bspline.options());
  test_bspline_evaluation<double>(bspline, xi, 1e-12);
}

TEST(BSpline, UniformBSpline_eval_degrees2_double)
{
  iganet::UniformBSpline<double, 1, 2> bspline({10}, iganet::BSplineInit::linear);
  auto xi  = iganet::to_tensorArray({0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0}, bspline.options());
  test_bspline_evaluation<double>(bspline, xi, 1e-12); 
}

TEST(BSpline, UniformBSpline_eval_degrees3_double)
{
  iganet::UniformBSpline<double, 1, 3> bspline({9}, iganet::BSplineInit::linear);
  auto xi  = iganet::to_tensorArray({0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0}, bspline.options());
  test_bspline_evaluation<double>(bspline, xi, 1e-12);  
}

TEST(BSpline, UniformBSpline_eval_degrees4_double)
{
  iganet::UniformBSpline<double, 1, 4> bspline({9}, iganet::BSplineInit::linear);
  auto xi  = iganet::to_tensorArray({0.0, 0.1, 0.24, 0.5, 0.75, 0.9, 1.0}, bspline.options());
  test_bspline_evaluation<double>(bspline, xi, 1e-11); 
}

TEST(BSpline, UniformBSpline_eval_degrees5_double)
{
  iganet::UniformBSpline<double, 1, 5> bspline({9}, iganet::BSplineInit::linear);
  auto xi  = iganet::to_tensorArray({0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0}, bspline.options());
  test_bspline_evaluation<double>(bspline, xi, 1e-11); 
}

TEST(BSpline, UniformBSpline_eval_degrees6_double)
{
  iganet::UniformBSpline<double, 1, 6> bspline({9}, iganet::BSplineInit::linear);
  auto xi  = iganet::to_tensorArray({0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0}, bspline.options());
  test_bspline_evaluation<double>(bspline, xi, 1e-11); 
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  iganet::init();
  return RUN_ALL_TESTS();
}
