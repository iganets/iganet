/**
   @file unittests/unittest_uniform_bspline_eval.cxx

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

auto trafo_parDim1_geoDim1_double = [](const std::array<double,1> xi){ return std::array<double,1>{ xi[0]*xi[0] }; };
auto trafo_parDim1_geoDim2_double = [](const std::array<double,1> xi){ return std::array<double,2>{ xi[0]*xi[0],
                                                                                                    sin(M_PI*xi[0]) }; };
auto trafo_parDim1_geoDim3_double = [](const std::array<double,1> xi){ return std::array<double,3>{ xi[0]*xi[0],
                                                                                                    sin(M_PI*xi[0]),
                                                                                                    xi[0] }; };
auto trafo_parDim1_geoDim4_double = [](const std::array<double,1> xi){ return std::array<double,4>{ xi[0]*xi[0],
                                                                                                    sin(M_PI*xi[0]),
                                                                                                    xi[0],
                                                                                                    cos(M_PI*xi[0])}; };

auto trafo_parDim2_geoDim1_double = [](const std::array<double,2> xi){ return std::array<double,1>{ xi[0]*xi[1] }; };
auto trafo_parDim2_geoDim2_double = [](const std::array<double,2> xi){ return std::array<double,2>{ xi[0]*xi[1],
                                                                                                    sin(M_PI*xi[0]) }; };
auto trafo_parDim2_geoDim3_double = [](const std::array<double,2> xi){ return std::array<double,3>{ xi[0]*xi[1],
                                                                                                    sin(M_PI*xi[0]),
                                                                                                    xi[1] }; };
auto trafo_parDim2_geoDim4_double = [](const std::array<double,2> xi){ return std::array<double,4>{ xi[0]*xi[1],
                                                                                                    sin(M_PI*xi[0]),
                                                                                                    xi[1],
                                                                                                    cos(M_PI*xi[1])}; };

auto trafo_parDim3_geoDim1_double = [](const std::array<double,3> xi){ return std::array<double,1>{ xi[0]*xi[1]*xi[2] }; };
auto trafo_parDim3_geoDim2_double = [](const std::array<double,3> xi){ return std::array<double,2>{ xi[0]*xi[1]*xi[2],
                                                                                                    sin(M_PI*xi[0]) }; };
auto trafo_parDim3_geoDim3_double = [](const std::array<double,3> xi){ return std::array<double,3>{ xi[0]*xi[1]*xi[2],
                                                                                                    sin(M_PI*xi[0]),
                                                                                                    xi[1]*xi[2] }; };
auto trafo_parDim3_geoDim4_double = [](const std::array<double,3> xi){ return std::array<double,4>{ xi[0]*xi[1]*xi[2],
                                                                                                    sin(M_PI*xi[0]),
                                                                                                    xi[1]*xi[2],
                                                                                                    cos(M_PI*xi[1])}; };

auto trafo_parDim4_geoDim1_double = [](const std::array<double,4> xi){ return std::array<double,1>{ xi[0]*xi[1]*xi[2]*xi[3] }; };
auto trafo_parDim4_geoDim2_double = [](const std::array<double,4> xi){ return std::array<double,2>{ xi[0]*xi[1]*xi[2]*xi[3],
                                                                                                    sin(M_PI*xi[0]) }; };
auto trafo_parDim4_geoDim3_double = [](const std::array<double,4> xi){ return std::array<double,3>{ xi[0]*xi[1]*xi[2]*xi[3],
                                                                                                    sin(M_PI*xi[0]),
                                                                                                    xi[1]*xi[2]*xi[3] }; };
auto trafo_parDim4_geoDim4_double = [](const std::array<double,4> xi){ return std::array<double,4>{ xi[0]*xi[1]*xi[2]*xi[3],
                                                                                                    sin(M_PI*xi[0]),
                                                                                                    xi[1]*xi[2]*xi[3],
                                                                                                    cos(M_PI*xi[1])}; };

TEST(BSpline, UniformBSpline_eval_parDim1_geoDim1_degrees1_double)
{
  iganet::UniformBSpline<iganet::core<double, true>, 1, 1> bspline({11}, iganet::init::linear);
  bspline.transform(trafo_parDim1_geoDim1_double);
  auto xi  = iganet::to_tensorArray<double>({0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0}, bspline.options());
  test_bspline_eval(bspline, xi, 1e-12);
}

TEST(BSpline, UniformBSpline_eval_parDim1_geoDim1_degrees2_double)
{
  iganet::UniformBSpline<iganet::core<double, true>, 1, 2> bspline({11}, iganet::init::linear);
  bspline.transform(trafo_parDim1_geoDim1_double);
  auto xi  = iganet::to_tensorArray<double>({0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0}, bspline.options());
  test_bspline_eval(bspline, xi, 1e-12); 
}

TEST(BSpline, UniformBSpline_eval_parDim1_geoDim1_degrees3_double)
{
  iganet::UniformBSpline<iganet::core<double, true>, 1, 3> bspline({11}, iganet::init::linear);
  bspline.transform(trafo_parDim1_geoDim1_double);
  auto xi  = iganet::to_tensorArray<double>({0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0}, bspline.options());
  test_bspline_eval(bspline, xi, 1e-12);  
}

TEST(BSpline, UniformBSpline_eval_parDim1_geoDim1_degrees4_double)
{
  iganet::UniformBSpline<iganet::core<double, true>, 1, 4> bspline({11}, iganet::init::linear);
  bspline.transform(trafo_parDim1_geoDim1_double);
  auto xi  = iganet::to_tensorArray<double>({0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0}, bspline.options());
  test_bspline_eval(bspline, xi, 1e-10); 
}

TEST(BSpline, UniformBSpline_eval_parDim1_geoDim1_degrees5_double)
{
  iganet::UniformBSpline<iganet::core<double, true>, 1, 5> bspline({11}, iganet::init::linear);
  bspline.transform(trafo_parDim1_geoDim1_double);
  auto xi  = iganet::to_tensorArray<double>({0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0}, bspline.options());
  test_bspline_eval(bspline, xi, 1e-10); 
}

TEST(BSpline, UniformBSpline_eval_parDim1_geoDim1_degrees6_double)
{
  iganet::UniformBSpline<iganet::core<double, true>, 1, 6> bspline({11}, iganet::init::linear);
  bspline.transform(trafo_parDim1_geoDim1_double);
  auto xi  = iganet::to_tensorArray<double>({0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0}, bspline.options());
  test_bspline_eval(bspline, xi, 1e-10); 
}

TEST(BSpline, UniformBSpline_eval_parDim1_geoDim2_degrees1_double)
{
  iganet::UniformBSpline<iganet::core<double, true>, 2, 1> bspline({11}, iganet::init::linear);
  bspline.transform(trafo_parDim1_geoDim2_double);
  auto xi  = iganet::to_tensorArray<double>({0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0}, bspline.options());
  test_bspline_eval(bspline, xi, 1e-12);
}

TEST(BSpline, UniformBSpline_eval_parDim1_geoDim2_degrees2_double)
{
  iganet::UniformBSpline<iganet::core<double, true>, 2, 2> bspline({11}, iganet::init::linear);
  bspline.transform(trafo_parDim1_geoDim2_double);
  auto xi  = iganet::to_tensorArray<double>({0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0}, bspline.options());
  test_bspline_eval(bspline, xi, 1e-12);
}

TEST(BSpline, UniformBSpline_eval_parDim1_geoDim2_degrees3_double)
{
  iganet::UniformBSpline<iganet::core<double, true>, 2, 3> bspline({11}, iganet::init::linear);
  bspline.transform(trafo_parDim1_geoDim2_double);
  auto xi  = iganet::to_tensorArray<double>({0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0}, bspline.options());
  test_bspline_eval(bspline, xi, 1e-12);
}

TEST(BSpline, UniformBSpline_eval_parDim1_geoDim2_degrees4_double)
{
  iganet::UniformBSpline<iganet::core<double, true>, 2, 4> bspline({11}, iganet::init::linear);
  bspline.transform(trafo_parDim1_geoDim2_double);
  auto xi  = iganet::to_tensorArray<double>({0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0}, bspline.options());
  test_bspline_eval(bspline, xi, 1e-10);
}

TEST(BSpline, UniformBSpline_eval_parDim1_geoDim2_degrees5_double)
{
  iganet::UniformBSpline<iganet::core<double, true>, 2, 5> bspline({11}, iganet::init::linear);
  bspline.transform(trafo_parDim1_geoDim2_double);
  auto xi  = iganet::to_tensorArray<double>({0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0}, bspline.options());
  test_bspline_eval(bspline, xi, 1e-10);
}

TEST(BSpline, UniformBSpline_eval_parDim1_geoDim2_degrees6_double)
{
  iganet::UniformBSpline<iganet::core<double, true>, 2, 6> bspline({11}, iganet::init::linear);
  bspline.transform(trafo_parDim1_geoDim2_double);
  auto xi  = iganet::to_tensorArray<double>({0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0}, bspline.options());
  test_bspline_eval(bspline, xi, 1e-10);
}

TEST(BSpline, UniformBSpline_eval_parDim1_geoDim3_degrees1_double)
{
  iganet::UniformBSpline<iganet::core<double, true>, 3, 1> bspline({11}, iganet::init::linear);
  bspline.transform(trafo_parDim1_geoDim3_double);
  auto xi  = iganet::to_tensorArray<double>({0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0}, bspline.options());
  test_bspline_eval(bspline, xi, 1e-12);
}

TEST(BSpline, UniformBSpline_eval_parDim1_geoDim3_degrees2_double)
{
  iganet::UniformBSpline<iganet::core<double, true>, 3, 2> bspline({11}, iganet::init::linear);
  bspline.transform(trafo_parDim1_geoDim3_double);
  auto xi  = iganet::to_tensorArray<double>({0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0}, bspline.options());
  test_bspline_eval(bspline, xi, 1e-12);
}

TEST(BSpline, UniformBSpline_eval_parDim1_geoDim3_degrees3_double)
{
  iganet::UniformBSpline<iganet::core<double, true>, 3, 3> bspline({11}, iganet::init::linear);
  bspline.transform(trafo_parDim1_geoDim3_double);
  auto xi  = iganet::to_tensorArray<double>({0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0}, bspline.options());
  test_bspline_eval(bspline, xi, 1e-12);
}

TEST(BSpline, UniformBSpline_eval_parDim1_geoDim3_degrees4_double)
{
  iganet::UniformBSpline<iganet::core<double, true>, 3, 4> bspline({11}, iganet::init::linear);
  bspline.transform(trafo_parDim1_geoDim3_double);
  auto xi  = iganet::to_tensorArray<double>({0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0}, bspline.options());
  test_bspline_eval(bspline, xi, 1e-10);
}

TEST(BSpline, UniformBSpline_eval_parDim1_geoDim3_degrees5_double)
{
  iganet::UniformBSpline<iganet::core<double, true>, 3, 5> bspline({11}, iganet::init::linear);
  bspline.transform(trafo_parDim1_geoDim3_double);
  auto xi  = iganet::to_tensorArray<double>({0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0}, bspline.options());
  test_bspline_eval(bspline, xi, 1e-10);
}

TEST(BSpline, UniformBSpline_eval_parDim1_geoDim3_degrees6_double)
{
  iganet::UniformBSpline<iganet::core<double, true>, 3, 6> bspline({11}, iganet::init::linear);
  bspline.transform(trafo_parDim1_geoDim3_double);
  auto xi  = iganet::to_tensorArray<double>({0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0}, bspline.options());
  test_bspline_eval(bspline, xi, 1e-10);
}

TEST(BSpline, UniformBSpline_eval_parDim1_geoDim4_degrees1_double)
{
  iganet::UniformBSpline<iganet::core<double, true>, 4, 1> bspline({11}, iganet::init::linear);
  bspline.transform(trafo_parDim1_geoDim4_double);
  auto xi  = iganet::to_tensorArray<double>({0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0}, bspline.options());
  test_bspline_eval(bspline, xi, 1e-12);
}

TEST(BSpline, UniformBSpline_eval_parDim1_geoDim4_degrees2_double)
{
  iganet::UniformBSpline<iganet::core<double, true>, 4, 2> bspline({11}, iganet::init::linear);
  bspline.transform(trafo_parDim1_geoDim4_double);
  auto xi  = iganet::to_tensorArray<double>({0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0}, bspline.options());
  test_bspline_eval(bspline, xi, 1e-12);
}

TEST(BSpline, UniformBSpline_eval_parDim1_geoDim4_degrees3_double)
{
  iganet::UniformBSpline<iganet::core<double, true>, 4, 3> bspline({11}, iganet::init::linear);
  bspline.transform(trafo_parDim1_geoDim4_double);
  auto xi  = iganet::to_tensorArray<double>({0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0}, bspline.options());
  test_bspline_eval(bspline, xi, 1e-12);
}

TEST(BSpline, UniformBSpline_eval_parDim1_geoDim4_degrees4_double)
{
  iganet::UniformBSpline<iganet::core<double, true>, 4, 4> bspline({11}, iganet::init::linear);
  bspline.transform(trafo_parDim1_geoDim4_double);
  auto xi  = iganet::to_tensorArray<double>({0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0}, bspline.options());
  test_bspline_eval(bspline, xi, 1e-10);
}

TEST(BSpline, UniformBSpline_eval_parDim1_geoDim4_degrees5_double)
{
  iganet::UniformBSpline<iganet::core<double, true>, 4, 5> bspline({11}, iganet::init::linear);
  bspline.transform(trafo_parDim1_geoDim4_double);
  auto xi  = iganet::to_tensorArray<double>({0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0}, bspline.options());
  test_bspline_eval(bspline, xi, 1e-10);
}

TEST(BSpline, UniformBSpline_eval_parDim1_geoDim4_degrees6_double)
{
  iganet::UniformBSpline<iganet::core<double, true>, 4, 6> bspline({11}, iganet::init::linear);
  bspline.transform(trafo_parDim1_geoDim4_double);
  auto xi  = iganet::to_tensorArray<double>({0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0}, bspline.options());
  test_bspline_eval(bspline, xi, 1e-10);
}

TEST(BSpline, UniformBSpline_eval_parDim2_geoDim1_degrees22_double)
{
  iganet::UniformBSpline<iganet::core<double, true>, 1, 2, 2> bspline({6, 5}, iganet::init::linear);
  bspline.transform(trafo_parDim2_geoDim1_double);
  auto xi  = iganet::to_tensorArray<double>({0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0},
                                            {0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0}, bspline.options());
  test_bspline_eval(bspline, xi, 1e-12);
}

TEST(BSpline, UniformBSpline_eval_parDim2_geoDim1_degrees46_double)
{
  iganet::UniformBSpline<iganet::core<double, true>, 1, 4, 6> bspline({5, 11}, iganet::init::linear);
  bspline.transform(trafo_parDim2_geoDim1_double);
  auto xi  = iganet::to_tensorArray<double>({0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0},
                                            {0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0}, bspline.options());
  test_bspline_eval(bspline, xi, 1e-10);
}

TEST(BSpline, UniformBSpline_eval_parDim2_geoDim1_degrees64_double)
{
  iganet::UniformBSpline<iganet::core<double, true>, 1, 6, 4> bspline({11, 5}, iganet::init::linear);
  bspline.transform(trafo_parDim2_geoDim1_double);
  auto xi  = iganet::to_tensorArray<double>({0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0},
                                            {0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0}, bspline.options());
  test_bspline_eval(bspline, xi, 1e-10);
}

TEST(BSpline, UniformBSpline_eval_parDim2_geoDim2_degrees22_double)
{
  iganet::UniformBSpline<iganet::core<double, true>, 2, 2, 2> bspline({6, 5}, iganet::init::linear);
  bspline.transform(trafo_parDim2_geoDim2_double);
  auto xi  = iganet::to_tensorArray<double>({0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0},
                                            {0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0}, bspline.options());
  test_bspline_eval(bspline, xi, 1e-12);
}

TEST(BSpline, UniformBSpline_eval_parDim2_geoDim2_degrees46_double)
{
  iganet::UniformBSpline<iganet::core<double, true>, 2, 4, 6> bspline({5, 11}, iganet::init::linear);
  bspline.transform(trafo_parDim2_geoDim2_double);
  auto xi  = iganet::to_tensorArray<double>({0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0},
                                            {0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0}, bspline.options());
  test_bspline_eval(bspline, xi, 1e-10);
}

TEST(BSpline, UniformBSpline_eval_parDim2_geoDim2_degrees64_double)
{
  iganet::UniformBSpline<iganet::core<double, true>, 2, 6, 4> bspline({11, 5}, iganet::init::linear);
  bspline.transform(trafo_parDim2_geoDim2_double);
  auto xi  = iganet::to_tensorArray<double>({0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0},
                                            {0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0}, bspline.options());
  test_bspline_eval(bspline, xi, 1e-10);
}

TEST(BSpline, UniformBSpline_eval_parDim2_geoDim3_degrees22_double)
{
  iganet::UniformBSpline<iganet::core<double, true>, 3, 2, 2> bspline({6, 5}, iganet::init::linear);
  bspline.transform(trafo_parDim2_geoDim3_double);
  auto xi  = iganet::to_tensorArray<double>({0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0},
                                            {0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0}, bspline.options());
  test_bspline_eval(bspline, xi, 1e-12);
}

TEST(BSpline, UniformBSpline_eval_parDim2_geoDim3_degrees46_double)
{
  iganet::UniformBSpline<iganet::core<double, true>, 3, 4, 6> bspline({5, 11}, iganet::init::linear);
  bspline.transform(trafo_parDim2_geoDim3_double);
  auto xi  = iganet::to_tensorArray<double>({0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0},
                                            {0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0}, bspline.options());
  test_bspline_eval(bspline, xi, 1e-10);
}

TEST(BSpline, UniformBSpline_eval_parDim2_geoDim3_degrees64_double)
{
  iganet::UniformBSpline<iganet::core<double, true>, 3, 6, 4> bspline({11, 5}, iganet::init::linear);
  bspline.transform(trafo_parDim2_geoDim3_double);
  auto xi  = iganet::to_tensorArray<double>({0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0},
                                            {0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0}, bspline.options());
  test_bspline_eval(bspline, xi, 1e-10);
}

TEST(BSpline, UniformBSpline_eval_parDim2_geoDim4_degrees22_double)
{
  iganet::UniformBSpline<iganet::core<double, true>, 4, 2, 2> bspline({6, 5}, iganet::init::linear);
  bspline.transform(trafo_parDim2_geoDim4_double);
  auto xi  = iganet::to_tensorArray<double>({0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0},
                                            {0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0}, bspline.options());
  test_bspline_eval(bspline, xi, 1e-12);
}

TEST(BSpline, UniformBSpline_eval_parDim2_geoDim4_degrees46_double)
{
  iganet::UniformBSpline<iganet::core<double, true>, 4, 4, 6> bspline({5, 11}, iganet::init::linear);
  bspline.transform(trafo_parDim2_geoDim4_double);
  auto xi  = iganet::to_tensorArray<double>({0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0},
                                            {0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0}, bspline.options());
  test_bspline_eval(bspline, xi, 1e-10);
}

TEST(BSpline, UniformBSpline_eval_parDim2_geoDim4_degrees64_double)
{
  iganet::UniformBSpline<iganet::core<double, true>, 4, 6, 4> bspline({11, 5}, iganet::init::linear);
  bspline.transform(trafo_parDim2_geoDim4_double);
  auto xi  = iganet::to_tensorArray<double>({0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0},
                                            {0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0}, bspline.options());
  test_bspline_eval(bspline, xi, 1e-10);
}

TEST(BSpline, UniformBSpline_eval_parDim3_geoDim1_degrees222_double)
{
  iganet::UniformBSpline<iganet::core<double, true>, 1, 2, 2, 2> bspline({11, 5, 3}, iganet::init::linear);
  bspline.transform(trafo_parDim3_geoDim1_double);
  auto xi  = iganet::to_tensorArray<double>({0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0},
                                            {0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0},
                                            {0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0}, bspline.options());
  test_bspline_eval(bspline, xi, 1e-12);
}

TEST(BSpline, UniformBSpline_eval_parDim3_geoDim1_degrees264_double)
{
  iganet::UniformBSpline<iganet::core<double, true>, 1, 2, 6, 4> bspline({3, 11, 5}, iganet::init::linear);
  bspline.transform(trafo_parDim3_geoDim1_double);
  auto xi  = iganet::to_tensorArray<double>({0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0},
                                            {0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0},
                                            {0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0}, bspline.options());
  test_bspline_eval(bspline, xi, 1e-10);
}

TEST(BSpline, UniformBSpline_eval_parDim3_geoDim2_degrees222_double)
{
  iganet::UniformBSpline<iganet::core<double, true>, 2, 2, 2, 2> bspline({11, 5, 3}, iganet::init::linear);
  bspline.transform(trafo_parDim3_geoDim2_double);
  auto xi  = iganet::to_tensorArray<double>({0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0},
                                            {0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0},
                                            {0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0}, bspline.options());
  test_bspline_eval(bspline, xi, 1e-12);
}

TEST(BSpline, UniformBSpline_eval_parDim3_geoDim2_degrees264_double)
{
  iganet::UniformBSpline<iganet::core<double, true>, 2, 2, 6, 4> bspline({3, 11, 5}, iganet::init::linear);
  bspline.transform(trafo_parDim3_geoDim2_double);
  auto xi  = iganet::to_tensorArray<double>({0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0},
                                            {0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0},
                                            {0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0}, bspline.options());
  test_bspline_eval(bspline, xi, 1e-10);
}

TEST(BSpline, UniformBSpline_eval_parDim3_geoDim3_degrees222_double)
{
  iganet::UniformBSpline<iganet::core<double, true>, 3, 2, 2, 2> bspline({11, 5, 3}, iganet::init::linear);
  bspline.transform(trafo_parDim3_geoDim3_double);
  auto xi  = iganet::to_tensorArray<double>({0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0},
                                            {0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0},
                                            {0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0}, bspline.options());
  test_bspline_eval(bspline, xi, 1e-12);
}

TEST(BSpline, UniformBSpline_eval_parDim3_geoDim3_degrees264_double)
{
  iganet::UniformBSpline<iganet::core<double, true>, 3, 2, 6, 4> bspline({3, 11, 5}, iganet::init::linear);
  bspline.transform(trafo_parDim3_geoDim3_double);
  auto xi  = iganet::to_tensorArray<double>({0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0},
                                            {0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0},
                                            {0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0}, bspline.options());
  test_bspline_eval(bspline, xi, 1e-10);
}

TEST(BSpline, UniformBSpline_eval_parDim3_geoDim4_degrees222_double)
{
  iganet::UniformBSpline<iganet::core<double, true>, 4, 2, 2, 2> bspline({11, 5, 3}, iganet::init::linear);
  bspline.transform(trafo_parDim3_geoDim4_double);
  auto xi  = iganet::to_tensorArray<double>({0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0},
                                            {0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0},
                                            {0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0}, bspline.options());
  test_bspline_eval(bspline, xi, 1e-12);
}

TEST(BSpline, UniformBSpline_eval_parDim3_geoDim4_degrees264_double)
{
  iganet::UniformBSpline<iganet::core<double, true>, 4, 2, 6, 4> bspline({3, 11, 5}, iganet::init::linear);
  bspline.transform(trafo_parDim3_geoDim4_double);
  auto xi  = iganet::to_tensorArray<double>({0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0},
                                            {0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0},
                                            {0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0}, bspline.options());
  test_bspline_eval(bspline, xi, 1e-10);
}

TEST(BSpline, UniformBSpline_eval_parDim4_geoDim1_degrees2222_double)
{
  iganet::UniformBSpline<iganet::core<double, true>, 1, 2, 2, 2, 2> bspline({11, 5, 3, 8}, iganet::init::linear);
  bspline.transform(trafo_parDim4_geoDim1_double);
  auto xi  = iganet::to_tensorArray<double>({0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0},
                                            {0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0},
                                            {0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0},
                                            {0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0}, bspline.options());
  test_bspline_eval(bspline, xi, 1e-12);
}

TEST(BSpline, UniformBSpline_eval_parDim4_geoDim1_degrees2643_double)
{
  iganet::UniformBSpline<iganet::core<double, true>, 1, 2, 6, 4, 3> bspline({3, 11, 5, 8}, iganet::init::linear);
  bspline.transform(trafo_parDim4_geoDim1_double);
  auto xi  = iganet::to_tensorArray<double>({0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0},
                                            {0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0},
                                            {0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0},
                                            {0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0}, bspline.options());
  test_bspline_eval(bspline, xi, 1e-12);
}

TEST(BSpline, UniformBSpline_eval_parDim4_geoDim2_degrees2222_double)
{
  iganet::UniformBSpline<iganet::core<double, true>, 2, 2, 2, 2, 2> bspline({11, 5, 3, 8}, iganet::init::linear);
  bspline.transform(trafo_parDim4_geoDim2_double);
  auto xi  = iganet::to_tensorArray<double>({0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0},
                                            {0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0},
                                            {0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0},
                                            {0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0}, bspline.options());
  test_bspline_eval(bspline, xi, 1e-12);
}

TEST(BSpline, UniformBSpline_eval_parDim4_geoDim2_degrees2643_double)
{
  iganet::UniformBSpline<iganet::core<double, true>, 2, 2, 6, 4, 3> bspline({3, 11, 5, 8}, iganet::init::linear);
  bspline.transform(trafo_parDim4_geoDim2_double);
  auto xi  = iganet::to_tensorArray<double>({0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0},
                                            {0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0},
                                            {0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0},
                                            {0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0}, bspline.options());
  test_bspline_eval(bspline, xi, 1e-12);
}

TEST(BSpline, UniformBSpline_eval_parDim4_geoDim3_degrees2222_double)
{
  iganet::UniformBSpline<iganet::core<double, true>, 3, 2, 2, 2, 2> bspline({11, 5, 3, 8}, iganet::init::linear);
  bspline.transform(trafo_parDim4_geoDim3_double);
  auto xi  = iganet::to_tensorArray<double>({0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0},
                                            {0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0},
                                            {0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0},
                                            {0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0}, bspline.options());
  test_bspline_eval(bspline, xi, 1e-12);
}

TEST(BSpline, UniformBSpline_eval_parDim4_geoDim3_degrees2643_double)
{
  iganet::UniformBSpline<iganet::core<double, true>, 3, 2, 6, 4, 3> bspline({3, 11, 5, 8}, iganet::init::linear);
  bspline.transform(trafo_parDim4_geoDim3_double);
  auto xi  = iganet::to_tensorArray<double>({0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0},
                                            {0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0},
                                            {0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0},
                                            {0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0}, bspline.options());
  test_bspline_eval(bspline, xi, 1e-12);
}

TEST(BSpline, UniformBSpline_eval_parDim4_geoDim4_degrees2222_double)
{
  iganet::UniformBSpline<iganet::core<double, true>, 4, 2, 2, 2, 2> bspline({11, 5, 3, 8}, iganet::init::linear);
  bspline.transform(trafo_parDim4_geoDim4_double);
  auto xi  = iganet::to_tensorArray<double>({0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0},
                                            {0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0},
                                            {0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0},
                                            {0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0}, bspline.options());
  test_bspline_eval(bspline, xi, 1e-12);
}

TEST(BSpline, UniformBSpline_eval_parDim4_geoDim4_degrees2643_double)
{
  iganet::UniformBSpline<iganet::core<double, true>, 4, 2, 6, 4, 3> bspline({3, 11, 5, 8}, iganet::init::linear);
  bspline.transform(trafo_parDim4_geoDim4_double);
  auto xi  = iganet::to_tensorArray<double>({0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0},
                                            {0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0},
                                            {0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0},
                                            {0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0}, bspline.options());
  test_bspline_eval(bspline, xi, 1e-12);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  iganet::init();
  return RUN_ALL_TESTS();
}
