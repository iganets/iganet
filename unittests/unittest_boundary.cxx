/**
   @file unittests/unittest_boundary.cxx

   @brief Boundary unittests

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include <iganet.h>
#include <filesystem>
#include <iostream>

#include <unittest_config.hpp>
#include <gtest/gtest.h>

class BoundaryTest
  : public ::testing::Test
{
protected:
  using real_t = iganet::unittests::real_t;
  iganet::Options<real_t> options;
};

TEST_F(BoundaryTest, Boundary_parDim1_geoDim1_degrees2)
{
  using iganet::side;
  using iganet::deriv;
  using BSpline_t = iganet::UniformBSpline<real_t, 1, 2>;
  iganet::Boundary<BSpline_t> boundary({0}, iganet::init::greville, options);

  auto xi = std::tuple{ std::array<torch::Tensor,0>{}, std::array<torch::Tensor,0>{} };

  // Evaluation
  auto eval = boundary.eval<deriv::func, false>(xi);

  EXPECT_TRUE(torch::equal(*std::get<side::left-1>(eval)[0], torch::ones(1, options)));
  EXPECT_TRUE(torch::equal(*std::get<side::right-1>(eval)[0], torch::ones(1, options)));
  
  eval = boundary.eval<deriv::dx, false>(xi);
  
  EXPECT_TRUE(torch::equal(*std::get<side::left -1>(eval)[0], torch::zeros(1, options)));
  EXPECT_TRUE(torch::equal(*std::get<side::right-1>(eval)[0], torch::zeros(1, options)));
  
  eval = boundary.eval<deriv::dx^2, false>(xi);
  
  EXPECT_TRUE(torch::equal(*std::get<side::left -1>(eval)[0], torch::zeros(1, options)));
  EXPECT_TRUE(torch::equal(*std::get<side::right-1>(eval)[0], torch::zeros(1, options)));

  // Evaluation from precomputed coefficients and basis functions
  auto knot_indices  = boundary.find_knot_indices(xi);
  auto coeff_indices = boundary.find_coeff_indices(knot_indices);

  auto numel = [](const auto& xi) { return std::tuple<int64_t, int64_t>{ 1, 1 }; };
  auto sizes = [](const auto& xi) {
    return std::tuple{
      torch::IntArrayRef{},
      torch::IntArrayRef{}
    }; };
    
  auto basfunc = boundary.template eval_basfunc<deriv::func, false>(xi, knot_indices);
  eval         = boundary.eval_from_precomputed(basfunc, coeff_indices, numel(xi), sizes(xi));
    
  EXPECT_TRUE(torch::equal(*std::get<side::left -1>(eval)[0], torch::ones({}, options)));
  EXPECT_TRUE(torch::equal(*std::get<side::right-1>(eval)[0], torch::ones({}, options)));

  basfunc = boundary.template eval_basfunc<deriv::dx, false>(xi, knot_indices);
  eval    = boundary.eval_from_precomputed(basfunc, coeff_indices, numel(xi), sizes(xi));

  EXPECT_TRUE(torch::equal(*std::get<side::left -1>(eval)[0], torch::zeros({}, options)));
  EXPECT_TRUE(torch::equal(*std::get<side::right-1>(eval)[0], torch::zeros({}, options)));

  basfunc = boundary.template eval_basfunc<deriv::dx^2, false>(xi, knot_indices);
  eval    = boundary.eval_from_precomputed(basfunc, coeff_indices, numel(xi), sizes(xi));
    
  EXPECT_TRUE(torch::equal(*std::get<side::left -1>(eval)[0], torch::zeros({}, options)));
  EXPECT_TRUE(torch::equal(*std::get<side::right-1>(eval)[0], torch::zeros({}, options)));
}

TEST_F(BoundaryTest, Boundary_parDim2_geoDim1_degrees23)
{
  using iganet::side;
  using iganet::deriv;
  using BSpline_t = iganet::UniformBSpline<real_t, 1, 2, 3>;
  iganet::Boundary<BSpline_t> boundary({5, 4}, iganet::init::greville, options);

  auto xi  = std::tuple{ iganet::utils::to_tensorArray<real_t>({1.0, 0.2, 0.1, 0.5, 0.9, 0.75, 0.0}, options) /* west  */,
                         iganet::utils::to_tensorArray<real_t>({1.0, 0.2, 0.1, 0.5, 0.9, 0.75, 0.0}, options) /* east  */,
                         iganet::utils::to_tensorArray<real_t>({0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0}, options) /* south */,
                         iganet::utils::to_tensorArray<real_t>({0.0, 0.1, 0.2, 0.5, 0.75, 0.9, 1.0}, options) /* north */};

  // Evaluation
  auto eval = boundary.eval<deriv::func, false>(xi);

  iganet::UniformBSpline<real_t, 1, 2> bspline_bdrNS({5}, iganet::init::greville, options);
  iganet::UniformBSpline<real_t, 1, 3> bspline_bdrEW({4}, iganet::init::greville, options);
  
  EXPECT_TRUE(torch::equal(*(std::get<side::north-1>(eval)[0]),
                           *(bspline_bdrNS.eval<deriv::func, false>(std::get<side::north-1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::south-1>(eval)[0]),
                           *(bspline_bdrNS.eval<deriv::func, false>(std::get<side::south-1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::east-1>(eval)[0]),
                           *(bspline_bdrEW.eval<deriv::func, false>(std::get<side::east-1>(xi))[0])));  
  EXPECT_TRUE(torch::equal(*(std::get<side::west-1>(eval)[0]),
                           *(bspline_bdrEW.eval<deriv::func, false>(std::get<side::west-1>(xi))[0])));

  eval = boundary.eval<deriv::dx, false>(xi);
  
  EXPECT_TRUE(torch::equal(*(std::get<side::north-1>(eval)[0]),
                           *(bspline_bdrNS.eval<deriv::dx, false>(std::get<side::north-1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::south-1>(eval)[0]),
                           *(bspline_bdrNS.eval<deriv::dx, false>(std::get<side::south-1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::east-1>(eval)[0]),
                           *(bspline_bdrEW.eval<deriv::dx, false>(std::get<side::east-1>(xi))[0])));  
  EXPECT_TRUE(torch::equal(*(std::get<side::west-1>(eval)[0]),
                           *(bspline_bdrEW.eval<deriv::dx, false>(std::get<side::west-1>(xi))[0])));

  eval = boundary.eval<deriv::dx^2, false>(xi);
  
  EXPECT_TRUE(torch::equal(*(std::get<side::north-1>(eval)[0]),
                           *(bspline_bdrNS.eval<deriv::dx^2, false>(std::get<side::north-1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::south-1>(eval)[0]),
                           *(bspline_bdrNS.eval<deriv::dx^2, false>(std::get<side::south-1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::east-1>(eval)[0]),
                           *(bspline_bdrEW.eval<deriv::dx^2, false>(std::get<side::east-1>(xi))[0])));  
  EXPECT_TRUE(torch::equal(*(std::get<side::west-1>(eval)[0]),
                           *(bspline_bdrEW.eval<deriv::dx^2, false>(std::get<side::west-1>(xi))[0])));

  eval = boundary.eval<deriv::dy, false>(xi);
  
  EXPECT_TRUE(torch::equal(*(std::get<side::north-1>(eval)[0]),
                           *(bspline_bdrNS.eval<deriv::dy, false>(std::get<side::north-1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::south-1>(eval)[0]),
                           *(bspline_bdrNS.eval<deriv::dy, false>(std::get<side::south-1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::east-1>(eval)[0]),
                           *(bspline_bdrEW.eval<deriv::dy, false>(std::get<side::east-1>(xi))[0])));  
  EXPECT_TRUE(torch::equal(*(std::get<side::west-1>(eval)[0]),
                           *(bspline_bdrEW.eval<deriv::dy, false>(std::get<side::west-1>(xi))[0])));

  eval = boundary.eval<deriv::dy^2, false>(xi);
  
  EXPECT_TRUE(torch::equal(*(std::get<side::north-1>(eval)[0]),
                           *(bspline_bdrNS.eval<deriv::dy^2, false>(std::get<side::north-1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::south-1>(eval)[0]),
                           *(bspline_bdrNS.eval<deriv::dy^2, false>(std::get<side::south-1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::east-1>(eval)[0]),
                           *(bspline_bdrEW.eval<deriv::dy^2, false>(std::get<side::east-1>(xi))[0])));  
  EXPECT_TRUE(torch::equal(*(std::get<side::west-1>(eval)[0]),
                           *(bspline_bdrEW.eval<deriv::dy^2, false>(std::get<side::west-1>(xi))[0])));

  eval = boundary.eval<deriv::dx+deriv::dy, false>(xi);
  
  EXPECT_TRUE(torch::equal(*(std::get<side::north-1>(eval)[0]),
                           *(bspline_bdrNS.eval<deriv::dx+deriv::dy, false>(std::get<side::north-1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::south-1>(eval)[0]),
                           *(bspline_bdrNS.eval<deriv::dx+deriv::dy, false>(std::get<side::south-1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::east-1>(eval)[0]),
                           *(bspline_bdrEW.eval<deriv::dx+deriv::dy, false>(std::get<side::east-1>(xi))[0])));  
  EXPECT_TRUE(torch::equal(*(std::get<side::west-1>(eval)[0]),
                           *(bspline_bdrEW.eval<deriv::dx+deriv::dy, false>(std::get<side::west-1>(xi))[0])));

  // Evaluation from precomputed coefficients and basis functions
  auto knot_indices  = boundary.find_knot_indices(xi);
  auto coeff_indices = boundary.find_coeff_indices(knot_indices);

  auto numel = [](const auto& xi) { return std::tuple{ std::get<0>(xi)[0].numel(),
                                                       std::get<1>(xi)[0].numel(),
                                                       std::get<2>(xi)[0].numel(),
                                                       std::get<3>(xi)[0].numel()}; };
  auto sizes = [](const auto& xi) { return std::tuple{ std::get<0>(xi)[0].sizes(),
                                                       std::get<1>(xi)[0].sizes(),
                                                       std::get<2>(xi)[0].sizes(),
                                                       std::get<3>(xi)[0].sizes()}; };
    
  auto basfunc = boundary.template eval_basfunc<deriv::func, false>(xi, knot_indices);
  eval         = boundary.eval_from_precomputed(basfunc, coeff_indices, numel(xi), sizes(xi));
    
  EXPECT_TRUE(torch::equal(*(std::get<side::north-1>(eval)[0]),
                           *(bspline_bdrNS.eval<deriv::func, false>(std::get<side::north-1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::south-1>(eval)[0]),
                           *(bspline_bdrNS.eval<deriv::func, false>(std::get<side::south-1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::east-1>(eval)[0]),
                           *(bspline_bdrEW.eval<deriv::func, false>(std::get<side::east-1>(xi))[0])));  
  EXPECT_TRUE(torch::equal(*(std::get<side::west-1>(eval)[0]),
                           *(bspline_bdrEW.eval<deriv::func, false>(std::get<side::west-1>(xi))[0])));

  basfunc = boundary.template eval_basfunc<deriv::dx, false>(xi, knot_indices);
  eval    = boundary.eval_from_precomputed(basfunc, coeff_indices, numel(xi), sizes(xi));
  
  EXPECT_TRUE(torch::equal(*(std::get<side::north-1>(eval)[0]),
                           *(bspline_bdrNS.eval<deriv::dx, false>(std::get<side::north-1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::south-1>(eval)[0]),
                           *(bspline_bdrNS.eval<deriv::dx, false>(std::get<side::south-1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::east-1>(eval)[0]),
                           *(bspline_bdrEW.eval<deriv::dx, false>(std::get<side::east-1>(xi))[0])));  
  EXPECT_TRUE(torch::equal(*(std::get<side::west-1>(eval)[0]),
                           *(bspline_bdrEW.eval<deriv::dx, false>(std::get<side::west-1>(xi))[0])));

  basfunc = boundary.template eval_basfunc<deriv::dx^2, false>(xi, knot_indices);
  eval    = boundary.eval_from_precomputed(basfunc, coeff_indices, numel(xi), sizes(xi));
  
  EXPECT_TRUE(torch::equal(*(std::get<side::north-1>(eval)[0]),
                           *(bspline_bdrNS.eval<deriv::dx^2, false>(std::get<side::north-1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::south-1>(eval)[0]),
                           *(bspline_bdrNS.eval<deriv::dx^2, false>(std::get<side::south-1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::east-1>(eval)[0]),
                           *(bspline_bdrEW.eval<deriv::dx^2, false>(std::get<side::east-1>(xi))[0])));  
  EXPECT_TRUE(torch::equal(*(std::get<side::west-1>(eval)[0]),
                           *(bspline_bdrEW.eval<deriv::dx^2, false>(std::get<side::west-1>(xi))[0])));

  basfunc = boundary.template eval_basfunc<deriv::dy, false>(xi, knot_indices);
  eval    = boundary.eval_from_precomputed(basfunc, coeff_indices, numel(xi), sizes(xi));
  
  EXPECT_TRUE(torch::equal(*(std::get<side::north-1>(eval)[0]),
                           *(bspline_bdrNS.eval<deriv::dy, false>(std::get<side::north-1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::south-1>(eval)[0]),
                           *(bspline_bdrNS.eval<deriv::dy, false>(std::get<side::south-1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::east-1>(eval)[0]),
                           *(bspline_bdrEW.eval<deriv::dy, false>(std::get<side::east-1>(xi))[0])));  
  EXPECT_TRUE(torch::equal(*(std::get<side::west-1>(eval)[0]),
                           *(bspline_bdrEW.eval<deriv::dy, false>(std::get<side::west-1>(xi))[0])));

  basfunc = boundary.template eval_basfunc<deriv::dy^2, false>(xi, knot_indices);
  eval    = boundary.eval_from_precomputed(basfunc, coeff_indices, numel(xi), sizes(xi));
  
  EXPECT_TRUE(torch::equal(*(std::get<side::north-1>(eval)[0]),
                           *(bspline_bdrNS.eval<deriv::dy^2, false>(std::get<side::north-1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::south-1>(eval)[0]),
                           *(bspline_bdrNS.eval<deriv::dy^2, false>(std::get<side::south-1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::east-1>(eval)[0]),
                           *(bspline_bdrEW.eval<deriv::dy^2, false>(std::get<side::east-1>(xi))[0])));  
  EXPECT_TRUE(torch::equal(*(std::get<side::west-1>(eval)[0]),
                           *(bspline_bdrEW.eval<deriv::dy^2, false>(std::get<side::west-1>(xi))[0])));

  basfunc = boundary.template eval_basfunc<deriv::dx+deriv::dy, false>(xi, knot_indices);
  eval    = boundary.eval_from_precomputed(basfunc, coeff_indices, numel(xi), sizes(xi));
  
  EXPECT_TRUE(torch::equal(*(std::get<side::north-1>(eval)[0]),
                           *(bspline_bdrNS.eval<deriv::dx+deriv::dy, false>(std::get<side::north-1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::south-1>(eval)[0]),
                           *(bspline_bdrNS.eval<deriv::dx+deriv::dy, false>(std::get<side::south-1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::east-1>(eval)[0]),
                           *(bspline_bdrEW.eval<deriv::dx+deriv::dy, false>(std::get<side::east-1>(xi))[0])));  
  EXPECT_TRUE(torch::equal(*(std::get<side::west-1>(eval)[0]),
                           *(bspline_bdrEW.eval<deriv::dx+deriv::dy, false>(std::get<side::west-1>(xi))[0])));
}

TEST_F(BoundaryTest, Boundary_parDim3_geoDim1_degrees234)
{
  using iganet::side;
  using iganet::deriv;
  using BSpline_t = iganet::UniformBSpline<real_t, 1, 2, 3, 4>;
  iganet::Boundary<BSpline_t> boundary({5, 4, 7}, iganet::init::greville, options);
  
  auto xi  = std::tuple{ iganet::utils::to_tensorArray<real_t>({1.0, 0.2, 0.1,  0.5, 0.9,  0.75, 0.0} /* v */,
                                                               {0.2, 0.5, 0.75, 0.9, 1.0,  0.0,  0.1} /* w */, options) /* west  */,
                         iganet::utils::to_tensorArray<real_t>({1.0, 0.2, 0.1,  0.5, 0.9,  0.75, 0.0} /* v */,
                                                               {0.2, 0.5, 0.75, 0.9, 1.0,  0.0,  0.1} /* w */, options) /* east  */,
                         iganet::utils::to_tensorArray<real_t>({0.0, 0.1, 0.2,  0.5, 0.75, 0.9,  1.0} /* u */,
                                                               {0.2, 0.5, 0.75, 0.9, 1.0,  0.0,  0.1} /* w */, options) /* south */,
                         iganet::utils::to_tensorArray<real_t>({0.0, 0.1, 0.2,  0.5, 0.75, 0.9,  1.0} /* u */,
                                                               {0.2, 0.5, 0.75, 0.9, 1.0,  0.0,  0.1} /* w */, options) /* north */,
                         iganet::utils::to_tensorArray<real_t>({0.0, 0.1, 0.2,  0.5, 0.75, 0.9,  1.0} /* u */,
                                                               {1.0, 0.2, 0.1,  0.5, 0.9,  0.75, 0.0} /* v */, options) /* front */,
                         iganet::utils::to_tensorArray<real_t>({0.0, 0.1, 0.2,  0.5, 0.75, 0.9,  1.0} /* u */,
                                                               {1.0, 0.2, 0.1,  0.5, 0.9,  0.75, 0.0} /* v */, options) /* back  */};

  // Evaluation
  auto eval = boundary.eval<deriv::func, false>(xi);

  iganet::UniformBSpline<real_t, 1, 2, 4> bspline_bdrNS({5, 7}, iganet::init::greville, options);
  iganet::UniformBSpline<real_t, 1, 3, 4> bspline_bdrEW({4, 7}, iganet::init::greville, options);
  iganet::UniformBSpline<real_t, 1, 2, 3> bspline_bdrFB({5, 4}, iganet::init::greville, options);
  
  EXPECT_TRUE(torch::equal(*(std::get<side::north-1>(eval)[0]),
                           *(bspline_bdrNS.eval<deriv::func, false>(std::get<side::north-1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::south-1>(eval)[0]),
                           *(bspline_bdrNS.eval<deriv::func, false>(std::get<side::south-1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::east-1>(eval)[0]),
                           *(bspline_bdrEW.eval<deriv::func, false>(std::get<side::east-1>(xi))[0])));  
  EXPECT_TRUE(torch::equal(*(std::get<side::west-1>(eval)[0]),
                           *(bspline_bdrEW.eval<deriv::func, false>(std::get<side::west-1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::front-1>(eval)[0]),
                           *(bspline_bdrFB.eval<deriv::func, false>(std::get<side::front-1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::back-1>(eval)[0]),
                           *(bspline_bdrFB.eval<deriv::func, false>(std::get<side::back-1>(xi))[0])));

  eval = boundary.eval<deriv::dx, false>(xi);

  EXPECT_TRUE(torch::equal(*(std::get<side::north-1>(eval)[0]),
                           *(bspline_bdrNS.eval<deriv::dx, false>(std::get<side::north-1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::south-1>(eval)[0]),
                           *(bspline_bdrNS.eval<deriv::dx, false>(std::get<side::south-1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::east-1>(eval)[0]),
                           *(bspline_bdrEW.eval<deriv::dx, false>(std::get<side::east-1>(xi))[0])));  
  EXPECT_TRUE(torch::equal(*(std::get<side::west-1>(eval)[0]),
                           *(bspline_bdrEW.eval<deriv::dx, false>(std::get<side::west-1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::front-1>(eval)[0]),
                           *(bspline_bdrFB.eval<deriv::dx, false>(std::get<side::front-1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::back-1>(eval)[0]),
                           *(bspline_bdrFB.eval<deriv::dx, false>(std::get<side::back-1>(xi))[0])));

  eval = boundary.eval<deriv::dx^2, false>(xi);

  EXPECT_TRUE(torch::equal(*(std::get<side::north-1>(eval)[0]),
                           *(bspline_bdrNS.eval<deriv::dx^2, false>(std::get<side::north-1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::south-1>(eval)[0]),
                           *(bspline_bdrNS.eval<deriv::dx^2, false>(std::get<side::south-1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::east-1>(eval)[0]),
                           *(bspline_bdrEW.eval<deriv::dx^2, false>(std::get<side::east-1>(xi))[0])));  
  EXPECT_TRUE(torch::equal(*(std::get<side::west-1>(eval)[0]),
                           *(bspline_bdrEW.eval<deriv::dx^2, false>(std::get<side::west-1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::front-1>(eval)[0]),
                           *(bspline_bdrFB.eval<deriv::dx^2, false>(std::get<side::front-1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::back-1>(eval)[0]),
                           *(bspline_bdrFB.eval<deriv::dx^2, false>(std::get<side::back-1>(xi))[0])));

  eval = boundary.eval<deriv::dy, false>(xi);

  EXPECT_TRUE(torch::equal(*(std::get<side::north-1>(eval)[0]),
                           *(bspline_bdrNS.eval<deriv::dy, false>(std::get<side::north-1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::south-1>(eval)[0]),
                           *(bspline_bdrNS.eval<deriv::dy, false>(std::get<side::south-1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::east-1>(eval)[0]),
                           *(bspline_bdrEW.eval<deriv::dy, false>(std::get<side::east-1>(xi))[0])));  
  EXPECT_TRUE(torch::equal(*(std::get<side::west-1>(eval)[0]),
                           *(bspline_bdrEW.eval<deriv::dy, false>(std::get<side::west-1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::front-1>(eval)[0]),
                           *(bspline_bdrFB.eval<deriv::dy, false>(std::get<side::front-1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::back-1>(eval)[0]),
                           *(bspline_bdrFB.eval<deriv::dy, false>(std::get<side::back-1>(xi))[0])));

  eval = boundary.eval<deriv::dy^2, false>(xi);

  EXPECT_TRUE(torch::equal(*(std::get<side::north-1>(eval)[0]),
                           *(bspline_bdrNS.eval<deriv::dy^2, false>(std::get<side::north-1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::south-1>(eval)[0]),
                           *(bspline_bdrNS.eval<deriv::dy^2, false>(std::get<side::south-1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::east-1>(eval)[0]),
                           *(bspline_bdrEW.eval<deriv::dy^2, false>(std::get<side::east-1>(xi))[0])));  
  EXPECT_TRUE(torch::equal(*(std::get<side::west-1>(eval)[0]),
                           *(bspline_bdrEW.eval<deriv::dy^2, false>(std::get<side::west-1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::front-1>(eval)[0]),
                           *(bspline_bdrFB.eval<deriv::dy^2, false>(std::get<side::front-1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::back-1>(eval)[0]),
                           *(bspline_bdrFB.eval<deriv::dy^2, false>(std::get<side::back-1>(xi))[0])));

  eval = boundary.eval<deriv::dz, false>(xi);

  EXPECT_TRUE(torch::equal(*(std::get<side::north-1>(eval)[0]),
                           *(bspline_bdrNS.eval<deriv::dz, false>(std::get<side::north-1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::south-1>(eval)[0]),
                           *(bspline_bdrNS.eval<deriv::dz, false>(std::get<side::south-1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::east-1>(eval)[0]),
                           *(bspline_bdrEW.eval<deriv::dz, false>(std::get<side::east-1>(xi))[0])));  
  EXPECT_TRUE(torch::equal(*(std::get<side::west-1>(eval)[0]),
                           *(bspline_bdrEW.eval<deriv::dz, false>(std::get<side::west-1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::front-1>(eval)[0]),
                           *(bspline_bdrFB.eval<deriv::dz, false>(std::get<side::front-1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::back-1>(eval)[0]),
                           *(bspline_bdrFB.eval<deriv::dz, false>(std::get<side::back-1>(xi))[0])));

  eval = boundary.eval<deriv::dz^2, false>(xi);

  EXPECT_TRUE(torch::equal(*(std::get<side::north-1>(eval)[0]),
                           *(bspline_bdrNS.eval<deriv::dz^2, false>(std::get<side::north-1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::south-1>(eval)[0]),
                           *(bspline_bdrNS.eval<deriv::dz^2, false>(std::get<side::south-1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::east-1>(eval)[0]),
                           *(bspline_bdrEW.eval<deriv::dz^2, false>(std::get<side::east-1>(xi))[0])));  
  EXPECT_TRUE(torch::equal(*(std::get<side::west-1>(eval)[0]),
                           *(bspline_bdrEW.eval<deriv::dz^2, false>(std::get<side::west-1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::front-1>(eval)[0]),
                           *(bspline_bdrFB.eval<deriv::dz^2, false>(std::get<side::front-1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::back-1>(eval)[0]),
                           *(bspline_bdrFB.eval<deriv::dz^2, false>(std::get<side::back-1>(xi))[0])));

  eval = boundary.eval<deriv::dx+deriv::dy, false>(xi);

  EXPECT_TRUE(torch::equal(*(std::get<side::north-1>(eval)[0]),
                           *(bspline_bdrNS.eval<deriv::dx+deriv::dy, false>(std::get<side::north-1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::south-1>(eval)[0]),
                           *(bspline_bdrNS.eval<deriv::dx+deriv::dy, false>(std::get<side::south-1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::east-1>(eval)[0]),
                           *(bspline_bdrEW.eval<deriv::dx+deriv::dy, false>(std::get<side::east-1>(xi))[0])));  
  EXPECT_TRUE(torch::equal(*(std::get<side::west-1>(eval)[0]),
                           *(bspline_bdrEW.eval<deriv::dx+deriv::dy, false>(std::get<side::west-1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::front-1>(eval)[0]),
                           *(bspline_bdrFB.eval<deriv::dx+deriv::dy, false>(std::get<side::front-1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::back-1>(eval)[0]),
                           *(bspline_bdrFB.eval<deriv::dx+deriv::dy, false>(std::get<side::back-1>(xi))[0])));

  eval = boundary.eval<deriv::dx+deriv::dz, false>(xi);

  EXPECT_TRUE(torch::equal(*(std::get<side::north-1>(eval)[0]),
                           *(bspline_bdrNS.eval<deriv::dx+deriv::dz, false>(std::get<side::north-1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::south-1>(eval)[0]),
                           *(bspline_bdrNS.eval<deriv::dx+deriv::dz, false>(std::get<side::south-1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::east-1>(eval)[0]),
                           *(bspline_bdrEW.eval<deriv::dx+deriv::dz, false>(std::get<side::east-1>(xi))[0])));  
  EXPECT_TRUE(torch::equal(*(std::get<side::west-1>(eval)[0]),
                           *(bspline_bdrEW.eval<deriv::dx+deriv::dz, false>(std::get<side::west-1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::front-1>(eval)[0]),
                           *(bspline_bdrFB.eval<deriv::dx+deriv::dz, false>(std::get<side::front-1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::back-1>(eval)[0]),
                           *(bspline_bdrFB.eval<deriv::dx+deriv::dz, false>(std::get<side::back-1>(xi))[0])));

  eval = boundary.eval<deriv::dy+deriv::dz, false>(xi);

  EXPECT_TRUE(torch::equal(*(std::get<side::north-1>(eval)[0]),
                           *(bspline_bdrNS.eval<deriv::dy+deriv::dz, false>(std::get<side::north-1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::south-1>(eval)[0]),
                           *(bspline_bdrNS.eval<deriv::dy+deriv::dz, false>(std::get<side::south-1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::east-1>(eval)[0]),
                           *(bspline_bdrEW.eval<deriv::dy+deriv::dz, false>(std::get<side::east-1>(xi))[0])));  
  EXPECT_TRUE(torch::equal(*(std::get<side::west-1>(eval)[0]),
                           *(bspline_bdrEW.eval<deriv::dy+deriv::dz, false>(std::get<side::west-1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::front-1>(eval)[0]),
                           *(bspline_bdrFB.eval<deriv::dy+deriv::dz, false>(std::get<side::front-1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::back-1>(eval)[0]),
                           *(bspline_bdrFB.eval<deriv::dy+deriv::dz, false>(std::get<side::back-1>(xi))[0])));

  // Evaluation from precomputed coefficients and basis functions
  auto knot_indices  = boundary.find_knot_indices(xi);
  auto coeff_indices = boundary.find_coeff_indices(knot_indices);
  
  auto numel = [](const auto& xi) { return std::tuple{ std::get<0>(xi)[0].numel(),
                                                       std::get<1>(xi)[0].numel(),
                                                       std::get<2>(xi)[0].numel(),
                                                       std::get<3>(xi)[0].numel(),
                                                       std::get<4>(xi)[0].numel(),
                                                       std::get<5>(xi)[0].numel()}; };
  auto sizes = [](const auto& xi) { return std::tuple{ std::get<0>(xi)[0].sizes(),
                                                       std::get<1>(xi)[0].sizes(),
                                                       std::get<2>(xi)[0].sizes(),
                                                       std::get<3>(xi)[0].sizes(),                                                         
                                                       std::get<4>(xi)[0].sizes(),
                                                       std::get<5>(xi)[0].sizes()}; };
    
  auto basfunc = boundary.template eval_basfunc<deriv::func, false>(xi, knot_indices);
  eval         = boundary.eval_from_precomputed(basfunc, coeff_indices, numel(xi), sizes(xi));

  EXPECT_TRUE(torch::equal(*(std::get<side::north-1>(eval)[0]),
                           *(bspline_bdrNS.eval<deriv::func, false>(std::get<side::north-1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::south-1>(eval)[0]),
                           *(bspline_bdrNS.eval<deriv::func, false>(std::get<side::south-1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::east-1>(eval)[0]),
                           *(bspline_bdrEW.eval<deriv::func, false>(std::get<side::east-1>(xi))[0])));  
  EXPECT_TRUE(torch::equal(*(std::get<side::west-1>(eval)[0]),
                           *(bspline_bdrEW.eval<deriv::func, false>(std::get<side::west-1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::front-1>(eval)[0]),
                           *(bspline_bdrFB.eval<deriv::func, false>(std::get<side::front-1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::back-1>(eval)[0]),
                           *(bspline_bdrFB.eval<deriv::func, false>(std::get<side::back-1>(xi))[0])));

  basfunc = boundary.template eval_basfunc<deriv::dx, false>(xi, knot_indices);
  eval    = boundary.eval_from_precomputed(basfunc, coeff_indices, numel(xi), sizes(xi));

  EXPECT_TRUE(torch::equal(*(std::get<side::north-1>(eval)[0]),
                           *(bspline_bdrNS.eval<deriv::dx, false>(std::get<side::north-1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::south-1>(eval)[0]),
                           *(bspline_bdrNS.eval<deriv::dx, false>(std::get<side::south-1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::east-1>(eval)[0]),
                           *(bspline_bdrEW.eval<deriv::dx, false>(std::get<side::east-1>(xi))[0])));  
  EXPECT_TRUE(torch::equal(*(std::get<side::west-1>(eval)[0]),
                           *(bspline_bdrEW.eval<deriv::dx, false>(std::get<side::west-1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::front-1>(eval)[0]),
                           *(bspline_bdrFB.eval<deriv::dx, false>(std::get<side::front-1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::back-1>(eval)[0]),
                           *(bspline_bdrFB.eval<deriv::dx, false>(std::get<side::back-1>(xi))[0])));

  basfunc = boundary.template eval_basfunc<deriv::dx^2, false>(xi, knot_indices);
  eval    = boundary.eval_from_precomputed(basfunc, coeff_indices, numel(xi), sizes(xi));

  EXPECT_TRUE(torch::equal(*(std::get<side::north-1>(eval)[0]),
                           *(bspline_bdrNS.eval<deriv::dx^2, false>(std::get<side::north-1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::south-1>(eval)[0]),
                           *(bspline_bdrNS.eval<deriv::dx^2, false>(std::get<side::south-1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::east-1>(eval)[0]),
                           *(bspline_bdrEW.eval<deriv::dx^2, false>(std::get<side::east-1>(xi))[0])));  
  EXPECT_TRUE(torch::equal(*(std::get<side::west-1>(eval)[0]),
                           *(bspline_bdrEW.eval<deriv::dx^2, false>(std::get<side::west-1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::front-1>(eval)[0]),
                           *(bspline_bdrFB.eval<deriv::dx^2, false>(std::get<side::front-1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::back-1>(eval)[0]),
                           *(bspline_bdrFB.eval<deriv::dx^2, false>(std::get<side::back-1>(xi))[0])));

  basfunc = boundary.template eval_basfunc<deriv::dy, false>(xi, knot_indices);
  eval    = boundary.eval_from_precomputed(basfunc, coeff_indices, numel(xi), sizes(xi));

  EXPECT_TRUE(torch::equal(*(std::get<side::north-1>(eval)[0]),
                           *(bspline_bdrNS.eval<deriv::dy, false>(std::get<side::north-1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::south-1>(eval)[0]),
                           *(bspline_bdrNS.eval<deriv::dy, false>(std::get<side::south-1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::east-1>(eval)[0]),
                           *(bspline_bdrEW.eval<deriv::dy, false>(std::get<side::east-1>(xi))[0])));  
  EXPECT_TRUE(torch::equal(*(std::get<side::west-1>(eval)[0]),
                           *(bspline_bdrEW.eval<deriv::dy, false>(std::get<side::west-1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::front-1>(eval)[0]),
                           *(bspline_bdrFB.eval<deriv::dy, false>(std::get<side::front-1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::back-1>(eval)[0]),
                           *(bspline_bdrFB.eval<deriv::dy, false>(std::get<side::back-1>(xi))[0])));

  basfunc = boundary.template eval_basfunc<deriv::dy^2, false>(xi, knot_indices);
  eval    = boundary.eval_from_precomputed(basfunc, coeff_indices, numel(xi), sizes(xi));       

  EXPECT_TRUE(torch::equal(*(std::get<side::north-1>(eval)[0]),
                           *(bspline_bdrNS.eval<deriv::dy^2, false>(std::get<side::north-1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::south-1>(eval)[0]),
                           *(bspline_bdrNS.eval<deriv::dy^2, false>(std::get<side::south-1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::east-1>(eval)[0]),
                           *(bspline_bdrEW.eval<deriv::dy^2, false>(std::get<side::east-1>(xi))[0])));  
  EXPECT_TRUE(torch::equal(*(std::get<side::west-1>(eval)[0]),
                           *(bspline_bdrEW.eval<deriv::dy^2, false>(std::get<side::west-1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::front-1>(eval)[0]),
                           *(bspline_bdrFB.eval<deriv::dy^2, false>(std::get<side::front-1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::back-1>(eval)[0]),
                           *(bspline_bdrFB.eval<deriv::dy^2, false>(std::get<side::back-1>(xi))[0])));

  basfunc = boundary.template eval_basfunc<deriv::dz, false>(xi, knot_indices);
  eval    = boundary.eval_from_precomputed(basfunc, coeff_indices, numel(xi), sizes(xi));
        
  EXPECT_TRUE(torch::equal(*(std::get<side::north-1>(eval)[0]),
                           *(bspline_bdrNS.eval<deriv::dz, false>(std::get<side::north-1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::south-1>(eval)[0]),
                           *(bspline_bdrNS.eval<deriv::dz, false>(std::get<side::south-1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::east-1>(eval)[0]),
                           *(bspline_bdrEW.eval<deriv::dz, false>(std::get<side::east-1>(xi))[0])));  
  EXPECT_TRUE(torch::equal(*(std::get<side::west-1>(eval)[0]),
                           *(bspline_bdrEW.eval<deriv::dz, false>(std::get<side::west-1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::front-1>(eval)[0]),
                           *(bspline_bdrFB.eval<deriv::dz, false>(std::get<side::front-1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::back-1>(eval)[0]),
                           *(bspline_bdrFB.eval<deriv::dz, false>(std::get<side::back-1>(xi))[0])));

  basfunc = boundary.template eval_basfunc<deriv::dz^2, false>(xi, knot_indices);
  eval    = boundary.eval_from_precomputed(basfunc, coeff_indices, numel(xi), sizes(xi));        

  EXPECT_TRUE(torch::equal(*(std::get<side::north-1>(eval)[0]),
                           *(bspline_bdrNS.eval<deriv::dz^2, false>(std::get<side::north-1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::south-1>(eval)[0]),
                           *(bspline_bdrNS.eval<deriv::dz^2, false>(std::get<side::south-1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::east-1>(eval)[0]),
                           *(bspline_bdrEW.eval<deriv::dz^2, false>(std::get<side::east-1>(xi))[0])));  
  EXPECT_TRUE(torch::equal(*(std::get<side::west-1>(eval)[0]),
                           *(bspline_bdrEW.eval<deriv::dz^2, false>(std::get<side::west-1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::front-1>(eval)[0]),
                           *(bspline_bdrFB.eval<deriv::dz^2, false>(std::get<side::front-1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::back-1>(eval)[0]),
                           *(bspline_bdrFB.eval<deriv::dz^2, false>(std::get<side::back-1>(xi))[0])));

  basfunc = boundary.template eval_basfunc<deriv::dx+deriv::dy, false>(xi, knot_indices);
  eval    = boundary.eval_from_precomputed(basfunc, coeff_indices, numel(xi), sizes(xi));        

  EXPECT_TRUE(torch::equal(*(std::get<side::north-1>(eval)[0]),
                           *(bspline_bdrNS.eval<deriv::dx+deriv::dy, false>(std::get<side::north-1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::south-1>(eval)[0]),
                           *(bspline_bdrNS.eval<deriv::dx+deriv::dy, false>(std::get<side::south-1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::east-1>(eval)[0]),
                           *(bspline_bdrEW.eval<deriv::dx+deriv::dy, false>(std::get<side::east-1>(xi))[0])));  
  EXPECT_TRUE(torch::equal(*(std::get<side::west-1>(eval)[0]),
                           *(bspline_bdrEW.eval<deriv::dx+deriv::dy, false>(std::get<side::west-1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::front-1>(eval)[0]),
                           *(bspline_bdrFB.eval<deriv::dx+deriv::dy, false>(std::get<side::front-1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::back-1>(eval)[0]),
                           *(bspline_bdrFB.eval<deriv::dx+deriv::dy, false>(std::get<side::back-1>(xi))[0])));

  basfunc = boundary.template eval_basfunc<deriv::dx+deriv::dz, false>(xi, knot_indices);
  eval    = boundary.eval_from_precomputed(basfunc, coeff_indices, numel(xi), sizes(xi));

  EXPECT_TRUE(torch::equal(*(std::get<side::north-1>(eval)[0]),
                           *(bspline_bdrNS.eval<deriv::dx+deriv::dz, false>(std::get<side::north-1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::south-1>(eval)[0]),
                           *(bspline_bdrNS.eval<deriv::dx+deriv::dz, false>(std::get<side::south-1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::east-1>(eval)[0]),
                           *(bspline_bdrEW.eval<deriv::dx+deriv::dz, false>(std::get<side::east-1>(xi))[0])));  
  EXPECT_TRUE(torch::equal(*(std::get<side::west-1>(eval)[0]),
                           *(bspline_bdrEW.eval<deriv::dx+deriv::dz, false>(std::get<side::west-1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::front-1>(eval)[0]),
                           *(bspline_bdrFB.eval<deriv::dx+deriv::dz, false>(std::get<side::front-1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::back-1>(eval)[0]),
                           *(bspline_bdrFB.eval<deriv::dx+deriv::dz, false>(std::get<side::back-1>(xi))[0])));

  basfunc = boundary.template eval_basfunc<deriv::dy+deriv::dz, false>(xi, knot_indices);
  eval    = boundary.eval_from_precomputed(basfunc, coeff_indices, numel(xi), sizes(xi));

  EXPECT_TRUE(torch::equal(*(std::get<side::north-1>(eval)[0]),
                           *(bspline_bdrNS.eval<deriv::dy+deriv::dz, false>(std::get<side::north-1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::south-1>(eval)[0]),
                           *(bspline_bdrNS.eval<deriv::dy+deriv::dz, false>(std::get<side::south-1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::east-1>(eval)[0]),
                           *(bspline_bdrEW.eval<deriv::dy+deriv::dz, false>(std::get<side::east-1>(xi))[0])));  
  EXPECT_TRUE(torch::equal(*(std::get<side::west-1>(eval)[0]),
                           *(bspline_bdrEW.eval<deriv::dy+deriv::dz, false>(std::get<side::west-1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::front-1>(eval)[0]),
                           *(bspline_bdrFB.eval<deriv::dy+deriv::dz, false>(std::get<side::front-1>(xi))[0])));
  EXPECT_TRUE(torch::equal(*(std::get<side::back-1>(eval)[0]),
                           *(bspline_bdrFB.eval<deriv::dy+deriv::dz, false>(std::get<side::back-1>(xi))[0])));
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  iganet::init();
  return RUN_ALL_TESTS();
}
