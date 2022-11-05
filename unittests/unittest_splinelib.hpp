/**
   @file unittests/unittest_splinelib.hpp

   @brief SplineLib helper functions

   @author Matthias Moller

   @copyright This file is part of the IgaNet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include <iganet.hpp>
#include <iostream>

#include <Sources/Splines/b_spline.hpp>
#include <gtest/gtest.h>

#pragma once

template<typename BSpline_t>
auto to_splinelib_bspline(BSpline_t& bspline)
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

  return splinelib_bspline;
}

template<iganet::BSplineDeriv deriv, bool precompute,
         typename BSpline_t, typename SplineLibBSpline_t, typename TensorArray_t>
void test_bspline_eval(BSpline_t& bspline, SplineLibBSpline_t splinelib_bspline,
                       TensorArray_t& xi, typename BSpline_t::value_type tol = 1e-12)
{
  // B-spline evaluation
  using ParametricCoordinate       = typename SplineLibBSpline_t::ParametricCoordinate_;
  using ScalarParametricCoordinate = typename ParametricCoordinate::value_type;
  using Derivative                 = typename SplineLibBSpline_t::ParameterSpace_::Derivative_;
  using ScalarDerivative           = typename Derivative::value_type;
  
  using BSplineValue_t = iganet::BlockTensor<torch::Tensor, 1, BSpline_t::geoDim()>;
  
  BSplineValue_t bspline_val;
  if constexpr (precompute) {
    auto knot_idx  = bspline.eval_knot_indices(xi);
    auto basfunc   = bspline.template eval_basfunc<deriv>(xi, knot_idx);
    auto coeff_idx = bspline.eval_coeff_indices(knot_idx);
    bspline_val = bspline.eval_from_precomputed(basfunc, coeff_idx, xi[0].numel(), xi[0].sizes());
  } else
    bspline_val = bspline.template eval<deriv>(xi);

  for (int64_t i=0; i<xi[0].size(0); ++i) {
    if constexpr (BSpline_t::parDim() == 1 &&
                  BSpline_t::geoDim() == 1)
      EXPECT_NEAR(bspline_val(0)[i].template item<typename BSpline_t::value_type>(),
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
        EXPECT_NEAR(bspline_val(k)[i].template item<typename BSpline_t::value_type>(),
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
      EXPECT_NEAR(bspline_val(0)[i].template item<typename BSpline_t::value_type>(),
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
        EXPECT_NEAR(bspline_val(k)[i].template item<typename BSpline_t::value_type>(),
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
      EXPECT_NEAR(bspline_val(0)[i].template item<typename BSpline_t::value_type>(),
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
        EXPECT_NEAR(bspline_val(k)[i].template item<typename BSpline_t::value_type>(),
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
      EXPECT_NEAR(bspline_val(0)[i].template item<typename BSpline_t::value_type>(),
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
        EXPECT_NEAR(bspline_val(k)[i].template item<typename BSpline_t::value_type>(),
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

template<bool precompute, typename BSpline_t, typename TensorArray_t>
void test_bspline_grad(BSpline_t& bspline, TensorArray_t& xi,
                       typename BSpline_t::value_type tol = 1e-12)
{
  iganet::BlockTensor<torch::Tensor, 1, BSpline_t::parDim()> bspline_grad_val;
  if constexpr (precompute) {
    auto knot_idx  = bspline.eval_knot_indices(xi);
    auto coeff_idx = bspline.eval_coeff_indices(knot_idx);
    bspline_grad_val = bspline.grad(xi, knot_idx, coeff_idx);
  } else
    bspline_grad_val = bspline.grad(xi);

  if constexpr (BSpline_t::parDim() == 1) {
    EXPECT_TRUE(torch::allclose(bspline_grad_val(0,0),
                                bspline.template eval<iganet::BSplineDeriv::dx>(xi)(0)
                                ));
  }
  else if constexpr (BSpline_t::parDim() == 2) {
    EXPECT_TRUE(torch::allclose(bspline_grad_val(0,0),
                                bspline.template eval<iganet::BSplineDeriv::dx>(xi)(0)
                                ));
    EXPECT_TRUE(torch::allclose(bspline_grad_val(0,1),
                                bspline.template eval<iganet::BSplineDeriv::dy>(xi)(0)
                                ));    
  }
  else if constexpr (BSpline_t::parDim() == 3) {
    EXPECT_TRUE(torch::allclose(bspline_grad_val(0,0),
                                bspline.template eval<iganet::BSplineDeriv::dx>(xi)(0)
                                ));
    EXPECT_TRUE(torch::allclose(bspline_grad_val(0,1),
                                bspline.template eval<iganet::BSplineDeriv::dy>(xi)(0)
                                ));
    EXPECT_TRUE(torch::allclose(bspline_grad_val(0,2),
                                bspline.template eval<iganet::BSplineDeriv::dz>(xi)(0)
                                ));
  }
  else if constexpr (BSpline_t::parDim() == 4) {
    EXPECT_TRUE(torch::allclose(bspline_grad_val(0,0),
                                bspline.template eval<iganet::BSplineDeriv::dx>(xi)(0)
                                ));
    EXPECT_TRUE(torch::allclose(bspline_grad_val(0,1),
                                bspline.template eval<iganet::BSplineDeriv::dy>(xi)(0)
                                ));
    EXPECT_TRUE(torch::allclose(bspline_grad_val(0,2),
                                bspline.template eval<iganet::BSplineDeriv::dz>(xi)(0)
                                ));
    EXPECT_TRUE(torch::allclose(bspline_grad_val(0,3),
                                bspline.template eval<iganet::BSplineDeriv::dt>(xi)(0)
                                ));
  }
}

template<bool precompute, typename BSpline_t, typename TensorArray_t>
void test_bspline_jac(BSpline_t& bspline, TensorArray_t& xi,
                      typename BSpline_t::value_type tol = 1e-12)
{
  iganet::BlockTensor<torch::Tensor, BSpline_t::geoDim(), BSpline_t::parDim()> bspline_jac_val;
  
  if constexpr (precompute) {
    auto knot_idx  = bspline.eval_knot_indices(xi);
    auto coeff_idx = bspline.eval_coeff_indices(knot_idx);
    bspline_jac_val = bspline.jac(xi, knot_idx, coeff_idx);
  } else
    bspline_jac_val = bspline.jac(xi);

  if constexpr (BSpline_t::parDim() >= 1) {
    for (short_t k=0; k<BSpline_t::geoDim(); ++k)
      EXPECT_TRUE(torch::allclose(bspline_jac_val(k,0),
                                  bspline.template eval<iganet::BSplineDeriv::dx>(xi)(k)
                                  ));
  }

  if constexpr (BSpline_t::parDim() >= 2) {
    for (short_t k=0; k<BSpline_t::geoDim(); ++k)
      EXPECT_TRUE(torch::allclose(bspline_jac_val(k,1),
                                  bspline.template eval<iganet::BSplineDeriv::dy>(xi)(k)
                                  ));
  }
  
  if constexpr (BSpline_t::parDim() >= 3) {
    for (short_t k=0; k<BSpline_t::geoDim(); ++k)
      EXPECT_TRUE(torch::allclose(bspline_jac_val(k,2),
                                  bspline.template eval<iganet::BSplineDeriv::dz>(xi)(k)
                                  ));
  }

  if constexpr (BSpline_t::parDim() >= 4) {
    for (short_t k=0; k<BSpline_t::geoDim(); ++k)
      EXPECT_TRUE(torch::allclose(bspline_jac_val(k,3),
                                  bspline.template eval<iganet::BSplineDeriv::dt>(xi)(k)
                                  ));
  }
}

template<bool precompute, typename BSpline_t, typename TensorArray_t>
void test_bspline_hess(BSpline_t& bspline, TensorArray_t& xi,
                       typename BSpline_t::value_type tol = 1e-12)
{
  iganet::BlockTensor<torch::Tensor,
                      BSpline_t::parDim(), BSpline_t::parDim(),
                      BSpline_t::geoDim()> bspline_hess_val;
  
  if constexpr (precompute) {
    auto knot_idx  = bspline.eval_knot_indices(xi);
    auto coeff_idx = bspline.eval_coeff_indices(knot_idx);
    bspline_hess_val = bspline.hess(xi, knot_idx, coeff_idx);
  } else
    bspline_hess_val = bspline.hess(xi);

  if constexpr (BSpline_t::parDim() >= 1) {
    for (short_t k=0; k<BSpline_t::geoDim(); ++k)
      EXPECT_TRUE(torch::allclose(bspline_hess_val(0,0,k),
                                  bspline.template eval<iganet::BSplineDeriv::dx2>(xi)(k)
                                  ));
  }

  if constexpr (BSpline_t::parDim() >= 2) {
    for (short_t k=0; k<BSpline_t::geoDim(); ++k) {
      EXPECT_TRUE(torch::allclose(bspline_hess_val(0,1,k),
                                  bspline.template eval<iganet::BSplineDeriv::dxdy>(xi)(k)
                                  ));
      EXPECT_TRUE(torch::allclose(bspline_hess_val(1,0,k),
                                  bspline.template eval<iganet::BSplineDeriv::dydx>(xi)(k)
                                  ));
      EXPECT_TRUE(torch::allclose(bspline_hess_val(1,1,k),
                                  bspline.template eval<iganet::BSplineDeriv::dy2>(xi)(k)
                                  ));
    }
  }
  
  if constexpr (BSpline_t::parDim() >= 3) {
    for (short_t k=0; k<BSpline_t::geoDim(); ++k) {
      EXPECT_TRUE(torch::allclose(bspline_hess_val(0,2,k),
                                  bspline.template eval<iganet::BSplineDeriv::dxdz>(xi)(k)
                                  ));
      EXPECT_TRUE(torch::allclose(bspline_hess_val(1,2,k),
                                  bspline.template eval<iganet::BSplineDeriv::dydz>(xi)(k)
                                  ));
      EXPECT_TRUE(torch::allclose(bspline_hess_val(2,0,k),
                                  bspline.template eval<iganet::BSplineDeriv::dzdx>(xi)(k)
                                  ));
      EXPECT_TRUE(torch::allclose(bspline_hess_val(2,1,k),
                                  bspline.template eval<iganet::BSplineDeriv::dzdy>(xi)(k)
                                  ));
      EXPECT_TRUE(torch::allclose(bspline_hess_val(2,2,k),
                                  bspline.template eval<iganet::BSplineDeriv::dz2>(xi)(k)
                                  ));
    }
  }

  if constexpr (BSpline_t::parDim() >= 4) {
    for (short_t k=0; k<BSpline_t::geoDim(); ++k) {
      EXPECT_TRUE(torch::allclose(bspline_hess_val(0,3,k),
                                  bspline.template eval<iganet::BSplineDeriv::dxdt>(xi)(k)
                                  ));
      EXPECT_TRUE(torch::allclose(bspline_hess_val(1,3,k),
                                  bspline.template eval<iganet::BSplineDeriv::dydt>(xi)(k)
                                  ));
      EXPECT_TRUE(torch::allclose(bspline_hess_val(2,3,k),
                                  bspline.template eval<iganet::BSplineDeriv::dzdt>(xi)(k)
                                  ));
      EXPECT_TRUE(torch::allclose(bspline_hess_val(3,0,k),
                                  bspline.template eval<iganet::BSplineDeriv::dtdx>(xi)(k)
                                  ));
      EXPECT_TRUE(torch::allclose(bspline_hess_val(3,1,k),
                                  bspline.template eval<iganet::BSplineDeriv::dtdy>(xi)(k)
                                  ));
      EXPECT_TRUE(torch::allclose(bspline_hess_val(3,2,k),
                                  bspline.template eval<iganet::BSplineDeriv::dtdz>(xi)(k)
                                  ));
      EXPECT_TRUE(torch::allclose(bspline_hess_val(3,3,k),
                                  bspline.template eval<iganet::BSplineDeriv::dt2>(xi)(k)
                                  ));                        
    }
  }
}

template<typename BSpline_t, typename TensorArray_t>
void test_bspline_eval(BSpline_t& bspline, TensorArray_t& xi, typename BSpline_t::value_type tol = 1e-12)
{
  // Create B-Spline
  auto splinelib_bspline = to_splinelib_bspline(bspline);

  // Evaluate function and derivatives
  test_bspline_eval<iganet::BSplineDeriv::func, false>(bspline, splinelib_bspline, xi, tol);

  if constexpr (BSpline_t::parDim() == 1) {
    test_bspline_eval<iganet::BSplineDeriv::dx1, false>(bspline, splinelib_bspline, xi, tol);
    test_bspline_eval<iganet::BSplineDeriv::dx2, false>(bspline, splinelib_bspline, xi, tol);
    test_bspline_eval<iganet::BSplineDeriv::dx3, false>(bspline, splinelib_bspline, xi, tol);
    test_bspline_eval<iganet::BSplineDeriv::dx4, false>(bspline, splinelib_bspline, xi, tol);
  }

  if constexpr (BSpline_t::parDim() == 2) {
    test_bspline_eval<iganet::BSplineDeriv::dy1, false>(bspline, splinelib_bspline, xi, tol);
    test_bspline_eval<iganet::BSplineDeriv::dy2, false>(bspline, splinelib_bspline, xi, tol);
    test_bspline_eval<iganet::BSplineDeriv::dy3, false>(bspline, splinelib_bspline, xi, tol);
    test_bspline_eval<iganet::BSplineDeriv::dy4, false>(bspline, splinelib_bspline, xi, tol);
  }

  if constexpr (BSpline_t::parDim() == 3) {
    test_bspline_eval<iganet::BSplineDeriv::dz1, false>(bspline, splinelib_bspline, xi, tol);
    test_bspline_eval<iganet::BSplineDeriv::dz2, false>(bspline, splinelib_bspline, xi, tol);
    test_bspline_eval<iganet::BSplineDeriv::dz3, false>(bspline, splinelib_bspline, xi, tol);
    test_bspline_eval<iganet::BSplineDeriv::dz4, false>(bspline, splinelib_bspline, xi, tol);
  }

  if constexpr (BSpline_t::parDim() == 4) {
    test_bspline_eval<iganet::BSplineDeriv::dt1, false>(bspline, splinelib_bspline, xi, tol);
    test_bspline_eval<iganet::BSplineDeriv::dt2, false>(bspline, splinelib_bspline, xi, tol);
    test_bspline_eval<iganet::BSplineDeriv::dt3, false>(bspline, splinelib_bspline, xi, tol);
    test_bspline_eval<iganet::BSplineDeriv::dt4, false>(bspline, splinelib_bspline, xi, tol);
  }
  
  // Evaluate function and derivatives from precomputed data
  test_bspline_eval<iganet::BSplineDeriv::func, true>(bspline, splinelib_bspline, xi, tol);

  if constexpr (BSpline_t::parDim() == 1) {
    test_bspline_eval<iganet::BSplineDeriv::dx1, true>(bspline, splinelib_bspline, xi, tol);
    test_bspline_eval<iganet::BSplineDeriv::dx2, true>(bspline, splinelib_bspline, xi, tol);
    test_bspline_eval<iganet::BSplineDeriv::dx3, true>(bspline, splinelib_bspline, xi, tol);
    test_bspline_eval<iganet::BSplineDeriv::dx4, true>(bspline, splinelib_bspline, xi, tol);
  }

  if constexpr (BSpline_t::parDim() == 2) {
    test_bspline_eval<iganet::BSplineDeriv::dy1, true>(bspline, splinelib_bspline, xi, tol);
    test_bspline_eval<iganet::BSplineDeriv::dy2, true>(bspline, splinelib_bspline, xi, tol);
    test_bspline_eval<iganet::BSplineDeriv::dy3, true>(bspline, splinelib_bspline, xi, tol);
    test_bspline_eval<iganet::BSplineDeriv::dy4, true>(bspline, splinelib_bspline, xi, tol);
  }

  if constexpr (BSpline_t::parDim() == 3) {
    test_bspline_eval<iganet::BSplineDeriv::dz1, true>(bspline, splinelib_bspline, xi, tol);
    test_bspline_eval<iganet::BSplineDeriv::dz2, true>(bspline, splinelib_bspline, xi, tol);
    test_bspline_eval<iganet::BSplineDeriv::dz3, true>(bspline, splinelib_bspline, xi, tol);
    test_bspline_eval<iganet::BSplineDeriv::dz4, true>(bspline, splinelib_bspline, xi, tol);
  }

  if constexpr (BSpline_t::parDim() == 4) {
    test_bspline_eval<iganet::BSplineDeriv::dt1, true>(bspline, splinelib_bspline, xi, tol);
    test_bspline_eval<iganet::BSplineDeriv::dt2, true>(bspline, splinelib_bspline, xi, tol);
    test_bspline_eval<iganet::BSplineDeriv::dt3, true>(bspline, splinelib_bspline, xi, tol);
    test_bspline_eval<iganet::BSplineDeriv::dt4, true>(bspline, splinelib_bspline, xi, tol);
  }

  // Evaluate gradients
  if constexpr (BSpline_t::geoDim() == 1) {
    test_bspline_grad<false>(bspline, xi, tol);
    test_bspline_grad<true>(bspline, xi, tol);
  }

  /// Evaluate Jacobian
  test_bspline_jac<false>(bspline, xi, tol);
  test_bspline_jac<true>(bspline, xi, tol);

  /// Evaluate Hessian
  if constexpr (BSpline_t::geoDim() == 1) {
    test_bspline_hess<false>(bspline, xi, tol);
    test_bspline_hess<true>(bspline, xi, tol);
  }
}
