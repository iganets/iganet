/**
   @file unittests/unittest_bsplinelib.hpp

   @brief BSplineLib helper functions

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include <iganet.h>
#include <iostream>

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunqualified-std-cast-call"
#include <BSplineLib/Splines/b_spline.hpp>
#pragma clang diagnostic pop

#include <gtest/gtest.h>

#pragma once

template<typename BSpline_t>
auto to_bsplinelib_bspline(const BSpline_t& bspline)
{
  // B-spline construction
  using BSpline          = bsplinelib::splines::BSpline<BSpline_t::parDim(), BSpline_t::geoDim()>;
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
  for (iganet::short_t k=0; k<bspline.parDim(); ++k)
    degrees[k] = Degree{ bspline.degree(k) };

  // Create knot vectors
  KnotVectors knot_vectors;
  for (iganet::short_t k=0; k<bspline.parDim(); ++k) {
    Knots knots_;
    for (int64_t i=0; i<bspline.nknots(k); ++i)
      knots_.emplace_back( Knot{bspline.knots(k)[i].template item<typename BSpline_t::value_type>()} );
    bsplinelib::SharedPointer<KnotVector> knot_vector{std::make_shared<KnotVector>(knots_)};
    knot_vectors[k] = std::move(knot_vector);
  }

  // Create parametric space
  bsplinelib::SharedPointer<ParameterSpace> parameter_space{std::make_shared<ParameterSpace>(knot_vectors, degrees)};

  // Create coordinate vector(s)
  Coordinates coordinates;
  for (int64_t i=0; i<bspline.ncumcoeffs(); ++i)
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
  bsplinelib::SharedPointer<VectorSpace> vector_space{std:: make_shared<VectorSpace>(coordinates)};

  // Create B-Spline
  BSpline bsplinelib_bspline{parameter_space, vector_space};

  return bsplinelib_bspline;
}

template<iganet::deriv deriv, bool memory_optimized, bool precompute,
         typename BSpline_t, typename BSplineLibBSpline_t, typename TensorArray_t>
void test_bspline_eval(const BSpline_t& bspline,
                       BSplineLibBSpline_t bsplinelib_bspline,
                       const TensorArray_t& xi, typename BSpline_t::value_type tol = 1e-12)
{
  // B-spline evaluation
  using ParametricCoordinate       = typename BSplineLibBSpline_t::ParametricCoordinate_;
  using ScalarParametricCoordinate = typename ParametricCoordinate::value_type;
  using Derivative                 = typename BSplineLibBSpline_t::ParameterSpace_::Derivative_;
  using ScalarDerivative           = typename Derivative::value_type;

  using BSplineValue_t = iganet::utils::BlockTensor<torch::Tensor, 1, BSpline_t::geoDim()>;

  BSplineValue_t bspline_val;
  if constexpr (precompute) {
    auto knot_indices  = bspline.find_knot_indices(xi);
    auto basfunc       = bspline.template eval_basfunc<deriv>(xi, knot_indices);
    auto coeff_indices = bspline.find_coeff_indices(knot_indices);
    bspline_val        = bspline.eval_from_precomputed(basfunc, coeff_indices, xi[0].numel(), xi[0].sizes());
  } else
    bspline_val        = bspline.template eval<deriv, memory_optimized>(xi);

  for (int64_t i=0; i<xi[0].size(0); ++i) {
    if constexpr (BSpline_t::parDim() == 1 &&
                  BSpline_t::geoDim() == 1)
      EXPECT_NEAR(bspline_val(0)[i].template item<typename BSpline_t::value_type>(),
                  bsplinelib_bspline(ParametricCoordinate
                                    {
                                      ScalarParametricCoordinate{(xi[0])[i].template item<typename BSpline_t::value_type>()}
                                    },
                                    Derivative
                                    {
                                      ScalarDerivative{(iganet::short_t) deriv % 10}
                                    }
                                    )[0], tol);
    else if constexpr (BSpline_t::parDim() == 1 &&
                       BSpline_t::geoDim() >  1)
      for (iganet::short_t k=0; k<bspline.geoDim(); ++k)
        EXPECT_NEAR(bspline_val(k)[i].template item<typename BSpline_t::value_type>(),
                    bsplinelib_bspline(ParametricCoordinate
                                      {
                                        ScalarParametricCoordinate{(xi[0])[i].template item<typename BSpline_t::value_type>()}
                                      },
                                      Derivative
                                      {
                                        ScalarDerivative{(iganet::short_t) deriv %10}
                                      }
                                      )[k], tol);
    else if constexpr (BSpline_t::parDim() == 2 &&
                       BSpline_t::geoDim() == 1)
      EXPECT_NEAR(bspline_val(0)[i].template item<typename BSpline_t::value_type>(),
                  bsplinelib_bspline(ParametricCoordinate
                                    {
                                      ScalarParametricCoordinate{(xi[0])[i].template item<typename BSpline_t::value_type>()},
                                      ScalarParametricCoordinate{(xi[1])[i].template item<typename BSpline_t::value_type>()}
                                    },
                                    Derivative
                                    {
                                      ScalarDerivative{ (iganet::short_t)deriv    %10},
                                      ScalarDerivative{((iganet::short_t)deriv/10)%10}
                                    }
                                    )[0], tol);
    else if constexpr (BSpline_t::parDim() == 2 &&
                       BSpline_t::geoDim() >  1)
      for (iganet::short_t k=0; k<bspline.geoDim(); ++k)
        EXPECT_NEAR(bspline_val(k)[i].template item<typename BSpline_t::value_type>(),
                    bsplinelib_bspline(ParametricCoordinate
                                      {
                                        ScalarParametricCoordinate{(xi[0])[i].template item<typename BSpline_t::value_type>()},
                                        ScalarParametricCoordinate{(xi[1])[i].template item<typename BSpline_t::value_type>()}
                                      },
                                      Derivative
                                      {
                                        ScalarDerivative{ (iganet::short_t)deriv    %10},
                                        ScalarDerivative{((iganet::short_t)deriv/10)%10}
                                      }
                                      )[k], tol);
    else if constexpr (BSpline_t::parDim() == 3 &&
                       BSpline_t::geoDim() == 1)
      EXPECT_NEAR(bspline_val(0)[i].template item<typename BSpline_t::value_type>(),
                  bsplinelib_bspline(ParametricCoordinate
                                    {
                                      ScalarParametricCoordinate{(xi[0])[i].template item<typename BSpline_t::value_type>()},
                                      ScalarParametricCoordinate{(xi[1])[i].template item<typename BSpline_t::value_type>()},
                                      ScalarParametricCoordinate{(xi[2])[i].template item<typename BSpline_t::value_type>()}
                                    },
                                    Derivative
                                    {
                                      ScalarDerivative{ (iganet::short_t)deriv     %10},
                                      ScalarDerivative{((iganet::short_t)deriv/ 10)%10},
                                      ScalarDerivative{((iganet::short_t)deriv/100)%10}
                                    }
                                    )[0], tol);
    else if constexpr (BSpline_t::parDim() == 3 &&
                       BSpline_t::geoDim() >  1)
      for (iganet::short_t k=0; k<bspline.geoDim(); ++k)
        EXPECT_NEAR(bspline_val(k)[i].template item<typename BSpline_t::value_type>(),
                    bsplinelib_bspline(ParametricCoordinate
                                      {
                                        ScalarParametricCoordinate{(xi[0])[i].template item<typename BSpline_t::value_type>()},
                                        ScalarParametricCoordinate{(xi[1])[i].template item<typename BSpline_t::value_type>()},
                                        ScalarParametricCoordinate{(xi[2])[i].template item<typename BSpline_t::value_type>()}
                                      },
                                      Derivative
                                      {
                                        ScalarDerivative{ (iganet::short_t)deriv     %10},
                                        ScalarDerivative{((iganet::short_t)deriv/ 10)%10},
                                        ScalarDerivative{((iganet::short_t)deriv/100)%10}
                                      }
                                      )[k], tol);
    else if constexpr (BSpline_t::parDim() == 4 &&
                       BSpline_t::geoDim() == 1)
      EXPECT_NEAR(bspline_val(0)[i].template item<typename BSpline_t::value_type>(),
                  bsplinelib_bspline(ParametricCoordinate
                                    {
                                      ScalarParametricCoordinate{(xi[0])[i].template item<typename BSpline_t::value_type>()},
                                      ScalarParametricCoordinate{(xi[1])[i].template item<typename BSpline_t::value_type>()},
                                      ScalarParametricCoordinate{(xi[2])[i].template item<typename BSpline_t::value_type>()},
                                      ScalarParametricCoordinate{(xi[3])[i].template item<typename BSpline_t::value_type>()}
                                    },
                                    Derivative
                                    {
                                      ScalarDerivative{ (iganet::short_t)deriv      %10},
                                      ScalarDerivative{((iganet::short_t)deriv/  10)%10},
                                      ScalarDerivative{((iganet::short_t)deriv/ 100)%10},
                                      ScalarDerivative{((iganet::short_t)deriv/1000)%10}
                                    }
                                    )[0], tol);
    else if constexpr (BSpline_t::parDim() == 4 &&
                       BSpline_t::geoDim() >  1)
      for (iganet::short_t k=0; k<bspline.geoDim(); ++k)
        EXPECT_NEAR(bspline_val(k)[i].template item<typename BSpline_t::value_type>(),
                    bsplinelib_bspline(ParametricCoordinate
                                      {
                                        ScalarParametricCoordinate{(xi[0])[i].template item<typename BSpline_t::value_type>()},
                                        ScalarParametricCoordinate{(xi[1])[i].template item<typename BSpline_t::value_type>()},
                                        ScalarParametricCoordinate{(xi[2])[i].template item<typename BSpline_t::value_type>()},
                                        ScalarParametricCoordinate{(xi[3])[i].template item<typename BSpline_t::value_type>()}
                                      },
                                      Derivative
                                      {
                                        ScalarDerivative{ (iganet::short_t)deriv      %10},
                                        ScalarDerivative{((iganet::short_t)deriv/  10)%10},
                                        ScalarDerivative{((iganet::short_t)deriv/ 100)%10},
                                        ScalarDerivative{((iganet::short_t)deriv/1000)%10}
                                      }
                                      )[k], tol);
    else
      throw std::runtime_error("Unsupported parametric/ geometric dimension");
  }
}

template<bool memory_optimized, bool precompute, typename BSpline_t, typename TensorArray_t>
void test_bspline_grad(const BSpline_t& bspline,
                       const TensorArray_t& xi,
                       typename BSpline_t::value_type tol = 1e-12)
{
  iganet::utils::BlockTensor<torch::Tensor, 1, BSpline_t::parDim()> bspline_grad_val;

  if constexpr (precompute) {
    auto knot_indices  = bspline.find_knot_indices(xi);
    auto coeff_indices = bspline.find_coeff_indices(knot_indices);
    bspline_grad_val   = bspline.grad(xi, knot_indices, coeff_indices);
  } else
    bspline_grad_val   = bspline.grad(xi);

  if constexpr (BSpline_t::parDim() == 1) {
    EXPECT_TRUE(torch::allclose(bspline_grad_val(0,0),
                                bspline.template eval<iganet::deriv::dx, memory_optimized>(xi)(0)
                                ));
  }
  else if constexpr (BSpline_t::parDim() == 2) {
    EXPECT_TRUE(torch::allclose(bspline_grad_val(0,0),
                                bspline.template eval<iganet::deriv::dx, memory_optimized>(xi)(0)
                                ));
    EXPECT_TRUE(torch::allclose(bspline_grad_val(0,1),
                                bspline.template eval<iganet::deriv::dy, memory_optimized>(xi)(0)
                                ));
  }
  else if constexpr (BSpline_t::parDim() == 3) {
    EXPECT_TRUE(torch::allclose(bspline_grad_val(0,0),
                                bspline.template eval<iganet::deriv::dx, memory_optimized>(xi)(0)
                                ));
    EXPECT_TRUE(torch::allclose(bspline_grad_val(0,1),
                                bspline.template eval<iganet::deriv::dy, memory_optimized>(xi)(0)
                                ));
    EXPECT_TRUE(torch::allclose(bspline_grad_val(0,2),
                                bspline.template eval<iganet::deriv::dz, memory_optimized>(xi)(0)
                                ));
  }
  else if constexpr (BSpline_t::parDim() == 4) {
    EXPECT_TRUE(torch::allclose(bspline_grad_val(0,0),
                                bspline.template eval<iganet::deriv::dx, memory_optimized>(xi)(0)
                                ));
    EXPECT_TRUE(torch::allclose(bspline_grad_val(0,1),
                                bspline.template eval<iganet::deriv::dy, memory_optimized>(xi)(0)
                                ));
    EXPECT_TRUE(torch::allclose(bspline_grad_val(0,2),
                                bspline.template eval<iganet::deriv::dz, memory_optimized>(xi)(0)
                                ));
    EXPECT_TRUE(torch::allclose(bspline_grad_val(0,3),
                                bspline.template eval<iganet::deriv::dt, memory_optimized>(xi)(0)
                                ));
  }
}

template<bool memory_optimized, bool precompute, typename Geometry_t, typename BSpline_t, typename TensorArray_t>
void test_bspline_igrad(const Geometry_t& geometry,
                        const BSpline_t& bspline,
                        const TensorArray_t& xi,
                        typename BSpline_t::value_type tol = 1e-12)
{
  iganet::utils::BlockTensor<torch::Tensor, 1, BSpline_t::parDim()> bspline_igrad_val;

  if constexpr (precompute) {
    auto knot_indices    = bspline.find_knot_indices(xi);
    auto coeff_indices   = bspline.find_coeff_indices(knot_indices);
    auto knot_indices_G  = geometry.find_knot_indices(xi);
    auto coeff_indices_G = geometry.find_coeff_indices(knot_indices_G);
    bspline_igrad_val    = bspline.igrad(geometry, xi,
                                         knot_indices, coeff_indices,
                                         knot_indices_G, coeff_indices_G);
  } else
    bspline_igrad_val    = bspline.igrad(geometry, xi);

  if constexpr (BSpline_t::parDim() == 1) {
    EXPECT_TRUE(torch::allclose(bspline_igrad_val(0,0),
                                bspline.template eval<iganet::deriv::dx, memory_optimized>(xi)(0)
                                ));
  }
  else if constexpr (BSpline_t::parDim() == 2) {
    EXPECT_TRUE(torch::allclose(bspline_igrad_val(0,0),
                                bspline.template eval<iganet::deriv::dx, memory_optimized>(xi)(0)
                                ));
    EXPECT_TRUE(torch::allclose(bspline_igrad_val(0,1),
                                bspline.template eval<iganet::deriv::dy, memory_optimized>(xi)(0)
                                ));
  }
  else if constexpr (BSpline_t::parDim() == 3) {
    EXPECT_TRUE(torch::allclose(bspline_igrad_val(0,0),
                                bspline.template eval<iganet::deriv::dx, memory_optimized>(xi)(0)
                                ));
    EXPECT_TRUE(torch::allclose(bspline_igrad_val(0,1),
                                bspline.template eval<iganet::deriv::dy, memory_optimized>(xi)(0)
                                ));
    EXPECT_TRUE(torch::allclose(bspline_igrad_val(0,2),
                                bspline.template eval<iganet::deriv::dz, memory_optimized>(xi)(0)
                                ));
  }
  else if constexpr (BSpline_t::parDim() == 4) {
    EXPECT_TRUE(torch::allclose(bspline_igrad_val(0,0),
                                bspline.template eval<iganet::deriv::dx, memory_optimized>(xi)(0)
                                ));
    EXPECT_TRUE(torch::allclose(bspline_igrad_val(0,1),
                                bspline.template eval<iganet::deriv::dy, memory_optimized>(xi)(0)
                                ));
    EXPECT_TRUE(torch::allclose(bspline_igrad_val(0,2),
                                bspline.template eval<iganet::deriv::dz, memory_optimized>(xi)(0)
                                ));
    EXPECT_TRUE(torch::allclose(bspline_igrad_val(0,3),
                                bspline.template eval<iganet::deriv::dt, memory_optimized>(xi)(0)
                                ));
  }
}

template<bool memory_optimized, bool precompute, typename BSpline_t, typename TensorArray_t>
void test_bspline_jac(const BSpline_t& bspline,
                      const TensorArray_t& xi,
                      typename BSpline_t::value_type tol = 1e-12)
{
  iganet::utils::BlockTensor<torch::Tensor,
                             BSpline_t::geoDim(), BSpline_t::parDim()> bspline_jac_val;

  if constexpr (precompute) {
    auto knot_indices  = bspline.find_knot_indices(xi);
    auto coeff_indices = bspline.find_coeff_indices(knot_indices);
    bspline_jac_val    = bspline.jac(xi, knot_indices, coeff_indices);
  } else
    bspline_jac_val    = bspline.jac(xi);

  if constexpr (BSpline_t::parDim() >= 1) {
    for (iganet::short_t k=0; k<BSpline_t::geoDim(); ++k)
      EXPECT_TRUE(torch::allclose(bspline_jac_val(k,0),
                                  bspline.template eval<iganet::deriv::dx, memory_optimized>(xi)(k)
                                  ));
  }

  if constexpr (BSpline_t::parDim() >= 2) {
    for (iganet::short_t k=0; k<BSpline_t::geoDim(); ++k)
      EXPECT_TRUE(torch::allclose(bspline_jac_val(k,1),
                                  bspline.template eval<iganet::deriv::dy, memory_optimized>(xi)(k)
                                  ));
  }

  if constexpr (BSpline_t::parDim() >= 3) {
    for (iganet::short_t k=0; k<BSpline_t::geoDim(); ++k)
      EXPECT_TRUE(torch::allclose(bspline_jac_val(k,2),
                                  bspline.template eval<iganet::deriv::dz, memory_optimized>(xi)(k)
                                  ));
  }

  if constexpr (BSpline_t::parDim() >= 4) {
    for (iganet::short_t k=0; k<BSpline_t::geoDim(); ++k)
      EXPECT_TRUE(torch::allclose(bspline_jac_val(k,3),
                                  bspline.template eval<iganet::deriv::dt, memory_optimized>(xi)(k)
                                  ));
  }
}

template<bool memory_optimized, bool precompute, typename Geometry_t, typename BSpline_t, typename TensorArray_t>
void test_bspline_ijac(const Geometry_t& geometry,
                       const BSpline_t& bspline,
                       const TensorArray_t& xi,
                       typename BSpline_t::value_type tol = 1e-12)
{
  iganet::utils::BlockTensor<torch::Tensor,
                             BSpline_t::geoDim(), BSpline_t::parDim()> bspline_ijac_val;

  if constexpr (precompute) {
    auto knot_indices    = bspline.find_knot_indices(xi);
    auto coeff_indices   = bspline.find_coeff_indices(knot_indices);
    auto knot_indices_G  = geometry.find_knot_indices(xi);
    auto coeff_indices_G = geometry.find_coeff_indices(knot_indices_G);
    bspline_ijac_val     = bspline.ijac(geometry, xi,
                                        knot_indices, coeff_indices,
                                        knot_indices_G, coeff_indices_G);
  } else
    bspline_ijac_val    = bspline.ijac(geometry, xi);

  if constexpr (BSpline_t::parDim() >= 1) {
    for (iganet::short_t k=0; k<BSpline_t::geoDim(); ++k)
      EXPECT_TRUE(torch::allclose(bspline_ijac_val(k,0),
                                  bspline.template eval<iganet::deriv::dx, memory_optimized>(xi)(k)
                                  ));
  }

  if constexpr (BSpline_t::parDim() >= 2) {
    for (iganet::short_t k=0; k<BSpline_t::geoDim(); ++k)
      EXPECT_TRUE(torch::allclose(bspline_ijac_val(k,1),
                                  bspline.template eval<iganet::deriv::dy, memory_optimized>(xi)(k)
                                  ));
  }

  if constexpr (BSpline_t::parDim() >= 3) {
    for (iganet::short_t k=0; k<BSpline_t::geoDim(); ++k)
      EXPECT_TRUE(torch::allclose(bspline_ijac_val(k,2),
                                  bspline.template eval<iganet::deriv::dz, memory_optimized>(xi)(k)
                                  ));
  }

  if constexpr (BSpline_t::parDim() >= 4) {
    for (iganet::short_t k=0; k<BSpline_t::geoDim(); ++k)
      EXPECT_TRUE(torch::allclose(bspline_ijac_val(k,3),
                                  bspline.template eval<iganet::deriv::dt, memory_optimized>(xi)(k)
                                  ));
  }
}

template<bool memory_optimized, bool precompute, typename BSpline_t, typename TensorArray_t>
void test_bspline_hess(const BSpline_t& bspline,
                       const TensorArray_t& xi,
                       typename BSpline_t::value_type tol = 1e-12)
{
  iganet::utils::BlockTensor<torch::Tensor,
                             BSpline_t::parDim(), BSpline_t::parDim(),
                             BSpline_t::geoDim()> bspline_hess_val;

  if constexpr (precompute) {
    auto knot_indices  = bspline.find_knot_indices(xi);
    auto coeff_indices = bspline.find_coeff_indices(knot_indices);
    bspline_hess_val   = bspline.hess(xi, knot_indices, coeff_indices);
  } else
    bspline_hess_val   = bspline.hess(xi);

  if constexpr (BSpline_t::parDim() >= 1) {
    for (iganet::short_t k=0; k<BSpline_t::geoDim(); ++k)
      EXPECT_TRUE(torch::allclose(bspline_hess_val(0,0,k),
                                  bspline.template eval<iganet::deriv::dx^2, memory_optimized>(xi)(k)
                                  ));
  }

  if constexpr (BSpline_t::parDim() >= 2) {
    for (iganet::short_t k=0; k<BSpline_t::geoDim(); ++k) {
      EXPECT_TRUE(torch::allclose(bspline_hess_val(0,1,k),
                                  bspline.template eval<iganet::deriv::dx+iganet::deriv::dy, memory_optimized>(xi)(k)
                                  ));
      EXPECT_TRUE(torch::allclose(bspline_hess_val(1,0,k),
                                  bspline.template eval<iganet::deriv::dy+iganet::deriv::dx, memory_optimized>(xi)(k)
                                  ));
      EXPECT_TRUE(torch::allclose(bspline_hess_val(1,1,k),
                                  bspline.template eval<iganet::deriv::dy^2, memory_optimized>(xi)(k)
                                  ));
    }
  }

  if constexpr (BSpline_t::parDim() >= 3) {
    for (iganet::short_t k=0; k<BSpline_t::geoDim(); ++k) {
      EXPECT_TRUE(torch::allclose(bspline_hess_val(0,2,k),
                                  bspline.template eval<iganet::deriv::dx+iganet::deriv::dz, memory_optimized>(xi)(k)
                                  ));
      EXPECT_TRUE(torch::allclose(bspline_hess_val(1,2,k),
                                  bspline.template eval<iganet::deriv::dy+iganet::deriv::dz, memory_optimized>(xi)(k)
                                  ));
      EXPECT_TRUE(torch::allclose(bspline_hess_val(2,0,k),
                                  bspline.template eval<iganet::deriv::dz+iganet::deriv::dx, memory_optimized>(xi)(k)
                                  ));
      EXPECT_TRUE(torch::allclose(bspline_hess_val(2,1,k),
                                  bspline.template eval<iganet::deriv::dz+iganet::deriv::dy, memory_optimized>(xi)(k)
                                  ));
      EXPECT_TRUE(torch::allclose(bspline_hess_val(2,2,k),
                                  bspline.template eval<iganet::deriv::dz^2, memory_optimized>(xi)(k)
                                  ));
    }
  }

  if constexpr (BSpline_t::parDim() >= 4) {
    for (iganet::short_t k=0; k<BSpline_t::geoDim(); ++k) {
      EXPECT_TRUE(torch::allclose(bspline_hess_val(0,3,k),
                                  bspline.template eval<iganet::deriv::dx+iganet::deriv::dt, memory_optimized>(xi)(k)
                                  ));
      EXPECT_TRUE(torch::allclose(bspline_hess_val(1,3,k),
                                  bspline.template eval<iganet::deriv::dy+iganet::deriv::dt, memory_optimized>(xi)(k)
                                  ));
      EXPECT_TRUE(torch::allclose(bspline_hess_val(2,3,k),
                                  bspline.template eval<iganet::deriv::dz+iganet::deriv::dt, memory_optimized>(xi)(k)
                                  ));
      EXPECT_TRUE(torch::allclose(bspline_hess_val(3,0,k),
                                  bspline.template eval<iganet::deriv::dt+iganet::deriv::dx, memory_optimized>(xi)(k)
                                  ));
      EXPECT_TRUE(torch::allclose(bspline_hess_val(3,1,k),
                                  bspline.template eval<iganet::deriv::dt+iganet::deriv::dy, memory_optimized>(xi)(k)
                                  ));
      EXPECT_TRUE(torch::allclose(bspline_hess_val(3,2,k),
                                  bspline.template eval<iganet::deriv::dt+iganet::deriv::dz, memory_optimized>(xi)(k)
                                  ));
      EXPECT_TRUE(torch::allclose(bspline_hess_val(3,3,k),
                                  bspline.template eval<iganet::deriv::dt^2, memory_optimized>(xi)(k)
                                  ));
    }
  }
}

template<bool memory_optimized, bool precompute, typename Geometry_t, typename BSpline_t, typename TensorArray_t>
void test_bspline_ihess(const Geometry_t& geometry,
                        const BSpline_t& bspline,
                        const TensorArray_t& xi,
                        typename BSpline_t::value_type tol = 1e-12)
{
  iganet::utils::BlockTensor<torch::Tensor,
                             BSpline_t::parDim(), BSpline_t::parDim(),
                             BSpline_t::geoDim()> bspline_ihess_val;

  if constexpr (precompute) {
    auto knot_indices    = bspline.find_knot_indices(xi);
    auto coeff_indices   = bspline.find_coeff_indices(knot_indices);
    auto knot_indices_G  = geometry.find_knot_indices(xi);
    auto coeff_indices_G = geometry.find_coeff_indices(knot_indices_G);
    bspline_ihess_val    = bspline.ihess(geometry, xi,
                                         knot_indices, coeff_indices,
                                         knot_indices_G, coeff_indices_G);
  } else
    bspline_ihess_val    = bspline.ihess(geometry, xi);

  if constexpr (BSpline_t::parDim() >= 1) {
    for (iganet::short_t k=0; k<BSpline_t::geoDim(); ++k)
      EXPECT_TRUE(torch::allclose(bspline_ihess_val(0,0,k),
                                  bspline.template eval<iganet::deriv::dx^2, memory_optimized>(xi)(k)
                                  ));
  }

  if constexpr (BSpline_t::parDim() >= 2) {
    for (iganet::short_t k=0; k<BSpline_t::geoDim(); ++k) {
      EXPECT_TRUE(torch::allclose(bspline_ihess_val(0,1,k),
                                  bspline.template eval<iganet::deriv::dx+iganet::deriv::dy, memory_optimized>(xi)(k)
                                  ));
      EXPECT_TRUE(torch::allclose(bspline_ihess_val(1,0,k),
                                  bspline.template eval<iganet::deriv::dy+iganet::deriv::dx, memory_optimized>(xi)(k)
                                  ));
      EXPECT_TRUE(torch::allclose(bspline_ihess_val(1,1,k),
                                  bspline.template eval<iganet::deriv::dy^2, memory_optimized>(xi)(k)
                                  ));
    }
  }

  if constexpr (BSpline_t::parDim() >= 3) {
    for (iganet::short_t k=0; k<BSpline_t::geoDim(); ++k) {
      EXPECT_TRUE(torch::allclose(bspline_ihess_val(0,2,k),
                                  bspline.template eval<iganet::deriv::dx+iganet::deriv::dz, memory_optimized>(xi)(k)
                                  ));
      EXPECT_TRUE(torch::allclose(bspline_ihess_val(1,2,k),
                                  bspline.template eval<iganet::deriv::dy+iganet::deriv::dz, memory_optimized>(xi)(k)
                                  ));
      EXPECT_TRUE(torch::allclose(bspline_ihess_val(2,0,k),
                                  bspline.template eval<iganet::deriv::dz+iganet::deriv::dx, memory_optimized>(xi)(k)
                                  ));
      EXPECT_TRUE(torch::allclose(bspline_ihess_val(2,1,k),
                                  bspline.template eval<iganet::deriv::dz+iganet::deriv::dy, memory_optimized>(xi)(k)
                                  ));
      EXPECT_TRUE(torch::allclose(bspline_ihess_val(2,2,k),
                                  bspline.template eval<iganet::deriv::dz^2, memory_optimized>(xi)(k)
                                  ));
    }
  }

  if constexpr (BSpline_t::parDim() >= 4) {
    for (iganet::short_t k=0; k<BSpline_t::geoDim(); ++k) {
      EXPECT_TRUE(torch::allclose(bspline_ihess_val(0,3,k),
                                  bspline.template eval<iganet::deriv::dx+iganet::deriv::dt, memory_optimized>(xi)(k)
                                  ));
      EXPECT_TRUE(torch::allclose(bspline_ihess_val(1,3,k),
                                  bspline.template eval<iganet::deriv::dy+iganet::deriv::dt, memory_optimized>(xi)(k)
                                  ));
      EXPECT_TRUE(torch::allclose(bspline_ihess_val(2,3,k),
                                  bspline.template eval<iganet::deriv::dz+iganet::deriv::dt, memory_optimized>(xi)(k)
                                  ));
      EXPECT_TRUE(torch::allclose(bspline_ihess_val(3,0,k),
                                  bspline.template eval<iganet::deriv::dt+iganet::deriv::dx, memory_optimized>(xi)(k)
                                  ));
      EXPECT_TRUE(torch::allclose(bspline_ihess_val(3,1,k),
                                  bspline.template eval<iganet::deriv::dt+iganet::deriv::dy, memory_optimized>(xi)(k)
                                  ));
      EXPECT_TRUE(torch::allclose(bspline_ihess_val(3,2,k),
                                  bspline.template eval<iganet::deriv::dt+iganet::deriv::dz, memory_optimized>(xi)(k)
                                  ));
      EXPECT_TRUE(torch::allclose(bspline_ihess_val(3,3,k),
                                  bspline.template eval<iganet::deriv::dt^2, memory_optimized>(xi)(k)
                                  ));
    }
  }
}

template<typename Geometry_t, typename BSpline_t, typename TensorArray_t>
void test_bspline_eval(const Geometry_t& geometry,
                       const BSpline_t& bspline,
                       const TensorArray_t& xi,
                       typename BSpline_t::value_type tol = 1e-12)
{
  // Create B-Spline
  auto bsplinelib_bspline = to_bsplinelib_bspline(bspline);

  // Evaluate function and derivatives (non-memory optimized)
  test_bspline_eval<iganet::deriv::func, false, false>(bspline, bsplinelib_bspline, xi, tol);

  if constexpr (BSpline_t::parDim() == 1) {
    test_bspline_eval<iganet::deriv::dx,   false, false>(bspline, bsplinelib_bspline, xi, tol);
    test_bspline_eval<iganet::deriv::dx^2, false, false>(bspline, bsplinelib_bspline, xi, tol);
    test_bspline_eval<iganet::deriv::dx^3, false, false>(bspline, bsplinelib_bspline, xi, tol);
    test_bspline_eval<iganet::deriv::dx^4, false, false>(bspline, bsplinelib_bspline, xi, tol);
  }

  if constexpr (BSpline_t::parDim() == 2) {
    test_bspline_eval<iganet::deriv::dy,   false, false>(bspline, bsplinelib_bspline, xi, tol);
    test_bspline_eval<iganet::deriv::dy^2, false, false>(bspline, bsplinelib_bspline, xi, tol);
    test_bspline_eval<iganet::deriv::dy^3, false, false>(bspline, bsplinelib_bspline, xi, tol);
    test_bspline_eval<iganet::deriv::dy^4, false, false>(bspline, bsplinelib_bspline, xi, tol);
  }

  if constexpr (BSpline_t::parDim() == 3) {
    test_bspline_eval<iganet::deriv::dz,   false, false>(bspline, bsplinelib_bspline, xi, tol);
    test_bspline_eval<iganet::deriv::dz^2, false, false>(bspline, bsplinelib_bspline, xi, tol);
    test_bspline_eval<iganet::deriv::dz^3, false, false>(bspline, bsplinelib_bspline, xi, tol);
    test_bspline_eval<iganet::deriv::dz^4, false, false>(bspline, bsplinelib_bspline, xi, tol);
  }

  if constexpr (BSpline_t::parDim() == 4) {
    test_bspline_eval<iganet::deriv::dt,   false, false>(bspline, bsplinelib_bspline, xi, tol);
    test_bspline_eval<iganet::deriv::dt^2, false, false>(bspline, bsplinelib_bspline, xi, tol);
    test_bspline_eval<iganet::deriv::dt^3, false, false>(bspline, bsplinelib_bspline, xi, tol);
    test_bspline_eval<iganet::deriv::dt^4, false, false>(bspline, bsplinelib_bspline, xi, tol);
  }

  // Evaluate function and derivatives (memory optimized)
  test_bspline_eval<iganet::deriv::func, true, false>(bspline, bsplinelib_bspline, xi, tol);

  if constexpr (BSpline_t::parDim() == 1) {
    test_bspline_eval<iganet::deriv::dx,   true, false>(bspline, bsplinelib_bspline, xi, tol);
    test_bspline_eval<iganet::deriv::dx^2, true, false>(bspline, bsplinelib_bspline, xi, tol);
    test_bspline_eval<iganet::deriv::dx^3, true, false>(bspline, bsplinelib_bspline, xi, tol);
    test_bspline_eval<iganet::deriv::dx^4, true, false>(bspline, bsplinelib_bspline, xi, tol);
  }

  if constexpr (BSpline_t::parDim() == 2) {
    test_bspline_eval<iganet::deriv::dy,   true, false>(bspline, bsplinelib_bspline, xi, tol);
    test_bspline_eval<iganet::deriv::dy^2, true, false>(bspline, bsplinelib_bspline, xi, tol);
    test_bspline_eval<iganet::deriv::dy^3, true, false>(bspline, bsplinelib_bspline, xi, tol);
    test_bspline_eval<iganet::deriv::dy^4, true, false>(bspline, bsplinelib_bspline, xi, tol);
  }

  if constexpr (BSpline_t::parDim() == 3) {
    test_bspline_eval<iganet::deriv::dz,   true, false>(bspline, bsplinelib_bspline, xi, tol);
    test_bspline_eval<iganet::deriv::dz^2, true, false>(bspline, bsplinelib_bspline, xi, tol);
    test_bspline_eval<iganet::deriv::dz^3, true, false>(bspline, bsplinelib_bspline, xi, tol);
    test_bspline_eval<iganet::deriv::dz^4, true, false>(bspline, bsplinelib_bspline, xi, tol);
  }

  if constexpr (BSpline_t::parDim() == 4) {
    test_bspline_eval<iganet::deriv::dt,   true, false>(bspline, bsplinelib_bspline, xi, tol);
    test_bspline_eval<iganet::deriv::dt^2, true, false>(bspline, bsplinelib_bspline, xi, tol);
    test_bspline_eval<iganet::deriv::dt^3, true, false>(bspline, bsplinelib_bspline, xi, tol);
    test_bspline_eval<iganet::deriv::dt^4, true, false>(bspline, bsplinelib_bspline, xi, tol);
  }

  // Evaluate function and derivatives from precomputed data (non-memory optimized)
  test_bspline_eval<iganet::deriv::func, false, true>(bspline, bsplinelib_bspline, xi, tol);

  if constexpr (BSpline_t::parDim() == 1) {
    test_bspline_eval<iganet::deriv::dx,   false, true>(bspline, bsplinelib_bspline, xi, tol);
    test_bspline_eval<iganet::deriv::dx^2, false, true>(bspline, bsplinelib_bspline, xi, tol);
    test_bspline_eval<iganet::deriv::dx^3, false, true>(bspline, bsplinelib_bspline, xi, tol);
    test_bspline_eval<iganet::deriv::dx^4, false, true>(bspline, bsplinelib_bspline, xi, tol);
  }

  if constexpr (BSpline_t::parDim() == 2) {
    test_bspline_eval<iganet::deriv::dy,   false, true>(bspline, bsplinelib_bspline, xi, tol);
    test_bspline_eval<iganet::deriv::dy^2, false, true>(bspline, bsplinelib_bspline, xi, tol);
    test_bspline_eval<iganet::deriv::dy^3, false, true>(bspline, bsplinelib_bspline, xi, tol);
    test_bspline_eval<iganet::deriv::dy^4, false, true>(bspline, bsplinelib_bspline, xi, tol);
  }

  if constexpr (BSpline_t::parDim() == 3) {
    test_bspline_eval<iganet::deriv::dz,   false, true>(bspline, bsplinelib_bspline, xi, tol);
    test_bspline_eval<iganet::deriv::dz^2, false, true>(bspline, bsplinelib_bspline, xi, tol);
    test_bspline_eval<iganet::deriv::dz^3, false, true>(bspline, bsplinelib_bspline, xi, tol);
    test_bspline_eval<iganet::deriv::dz^4, false, true>(bspline, bsplinelib_bspline, xi, tol);
  }

  if constexpr (BSpline_t::parDim() == 4) {
    test_bspline_eval<iganet::deriv::dt,   false, true>(bspline, bsplinelib_bspline, xi, tol);
    test_bspline_eval<iganet::deriv::dt^2, false, true>(bspline, bsplinelib_bspline, xi, tol);
    test_bspline_eval<iganet::deriv::dt^3, false, true>(bspline, bsplinelib_bspline, xi, tol);
    test_bspline_eval<iganet::deriv::dt^4, false, true>(bspline, bsplinelib_bspline, xi, tol);
  }

  // Evaluate function and derivatives from precomputed data (memory optimized)
  test_bspline_eval<iganet::deriv::func, true, true>(bspline, bsplinelib_bspline, xi, tol);

  if constexpr (BSpline_t::parDim() == 1) {
    test_bspline_eval<iganet::deriv::dx,   true, true>(bspline, bsplinelib_bspline, xi, tol);
    test_bspline_eval<iganet::deriv::dx^2, true, true>(bspline, bsplinelib_bspline, xi, tol);
    test_bspline_eval<iganet::deriv::dx^3, true, true>(bspline, bsplinelib_bspline, xi, tol);
    test_bspline_eval<iganet::deriv::dx^4, true, true>(bspline, bsplinelib_bspline, xi, tol);
  }

  if constexpr (BSpline_t::parDim() == 2) {
    test_bspline_eval<iganet::deriv::dy,   true, true>(bspline, bsplinelib_bspline, xi, tol);
    test_bspline_eval<iganet::deriv::dy^2, true, true>(bspline, bsplinelib_bspline, xi, tol);
    test_bspline_eval<iganet::deriv::dy^3, true, true>(bspline, bsplinelib_bspline, xi, tol);
    test_bspline_eval<iganet::deriv::dy^4, true, true>(bspline, bsplinelib_bspline, xi, tol);
  }

  if constexpr (BSpline_t::parDim() == 3) {
    test_bspline_eval<iganet::deriv::dz,   true, true>(bspline, bsplinelib_bspline, xi, tol);
    test_bspline_eval<iganet::deriv::dz^2, true, true>(bspline, bsplinelib_bspline, xi, tol);
    test_bspline_eval<iganet::deriv::dz^3, true, true>(bspline, bsplinelib_bspline, xi, tol);
    test_bspline_eval<iganet::deriv::dz^4, true, true>(bspline, bsplinelib_bspline, xi, tol);
  }

  if constexpr (BSpline_t::parDim() == 4) {
    test_bspline_eval<iganet::deriv::dt,   true, true>(bspline, bsplinelib_bspline, xi, tol);
    test_bspline_eval<iganet::deriv::dt^2, true, true>(bspline, bsplinelib_bspline, xi, tol);
    test_bspline_eval<iganet::deriv::dt^3, true, true>(bspline, bsplinelib_bspline, xi, tol);
    test_bspline_eval<iganet::deriv::dt^4, true, true>(bspline, bsplinelib_bspline, xi, tol);
  }

  // Evaluate gradients
  if constexpr (BSpline_t::geoDim() == 1) {
    test_bspline_grad<false, false>(bspline, xi, tol);
    test_bspline_grad<false,  true>(bspline, xi, tol);
    test_bspline_grad<true,  false>(bspline, xi, tol);
    test_bspline_grad<true,   true>(bspline, xi, tol);

    test_bspline_igrad<false, false>(geometry, bspline, xi, tol);
    test_bspline_igrad<false,  true>(geometry, bspline, xi, tol);
    test_bspline_igrad<true,  false>(geometry, bspline, xi, tol);
    test_bspline_igrad<true,   true>(geometry, bspline, xi, tol);
  }

  /// Evaluate Jacobian
  test_bspline_jac<false, false>(bspline, xi, tol);
  test_bspline_jac<false,  true>(bspline, xi, tol);
  test_bspline_jac<true,  false>(bspline, xi, tol);
  test_bspline_jac<true,   true>(bspline, xi, tol);

  test_bspline_ijac<false, false>(geometry, bspline, xi, tol);
  test_bspline_ijac<false,  true>(geometry, bspline, xi, tol);
  test_bspline_ijac<true,  false>(geometry, bspline, xi, tol);
  test_bspline_ijac<true,   true>(geometry, bspline, xi, tol);

  /// Evaluate Hessian
  if constexpr (BSpline_t::geoDim() == 1) {
    test_bspline_hess<false, false>(bspline, xi, tol);
    test_bspline_hess<false,  true>(bspline, xi, tol);
    test_bspline_hess<true,  false>(bspline, xi, tol);
    test_bspline_hess<true,   true>(bspline, xi, tol);

    test_bspline_ihess<false, false>(geometry, bspline, xi, tol);
    test_bspline_ihess<false,  true>(geometry, bspline, xi, tol);
    test_bspline_ihess<true,  false>(geometry, bspline, xi, tol);
    test_bspline_ihess<true,   true>(geometry, bspline, xi, tol);
  }
}
