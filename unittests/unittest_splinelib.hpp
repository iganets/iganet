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

template<iganet::BSplineDeriv deriv,
         typename BSpline_t, typename SplineLibBSpline_t, typename TensorArray_t>
void test_bspline_eval_(BSpline_t& bspline, SplineLibBSpline_t splinelib_bspline,
                        TensorArray_t& xi, typename BSpline_t::value_type tol = 1e-12)
{
  // B-spline evaluation
  using ParametricCoordinate       = typename SplineLibBSpline_t::ParametricCoordinate_;
  using ScalarParametricCoordinate = typename ParametricCoordinate::value_type;
  using Derivative                 = typename SplineLibBSpline_t::ParameterSpace_::Derivative_;
  using ScalarDerivative           = typename Derivative::value_type;

  auto bspline_val = bspline.template eval_<deriv>(xi);

  for (int64_t i=0; i<xi[0].size(0); ++i) {
    if constexpr (BSpline_t::parDim() == 1 &&
                  BSpline_t::geoDim() == 1)
      EXPECT_NEAR(bspline_val[i].template item<typename BSpline_t::value_type>(),
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
        EXPECT_NEAR(bspline_val[k][i].template item<typename BSpline_t::value_type>(),
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
      EXPECT_NEAR(bspline_val[i].template item<typename BSpline_t::value_type>(),
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
        EXPECT_NEAR(bspline_val[k][i].template item<typename BSpline_t::value_type>(),
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
      EXPECT_NEAR(bspline_val[i].template item<typename BSpline_t::value_type>(),
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
        EXPECT_NEAR(bspline_val[k][i].template item<typename BSpline_t::value_type>(),
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
      EXPECT_NEAR(bspline_val[i].template item<typename BSpline_t::value_type>(),
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
        EXPECT_NEAR(bspline_val[k][i].template item<typename BSpline_t::value_type>(),
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

template<typename BSpline_t, typename TensorArray_t>
void test_bspline_eval(BSpline_t& bspline, TensorArray_t& xi, typename BSpline_t::value_type tol = 1e-12)
{
  // Create B-Spline
  auto splinelib_bspline = to_splinelib_bspline(bspline);

  // Evaluate function and derivatives
  test_bspline_eval_<iganet::BSplineDeriv::func>(bspline, splinelib_bspline, xi, tol);
  test_bspline_eval_<iganet::BSplineDeriv::dx1 >(bspline, splinelib_bspline, xi, tol);
  test_bspline_eval_<iganet::BSplineDeriv::dx2 >(bspline, splinelib_bspline, xi, tol);
  test_bspline_eval_<iganet::BSplineDeriv::dx3 >(bspline, splinelib_bspline, xi, tol);
  test_bspline_eval_<iganet::BSplineDeriv::dx4 >(bspline, splinelib_bspline, xi, tol);
}
