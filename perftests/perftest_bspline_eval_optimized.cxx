/**
   @file perftests/perftest_bspline_eval.cxx

   @brief B-spline evaluation performance unittests

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include <iganet.hpp>

#include <iostream>
#include <chrono>

#include "../unittests/unittest_splinelib.hpp"
#include <gtest/gtest.h>

#define SPLINELIB
static constexpr iganet::deriv deriv = iganet::deriv::func;
static constexpr bool precompute = false;

namespace perftest {

  template<typename real_t, short_t GeoDim,
           short_t Degree0,
           iganet::deriv deriv,
           bool precompute = false>
  auto test_UniformBSpline(int64_t ncoeffs, int64_t nsamples)
  {
    iganet::core<real_t> core_(false);
    iganet::UniformBSpline<iganet::core<real_t, true>, GeoDim, Degree0> bspline({ncoeffs}, iganet::init::linear);
    iganet::TensorArray1 xi = {torch::rand(nsamples, core_.options())};
    
    auto t1 = std::chrono::high_resolution_clock::now();
    if constexpr (precompute)
    {
      auto knot_idx  = bspline.find_knot_indices(xi);
      auto basfunc   = bspline.template eval_basfunc<deriv>(xi, knot_idx);
      auto coeff_idx = bspline.eval_coeff_indices(knot_idx);
      for (int i=0; i<10; i++)
        auto bspline_val = bspline.eval_from_precomputed(basfunc, coeff_idx, xi[0].numel(), xi[0].sizes());
    }
    else
      for (int i=0; i<10; i++)
        bspline.template eval<deriv>(xi);
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << std::right << std::setw(10)
              << std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count() / double(nsamples*10)
              << " (ns/entry)";
    
#ifdef SPLINELIB
    if (nsamples == 1) {
      auto splinelib_bspline = to_splinelib_bspline(bspline);
      
      // B-spline evaluation
      using ParametricCoordinate       = typename decltype(splinelib_bspline)::ParametricCoordinate_;
      using ScalarParametricCoordinate = typename ParametricCoordinate::value_type;
      using Derivative                 = typename decltype(splinelib_bspline)::ParameterSpace_::Derivative_;
      using ScalarDerivative           = typename Derivative::value_type;
      
      auto t1 = std::chrono::high_resolution_clock::now();
      for (int i=0; i<1000; i++)
        splinelib_bspline(ParametricCoordinate
                          {
                            ScalarParametricCoordinate{0.5}
                          },
                          Derivative
                          {
                            ScalarDerivative{(short_t) deriv % 10}
                          });
      auto t2 = std::chrono::high_resolution_clock::now();
      std::cout << std::right << std::setw(10)
                << std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count() / double(1000)
                << " (ns/entry)";
    } else {
      std::cout << std::right << std::setw(10)
                << "--"
                << " (ns/entry)";
    }
#endif
  }

  template<typename real_t, short_t GeoDim,
           short_t Degree0, short_t Degree1,
           iganet::deriv deriv,
           bool precompute = false>
  auto test_UniformBSpline(int64_t ncoeffs, int64_t nsamples)
  {
    iganet::core<real_t> core_(false);
    iganet::UniformBSpline<iganet::core<real_t, true>, GeoDim, Degree0, Degree1> bspline({ncoeffs,
                                                                                          ncoeffs}, iganet::init::linear);
    iganet::TensorArray2 xi = {torch::rand(nsamples, core_.options()),
                               torch::rand(nsamples, core_.options())};
    
    auto t1 = std::chrono::high_resolution_clock::now();
    if constexpr (precompute)
    {
      auto knot_idx  = bspline.find_knot_indices(xi);
      auto basfunc   = bspline.template eval_basfunc<deriv>(xi, knot_idx);
      auto coeff_idx = bspline.eval_coeff_indices(knot_idx);
      for (int i=0; i<10; i++)
        auto bspline_val = bspline.eval_from_precomputed(basfunc, coeff_idx, xi[0].numel(), xi[0].sizes());
    }
    else
      for (int i=0; i<10; i++)
        bspline.template eval<deriv>(xi);
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << std::right << std::setw(10)
              << std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count() / double(nsamples*10)
              << " (ns/entry)";
    
#ifdef SPLINELIB
    if (nsamples == 1) {
      auto splinelib_bspline = to_splinelib_bspline(bspline);
      
      // B-spline evaluation
      using ParametricCoordinate       = typename decltype(splinelib_bspline)::ParametricCoordinate_;
      using ScalarParametricCoordinate = typename ParametricCoordinate::value_type;
      using Derivative                 = typename decltype(splinelib_bspline)::ParameterSpace_::Derivative_;
      using ScalarDerivative           = typename Derivative::value_type;
      
      auto t1 = std::chrono::high_resolution_clock::now();
      for (int i=0; i<1000; i++)
        splinelib_bspline(ParametricCoordinate
                          {
                            ScalarParametricCoordinate{0.5},
                            ScalarParametricCoordinate{0.5}
                          },
                          Derivative
                          {
                            ScalarDerivative{ (short_t)deriv    %10},
                            ScalarDerivative{((short_t)deriv/10)%10}
                          });
      auto t2 = std::chrono::high_resolution_clock::now();
      std::cout << std::right << std::setw(10)
                << std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count() / double(1000)
                << " (ns/entry)";
    } else {
      std::cout << std::right << std::setw(10)
                << "--"
                << " (ns/entry)";
    }
#endif
  }

  template<typename real_t, short_t GeoDim,
           short_t Degree0, short_t Degree1, short_t Degree2,
           iganet::deriv deriv,
           bool precompute = false>
  auto test_UniformBSpline(int64_t ncoeffs, int64_t nsamples)
  {
    iganet::core<real_t> core_(false);
    iganet::UniformBSpline<iganet::core<real_t, true>, GeoDim, Degree0, Degree1, Degree2> bspline({ncoeffs,
                                                                                                   ncoeffs,
                                                                                                   ncoeffs}, iganet::init::linear);
    iganet::TensorArray3 xi = {torch::rand(nsamples, core_.options()),
                               torch::rand(nsamples, core_.options()),
                               torch::rand(nsamples, core_.options())};
    
    auto t1 = std::chrono::high_resolution_clock::now();
    if constexpr (precompute)
    {
      auto knot_idx  = bspline.find_knot_indices(xi);
      auto basfunc   = bspline.template eval_basfunc<deriv>(xi, knot_idx);
      auto coeff_idx = bspline.eval_coeff_indices(knot_idx);
      for (int i=0; i<10; i++)
        auto bspline_val = bspline.eval_from_precomputed(basfunc, coeff_idx, xi[0].numel(), xi[0].sizes());
    }
    else
      for (int i=0; i<10; i++)
        bspline.template eval<deriv>(xi);
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << std::right << std::setw(10)
              << std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count() / double(nsamples*10)
              << " (ns/entry)";
    
#ifdef SPLINELIB
    if (nsamples == 1) {
      auto splinelib_bspline = to_splinelib_bspline(bspline);
      
      // B-spline evaluation
      using ParametricCoordinate       = typename decltype(splinelib_bspline)::ParametricCoordinate_;
      using ScalarParametricCoordinate = typename ParametricCoordinate::value_type;
      using Derivative                 = typename decltype(splinelib_bspline)::ParameterSpace_::Derivative_;
      using ScalarDerivative           = typename Derivative::value_type;
      
      auto t1 = std::chrono::high_resolution_clock::now();
      for (int i=0; i<1000; i++)
        splinelib_bspline(ParametricCoordinate
                          {
                            ScalarParametricCoordinate{0.5},
                            ScalarParametricCoordinate{0.5},
                            ScalarParametricCoordinate{0.5}
                          },
                          Derivative
                          {
                            ScalarDerivative{ (short_t)deriv     %10},
                            ScalarDerivative{((short_t)deriv/ 10)%10},
                            ScalarDerivative{((short_t)deriv/100)%10}
                          });
      auto t2 = std::chrono::high_resolution_clock::now();
      std::cout << std::right << std::setw(10)
                << std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count() / double(1000)
                << " (ns/entry)";
    } else {
      std::cout << std::right << std::setw(10)
                << "--"
                << " (ns/entry)";
    }
#endif
  }

  template<typename real_t, short_t GeoDim,
           short_t Degree0, short_t Degree1, short_t Degree2, short_t Degree3,
           iganet::deriv deriv,
           bool precompute = false>
  auto test_UniformBSpline(int64_t ncoeffs, int64_t nsamples)
  {
    iganet::core<real_t> core_(false);
    iganet::UniformBSpline<iganet::core<real_t, true>, GeoDim, Degree0, Degree1, Degree2, Degree3> bspline({ncoeffs,
                                                                                                            ncoeffs,
                                                                                                            ncoeffs,
                                                                                                            ncoeffs}, iganet::init::linear);
    iganet::TensorArray4 xi = {torch::rand(nsamples, core_.options()),
                               torch::rand(nsamples, core_.options()),
                               torch::rand(nsamples, core_.options()),
                               torch::rand(nsamples, core_.options())};
    
    auto t1 = std::chrono::high_resolution_clock::now();
    if constexpr (precompute)
    {
      auto knot_idx  = bspline.find_knot_indices(xi);
      auto basfunc   = bspline.template eval_basfunc<deriv>(xi, knot_idx);
      auto coeff_idx = bspline.eval_coeff_indices(knot_idx);
      for (int i=0; i<10; i++)
        auto bspline_val = bspline.eval_from_precomputed(basfunc, coeff_idx, xi[0].numel(), xi[0].sizes());
    }
    else
      for (int i=0; i<10; i++)
        bspline.template eval<deriv>(xi);
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << std::right << std::setw(10)
              << std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count() / double(nsamples*10)
              << " (ns/entry)";
    
#ifdef SPLINELIB
    if (nsamples == 1) {
      auto splinelib_bspline = to_splinelib_bspline(bspline);
      
      // B-spline evaluation
      using ParametricCoordinate       = typename decltype(splinelib_bspline)::ParametricCoordinate_;
      using ScalarParametricCoordinate = typename ParametricCoordinate::value_type;
      using Derivative                 = typename decltype(splinelib_bspline)::ParameterSpace_::Derivative_;
      using ScalarDerivative           = typename Derivative::value_type;
      
      auto t1 = std::chrono::high_resolution_clock::now();
      for (int i=0; i<1000; i++)
        splinelib_bspline(ParametricCoordinate
                          {
                            ScalarParametricCoordinate{0.5},
                            ScalarParametricCoordinate{0.5},
                            ScalarParametricCoordinate{0.5},
                            ScalarParametricCoordinate{0.5}
                          },
                          Derivative
                          {
                            ScalarDerivative{ (short_t)deriv      %10},
                            ScalarDerivative{((short_t)deriv/  10)%10},
                            ScalarDerivative{((short_t)deriv/ 100)%10},
                            ScalarDerivative{((short_t)deriv/1000)%10}
                          });
      auto t2 = std::chrono::high_resolution_clock::now();
      std::cout << std::right << std::setw(10)
                << std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count() / double(1000)
                << " (ns/entry)";
    } else {
      std::cout << std::right << std::setw(10)
                << "--"
                << " (ns/entry)";
    }
#endif
  }
  
} // namespace perftest

TEST(Performance, UniformBSpline_parDim1_double)
{
  std::cout << std::scientific << std::setprecision(3);  
  for (int64_t ncoeffs : {10, 100}) {
    for (int64_t nsamples : {1, 10, 100, 1000, 10000, 25000, 50000, 100000, 250000, 500000, 1000000}) {

      std::cout << "("
                << std::right << std::setw(8) << ncoeffs << ","
                << std::right << std::setw(8) << nsamples << ") ";
      perftest::template test_UniformBSpline<double, 1, 1, deriv, precompute>(ncoeffs, nsamples);
      perftest::template test_UniformBSpline<double, 1, 2, deriv, precompute>(ncoeffs, nsamples);
      perftest::template test_UniformBSpline<double, 1, 3, deriv, precompute>(ncoeffs, nsamples);
      perftest::template test_UniformBSpline<double, 1, 4, deriv, precompute>(ncoeffs, nsamples);
      perftest::template test_UniformBSpline<double, 1, 5, deriv, precompute>(ncoeffs, nsamples);
      std::cout << std::endl;
    }
  }
}

TEST(Performance, UniformBSpline_parDim2_double)
{
  std::cout << std::scientific << std::setprecision(3);  
  for (int64_t ncoeffs : {10, 100}) {
    for (int64_t nsamples : {1, 10, 100, 1000, 10000, 25000, 50000, 100000, 250000, 500000, 1000000}) {
      
      std::cout << "("
                << std::right << std::setw(8) << ncoeffs << ","
                << std::right << std::setw(8) << nsamples << ") ";
      
      perftest::template test_UniformBSpline<double, 1, 1, 1, deriv, precompute>(ncoeffs, nsamples);
      perftest::template test_UniformBSpline<double, 1, 2, 2, deriv, precompute>(ncoeffs, nsamples);
      perftest::template test_UniformBSpline<double, 1, 3, 3, deriv, precompute>(ncoeffs, nsamples);
      perftest::template test_UniformBSpline<double, 1, 4, 4, deriv, precompute>(ncoeffs, nsamples);
      perftest::template test_UniformBSpline<double, 1, 5, 5, deriv, precompute>(ncoeffs, nsamples);
      std::cout << std::endl;
    }
  }
}

TEST(Performance, UniformBSpline_parDim3_double)
{
  std::cout << std::scientific << std::setprecision(3);  
  for (int64_t ncoeffs : {10, 100}) {
    for (int64_t nsamples : {1, 10, 100, 1000, 10000, 25000, 50000, 100000, 250000, 500000, 1000000}) {
      
      std::cout << "("
                << std::right << std::setw(8) << ncoeffs << ","
                << std::right << std::setw(8) << nsamples << ") ";
      
      perftest::template test_UniformBSpline<double, 1, 1, 1, 1, deriv, precompute>(ncoeffs, nsamples);
      perftest::template test_UniformBSpline<double, 1, 2, 2, 2, deriv, precompute>(ncoeffs, nsamples);
      perftest::template test_UniformBSpline<double, 1, 3, 3, 3, deriv, precompute>(ncoeffs, nsamples);
      perftest::template test_UniformBSpline<double, 1, 4, 4, 4, deriv, precompute>(ncoeffs, nsamples);
      perftest::template test_UniformBSpline<double, 1, 5, 5, 5, deriv, precompute>(ncoeffs, nsamples);
      std::cout << std::endl;
    }
  }
}

TEST(Performance, UniformBSpline_parDim4_double)
{
  std::cout << std::scientific << std::setprecision(3);  
  for (int64_t ncoeffs : {10, 100}) {
    for (int64_t nsamples : {1, 10, 100, 1000, 10000, 25000, 50000, 100000, 250000, 500000, 1000000}) {
      
      std::cout << "("
                << std::right << std::setw(8) << ncoeffs << ","
                << std::right << std::setw(8) << nsamples << ") ";
      
      perftest::template test_UniformBSpline<double, 1, 1, 1, 1, 1, deriv, precompute>(ncoeffs, nsamples);
      perftest::template test_UniformBSpline<double, 1, 2, 2, 2, 2, deriv, precompute>(ncoeffs, nsamples);
      perftest::template test_UniformBSpline<double, 1, 3, 3, 3, 3, deriv, precompute>(ncoeffs, nsamples);
      perftest::template test_UniformBSpline<double, 1, 4, 4, 4, 4, deriv, precompute>(ncoeffs, nsamples);
      perftest::template test_UniformBSpline<double, 1, 5, 5, 5, 5, deriv, precompute>(ncoeffs, nsamples);
      std::cout << std::endl;
    }
  } 
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  iganet::init();
  return RUN_ALL_TESTS();
}
