/**
   @file include/utils/spsolver.hpp

   @brief Sparse matrix solvers

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#pragma once

#include <core.hpp>

namespace iganet::utils {

  /// @brief Solves the linear system A * x = b using the Conjugate Gradient method
  auto spsolve_cg(const torch::Tensor& A, const torch::Tensor b, int max_iter = 1000, double tol = 1e-10) {

    auto x = torch::zeros_like(b);

    if (b.norm().item<double>() < tol)
      return std::make_tuple(x, -1, b.norm().item<double>());
    
    auto r = b.clone();  
    auto p = b.clone();

    for (int iter = 0; iter < max_iter; iter++) {

      auto Ap = A.matmul(p);
      auto beta = torch::dot(r, r);
      auto alpha = beta / torch::dot(p, Ap);
      
      x = x + alpha * p;
      r = r - alpha * Ap;

      if (r.norm().item<double>() < tol)
        return std::make_tuple(x, iter, r.norm().item<double>());

      beta = torch::dot(r, r) / beta;

      p = r + beta * p;      
    }

    return std::make_tuple(x, max_iter, r.norm().item<double>());    
  }
  
} // namespace iganet::utils
