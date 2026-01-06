/**
   @file utils/solver.hpp

   @brief Solver utility functions

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#pragma once

#include <core/core.hpp>

namespace iganet::utils {

  /// @brief Solves the linear system A * x = b using the Conjugate
  /// Gradient (CG) method
  auto solve_cg(const torch::Tensor& A,
                const torch::Tensor b,
                int max_iter = 1000,
                double tol = 1e-10) {

    auto x = torch::zeros_like(b);

    if (b.norm().item<double>() < tol)
      return std::make_tuple(x, -1, b.norm().item<double>());
    
    auto r = b.clone();  
    auto p = b.clone();

    for (int iter = 0; iter < max_iter; iter++) {

      auto Ap = A.matmul(p);
      auto beta = torch::dot(r, r);
      auto alpha = beta / torch::dot(Ap, p);
      
      x += alpha * p;
      r -= alpha * Ap;
      
      if (r.norm().item<double>() < tol)
        return std::make_tuple(x, iter, r.norm().item<double>());

      beta = torch::dot(r, r) / beta;
      p = r + beta * p;      
    }

    return std::make_tuple(x, max_iter, r.norm().item<double>());    
  }

  /// @brief Solves the linear system A * x = b using the Bi-Conjugate
  /// Gradient Stabilized (BiCGStab) method
  auto solve_bicgstab(const torch::Tensor& A,
                      const torch::Tensor b,
                      int max_iter = 1000,
                      double tol = 1e-10) {

    auto x = torch::zeros_like(b);

    if (b.norm().item<double>() < tol)
      return std::make_tuple(x, -1, b.norm().item<double>());
    
    auto r = b.clone();  
    auto r_hat = b.clone();

    auto alpha = torch::scalar_tensor(1.0, b.options());
    auto omega = torch::scalar_tensor(1.0, b.options());
    auto rho = torch::scalar_tensor(1.0, b.options());

    auto p = torch::zeros_like(b);
    auto v = torch::zeros_like(b);
    
    for (int iter = 0; iter < max_iter; iter++) {
      
      auto rho_hat = torch::dot(r_hat, r);
      auto beta = rho_hat / rho * alpha / omega;

      p = r + beta * (p - omega * v);
      v = A.matmul(p);

      alpha = rho_hat / torch::dot(r_hat, v);
      auto s = r - alpha * v;

      if (s.norm().item<double>() < tol) {
        x += alpha * p;
        return std::make_tuple(x, iter, s.norm().item<double>());
      }

      auto t = A.matmul(s);
      omega = torch::dot(s, t) / torch::dot(t, t);
      x += alpha * p + omega * s;
      r = s - omega * t;
    }

    return std::make_tuple(x, max_iter, r.norm().item<double>());    
  }
} // namespace iganet::utils
