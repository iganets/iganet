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

#include <cmath>

#include <core/core.hpp>

namespace iganet::utils {

namespace detail {

inline bool tensor_values_are_finite(const torch::Tensor &tensor) {
  const auto &values = tensor.layout() == torch::kStrided
                           ? tensor
                           : tensor.values();
  return torch::isfinite(values).all().item<bool>();
}

inline void validate_iterative_solver_inputs(const torch::Tensor &A,
                                             const torch::Tensor &b,
                                             int max_iter, double tol) {
  TORCH_CHECK(A.dim() == 2 && A.size(0) == A.size(1),
              "iterative solver requires a square rank-2 matrix");
  TORCH_CHECK(b.dim() == 1 && b.size(0) == A.size(0),
              "iterative solver requires a compatible rank-1 right-hand side");
  TORCH_CHECK(A.scalar_type() == b.scalar_type(),
              "matrix and right-hand side must have the same dtype");
  TORCH_CHECK(A.device() == b.device(),
              "matrix and right-hand side must be on the same device");
  TORCH_CHECK(A.is_floating_point() && b.is_floating_point(),
              "iterative solver requires floating-point inputs");
  TORCH_CHECK(max_iter >= 0, "max_iter must be non-negative");
  TORCH_CHECK(std::isfinite(tol) && tol > 0.0,
              "tolerance must be finite and positive");
  TORCH_CHECK(tensor_values_are_finite(A),
              "matrix contains NaN or Inf");
  TORCH_CHECK(tensor_values_are_finite(b),
              "right-hand side contains NaN or Inf");
}

inline void check_finite_scalar(const torch::Tensor &value,
                                const char *message) {
  TORCH_CHECK(torch::isfinite(value).item<bool>(), message);
}

inline void check_nonzero_finite_scalar(const torch::Tensor &value,
                                        const char *message) {
  check_finite_scalar(value, message);
  TORCH_CHECK(value.item<double>() != 0.0, message);
}

} // namespace detail

  /// @brief Solves the linear system A * x = b using the Conjugate
  /// Gradient (CG) method
  inline auto solve_cg(const torch::Tensor& A,
                       const torch::Tensor b,
                       int max_iter = 1000,
                       double tol = 1e-10) {

    detail::validate_iterative_solver_inputs(A, b, max_iter, tol);

    auto x = torch::zeros_like(b);

    if (b.norm().item<double>() < tol)
      return std::make_tuple(x, -1, b.norm().item<double>());
    
    auto r = b.clone();  
    auto p = b.clone();

    for (int iter = 0; iter < max_iter; iter++) {

      auto Ap = A.matmul(p);
      auto beta = torch::dot(r, r);
      detail::check_nonzero_finite_scalar(beta,
                                          "CG numerical breakdown: r.r is zero");
      auto denominator = torch::dot(Ap, p);
      detail::check_nonzero_finite_scalar(
          denominator, "CG numerical breakdown: p.A.p is zero or non-finite");
      auto alpha = beta / denominator;
      detail::check_finite_scalar(alpha,
                                  "CG numerical breakdown: alpha is non-finite");
      
      x += alpha * p;
      r -= alpha * Ap;
      
      if (r.norm().item<double>() < tol)
        return std::make_tuple(x, iter, r.norm().item<double>());

      beta = torch::dot(r, r) / beta;
      detail::check_finite_scalar(beta,
                                  "CG numerical breakdown: beta is non-finite");
      p = r + beta * p;      
    }

    return std::make_tuple(x, max_iter, r.norm().item<double>());    
  }

  /// @brief Solves the linear system A * x = b using the Bi-Conjugate
  /// Gradient Stabilized (BiCGStab) method
  inline auto solve_bicgstab(const torch::Tensor& A,
                             const torch::Tensor b,
                             int max_iter = 1000,
                             double tol = 1e-10) {

    detail::validate_iterative_solver_inputs(A, b, max_iter, tol);

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
      detail::check_nonzero_finite_scalar(
          rho_hat, "BiCGStab numerical breakdown: rho is zero or non-finite");
      detail::check_nonzero_finite_scalar(
          rho, "BiCGStab numerical breakdown: previous rho is zero or non-finite");
      detail::check_nonzero_finite_scalar(
          omega,
          "BiCGStab numerical breakdown: omega is zero or non-finite");
      auto beta = rho_hat / rho * alpha / omega;
      detail::check_finite_scalar(
          beta, "BiCGStab numerical breakdown: beta is non-finite");

      p = r + beta * (p - omega * v);
      v = A.matmul(p);

      auto alpha_denominator = torch::dot(r_hat, v);
      detail::check_nonzero_finite_scalar(
          alpha_denominator,
          "BiCGStab numerical breakdown: alpha denominator is zero or non-finite");
      alpha = rho_hat / alpha_denominator;
      detail::check_finite_scalar(
          alpha, "BiCGStab numerical breakdown: alpha is non-finite");
      auto s = r - alpha * v;

      if (s.norm().item<double>() < tol) {
        x += alpha * p;
        return std::make_tuple(x, iter, s.norm().item<double>());
      }

      auto t = A.matmul(s);
      auto omega_denominator = torch::dot(t, t);
      detail::check_nonzero_finite_scalar(
          omega_denominator,
          "BiCGStab numerical breakdown: t.t is zero or non-finite");
      omega = torch::dot(s, t) / omega_denominator;
      detail::check_nonzero_finite_scalar(
          omega, "BiCGStab numerical breakdown: omega is zero or non-finite");
      x += alpha * p + omega * s;
      r = s - omega * t;
      TORCH_CHECK(torch::isfinite(r).all().item<bool>(),
                  "BiCGStab numerical breakdown: residual is non-finite");
      rho = rho_hat;
    }

    return std::make_tuple(x, max_iter, r.norm().item<double>());    
  }
} // namespace iganet::utils
