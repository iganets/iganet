/**
   @file utils/solver.hpp

   @brief Solver utility functions.

   @author Matthias Moller.

   @copyright This file is part of the IgANet project.

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#pragma once

#include <algorithm>
#include <cmath>
#include <concepts>
#include <vector>

#include <core/core.hpp>

namespace iganet::utils {

/// @brief Specifies the callable interface required by the preconditioned
/// iterative solvers.
template <typename T>
concept IterativeSolverPreconditioner =
    requires(T &preconditioner, const torch::Tensor &residual) {
      { preconditioner(residual) } -> std::same_as<torch::Tensor>;
    };

namespace detail {

#ifndef NDEBUG
/// @brief Provides the `tensor_values_are_finite` operation.
/// @param tensor Tensor to process.
/// @return Result of the operation.
inline bool tensor_values_are_finite(const torch::Tensor &tensor) {
  const auto &values = tensor.layout() == torch::kStrided
                           ? tensor
                           : tensor.values();
  return torch::isfinite(values).all().item<bool>();
}

/// @brief Provides the `validate_iterative_solver_inputs` operation.
/// @param A Value of `A`.
/// @param b Value of `b`.
/// @param max_iter Value of `max_iter`.
/// @param tol Value of `tol`.
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

/// @brief Provides the `check_finite_scalar` operation.
/// @param value Value to process.
/// @param message Value of `message`.
inline void check_finite_scalar(const torch::Tensor &value,
                                const char *message) {
  TORCH_CHECK(torch::isfinite(value).item<bool>(), message);
}

/// @brief Provides the `check_finite_tensor` operation.
/// @param value Value to process.
/// @param message Value of `message`.
inline void check_finite_tensor(const torch::Tensor &value,
                                const char *message) {
  TORCH_CHECK(tensor_values_are_finite(value), message);
}

/// @brief Provides the `validate_preconditioner_output` operation.
/// @param value Value to process.
/// @param residual Value of `residual`.
inline void validate_preconditioner_output(const torch::Tensor &value,
                                           const torch::Tensor &residual) {
  TORCH_CHECK(value.sizes() == residual.sizes(),
              "preconditioner must preserve the residual shape");
  TORCH_CHECK(value.scalar_type() == residual.scalar_type(),
              "preconditioner must preserve the residual dtype");
  TORCH_CHECK(value.device() == residual.device(),
              "preconditioner must preserve the residual device");
  TORCH_CHECK(value.layout() == torch::kStrided,
              "preconditioner must return a strided tensor");
  TORCH_CHECK(tensor_values_are_finite(value),
              "preconditioner returned NaN or Inf");
}

/// @brief Provides the `validate_inverse_preconditioner` operation.
/// @param inverse_preconditioner Value of `inverse_preconditioner`.
/// @param A Value of `A`.
inline void validate_inverse_preconditioner(
    const torch::Tensor &inverse_preconditioner, const torch::Tensor &A) {
  TORCH_CHECK(inverse_preconditioner.dim() == 2 &&
                  inverse_preconditioner.size(0) == A.size(0) &&
                  inverse_preconditioner.size(1) == A.size(1),
              "inverse preconditioner must have the same square shape as A");
  TORCH_CHECK(inverse_preconditioner.scalar_type() == A.scalar_type(),
              "inverse preconditioner and A must have the same dtype");
  TORCH_CHECK(inverse_preconditioner.device() == A.device(),
              "inverse preconditioner and A must be on the same device");
  TORCH_CHECK(tensor_values_are_finite(inverse_preconditioner),
              "inverse preconditioner contains NaN or Inf");
}

/// @brief Provides the `validate_gmres_parameters` operation.
/// @param restart Value of `restart`.
inline void validate_gmres_parameters(int restart) {
  TORCH_CHECK(restart > 0, "GMRES restart must be positive");
}

/// @brief Provides the `check_nonzero_finite_scalar` operation.
/// @param value Value to process.
/// @param message Value of `message`.
inline void check_nonzero_finite_scalar(const torch::Tensor &value,
                                        const char *message) {
  check_finite_scalar(value, message);
  TORCH_CHECK(value.item<double>() != 0.0, message);
}

/// @brief Provides the `check_positive_finite_scalar` operation.
/// @param value Value to process.
/// @param message Value of `message`.
inline void check_positive_finite_scalar(const torch::Tensor &value,
                                         const char *message) {
  check_finite_scalar(value, message);
  TORCH_CHECK(value.item<double>() > 0.0, message);
}

/// @brief Provides the `check_nonnegative_finite_scalar` operation.
/// @param value Value to process.
/// @param message Value of `message`.
inline void check_nonnegative_finite_scalar(const torch::Tensor &value,
                                            const char *message) {
  check_finite_scalar(value, message);
  TORCH_CHECK(value.item<double>() >= 0.0, message);
}
#else
inline void validate_iterative_solver_inputs(const torch::Tensor &,
                                             const torch::Tensor &, int,
                                             double) {}

inline void check_finite_scalar(const torch::Tensor &, const char *) {}

inline void check_finite_tensor(const torch::Tensor &, const char *) {}

inline void validate_preconditioner_output(const torch::Tensor &,
                                           const torch::Tensor &) {}

inline void validate_inverse_preconditioner(const torch::Tensor &,
                                            const torch::Tensor &) {}

inline void validate_gmres_parameters(int) {}

inline void check_nonzero_finite_scalar(const torch::Tensor &,
                                        const char *) {}

inline void check_positive_finite_scalar(const torch::Tensor &,
                                         const char *) {}

inline void check_nonnegative_finite_scalar(const torch::Tensor &,
                                            const char *) {}
#endif

} // namespace detail

  /// @brief Solves the linear system A * x = b using the Conjugate
  /// Gradient (CG) method.
  /// @param A Value of `A`.
  /// @param b Value of `b`.
  /// @param max_iter Value of `max_iter`.
  /// @param tol Value of `tol`.
  /// @return Result of the operation.
  inline auto cg(const torch::Tensor& A,
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

  /// @brief Solves the linear system A * x = b using the preconditioned
  /// Conjugate Gradient (PCG) method.
  ///
  /// The preconditioner must be callable with a residual tensor and return
  /// the action of the inverse preconditioner on that residual.
  /// @tparam Preconditioner Template parameter `Preconditioner`.
  /// @param A Value of `A`.
  /// @param b Value of `b`.
  /// @param preconditioner Value of `preconditioner`.
  /// @param max_iter Value of `max_iter`.
  /// @param tol Value of `tol`.
  /// @return Result of the operation.
  template <IterativeSolverPreconditioner Preconditioner>
  inline auto pcg(const torch::Tensor &A, const torch::Tensor b,
                        Preconditioner &&preconditioner,
                        int max_iter = 1000, double tol = 1e-10) {
    detail::validate_iterative_solver_inputs(A, b, max_iter, tol);

    auto x = torch::zeros_like(b);
    const auto initial_residual = b.norm().item<double>();
    if (initial_residual < tol)
      return std::make_tuple(x, -1, initial_residual);
    if (max_iter == 0)
      return std::make_tuple(x, 0, initial_residual);

    auto r = b.clone();
    auto z = preconditioner(r);
    detail::validate_preconditioner_output(z, r);
    auto p = z.clone();
    auto rz = torch::dot(r, z);
    detail::check_nonzero_finite_scalar(
        rz, "PCG numerical breakdown: r.z is zero or non-finite");

    for (int iter = 0; iter < max_iter; ++iter) {
      auto Ap = A.matmul(p);
      auto denominator = torch::dot(p, Ap);
      detail::check_nonzero_finite_scalar(
          denominator, "PCG numerical breakdown: p.A.p is zero or non-finite");
      auto alpha = rz / denominator;
      detail::check_finite_scalar(
          alpha, "PCG numerical breakdown: alpha is non-finite");

      x += alpha * p;
      r -= alpha * Ap;

      const auto residual = r.norm().item<double>();
      if (residual < tol)
        return std::make_tuple(x, iter, residual);

      z = preconditioner(r);
      detail::validate_preconditioner_output(z, r);
      auto rz_next = torch::dot(r, z);
      detail::check_nonzero_finite_scalar(
          rz_next, "PCG numerical breakdown: next r.z is zero or non-finite");
      auto beta = rz_next / rz;
      detail::check_finite_scalar(
          beta, "PCG numerical breakdown: beta is non-finite");
      p = z + beta * p;
      rz = rz_next;
    }

    return std::make_tuple(x, max_iter, r.norm().item<double>());
  }

  /// @brief PCG overload taking the inverse preconditioner as a tensor.
  /// @param A Value of `A`.
  /// @param b Value of `b`.
  /// @param inverse_preconditioner Value of `inverse_preconditioner`.
  /// @param max_iter Value of `max_iter`.
  /// @param tol Value of `tol`.
  /// @return Result of the operation.
  inline auto pcg(const torch::Tensor &A, const torch::Tensor b,
                        const torch::Tensor &inverse_preconditioner,
                        int max_iter = 1000, double tol = 1e-10) {
    detail::validate_inverse_preconditioner(inverse_preconditioner, A);
    auto apply = [&inverse_preconditioner](const torch::Tensor &residual) {
      return inverse_preconditioner.matmul(residual);
    };
    return pcg(A, b, apply, max_iter, tol);
  }

  /// @brief Solves the linear system A * x = b using the Bi-Conjugate
  /// Gradient Stabilized (BiCGStab) method.
  /// @param A Value of `A`.
  /// @param b Value of `b`.
  /// @param max_iter Value of `max_iter`.
  /// @param tol Value of `tol`.
  /// @return Result of the operation.
  inline auto bicgstab(const torch::Tensor& A,
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
      detail::check_finite_tensor(
          r, "BiCGStab numerical breakdown: residual is non-finite");
      rho = rho_hat;
    }

    return std::make_tuple(x, max_iter, r.norm().item<double>());    
  }

  /// @brief Solves the linear system A * x = b using the preconditioned
  /// Bi-Conjugate Gradient Stabilized (PBiCGStab) method.
  ///
  /// The preconditioner must be callable with a residual tensor and return
  /// the action of the inverse preconditioner on that residual.
  /// @tparam Preconditioner Template parameter `Preconditioner`.
  /// @param A Value of `A`.
  /// @param b Value of `b`.
  /// @param preconditioner Value of `preconditioner`.
  /// @param max_iter Value of `max_iter`.
  /// @param tol Value of `tol`.
  /// @return Result of the operation.
  template <IterativeSolverPreconditioner Preconditioner>
  inline auto pbicgstab(const torch::Tensor &A, const torch::Tensor b,
                             Preconditioner &&preconditioner,
                             int max_iter = 1000, double tol = 1e-10) {
    detail::validate_iterative_solver_inputs(A, b, max_iter, tol);

    auto x = torch::zeros_like(b);
    const auto initial_residual = b.norm().item<double>();
    if (initial_residual < tol)
      return std::make_tuple(x, -1, initial_residual);

    auto r = b.clone();
    auto r_hat = b.clone();
    auto alpha = torch::scalar_tensor(1.0, b.options());
    auto omega = torch::scalar_tensor(1.0, b.options());
    auto rho = torch::scalar_tensor(1.0, b.options());
    auto p = torch::zeros_like(b);
    auto v = torch::zeros_like(b);

    for (int iter = 0; iter < max_iter; ++iter) {
      auto rho_hat = torch::dot(r_hat, r);
      detail::check_nonzero_finite_scalar(
          rho_hat, "PBiCGStab numerical breakdown: rho is zero or non-finite");
      detail::check_nonzero_finite_scalar(
          rho, "PBiCGStab numerical breakdown: previous rho is zero or non-finite");
      detail::check_nonzero_finite_scalar(
          omega, "PBiCGStab numerical breakdown: omega is zero or non-finite");
      auto beta = rho_hat / rho * alpha / omega;
      detail::check_finite_scalar(
          beta, "PBiCGStab numerical breakdown: beta is non-finite");

      p = r + beta * (p - omega * v);
      auto p_hat = preconditioner(p);
      detail::validate_preconditioner_output(p_hat, p);
      v = A.matmul(p_hat);

      auto alpha_denominator = torch::dot(r_hat, v);
      detail::check_nonzero_finite_scalar(
          alpha_denominator,
          "PBiCGStab numerical breakdown: alpha denominator is zero or non-finite");
      alpha = rho_hat / alpha_denominator;
      detail::check_finite_scalar(
          alpha, "PBiCGStab numerical breakdown: alpha is non-finite");
      auto s = r - alpha * v;

      const auto s_residual = s.norm().item<double>();
      if (s_residual < tol) {
        x += alpha * p_hat;
        return std::make_tuple(x, iter, s_residual);
      }

      auto s_hat = preconditioner(s);
      detail::validate_preconditioner_output(s_hat, s);
      auto t = A.matmul(s_hat);
      auto omega_denominator = torch::dot(t, t);
      detail::check_nonzero_finite_scalar(
          omega_denominator,
          "PBiCGStab numerical breakdown: t.t is zero or non-finite");
      omega = torch::dot(s, t) / omega_denominator;
      detail::check_nonzero_finite_scalar(
          omega, "PBiCGStab numerical breakdown: omega is zero or non-finite");
      x += alpha * p_hat + omega * s_hat;
      r = s - omega * t;
      detail::check_finite_tensor(
          r, "PBiCGStab numerical breakdown: residual is non-finite");
      rho = rho_hat;
    }

    return std::make_tuple(x, max_iter, r.norm().item<double>());
  }

  /// @brief PBiCGStab overload taking the inverse preconditioner as a tensor.
  /// @param A Value of `A`.
  /// @param b Value of `b`.
  /// @param inverse_preconditioner Value of `inverse_preconditioner`.
  /// @param max_iter Value of `max_iter`.
  /// @param tol Value of `tol`.
  /// @return Result of the operation.
  inline auto pbicgstab(
      const torch::Tensor &A, const torch::Tensor b,
      const torch::Tensor &inverse_preconditioner, int max_iter = 1000,
      double tol = 1e-10) {
    detail::validate_inverse_preconditioner(inverse_preconditioner, A);
    auto apply = [&inverse_preconditioner](const torch::Tensor &residual) {
      return inverse_preconditioner.matmul(residual);
    };
    return pbicgstab(A, b, apply, max_iter, tol);
  }

  /// @brief Solves A * x = b using preconditioned MINRES.
  ///
  /// A must be symmetric and the preconditioner must be symmetric positive
  /// definite. The preconditioner returns the action of its inverse.
  /// @tparam Preconditioner Template parameter `Preconditioner`.
  /// @param A Value of `A`.
  /// @param b Value of `b`.
  /// @param preconditioner Value of `preconditioner`.
  /// @param max_iter Value of `max_iter`.
  /// @param tol Value of `tol`.
  /// @return Result of the operation.
  template <IterativeSolverPreconditioner Preconditioner>
  inline auto pminres(const torch::Tensor &A, const torch::Tensor b,
                            Preconditioner &&preconditioner,
                            int max_iter = 1000, double tol = 1e-10) {
    detail::validate_iterative_solver_inputs(A, b, max_iter, tol);

    auto x = torch::zeros_like(b);
    auto residual = b.norm().item<double>();
    if (residual < tol)
      return std::make_tuple(x, -1, residual);
    if (max_iter == 0)
      return std::make_tuple(x, 0, residual);

    auto r1 = b.clone();
    auto r2 = r1.clone();
    auto y = preconditioner(r1);
    detail::validate_preconditioner_output(y, r1);
    auto beta_squared = torch::dot(r1, y);
    detail::check_positive_finite_scalar(
        beta_squared,
        "MINRES requires a symmetric positive-definite preconditioner");
    auto beta = torch::sqrt(beta_squared);
    auto old_beta = torch::zeros_like(beta);
    auto dbar = torch::zeros_like(beta);
    auto epsilon = torch::zeros_like(beta);
    auto cosine = -torch::ones_like(beta);
    auto sine = torch::zeros_like(beta);
    auto phibar = beta.clone();
    auto w = torch::zeros_like(b);
    auto w_older = torch::zeros_like(b);

    for (int iter = 0; iter < max_iter; ++iter) {
      auto v = y / beta;
      y = A.matmul(v);
      if (iter > 0)
        y -= (beta / old_beta) * r1;
      auto alpha = torch::dot(v, y);
      y -= (alpha / beta) * r2;
      r1 = r2;
      r2 = y;
      y = preconditioner(r2);
      detail::validate_preconditioner_output(y, r2);

      old_beta = beta;
      beta_squared = torch::dot(r2, y);
      detail::check_nonnegative_finite_scalar(
          beta_squared,
          "MINRES requires a symmetric positive-definite preconditioner");
      beta = torch::sqrt(torch::clamp_min(beta_squared, 0.0));

      auto old_epsilon = epsilon;
      auto delta = cosine * dbar + sine * alpha;
      auto gbar = sine * dbar - cosine * alpha;
      epsilon = sine * beta;
      dbar = -cosine * beta;
      auto gamma = torch::sqrt(gbar * gbar + beta * beta);
      detail::check_nonzero_finite_scalar(
          gamma, "MINRES numerical breakdown: rotation norm is zero or non-finite");
      cosine = gbar / gamma;
      sine = beta / gamma;
      auto phi = cosine * phibar;
      phibar = sine * phibar;

      auto w_old = w;
      w = (v - old_epsilon * w_older - delta * w_old) / gamma;
      w_older = w_old;
      x += phi * w;

      if (phibar.abs().template item<double>() < tol) {
        residual = (b - A.matmul(x)).norm().item<double>();
        if (residual < tol)
          return std::make_tuple(x, iter, residual);
      }
    }

    residual = (b - A.matmul(x)).norm().item<double>();
    return std::make_tuple(x, max_iter, residual);
  }

  /// @brief PMINRES overload taking the inverse preconditioner as a tensor.
  /// @param A Value of `A`.
  /// @param b Value of `b`.
  /// @param inverse_preconditioner Value of `inverse_preconditioner`.
  /// @param max_iter Value of `max_iter`.
  /// @param tol Value of `tol`.
  /// @return Result of the operation.
  inline auto pminres(const torch::Tensor &A, const torch::Tensor b,
                            const torch::Tensor &inverse_preconditioner,
                            int max_iter = 1000, double tol = 1e-10) {
    detail::validate_inverse_preconditioner(inverse_preconditioner, A);
    auto apply = [&inverse_preconditioner](const torch::Tensor &residual) {
      return inverse_preconditioner.matmul(residual);
    };
    return pminres(A, b, apply, max_iter, tol);
  }

  /// @brief Solves A * x = b using MINRES.
  /// @param A Value of `A`.
  /// @param b Value of `b`.
  /// @param max_iter Value of `max_iter`.
  /// @param tol Value of `tol`.
  /// @return Result of the operation.
  inline auto minres(const torch::Tensor &A, const torch::Tensor b,
                           int max_iter = 1000, double tol = 1e-10) {
    auto identity = [](const torch::Tensor &residual) {
      return residual.clone();
    };
    return pminres(A, b, identity, max_iter, tol);
  }

  /// @brief Solves A * x = b using restarted, right-preconditioned GMRES.
  ///
  /// The preconditioner must return the action of the inverse preconditioner.
  /// Setting restart equal to max_iter gives unrestarted GMRES.
  /// @tparam Preconditioner Template parameter `Preconditioner`.
  /// @param A Value of `A`.
  /// @param b Value of `b`.
  /// @param preconditioner Value of `preconditioner`.
  /// @param max_iter Value of `max_iter`.
  /// @param tol Value of `tol`.
  /// @param restart Value of `restart`.
  /// @return Result of the operation.
  template <IterativeSolverPreconditioner Preconditioner>
  inline auto fgmres(const torch::Tensor &A, const torch::Tensor b,
                           Preconditioner &&preconditioner,
                           int max_iter = 1000, double tol = 1e-10,
                           int restart = 30) {
    detail::validate_iterative_solver_inputs(A, b, max_iter, tol);
    detail::validate_gmres_parameters(restart);

    auto x = torch::zeros_like(b);
    auto r = b.clone();
    auto residual = r.norm().item<double>();
    if (residual < tol)
      return std::make_tuple(x, -1, residual);
    if (max_iter == 0)
      return std::make_tuple(x, 0, residual);

    int iterations = 0;
    while (iterations < max_iter) {
      const int cycle_size = std::min(restart, max_iter - iterations);
      auto beta = r.norm();
      std::vector<torch::Tensor> basis;
      std::vector<torch::Tensor> preconditioned_basis;
      std::vector<torch::Tensor> cosines;
      std::vector<torch::Tensor> sines;
      basis.reserve(cycle_size + 1);
      preconditioned_basis.reserve(cycle_size);
      cosines.reserve(cycle_size);
      sines.reserve(cycle_size);
      basis.emplace_back(r / beta);

      auto hessenberg = torch::zeros(
          {cycle_size + 1, cycle_size}, b.options());
      auto transformed_rhs = torch::zeros({cycle_size + 1}, b.options());
      transformed_rhs.index_put_({0}, beta);

      int inner_steps = 0;
      bool estimated_convergence = false;
      for (int j = 0; j < cycle_size; ++j) {
        auto z = preconditioner(basis[j]);
        detail::validate_preconditioner_output(z, basis[j]);
        preconditioned_basis.emplace_back(z);
        auto w = A.matmul(z);

        for (int i = 0; i <= j; ++i) {
          auto coefficient = torch::dot(basis[i], w);
          hessenberg.index_put_({i, j}, coefficient);
          w -= coefficient * basis[i];
        }

        auto next_norm = w.norm();
        hessenberg.index_put_({j + 1, j}, next_norm);
        const bool happy_breakdown = next_norm.template item<double>() == 0.0;
        if (!happy_breakdown)
          basis.emplace_back(w / next_norm);

        for (int i = 0; i < j; ++i) {
          auto upper = hessenberg.index({i, j}).clone();
          auto lower = hessenberg.index({i + 1, j}).clone();
          hessenberg.index_put_({i, j},
                                cosines[i] * upper + sines[i] * lower);
          hessenberg.index_put_({i + 1, j},
                                -sines[i] * upper + cosines[i] * lower);
        }

        auto diagonal = hessenberg.index({j, j}).clone();
        auto subdiagonal = hessenberg.index({j + 1, j}).clone();
        auto rotation_norm = torch::sqrt(diagonal * diagonal +
                                         subdiagonal * subdiagonal);
        detail::check_nonzero_finite_scalar(
            rotation_norm,
            "GMRES numerical breakdown: Givens rotation norm is zero or non-finite");
        auto cosine = diagonal / rotation_norm;
        auto sine = subdiagonal / rotation_norm;
        cosines.emplace_back(cosine);
        sines.emplace_back(sine);
        hessenberg.index_put_({j, j},
                              cosine * diagonal + sine * subdiagonal);
        hessenberg.index_put_({j + 1, j}, torch::zeros_like(subdiagonal));

        auto rhs_entry = transformed_rhs.index({j}).clone();
        auto rhs_next = transformed_rhs.index({j + 1}).clone();
        transformed_rhs.index_put_({j},
                                   cosine * rhs_entry + sine * rhs_next);
        transformed_rhs.index_put_({j + 1},
                                   -sine * rhs_entry + cosine * rhs_next);

        ++iterations;
        inner_steps = j + 1;
        residual = transformed_rhs.index({j + 1}).abs().item<double>();
        if (residual < tol || happy_breakdown) {
          estimated_convergence = true;
          break;
        }
      }

      using torch::indexing::Slice;
      auto upper = hessenberg.index(
          {Slice(0, inner_steps), Slice(0, inner_steps)});
      auto rhs = transformed_rhs.index({Slice(0, inner_steps)}).unsqueeze(1);
      auto coefficients =
          torch::linalg_solve_triangular(upper, rhs, true).squeeze(1);
      for (int i = 0; i < inner_steps; ++i)
        x += coefficients.index({i}) * preconditioned_basis[i];

      r = b - A.matmul(x);
      residual = r.norm().item<double>();
      if (residual < tol)
        return std::make_tuple(x, iterations - 1, residual);

      if (estimated_convergence)
        detail::check_finite_tensor(
            r, "GMRES numerical breakdown: true residual is non-finite");
    }

    return std::make_tuple(x, max_iter, residual);
  }

  /// @brief FGMRES overload taking the inverse preconditioner as a tensor.
  /// @param A Value of `A`.
  /// @param b Value of `b`.
  /// @param inverse_preconditioner Value of `inverse_preconditioner`.
  /// @param max_iter Value of `max_iter`.
  /// @param tol Value of `tol`.
  /// @param restart Value of `restart`.
  /// @return Result of the operation.
  inline auto fgmres(const torch::Tensor &A, const torch::Tensor b,
                           const torch::Tensor &inverse_preconditioner,
                           int max_iter = 1000, double tol = 1e-10,
                           int restart = 30) {
    detail::validate_inverse_preconditioner(inverse_preconditioner, A);
    auto apply = [&inverse_preconditioner](const torch::Tensor &residual) {
      return inverse_preconditioner.matmul(residual);
    };
    return fgmres(A, b, apply, max_iter, tol, restart);
  }

  /// @brief Solves A * x = b using restarted GMRES.
  /// @param A Value of `A`.
  /// @param b Value of `b`.
  /// @param max_iter Value of `max_iter`.
  /// @param tol Value of `tol`.
  /// @param restart Value of `restart`.
  /// @return Result of the operation.
  inline auto gmres(const torch::Tensor &A, const torch::Tensor b,
                          int max_iter = 1000, double tol = 1e-10,
                          int restart = 30) {
    auto identity = [](const torch::Tensor &residual) {
      return residual.clone();
    };
    return fgmres(A, b, identity, max_iter, tol, restart);
  }
} // namespace iganet::utils
