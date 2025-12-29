/**
   @file include/igasolver.hpp

   @brief Isogeometric analysis solver

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#pragma once

#include <boundary.hpp>
#include <functionspace.hpp>
#include <igabase.hpp>

#include <utils/spsolver.hpp>

namespace iganet {

/// @brief IgA solver
///
/// This class implements the core functionality of IgA solvers
template <typename Inputs, typename Outputs, typename CollPts = void>
class IgASolver : public IgABase2<Inputs, Outputs, CollPts>,
                  private utils::FullQualifiedName {

protected:
  /// @brief Left-hand side tensor
  torch::Tensor lhs_;

  /// @brief Right-hand side tensor
  torch::Tensor rhs_;
  
public:
  /// @brief Base type
  using Base = IgABase2<Inputs, Outputs, CollPts>;

  /// @brief Base class constructor
  using Base::IgABase2;

  /// @brief Returns a constant reference to the left-hand side object
  inline constexpr const auto &lhs() const { return lhs_; }

  /// @brief Returns a non-constant reference to the left-hand side object
  inline constexpr auto &lhs() { return lhs_; }

  /// @brief Returns a constant reference to the right-hand side object
  inline constexpr const auto &rhs() const { return rhs_; }

  /// @brief Returns a non-constant reference to the right-hand side object
  inline constexpr auto &rhs() { return rhs_; }

  /// @brief Initializes the solver
  virtual void init() = 0;
  
  /// @brief Assembles the solver
  virtual void assemble() {
    assembleLhs();
    assembleRhs();
  }
  
  /// @brief Assembles the left-hand side of the solver
  virtual void assembleLhs() = 0;

  /// @brief Assembles the right-hand side of the solver
  virtual void assembleRhs() = 0;

  /// @brief Computes the solution vector
  torch::Tensor solve() const {
    auto [x, iter, res] = utils::spsolve_bicgstab(lhs(), rhs());    
    return x;
  }
  
  /// @brief Returns a string representation of the IgASolver object
  inline void pretty_print(std::ostream &os) const noexcept override {
    os << name() << "(\n";
    
    os << "inputs[" << Base::ninputs() << "] = (";
    std::apply([&os](const auto &...elems) { ((os << elems << "\n"), ...); },
               Base::inputs());
    os << ")";

    os << "outputs [" << Base::noutputs() << "]= (";
    std::apply([&os](const auto &...elems) { ((os << elems << "\n"), ...); },
               Base::inputs());
    os << ")";

    os << "collPts [" << Base::ncollPts() << "]= (";
    std::apply([&os](const auto &...elems) { ((os << elems << "\n"), ...); },
               Base::collPts());
    os << ")";
  }  
};
  
/// @brief Print (as string) a IgASolver object
template <typename Inputs, typename Outputs, typename CollPts>
inline std::ostream &
operator<<(std::ostream &os,
           const IgASolver<Inputs, Outputs, CollPts> &obj) {
  obj.pretty_print(os);
  return os;
}  
  
} // namespace iganet
