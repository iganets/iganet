/**
   @file solver/ezinterp.hpp

   @brief Isogeometric analysis solver

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#pragma once

#include <net/igabase.hpp>
#include <solver/igasolver.hpp>
#include <splines/bspline.hpp>
#include <utils/matrix.hpp>

namespace iganet {

/// @brief Easy-to-use solver base class
///
/// This abstract class implements an easy-to-use solver that takes a
/// geometry map and a variable function space as inputs. Note that
/// this is an abstract class. A derived class needs to implement the
/// member functions assembleLhs and assembleRhs.
template <FunctionSpaceType GeometryMap, FunctionSpaceType Variable>
class EZSolverBase : public iganet::IgASolver<std::tuple<GeometryMap>, std::tuple<Variable>>,
                     public iganet::IgANetCustomizable2<std::tuple<GeometryMap>, std::tuple<Variable>> {
  
protected:
  /// @brief Type of the base class
  using Base = iganet::IgASolver<std::tuple<GeometryMap>, std::tuple<Variable>>;
  
  /// @brief Collocation points
  Base::template collPts_t<0> collPts_;

  /// @brief Type of the customizable class
  using Customizable = iganet::IgANetCustomizable2<std::tuple<GeometryMap>, std::tuple<Variable>>;

  /// @brief Knot indices of the geometry map
  Customizable::template input_interior_knot_indices_t<0> G_knot_indices_;

  /// @brief Knot indices of variables
  Customizable::template output_interior_knot_indices_t<0> var_knot_indices_;

public:
  /// @brief Constructor
  template <std::size_t GeometryMapNumCoeffs, std::size_t VariableNumCoeffs>
  EZSolverBase(const std::array<int64_t, GeometryMapNumCoeffs> &geometryMapNumCoeffs,
               const std::array<int64_t, VariableNumCoeffs> &variableNumCoeffs)
    : Base(std::make_tuple(geometryMapNumCoeffs),
           std::make_tuple(variableNumCoeffs)) {}
  
  /// @brief Returns a constant reference to the collocation points
  auto const &collPts() const { return collPts_; }
  
  
  /// @brief Returns a constant reference to the geometry
  auto const &G() const { return Base::template input<0>(); }
  
  /// @brief Returns a non-constant reference to the geometry
  auto &G() { return Base::template input<0>(); }
  
  /// @brief Returns a constant reference to the variable
  auto const &u() const { return Base::template output<0>(); }
  
  /// @brief Returns a non-constant reference to the variable
  auto &u() { return Base::template output<0>(); }
  
  /// @brief Initializes the solver
  void init() override {
    collPts_ =
      Base::template collPts<0>(iganet::collPts::greville);
    G_knot_indices_ =
      G().template find_knot_indices<iganet::functionspace::interior>(collPts_.first);    
    var_knot_indices_ =
      u().template find_knot_indices<iganet::functionspace::interior>(collPts_.first);    
  }
};

/// @brief Easy-to-use solver class
///
/// This class implements an easy-to-use solver that takes a
/// geometry map and a variable function space as inputs.
template <FunctionSpaceType GeometryMap, FunctionSpaceType Variable>  
class EZSolver : public EZSolverBase<GeometryMap, Variable> {
private:
  /// @brief Base class
  using Base = EZSolverBase<GeometryMap, Variable>;
  
  /// @brief Left-hand side function
  std::function<
    std::array<torch::Tensor, Variable::template geoDim<0>()>(
                                                              const std::array<torch::Tensor, Variable::template parDim<0>()>&
                                                              )
    > lhs_;

  /// @brief Right-hand side function
  std::function<
    std::array<torch::Tensor, Variable::template geoDim<0>()>(
                                                              const std::array<torch::Tensor, Variable::template parDim<0>()>&
                                                              )
    > rhs_;

public:
  /// @brief Constructor
  EZSolver(const GeometryMap& geometryMap, const Variable& variable,
           const std::function<
           std::array<torch::Tensor, Variable::template geoDim<0>()>(
                                                                     const std::array<torch::Tensor, Variable::template parDim<0>()>&
                                                                     )
           >& lhs,
           const std::function<
           std::array<torch::Tensor, Variable::template geoDim<0>()>(
                                                                     const std::array<torch::Tensor, Variable::template parDim<0>()>&
                                                                     )
           >& rhs)
  : EZSolverBase<GeometryMap, Variable>(geometryMap.template space<0>().ncoeffs(), variable.template space<0>().ncoeffs()),
    lhs_(lhs), rhs_(rhs) {}
  
  /// @brief Assembles the left-hand side as the mass matrix
  void assembleLhs() override {    
    Base::lhs_ = iganet::utils::to_sparseCsrTensor(Base::var_knot_indices_,
                                                   this->u().template space<0>().degrees(),
                                                   this->u().template space<0>().ncoeffs(),
                                                   this->u().template eval_basfunc(Base::collPts_.first, Base::var_knot_indices_).t(),
                                                   { this->u().template space<0>().ncumcoeffs(),
                                                      this->u().template space<0>().ncumcoeffs() });
  }
  
  /// @brief Assembles the right-hand side from the given mapping
  void assembleRhs() override {
    Base::rhs_ = rhs_(Base::collPts().first)[0];     
  }
};
  
/// @brief Easy-to-use interpolation class
///
/// This class implements an easy-to-use interpolation that takes a
/// geometry map and a variable function space as inputs.
template <FunctionSpaceType GeometryMap, FunctionSpaceType Variable>  
class EZInterpolation : public EZSolverBase<GeometryMap, Variable> {
private:
  /// @brief Base class
  using Base = EZSolverBase<GeometryMap, Variable>;
  
  /// @brief Mapping
  std::function<
    std::array<torch::Tensor, Variable::template geoDim<0>()>(
                                                              const std::array<torch::Tensor, Variable::template parDim<0>()>&
                                                              )
    > mapping_;
  
public:
  /// @brief Constructor
  EZInterpolation(const GeometryMap& geometryMap, const Variable& variable,
                  const std::function<
                  std::array<torch::Tensor, Variable::template geoDim<0>()>(
                                                                            const std::array<torch::Tensor, Variable::template parDim<0>()>&
                                                                            )
                  >& mapping)
    : EZSolverBase<GeometryMap, Variable>(geometryMap.template space<0>().ncoeffs(), variable.template space<0>().ncoeffs()), mapping_(mapping) {}
  
  /// @brief Assembles the left-hand side as the mass matrix
  void assembleLhs() override {    
    Base::lhs_ = iganet::utils::to_sparseCsrTensor(Base::var_knot_indices_,
                                                   this->u().template space<0>().degrees(),
                                                   this->u().template space<0>().ncoeffs(),
                                                   this->u().template eval_basfunc(Base::collPts_.first, Base::var_knot_indices_).t(),
                                                   { this->u().template space<0>().ncumcoeffs(),
                                                      this->u().template space<0>().ncumcoeffs() });
  }
  
  /// @brief Assembles the right-hand side from the given mapping
  void assembleRhs() override {
    Base::rhs_ = mapping_(Base::collPts().first)[0];     
  }
};
  
/// @brief Easy-to-use interpolation function
///
/// This function interpolates the given mapping in the Greville
/// points of the variable function space
template <FunctionSpaceType GeometryMap, FunctionSpaceType Variable>
auto ezinterp(const GeometryMap& geometryMap,
              const Variable& variable,
              const std::function<std::array<torch::Tensor, Variable::template geoDim<0>()>(const std::array<torch::Tensor, Variable::template parDim<0>()> &)>
              mapping) {

  EZInterpolation interp(geometryMap, variable, mapping);
  interp.init();
  interp.assemble();
  return interp.solve().clone();
}
  
} // namespace iganet
