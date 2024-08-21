/**
   @file webapps/models/gismo/GismoPoissonModel.hpp

   @brief G+Smo Poisson model

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#pragma once

#include <GismoPdeModel.hpp>

using namespace std::string_literals;

namespace iganet {

namespace webapp {

/// @brief G+Smo Poisson model
template <short_t d, typename T>
class GismoPoissonModel : public GismoPdeModel<d, T> {

  static_assert(d >= 1 && d <= 3, "Spatial dimension must be between 1 and 3");

private:
  /// @brief Base class
  using Base = GismoPdeModel<d, T>;

  /// @brief Type of the geometry mapping
  using geometryMap_type = typename gsExprAssembler<T>::geometryMap;

  /// @brief Type of the variable
  using variable_type = typename gsExprAssembler<T>::variable;

  /// @brief Type of the function space
  using space_type = typename gsExprAssembler<T>::space;

  /// @brief Type of the solution
  using solution_type = typename gsExprAssembler<T>::solution;

  /// @brief Multi-patch basis
  gsMultiBasis<T> basis_;

  /// @brief Boundary conditions
  gsBoundaryConditions<T> bc_;

  /// @brief Right-hand side function
  gsFunctionExpr<T> rhsFunc_;

  /// @brief Right-hand side function defined on parametric domain (default
  /// false)
  bool rhsFuncParametric_;

  /// @brief Boundary condition values
  std::array<gsFunctionExpr<T>, 2 * d> bcFunc_;

  /// @brief Boundary values defined on parametric domain (default false)
  std::array<bool, 2 * d> bcFuncParametric_;

  /// @brief Boundary condition type
  std::array<gismo::condition_type::type, 2 * d> bcType_;

  /// @brief Expression assembler
  gsExprAssembler<T> assembler_;

  /// @brief Solution
  gsMultiPatch<T> solution_;

  /// @brief Solve the Poisson problem
  void solve() {

    // Set up expression assembler
    auto G = assembler_.getMap(Base::geo_);
    auto u = assembler_.getSpace(basis_);

    // Impose boundary conditions
    u.setup(bc_, gismo::dirichlet::l2Projection, 0);

    // Set up system
    assembler_.initSystem();
    if (rhsFuncParametric_) {
      auto f = assembler_.getCoeff(rhsFunc_);
      assembler_.assemble(igrad(u, G) * igrad(u, G).tr() * meas(G) // matrix
                          ,
                          u * f * meas(G) // rhs vector
      );
    } else {
      auto f = assembler_.getCoeff(rhsFunc_, G);
      assembler_.assemble(igrad(u, G) * igrad(u, G).tr() * meas(G) // matrix
                          ,
                          u * f * meas(G) // rhs vector
      );
    }

    // Compute the Neumann terms defined on physical space
    auto bcNeumann = bc_.get("Neumann");
    if (!bcNeumann.empty()) {
      auto g = assembler_.getBdrFunction(G);
      assembler_.assembleBdr(bcNeumann, u * g * meas(G));
    }

    // Compute the Neumann terms defined on parametric space
    auto bcNeumannParametric = bc_.get("NeumannParametric");
    if (!bcNeumannParametric.empty()) {
      auto g = assembler_.getBdrFunction();
      assembler_.assembleBdr(bcNeumannParametric, u * g * meas(G));
    }

    // Solve system
    typename gismo::gsSparseSolver<T>::CGDiagonal solver;
    solver.compute(assembler_.matrix());

    gsMatrix<T> solutionVector;
    solution_type solution = assembler_.getSolution(u, solutionVector);
    solutionVector = solver.solve(assembler_.rhs());

    // Extract solution
    solution.extract(solution_);
  }

public:
  /// @brief Default constructor
  GismoPoissonModel() = delete;

  /// @brief Constructor for equidistant knot vectors
  GismoPoissonModel(const std::array<short_t, d> degrees,
                    const std::array<int64_t, d> ncoeffs,
                    const std::array<int64_t, d> npatches,
                    const std::array<T, d> dimensions)
      : Base(degrees, ncoeffs, npatches, dimensions), basis_(Base::geo_, true),
        rhsFuncParametric_(true),
        rhsFunc_(d == 1   ? "2*pi^2*sin(pi*x)"
                 : d == 2 ? "2*pi^2*sin(pi*x)*sin(pi*y)"
                          : "2*pi^2*sin(pi*x)*sin(pi*y)*sin(pi*z)",
                 /* rhsFuncParametric_ == false */ d),
        assembler_(1, 1) {

    // Specify assembler options
    gsOptionList Aopt = gsExprAssembler<>::defaultOptions();

    // Set assembler options
    assembler_.setOptions(Aopt);

    // Set assembler basis
    assembler_.setIntegrationElements(basis_);

    // Set all boundary conditions to parametric
    bcFuncParametric_.fill(true);

    // Set boundary conditions type and expression
    for (const auto &side : GismoBoundarySides<d>) {
      bcType_[side - 1] = gismo::condition_type::dirichlet;
      bcFunc_[side - 1] = gismo::give(
          gsFunctionExpr<T>("0", bcFuncParametric_[side - 1] ? d : 3));
      bc_.addCondition(side, bcType_[side - 1], &bcFunc_[side - 1], 0,
                       bcFuncParametric_[side - 1]);
    }

    // Set geometry
    bc_.setGeoMap(Base::geo_);

    // Generate solution
    solve();
  }

  /// @brief Destructor
  ~GismoPoissonModel() {}

  /// @brief Returns the model's name
  std::string getName() const override {
    return "GismoPoisson" + std::to_string(d) + "d";
  }

  /// @brief Returns the model's description
  std::string getDescription() const override {
    return "G+Smo Poisson model in " + std::to_string(d) + " dimensions";
  };

  /// @brief Returns the model's outputs
  nlohmann::json getOutputs() const override {
    auto json = R"([{
           "name" : "Solution",
           "description" : "Solution of the Poisson equation",
           "type" : 1},{
           "name" : "Rhs",
           "description" : "Right-hand side function",
           "type" : 1}])"_json;

    for (auto const &output : Base::getOutputs())
      json.push_back(output);

    return json;
  }

  /// @brief Returns the model's parameters
  nlohmann::json getParameters() const override {

    auto json = nlohmann::json::array();
    int uiid = 0;

    // Lambda expression to add a JSON entry
    auto add_json = [&json, &uiid]<typename Type, typename Value>(
                        const std::string &name, const std::string &description,
                        const Type &type, const Value &value) {
      nlohmann::json item;
      item["name"] = name;
      item["description"] = description;
      item["type"] = type;
      item["value"] = value;
      item["default"] = value;
      item["uuid"] = uiid++;
      json.push_back(item);
    };

    // Lambda expression to add a JSON entry with different default type
    auto add_json_default =
        [&json, &uiid]<typename Type, typename Value, typename DefaultValue>(
            const std::string &name, const std::string &description,
            const Type &type, const Value &value,
            const DefaultValue &defaultValue) {
          nlohmann::json item;
          item["name"] = name;
          item["description"] = description;
          item["type"] = type;
          item["value"] = value;
          item["default"] = defaultValue;
          item["uuid"] = uiid++;
          json.push_back(item);
        };

    add_json("rhs", "Right-hand side function", "text", rhsFunc_.expression(0));
    add_json("rhs_parametric",
             "Right-hand side function defined in parametric domain", "bool",
             rhsFuncParametric_);

    for (const auto &[side, str] :
         utils::zip(GismoBoundarySides<d>, GismoBoundarySideStrings<d>)) {
      add_json("bc["s + str + "]"s,
               "Boundary value at the "s + str + " boundary"s, "text",
               bcFunc_[side - 1].expression(0));
      add_json("bc_parametric["s + str + "]",
               "Boundary value at the "s + str +
                   " boundary defined in parametric domain"s,
               "bool", bcFuncParametric_[side - 1]);
      add_json_default(
          "bc_type["s + str + "]",
          "Type of boundary condition at the "s + str + " boundary"s, "select",
          R"([ "Dirichlet", "Neumann" ])"_json, "Dirichlet");
    }

    return json;
  }

  /// @brief Updates the attributes of the model
  nlohmann::json updateAttribute(const std::string &component,
                                 const std::string &attribute,
                                 const nlohmann::json &json) override {

    bool updateBC(false);
    nlohmann::json result = R"({})"_json;

    for (const auto &[side, str] :
         utils::zip(GismoBoundarySides<d>, GismoBoundarySideStrings<d>)) {

      // bc_parametric[*]
      if (attribute == "bc_parametric["s + str + "]"s) {
        if (!json.contains("data"))
          throw InvalidModelAttributeException();
        if (!json["data"].contains("bc_parametric["s + str + "]"s))
          throw InvalidModelAttributeException();

        bcFuncParametric_[side - 1] =
            json["data"]["bc_parametric["s + str + "]"s].template get<bool>();

        bcFunc_[side - 1] =
            gismo::give(gsFunctionExpr<T>(bcFunc_[side - 1].expression(0),
                                          bcFuncParametric_[side - 1] ? d : 3));

        updateBC = true;
        break;
      }

      // bc_type[*]
      if (attribute == "bc_type["s + str + "]"s) {
        if (!json.contains("data"))
          throw InvalidModelAttributeException();
        if (!json["data"].contains("bc_type["s + str + "]"s))
          throw InvalidModelAttributeException();

        std::string bc_type =
            json["data"]["bc_type["s + str + "]"s].template get<std::string>();

        if (bc_type == "Dirichlet")
          bcType_[side - 1] = gismo::condition_type::type::dirichlet;
        else if (bc_type == "Neumann")
          bcType_[side - 1] = gismo::condition_type::type::neumann;
        else
          throw InvalidModelAttributeException();

        updateBC = true;
        break;
      }

      // bc[*]
      if (attribute == "bc["s + str + "]"s) {
        if (!json.contains("data"))
          throw InvalidModelAttributeException();
        if (!json["data"].contains("bc["s + str + "]"s))
          throw InvalidModelAttributeException();

        bcFunc_[side - 1] = gismo::give(gsFunctionExpr<T>(
            json["data"]["bc["s + str + "]"s].template get<std::string>(),
            bcFuncParametric_[side - 1] ? d : 3));

        updateBC = true;
        break;
      }
    }

    // rhs_parametric
    if (attribute == "rhs_parametric") {
      if (!json.contains("data"))
        throw InvalidModelAttributeException();
      if (!json["data"].contains("rhs_parametric"))
        throw InvalidModelAttributeException();
      rhsFuncParametric_ = json["data"]["rhs_parametric"].get<bool>();
      rhsFunc_ = gismo::give(gsFunctionExpr<T>(rhsFunc_.expression(0),
                                               rhsFuncParametric_ ? d : 3));
    }

    // rhs
    else if (attribute == "rhs") {
      if (!json.contains("data"))
        throw InvalidModelAttributeException();
      if (!json["data"].contains("rhs"))
        throw InvalidModelAttributeException();
      rhsFunc_ = gismo::give(gsFunctionExpr<T>(
          json["data"]["rhs"].get<std::string>(), rhsFuncParametric_ ? d : 3));
    }

    else if (!updateBC)
      result = Base::updateAttribute(component, attribute, json);

    if (updateBC) {
      bc_.clear();

      // Set boundary conditions
      for (const auto &side : GismoBoundarySides<d>) {
        switch (bcType_[side - 1]) {
        case gismo::condition_type::type::dirichlet:
          bc_.add(0, side, "Dirichlet", &bcFunc_[side - 1], 0, -1,
                  bcFuncParametric_[side - 1]);
          break;
        case gismo::condition_type::type::neumann:
          bc_.add(0, side,
                  bcFuncParametric_[side - 1] ? "NeumannParametric" : "Neumann",
                  &bcFunc_[side - 1], 0, -1, bcFuncParametric_[side - 1]);
          break;
        default:
          break;
        }
      }
    }

    // Solve updated problem
    solve();

    return result;
  }

  /// @brief Evaluates the model
  nlohmann::json eval(const std::string &component,
                      const nlohmann::json &json) const override {

    if (component == "Solution" || component == "Rhs") {

      // Get grid resolution
      gsVector<unsigned> npts(Base::geo_.parDim());
      npts.setConstant(25);

      if (json.contains("data"))
        if (json["data"].contains("resolution")) {
          auto res = json["data"]["resolution"].get<std::array<int64_t, d>>();

          for (std::size_t i = 0; i < d; ++i)
            npts(i) = res[i];
        }

      if (component == "Solution" ||
          component == "Rhs" && !rhsFuncParametric_) {

        // Create uniform grid in physical domain
        gsMatrix<T> ab = Base::geo_.patch(0).support();
        gsVector<T> a = ab.col(0);
        gsVector<T> b = ab.col(1);
        gsMatrix<T> pts = gsPointGrid(a, b, npts);

        if (component == "Solution") {
          gsMatrix<T> eval = solution_.patch(0).eval(pts);
          return utils::to_json(eval, true, false);
        } else {
          gsMatrix<T> eval = rhsFunc_.eval(Base::geo_.patch(0).eval(pts));
          return utils::to_json(eval, true, false);
        }

      } else {

        // Create uniform grid in parametric domain
        gsMatrix<T> ab = Base::geo_.patch(0).parameterRange();
        gsVector<T> a = ab.col(0);
        gsVector<T> b = ab.col(1);
        gsMatrix<T> pts = gsPointGrid(a, b, npts);

        gsMatrix<T> eval = rhsFunc_.eval(pts);
        return utils::to_json(eval, true, false);
      }
    }

    else
      return Base::eval(component, json);
  }

  /// @brief Elevates the model's degrees, preserves smoothness
  void elevate(const nlohmann::json &json = NULL) override {

    bool geometry = true;

    if (json.contains("data"))
      if (json["data"].contains("num"))
        geometry = json["data"]["geometry"].get<bool>();

    if (geometry) {
      // Elevate geometry
      Base::elevate(json);

      // Set geometry
      bc_.setGeoMap(Base::geo_);
    }

    int num = 1, dim = -1;

    if (json.contains("data")) {
      if (json["data"].contains("num"))
        num = json["data"]["num"].get<int>();

      if (json["data"].contains("dim"))
        dim = json["data"]["dim"].get<int>();
    }

    // Degree elevate basis of solution space
    basis_.basis(0).degreeElevate(num, dim);

    // Set assembler basis
    assembler_.setIntegrationElements(basis_);

    // Generate solution
    solve();
  }

  /// @brief Increases the model's degrees, preserves multiplicity
  void increase(const nlohmann::json &json = NULL) override {

    bool geometry = true;

    if (json.contains("data"))
      if (json["data"].contains("num"))
        geometry = json["data"]["geometry"].get<bool>();

    if (geometry) {
      // Increase geometry
      Base::increase(json);

      // Set geometry
      bc_.setGeoMap(Base::geo_);
    }

    int num = 1, dim = -1;

    if (json.contains("data")) {
      if (json["data"].contains("num"))
        num = json["data"]["num"].get<int>();

      if (json["data"].contains("dim"))
        dim = json["data"]["dim"].get<int>();
    }

    // Degree increase basis of solution space
    basis_.basis(0).degreeIncrease(num, dim);

    // Set assembler basis
    assembler_.setIntegrationElements(basis_);

    // Generate solution
    solve();
  }

  /// @brief Refines the model
  void refine(const nlohmann::json &json = NULL) override {

    bool geometry = true;

    if (json.contains("data"))
      if (json["data"].contains("num"))
        geometry = json["data"]["geometry"].get<bool>();

    if (geometry) {
      // Refine geometry
      Base::refine(json);

      // Set geometry
      bc_.setGeoMap(Base::geo_);
    }

    int num = 1, dim = -1;

    if (json.contains("data")) {
      if (json["data"].contains("num"))
        num = json["data"]["num"].get<int>();

      if (json["data"].contains("dim"))
        dim = json["data"]["dim"].get<int>();
    }

    // Refine basis of solution space
    basis_.basis(0).uniformRefine(num, 1, dim);

    // Set assembler basis
    assembler_.setIntegrationElements(basis_);

    // Generate solution
    solve();
  }
};

} // namespace webapp
} // namespace iganet
