/**
   @file webapps/models/gismo/GismoPoissonModel.hpp

   @brief G+Smo Poisson model

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include <GismoPdeModel.hpp>

namespace iganet {

namespace webapp {

/// @brief G+Smo Poisson model
template <short_t d, typename T>
class GismoPoissonModel : public GismoPdeModel<d, T>, public ModelEval {

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

  /// @brief Right-hand side and boundary values
  gsFunctionExpr<T> rhsFunc_, bdrFunc_;

  /// @brief Expression assembler
  gsExprAssembler<T> A_;

  /// @brief Geometry map
  geometryMap_type G_;

  /// @brief Discretization space
  space_type u_;

  /// @brief Right-hand side vector
  gismo::expr::gsComposition<T> f_;

  /// @brief Solution
  gsMultiPatch<T> sol_;

public:
  /// @brief Default constructor
  GismoPoissonModel() = delete;

  /// @brief Constructor for equidistant knot vectors
  GismoPoissonModel(const std::array<short_t, d> degrees,
                    const std::array<int64_t, d> ncoeffs,
                    const std::array<int64_t, d> npatches)
      : Base(degrees, ncoeffs, npatches), basis_(Base::geo_, true),
        rhsFunc_("2*pi^2*sin(pi*x)*sin(pi*y)", d),
        bdrFunc_("sin(pi*x) * sin(pi*y)", d), A_(1, 1),
        G_(A_.getMap(Base::geo_)), u_(A_.getSpace(basis_)),
        f_(A_.getCoeff(rhsFunc_, G_)) {
    // Specify assembler options
    gsOptionList Aopt;

    Aopt.addInt("DirichletStrategy",
                "Method for enforcement of Dirichlet BCs [11..14]", 11);
    Aopt.addInt("DirichletValues",
                "Method for computation of Dirichlet DoF values [100..103]",
                101);
    Aopt.addInt("InterfaceStrategy",
                "Method of treatment of patch interfaces [0..3]", 1);
    Aopt.addReal(
        "bdA", "Estimated nonzeros per column of the matrix: bdA*deg + bdB", 2);
    Aopt.addInt(
        "bdB", "Estimated nonzeros per column of the matrix: bdA*deg + bdB", 1);
    Aopt.addReal(
        "bdO",
        "Overhead of sparse mem. allocation: (1+bdO)(bdA*deg + bdB) [0..1]",
        0.333);
    Aopt.addReal("quA", "Number of quadrature points: quA*deg + quB", 1);
    Aopt.addInt("quB", "Number of quadrature points: quA*deg + quB", 1);
    Aopt.addInt("quRule", "Quadrature rule [1:GaussLegendre, 2:GaussLobatto]",
                1);

    // Set assembler options
    A_.setOptions(Aopt);
    A_.setIntegrationElements(basis_);

    // Set boundary conditions
    bc_.addCondition(gismo::boundary::west, gismo::condition_type::dirichlet,
                     &bdrFunc_);
    bc_.addCondition(gismo::boundary::east, gismo::condition_type::dirichlet,
                     &bdrFunc_);
    bc_.addCondition(gismo::boundary::north, gismo::condition_type::dirichlet,
                     &bdrFunc_);
    bc_.addCondition(gismo::boundary::south, gismo::condition_type::dirichlet,
                     &bdrFunc_);

    // Set geometry
    bc_.setGeoMap(Base::geo_);

    // Impose boundary conditions
    u_.setup(bc_, gismo::dirichlet::l2Projection, 0);

    // Generate solution
    solve();
  }

  /// @brief Destructor
  ~GismoPoissonModel() {}

  /// @brief Solve the Poisson problem
  void solve() {
    // Set up system
    A_.initSystem();
    A_.assemble(igrad(u_, G_) * igrad(u_, G_).tr() * meas(G_) // matrix
                ,
                u_ * f_ * meas(G_) // rhs vector
    );

    // Solve system
    typename gismo::gsSparseSolver<T>::CGDiagonal solver;
    solver.compute(A_.matrix());

    gsMatrix<T> solVector;
    solution_type sol = A_.getSolution(u_, solVector);
    solVector = solver.solve(A_.rhs());

    // Extract solution
    sol.extract(sol_);
  }

  /// @brief Returns the model's name
  std::string getName() const override {
    return "GismoPoisson" + std::to_string(d) + "d";
  }

  /// @brief Returns the model's description
  std::string getDescription() const override {
    return "G+Smo Poisson model in " + std::to_string(d) + " dimensions";
  };

  /// @brief Returns the model's outputs
  std::string getOutputs() const override {
    return "["
           "{\"name\" : \"Solution\","
           " \"description\" : \"Solution of the Poisson equation\","
           " \"type\" : 1}"
           "]";
  }

  /// @brief Updates the attributes of the model
  nlohmann::json updateAttribute(const std::string &component,
                                 const std::string &attribute,
                                 const nlohmann::json &json) override {

    auto result = Base::updateAttribute(component, attribute, json);
    solve();

    return result;
  }

  /// @brief Evaluates the model
  nlohmann::json eval(const std::string &component,
                      const nlohmann::json &json) const override {

    // Create uniform grid
    gsMatrix<T> ab = Base::geo_.patch(0).support();
    gsVector<T> a = ab.col(0);
    gsVector<T> b = ab.col(1);

    gsVector<unsigned> np(Base::geo_.parDim());
    np.setConstant(25);

    if (json.contains("data"))
      if (json["data"].contains("resolution")) {
        auto res = json["data"]["resolution"].get<std::array<int64_t, d>>();

        for (std::size_t i = 0; i < d; ++i)
          np(i) = res[i];
      }

    // Uniform parameters for evaluation
    gsMatrix<T> pts = gsPointGrid(a, b, np);
    gsMatrix<T> eval = sol_.patch(0).eval(pts);

    return utils::to_json(eval, true);
  }

  /// @brief Refines the model
  void refine(const nlohmann::json &json = NULL) override {

    Base::refine(json);

    int num = 1, dim = -1;

    if (json.contains("data")) {
      if (json["data"].contains("num"))
        num = json["data"]["num"].get<int>();

      if (json["data"].contains("dim"))
        dim = json["data"]["dim"].get<int>();
    }

    basis_.basis(0).uniformRefine(num, 1, dim);
  }
};

} // namespace webapp
} // namespace iganet
