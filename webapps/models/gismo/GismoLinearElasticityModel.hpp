/**
   @file webapps/models/gismo/GismoLinearElasticityModel.hpp

   @brief G+Smo Linear elasticity model

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include <GismoPdeModel.hpp>

namespace iganet {

namespace webapp {

/// @brief G+Smo Linear elasticity model
template <short_t d, typename T>
class GismoLinearElasticityModel : public GismoPdeModel<d, T>,
                                   public ModelEval {

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
  gsExprAssembler<T> assembler_;

  /// @brief Solution
  gsMultiPatch<T> solution_;

public:
  /// @brief Default constructor
  GismoLinearElasticityModel() = delete;

  /// @brief Constructor for equidistant knot vectors
  GismoLinearElasticityModel(const std::array<short_t, d> degrees,
                             const std::array<int64_t, d> ncoeffs,
                             const std::array<int64_t, d> npatches)
      : Base(degrees, ncoeffs, npatches), basis_(Base::geo_, true),
        rhsFunc_("2*pi^2*sin(pi*x)*sin(pi*y)", d),
        bdrFunc_("sin(pi*x) * sin(pi*y)", d), assembler_(1, 1) {
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
    assembler_.setOptions(Aopt);

    // Set assembler basis
    assembler_.setIntegrationElements(basis_);

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

    // Generate solution
    solve();
  }

  /// @brief Destructor
  ~GismoLinearElasticityModel() {}

  /// @brief Solve the Linear elasticity problem
  void solve() {

    // Set up expression assembler
    auto G = assembler_.getMap(Base::geo_);
    auto u = assembler_.getSpace(basis_);
    auto f = assembler_.getCoeff(rhsFunc_, G);

    // Impose boundary conditions
    u.setup(bc_, gismo::dirichlet::l2Projection, 0);

    // Set up system
    assembler_.initSystem();
    assembler_.assemble(igrad(u, G) * igrad(u, G).tr() * meas(G) // matrix
                        ,
                        u * f * meas(G) // rhs vector
    );

    // Solve system
    typename gismo::gsSparseSolver<T>::CGDiagonal solver;
    solver.compute(assembler_.matrix());

    gsMatrix<T> solutionVector;
    solution_type solution = assembler_.getSolution(u, solutionVector);
    solutionVector = solver.solve(assembler_.rhs());

    // Extract solution
    solution.extract(solution_);
  }

  /// @brief Returns the model's name
  std::string getName() const override {
    return "GismoLinearElasticity" + std::to_string(d) + "d";
  }

  /// @brief Returns the model's description
  std::string getDescription() const override {
    return "G+Smo linear elasticity model in " + std::to_string(d) +
           " dimensions";
  };

  /// @brief Returns the model's outputs
  nlohmann::json getOutputs() const override {
    return R"([{
           "name" : "Solution",
           "description" : "Solution of the linear elasticity equation",
           "type" : 1}])"_json;
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
    gsMatrix<T> eval = solution_.patch(0).eval(pts);

    return utils::to_json(eval, true);
  }

  /// @brief Refines the model
  void refine(const nlohmann::json &json = NULL) override {

    // Refine geometry
    Base::refine(json);

    // Set geometry
    bc_.setGeoMap(Base::geo_);

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
  }
};

} // namespace webapp
} // namespace iganet
