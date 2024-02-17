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
  class GismoPoissonModel : public GismoPdeModel<d, T> {

  protected:
    /// @brief Option list
    gsOptionList Aopt_;
    
    /// @brief Multi-patch basis
    gsMultiBasis<T> basis_;

    /// @brief Expression assembler
    gsExprAssembler<T> A_;

    /// @brief Expression evaluator
    gsExprEvaluator<T> ev_;

    /// @brief Type of the geometry mapping
    using geometryMap = typename gsExprAssembler<T>::geometryMap;

    /// @brief Type of the variable
    using variable = typename gsExprAssembler<T>::variable;

    /// @brief Type of the function space
    using space = typename gsExprAssembler<T>::space;

    /// @brief Type of the solution
    using solution = typename gsExprAssembler<T>::solution;

    /// @brief Geoemtry map
    geometryMap G_;

    /// @brief Discretization space
    space u_;
    
  public:
    /// @brief Default constructor
    GismoPoissonModel() = delete;

    /// @brief Constructor for equidistant knot vectors
    GismoPoissonModel(const std::array<short_t, d> degrees,
                      const std::array<int64_t, d> ncoeffs,
                      const std::array<int64_t, d> npatches)
      : GismoPdeModel<d, T>(degrees, ncoeffs, npatches),
        basis_(GismoPdeModel<d, T>::geo_, true),
        A_(1,1),
        ev_(A_),
        G_(A_.getMap(GismoPdeModel<d, T>::geo_)),
        u_(A_.getSpace(basis_))
    {
      A_.setOptions(Aopt_);
      A_.setIntegrationElements(basis_);
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
    std::string getOutputs() const override {
      return "["
        "{\"name\" : \"Solution\","
        " \"description\" : \"Solution of the Poisson equation\","
        " \"type\" : 1}"
        "]";
    }

  };

  } // namespace webapp
} // namespace iganet
