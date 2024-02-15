/**
   @file webapps/models/gismo/GismoPoissonModel.hpp

   @brief G+Smo Poisson model

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include "GismoPdeModel.hpp"

namespace iganet {

  namespace webapp {


  /// @brief G+Smo Poisson model
  template <short_t d, typename T>
  class GismoPoissonModel : public GismoPdeModel<d, T> {

  protected:
      
  public:
    /// @brief Default constructor
    GismoPoissonModel() = default;

    /// @brief Constructor for equidistant knot vectors
    GismoPoissonModel(const std::array<short_t, d> degrees,
                      const std::array<int64_t, d> ncoeffs,
                      const std::array<int64_t, d> npatches)
      : GismoPdeModel<d, T>(degrees, ncoeffs, npatches)
    {}
      
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
