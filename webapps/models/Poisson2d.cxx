/**
   @file models/Poisson2d.cxx

   @brief Poisson equation in 2d model

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include <iganet.hpp>
#include <model.hpp>

namespace iganet {

  /// @brief Poisson equation in 2d model
  class Poisson2dModel : public Model,
                         public UniformBSpline<double,1,1,1> {
  public:
    /// @brief Default constructor
    Poisson2dModel() = default;
    
    /// @brief Destructor
    ~Poisson2dModel() {}

    /// @brief Returns the model's name
    std::string getName() const override {
      return "Poisson2d";
    }
    
    /// @brief Returns the model's description
    std::string getDescription() const override {
      return "Poisson 2d model";
    }

    /// @brief Returns the model's options
    std::string getOptions() const override {
      return "[]";
    }

    /// @brief Returns the model's inputs
    std::string getInputs() const override {
      return "[]";
    }

    /// @brief Returns the model's outputs
    std::string getOutputs() const override {
      return "[]";
    }
    
    /// @brief Serializes the model to JSON
    nlohmann::json to_json(const std::string& attribute = "") const override {
      return UniformBSpline<double,1,1,1>::to_json();
    }

    /// @brief Updates the attrbutes of the model
    nlohmann::json updateAttribute(const std::string& attribute,
                                   const nlohmann::json& json) override {
      return "\"reason\" : \"Not implemented yet\"";
    }

  };
} // namespace iganet
  
#ifdef _WIN32
extern "C" __declspec(dllexport)
#else
extern "C"
#endif
{
  std::shared_ptr<iganet::Model> create(const nlohmann::json& json) {
    return std::make_shared<iganet::Poisson2dModel>();
  }
}
