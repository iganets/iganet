/**
   @file plugins/Poisson2d.cxx

   @brief Poisson equation in 2d pluging

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include <iganet.hpp>
#include <pluginmanager.hpp>

namespace iganet {

  /// @brief Poisson equation in 2d plugin
  class Poisson2dPlugin : public Plugin,
                          public UniformBSpline<double,1,1,1> {
  public:
    /// @brief Default constructor
    Poisson2dPlugin() = default;
    
    /// @brief Destructor
    ~Poisson2dPlugin() {}

    /// @brief Returns the plugin's name
    std::string getName() const override {
      return "Poisson2d";
    }
    
    /// @brief Returns the plugin's description
    std::string getDescription() const override {
      return "Poisson 2d pluging";
    }

    /// @brief Returns the plugin's Options
    std::string getOptions() const override {
      return "{\"ncoeffs\" : \"int\"}";
    }
    
    /// @brief Serializes the plugin to JSON
    nlohmann::json to_json() const override {
      return UniformBSpline<double,1,1,1>::to_json();
    }

  };
} // namespace iganet
  
#ifdef _WIN32
extern "C" __declspec(dllexport)
#else
extern "C"
#endif
{
  std::shared_ptr<iganet::Plugin> create(const nlohmann::json& config) {
    return std::make_shared<iganet::Poisson2dPlugin>();
  }
}
