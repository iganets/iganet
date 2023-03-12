/**
   @file plugins/BSpline.cxx

   @brief BSpline test pluging

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include <iganet.hpp>
#include <pluginmanager.hpp>

namespace iganet {

  /// @brief BSpline plugin
  class BSplinePlugin : public PluginEval<1,2>,
                        public UniformBSpline<double,1,1,1> {
  public:
    /// @brief Default constructor
    BSplinePlugin()
      : UniformBSpline<double,1,1,1>({5, 5}, iganet::init::zeros)
    {}
    
    BSplinePlugin(const std::array<int64_t, 2> ncoeffs,
                  enum iganet::init init = iganet::init::zeros)
      : UniformBSpline<double,1,1,1>(ncoeffs, init)
    {}

    /// @brief Destructor
    ~BSplinePlugin() {}

    /// @brief Returns the plugin's name
    std::string getName() const override {
      return "BSpline";
    }

    /// @brief Returns the plugin's description
    std::string getDescription() const override {
      return "B-Spline pluging";
    }

    /// @brief Returns the plugin's Options
    std::string getOptions() const override {
      return "{\"ncoeffs\" : \"int\"}";
    }
    
    /// @brief Serializes the plugin to JSON
    nlohmann::json to_json() const override {
      return UniformBSpline<double,1,1,1>::to_json();
    }

    /// @brief Evaluates the plugin
    BlockTensor<torch::Tensor, 1, 1> eval(const nlohmann::json& config = NULL) const override {
      iganet::TensorArray2 xi = {torch::linspace(0,1,100),
                                 torch::linspace(0,1,100)};
      return UniformBSpline<double,1,1,1>::eval(xi);
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
    if (config.contains("data")) {
      std::array<int64_t, 2> ncoeffs = {1,1};
      enum iganet::init init = iganet::init::zeros;
      
      if (config["data"].contains("ncoeffs"))
        ncoeffs = config["data"]["ncoeffs"].get<std::array<int64_t,2>>();

      if (config["data"].contains("init"))
        init = config["data"]["init"].get<enum iganet::init>();
      return std::make_shared<iganet::BSplinePlugin>(ncoeffs, init);
    }
    else
      return std::make_shared<iganet::BSplinePlugin>();    
  }
}
