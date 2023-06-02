/**
   @file models/BSplineSurface.cxx

   @brief B-Spline surface

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include "BSpline.hpp"
#include <model.hpp>

#ifdef _WIN32
extern "C" __declspec(dllexport)
#else
extern "C"
#endif
{
  /// @brief Create a B-spline surface
  std::shared_ptr<iganet::Model> create(const nlohmann::json& json) {
    enum iganet::webapp::degree degree = iganet::webapp::degree::linear;
    enum iganet::init init = iganet::init::linear;
    std::array<int64_t, 2> ncoeffs = {4,4};
    bool nonuniform = false;

    if (json.contains("data")) {

      if (json["data"].contains("degree"))
        degree = json["data"]["degree"].get<enum iganet::webapp::degree>();

      if (json["data"].contains("init"))
        init = json["data"]["init"].get<enum iganet::init>();

      if (json["data"].contains("ncoeffs"))
        ncoeffs = json["data"]["ncoeffs"].get<std::array<int64_t,2>>();

      if (json["data"].contains("nonuniform"))
        nonuniform = json["data"]["nonuniform"].get<bool>();

      switch (degree) {
      case iganet::webapp::degree::constant:
        if (nonuniform)
          return std::make_shared<iganet::webapp::BSplineModel<iganet::NonUniformBSpline<iganet::real_t, 3, 0,0>>>(ncoeffs, init);
        else
          return std::make_shared<iganet::webapp::BSplineModel<iganet::   UniformBSpline<iganet::real_t, 3, 0,0>>>(ncoeffs, init);
      case iganet::webapp::degree::linear:
        if (nonuniform)
          return std::make_shared<iganet::webapp::BSplineModel<iganet::NonUniformBSpline<iganet::real_t, 3, 1,1>>>(ncoeffs, init);
        else
          return std::make_shared<iganet::webapp::BSplineModel<iganet::   UniformBSpline<iganet::real_t, 3, 1,1>>>(ncoeffs, init);
      case iganet::webapp::degree::quadratic:
        if (nonuniform)
          return std::make_shared<iganet::webapp::BSplineModel<iganet::NonUniformBSpline<iganet::real_t, 3, 2,2>>>(ncoeffs, init);
        else
          return std::make_shared<iganet::webapp::BSplineModel<iganet::   UniformBSpline<iganet::real_t, 3, 2,2>>>(ncoeffs, init);
      case iganet::webapp::degree::cubic:
        if (nonuniform)
          return std::make_shared<iganet::webapp::BSplineModel<iganet::NonUniformBSpline<iganet::real_t, 3, 3,3>>>(ncoeffs, init);
        else
          return std::make_shared<iganet::webapp::BSplineModel<iganet::   UniformBSpline<iganet::real_t, 3, 3,3>>>(ncoeffs, init);
      case iganet::webapp::degree::quartic:
        if (nonuniform)
          return std::make_shared<iganet::webapp::BSplineModel<iganet::NonUniformBSpline<iganet::real_t, 3, 4,4>>>(ncoeffs, init);
        else
          return std::make_shared<iganet::webapp::BSplineModel<iganet::   UniformBSpline<iganet::real_t, 3, 4,4>>>(ncoeffs, init);
      case iganet::webapp::degree::quintic:
        if (nonuniform)
          return std::make_shared<iganet::webapp::BSplineModel<iganet::NonUniformBSpline<iganet::real_t, 3, 5,5>>>(ncoeffs, init);
        else
          return std::make_shared<iganet::webapp::BSplineModel<iganet::   UniformBSpline<iganet::real_t, 3, 5,5>>>(ncoeffs, init);
      default:
        throw std::runtime_error("Invalid degree");
      }
    }
    else
      if (nonuniform)
        return std::make_shared<iganet::webapp::BSplineModel<iganet::NonUniformBSpline<iganet::real_t, 3, 1,1>>>(ncoeffs, init);
      else
        return std::make_shared<iganet::webapp::BSplineModel<iganet::   UniformBSpline<iganet::real_t, 3, 1,1>>>(ncoeffs, init);
  }

  /// @brief Load a B-spline surface
  std::shared_ptr<iganet::Model> load(const nlohmann::json& json) {

    if (json.contains("data")) {
      if (json["data"].contains("binary")) {

        // get binary vector from JSON object
        auto binary = json["data"]["binary"].get<std::vector<std::uint8_t>>();

        // recover input archive from binary vector
        torch::serialize::InputArchive archive;
        archive.load_from(reinterpret_cast<const char*>(binary.data()), binary.size());

        try {
          // get model hash
          c10::IValue model;
          archive.read("model", model);

          // check if model can be processed
          if (model == static_cast<int64_t>(std::hash<std::string>{}("BSplineSurface"))) {

            torch::Tensor tensor;
            archive.read("geometry.parDim", tensor); int64_t parDim = tensor.item<int64_t>();
            archive.read("geometry.geoDim", tensor); int64_t geoDim = tensor.item<int64_t>();

            if (parDim != 2)
              throw iganet::InvalidModelException();

            std::array<short_t, 2> degrees;
            for (short_t i = 0; i < parDim; ++i) {
              archive.read("geometry.degree[" + std::to_string(i) + "]", tensor);
              degrees[i] = tensor.item<int64_t>();
            }

            std::array<int64_t, 2> nknots;
            for (short_t i = 0; i < parDim; ++i) {
              archive.read("geometry.nknots[" + std::to_string(i) + "]", tensor);
              nknots[i] = tensor.item<int64_t>();
            }

            std::array<int64_t, 2> ncoeffs;
            for (short_t i = 0; i < parDim; ++i) {
              archive.read("geometry.ncoeffs[" + std::to_string(i) + "]", tensor);
              ncoeffs[i] = tensor.item<int64_t>();
            }

            
          }

          else
            throw iganet::InvalidModelException();

        } catch(...) {
          throw iganet::InvalidModelException();
        }
      }
    }

    throw iganet::InvalidModelException();
  }
}
