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
          if (model.toInt() ==
              static_cast<int64_t>(std::hash<std::string>{}("BSplineSurface"))) {

            torch::Tensor tensor;

            // get parametric and geometric dimensions
            archive.read("geometry.parDim", tensor); iganet::short_t parDim = tensor.item<int64_t>();
            archive.read("geometry.geoDim", tensor); iganet::short_t geoDim = tensor.item<int64_t>();

            if (parDim != 2)
              throw iganet::InvalidModelException();

            // get degrees
            std::array<iganet::short_t, 2> degrees;
            for (iganet::short_t i = 0; i < parDim; ++i) {
              archive.read("geometry.degree[" + std::to_string(i) + "]", tensor);
              degrees[i] = tensor.item<int64_t>();
            }

            bool nonuniform = false;
            std::shared_ptr<iganet::Model> m;

            if (nonuniform) {

              // Non-uniform B-splines
              switch (static_cast<enum iganet::webapp::degree>(degrees[0])) {
              case iganet::webapp::degree::constant:
                switch (static_cast<enum iganet::webapp::degree>(degrees[1])) {
                case iganet::webapp::degree::constant:
                  m = std::make_shared<iganet::webapp::BSplineModel<iganet::NonUniformBSpline<iganet::real_t, 3, 0, 0>>>(); break;
                case iganet::webapp::degree::linear:
                  m = std::make_shared<iganet::webapp::BSplineModel<iganet::NonUniformBSpline<iganet::real_t, 3, 0, 1>>>(); break;
                case iganet::webapp::degree::quadratic:
                  m = std::make_shared<iganet::webapp::BSplineModel<iganet::NonUniformBSpline<iganet::real_t, 3, 0, 2>>>(); break;
                case iganet::webapp::degree::cubic:
                  m = std::make_shared<iganet::webapp::BSplineModel<iganet::NonUniformBSpline<iganet::real_t, 3, 0, 3>>>(); break;
                case iganet::webapp::degree::quartic:
                  m = std::make_shared<iganet::webapp::BSplineModel<iganet::NonUniformBSpline<iganet::real_t, 3, 0, 4>>>(); break;
                case iganet::webapp::degree::quintic:
                  m = std::make_shared<iganet::webapp::BSplineModel<iganet::NonUniformBSpline<iganet::real_t, 3, 0, 5>>>(); break;
                default:
                  throw std::runtime_error("Invalid degree");
                }
                break;
              case iganet::webapp::degree::linear:
                switch (static_cast<enum iganet::webapp::degree>(degrees[1])) {
                case iganet::webapp::degree::constant:
                  m = std::make_shared<iganet::webapp::BSplineModel<iganet::NonUniformBSpline<iganet::real_t, 3, 1, 0>>>(); break;
                case iganet::webapp::degree::linear:
                  m = std::make_shared<iganet::webapp::BSplineModel<iganet::NonUniformBSpline<iganet::real_t, 3, 1, 1>>>(); break;
                case iganet::webapp::degree::quadratic:
                  m = std::make_shared<iganet::webapp::BSplineModel<iganet::NonUniformBSpline<iganet::real_t, 3, 1, 2>>>(); break;
                case iganet::webapp::degree::cubic:
                  m = std::make_shared<iganet::webapp::BSplineModel<iganet::NonUniformBSpline<iganet::real_t, 3, 1, 3>>>(); break;
                case iganet::webapp::degree::quartic:
                  m = std::make_shared<iganet::webapp::BSplineModel<iganet::NonUniformBSpline<iganet::real_t, 3, 1, 4>>>(); break;
                case iganet::webapp::degree::quintic:
                  m = std::make_shared<iganet::webapp::BSplineModel<iganet::NonUniformBSpline<iganet::real_t, 3, 1, 5>>>(); break;
                default:
                  throw std::runtime_error("Invalid degree");
                }
                break;
              case iganet::webapp::degree::quadratic:
                switch (static_cast<enum iganet::webapp::degree>(degrees[1])) {
                case iganet::webapp::degree::constant:
                  m = std::make_shared<iganet::webapp::BSplineModel<iganet::NonUniformBSpline<iganet::real_t, 3, 2, 0>>>(); break;
                case iganet::webapp::degree::linear:
                  m = std::make_shared<iganet::webapp::BSplineModel<iganet::NonUniformBSpline<iganet::real_t, 3, 2, 1>>>(); break;
                case iganet::webapp::degree::quadratic:
                  m = std::make_shared<iganet::webapp::BSplineModel<iganet::NonUniformBSpline<iganet::real_t, 3, 2, 2>>>(); break;
                case iganet::webapp::degree::cubic:
                  m = std::make_shared<iganet::webapp::BSplineModel<iganet::NonUniformBSpline<iganet::real_t, 3, 2, 3>>>(); break;
                case iganet::webapp::degree::quartic:
                  m = std::make_shared<iganet::webapp::BSplineModel<iganet::NonUniformBSpline<iganet::real_t, 3, 2, 4>>>(); break;
                case iganet::webapp::degree::quintic:
                  m = std::make_shared<iganet::webapp::BSplineModel<iganet::NonUniformBSpline<iganet::real_t, 3, 2, 5>>>(); break;
                default:
                  throw std::runtime_error("Invalid degree");
                }
                break;
              case iganet::webapp::degree::cubic:
                switch (static_cast<enum iganet::webapp::degree>(degrees[1])) {
                case iganet::webapp::degree::constant:
                  m = std::make_shared<iganet::webapp::BSplineModel<iganet::NonUniformBSpline<iganet::real_t, 3, 3, 0>>>(); break;
                case iganet::webapp::degree::linear:
                  m = std::make_shared<iganet::webapp::BSplineModel<iganet::NonUniformBSpline<iganet::real_t, 3, 3, 1>>>(); break;
                case iganet::webapp::degree::quadratic:
                  m = std::make_shared<iganet::webapp::BSplineModel<iganet::NonUniformBSpline<iganet::real_t, 3, 3, 2>>>(); break;
                case iganet::webapp::degree::cubic:
                  m = std::make_shared<iganet::webapp::BSplineModel<iganet::NonUniformBSpline<iganet::real_t, 3, 3, 3>>>(); break;
                case iganet::webapp::degree::quartic:
                  m = std::make_shared<iganet::webapp::BSplineModel<iganet::NonUniformBSpline<iganet::real_t, 3, 3, 4>>>(); break;
                case iganet::webapp::degree::quintic:
                  m = std::make_shared<iganet::webapp::BSplineModel<iganet::NonUniformBSpline<iganet::real_t, 3, 3, 5>>>(); break;
                default:
                  throw std::runtime_error("Invalid degree");
                }
                break;
              case iganet::webapp::degree::quartic:
                switch (static_cast<enum iganet::webapp::degree>(degrees[1])) {
                case iganet::webapp::degree::constant:
                  m = std::make_shared<iganet::webapp::BSplineModel<iganet::NonUniformBSpline<iganet::real_t, 3, 4, 0>>>(); break;
                case iganet::webapp::degree::linear:
                  m = std::make_shared<iganet::webapp::BSplineModel<iganet::NonUniformBSpline<iganet::real_t, 3, 4, 1>>>(); break;
                case iganet::webapp::degree::quadratic:
                  m = std::make_shared<iganet::webapp::BSplineModel<iganet::NonUniformBSpline<iganet::real_t, 3, 4, 2>>>(); break;
                case iganet::webapp::degree::cubic:
                  m = std::make_shared<iganet::webapp::BSplineModel<iganet::NonUniformBSpline<iganet::real_t, 3, 4, 3>>>(); break;
                case iganet::webapp::degree::quartic:
                  m = std::make_shared<iganet::webapp::BSplineModel<iganet::NonUniformBSpline<iganet::real_t, 3, 4, 4>>>(); break;
                case iganet::webapp::degree::quintic:
                  m = std::make_shared<iganet::webapp::BSplineModel<iganet::NonUniformBSpline<iganet::real_t, 3, 4, 5>>>(); break;
                default:
                  throw std::runtime_error("Invalid degree");
                }
                break;
              case iganet::webapp::degree::quintic:
                switch (static_cast<enum iganet::webapp::degree>(degrees[1])) {
                case iganet::webapp::degree::constant:
                  m = std::make_shared<iganet::webapp::BSplineModel<iganet::NonUniformBSpline<iganet::real_t, 3, 5, 0>>>(); break;
                case iganet::webapp::degree::linear:
                  m = std::make_shared<iganet::webapp::BSplineModel<iganet::NonUniformBSpline<iganet::real_t, 3, 5, 1>>>(); break;
                case iganet::webapp::degree::quadratic:
                  m = std::make_shared<iganet::webapp::BSplineModel<iganet::NonUniformBSpline<iganet::real_t, 3, 5, 2>>>(); break;
                case iganet::webapp::degree::cubic:
                  m = std::make_shared<iganet::webapp::BSplineModel<iganet::NonUniformBSpline<iganet::real_t, 3, 5, 3>>>(); break;
                case iganet::webapp::degree::quartic:
                  m = std::make_shared<iganet::webapp::BSplineModel<iganet::NonUniformBSpline<iganet::real_t, 3, 5, 4>>>(); break;
                case iganet::webapp::degree::quintic:
                  m = std::make_shared<iganet::webapp::BSplineModel<iganet::NonUniformBSpline<iganet::real_t, 3, 5, 5>>>(); break;
                default:
                  throw std::runtime_error("Invalid degree");
                }
                break;
              default:
                throw std::runtime_error("Invalid degree");
              }

            } else {

              // Uniform B-splines
              switch (static_cast<enum iganet::webapp::degree>(degrees[0])) {
              case iganet::webapp::degree::constant:
                switch (static_cast<enum iganet::webapp::degree>(degrees[1])) {
                case iganet::webapp::degree::constant:
                  m = std::make_shared<iganet::webapp::BSplineModel<iganet::UniformBSpline<iganet::real_t, 3, 0, 0>>>(); break;
                case iganet::webapp::degree::linear:
                  m = std::make_shared<iganet::webapp::BSplineModel<iganet::UniformBSpline<iganet::real_t, 3, 0, 1>>>(); break;
                case iganet::webapp::degree::quadratic:
                  m = std::make_shared<iganet::webapp::BSplineModel<iganet::UniformBSpline<iganet::real_t, 3, 0, 2>>>(); break;
                case iganet::webapp::degree::cubic:
                  m = std::make_shared<iganet::webapp::BSplineModel<iganet::UniformBSpline<iganet::real_t, 3, 0, 3>>>(); break;
                case iganet::webapp::degree::quartic:
                  m = std::make_shared<iganet::webapp::BSplineModel<iganet::UniformBSpline<iganet::real_t, 3, 0, 4>>>(); break;
                case iganet::webapp::degree::quintic:
                  m = std::make_shared<iganet::webapp::BSplineModel<iganet::UniformBSpline<iganet::real_t, 3, 0, 5>>>(); break;
                default:
                  throw std::runtime_error("Invalid degree");
                }
                break;
              case iganet::webapp::degree::linear:
                switch (static_cast<enum iganet::webapp::degree>(degrees[1])) {
                case iganet::webapp::degree::constant:
                  m = std::make_shared<iganet::webapp::BSplineModel<iganet::UniformBSpline<iganet::real_t, 3, 1, 0>>>(); break;
                case iganet::webapp::degree::linear:
                  m = std::make_shared<iganet::webapp::BSplineModel<iganet::UniformBSpline<iganet::real_t, 3, 1, 1>>>(); break;
                case iganet::webapp::degree::quadratic:
                  m = std::make_shared<iganet::webapp::BSplineModel<iganet::UniformBSpline<iganet::real_t, 3, 1, 2>>>(); break;
                case iganet::webapp::degree::cubic:
                  m = std::make_shared<iganet::webapp::BSplineModel<iganet::UniformBSpline<iganet::real_t, 3, 1, 3>>>(); break;
                case iganet::webapp::degree::quartic:
                  m = std::make_shared<iganet::webapp::BSplineModel<iganet::UniformBSpline<iganet::real_t, 3, 1, 4>>>(); break;
                case iganet::webapp::degree::quintic:
                  m = std::make_shared<iganet::webapp::BSplineModel<iganet::UniformBSpline<iganet::real_t, 3, 1, 5>>>(); break;
                default:
                  throw std::runtime_error("Invalid degree");
                }
                break;
              case iganet::webapp::degree::quadratic:
                switch (static_cast<enum iganet::webapp::degree>(degrees[1])) {
                case iganet::webapp::degree::constant:
                  m = std::make_shared<iganet::webapp::BSplineModel<iganet::UniformBSpline<iganet::real_t, 3, 2, 0>>>(); break;
                case iganet::webapp::degree::linear:
                  m = std::make_shared<iganet::webapp::BSplineModel<iganet::UniformBSpline<iganet::real_t, 3, 2, 1>>>(); break;
                case iganet::webapp::degree::quadratic:
                  m = std::make_shared<iganet::webapp::BSplineModel<iganet::UniformBSpline<iganet::real_t, 3, 2, 2>>>(); break;
                case iganet::webapp::degree::cubic:
                  m = std::make_shared<iganet::webapp::BSplineModel<iganet::UniformBSpline<iganet::real_t, 3, 2, 3>>>(); break;
                case iganet::webapp::degree::quartic:
                  m = std::make_shared<iganet::webapp::BSplineModel<iganet::UniformBSpline<iganet::real_t, 3, 2, 4>>>(); break;
                case iganet::webapp::degree::quintic:
                  m = std::make_shared<iganet::webapp::BSplineModel<iganet::UniformBSpline<iganet::real_t, 3, 2, 5>>>(); break;
                default:
                  throw std::runtime_error("Invalid degree");
                }
                break;
              case iganet::webapp::degree::cubic:
                switch (static_cast<enum iganet::webapp::degree>(degrees[1])) {
                case iganet::webapp::degree::constant:
                  m = std::make_shared<iganet::webapp::BSplineModel<iganet::UniformBSpline<iganet::real_t, 3, 3, 0>>>(); break;
                case iganet::webapp::degree::linear:
                  m = std::make_shared<iganet::webapp::BSplineModel<iganet::UniformBSpline<iganet::real_t, 3, 3, 1>>>(); break;
                case iganet::webapp::degree::quadratic:
                  m = std::make_shared<iganet::webapp::BSplineModel<iganet::UniformBSpline<iganet::real_t, 3, 3, 2>>>(); break;
                case iganet::webapp::degree::cubic:
                  m = std::make_shared<iganet::webapp::BSplineModel<iganet::UniformBSpline<iganet::real_t, 3, 3, 3>>>(); break;
                case iganet::webapp::degree::quartic:
                  m = std::make_shared<iganet::webapp::BSplineModel<iganet::UniformBSpline<iganet::real_t, 3, 3, 4>>>(); break;
                case iganet::webapp::degree::quintic:
                  m = std::make_shared<iganet::webapp::BSplineModel<iganet::UniformBSpline<iganet::real_t, 3, 3, 5>>>(); break;
                default:
                  throw std::runtime_error("Invalid degree");
                }
                break;
              case iganet::webapp::degree::quartic:
                switch (static_cast<enum iganet::webapp::degree>(degrees[1])) {
                case iganet::webapp::degree::constant:
                  m = std::make_shared<iganet::webapp::BSplineModel<iganet::UniformBSpline<iganet::real_t, 3, 4, 0>>>(); break;
                case iganet::webapp::degree::linear:
                  m = std::make_shared<iganet::webapp::BSplineModel<iganet::UniformBSpline<iganet::real_t, 3, 4, 1>>>(); break;
                case iganet::webapp::degree::quadratic:
                  m = std::make_shared<iganet::webapp::BSplineModel<iganet::UniformBSpline<iganet::real_t, 3, 4, 2>>>(); break;
                case iganet::webapp::degree::cubic:
                  m = std::make_shared<iganet::webapp::BSplineModel<iganet::UniformBSpline<iganet::real_t, 3, 4, 3>>>(); break;
                case iganet::webapp::degree::quartic:
                  m = std::make_shared<iganet::webapp::BSplineModel<iganet::UniformBSpline<iganet::real_t, 3, 4, 4>>>(); break;
                case iganet::webapp::degree::quintic:
                  m = std::make_shared<iganet::webapp::BSplineModel<iganet::UniformBSpline<iganet::real_t, 3, 4, 5>>>(); break;
                default:
                  throw std::runtime_error("Invalid degree");
                }
                break;
              case iganet::webapp::degree::quintic:
                switch (static_cast<enum iganet::webapp::degree>(degrees[1])) {
                case iganet::webapp::degree::constant:
                  m = std::make_shared<iganet::webapp::BSplineModel<iganet::UniformBSpline<iganet::real_t, 3, 5, 0>>>(); break;
                case iganet::webapp::degree::linear:
                  m = std::make_shared<iganet::webapp::BSplineModel<iganet::UniformBSpline<iganet::real_t, 3, 5, 1>>>(); break;
                case iganet::webapp::degree::quadratic:
                  m = std::make_shared<iganet::webapp::BSplineModel<iganet::UniformBSpline<iganet::real_t, 3, 5, 2>>>(); break;
                case iganet::webapp::degree::cubic:
                  m = std::make_shared<iganet::webapp::BSplineModel<iganet::UniformBSpline<iganet::real_t, 3, 5, 3>>>(); break;
                case iganet::webapp::degree::quartic:
                  m = std::make_shared<iganet::webapp::BSplineModel<iganet::UniformBSpline<iganet::real_t, 3, 5, 4>>>(); break;
                case iganet::webapp::degree::quintic:
                  m = std::make_shared<iganet::webapp::BSplineModel<iganet::UniformBSpline<iganet::real_t, 3, 5, 5>>>(); break;
                default:
                  throw std::runtime_error("Invalid degree");
                }
                break;
              default:
                throw std::runtime_error("Invalid degree");
              }
            }

            if (auto m_ = std::dynamic_pointer_cast<iganet::ModelSerialize>(m))
              m_->load(json);
            return m;
          }

          else {
            throw iganet::InvalidModelException();
          }

        } catch(...) {
          throw iganet::InvalidModelException();
        }
      }
    }

    throw iganet::InvalidModelException();
  }
}
