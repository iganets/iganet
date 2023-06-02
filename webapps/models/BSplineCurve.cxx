/**
   @file models/BSplineCurve.cxx

   @brief B-Spline curve

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
  /// @brief Create a B-spline curve
  std::shared_ptr<iganet::Model> create(const nlohmann::json& json) {
    enum iganet::webapp::degree degree = iganet::webapp::degree::linear;
    enum iganet::init init = iganet::init::linear;
    std::array<int64_t, 1> ncoeffs = {4};
    bool nonuniform = false;

    if (json.contains("data")) {

      if (json["data"].contains("degree"))
        degree = json["data"]["degree"].get<enum iganet::webapp::degree>();

      if (json["data"].contains("init"))
        init = json["data"]["init"].get<enum iganet::init>();

      if (json["data"].contains("ncoeffs"))
        ncoeffs = json["data"]["ncoeffs"].get<std::array<int64_t,1>>();

      if (json["data"].contains("nonuniform"))
        nonuniform = json["data"]["nonuniform"].get<bool>();

      switch (degree) {
      case iganet::webapp::degree::constant:
        if (nonuniform)
          return std::make_shared<iganet::webapp::BSplineModel<iganet::NonUniformBSpline<iganet::real_t, 3, 0>>>(ncoeffs, init);
        else
          return std::make_shared<iganet::webapp::BSplineModel<iganet::   UniformBSpline<iganet::real_t, 3, 0>>>(ncoeffs, init);
      case iganet::webapp::degree::linear:
        if (nonuniform)
          return std::make_shared<iganet::webapp::BSplineModel<iganet::NonUniformBSpline<iganet::real_t, 3, 1>>>(ncoeffs, init);
        else
          return std::make_shared<iganet::webapp::BSplineModel<iganet::   UniformBSpline<iganet::real_t, 3, 1>>>(ncoeffs, init);
      case iganet::webapp::degree::quadratic:
        if (nonuniform)
          return std::make_shared<iganet::webapp::BSplineModel<iganet::NonUniformBSpline<iganet::real_t, 3, 2>>>(ncoeffs, init);
        else
          return std::make_shared<iganet::webapp::BSplineModel<iganet::   UniformBSpline<iganet::real_t, 3, 2>>>(ncoeffs, init);
      case iganet::webapp::degree::cubic:
        if (nonuniform)
          return std::make_shared<iganet::webapp::BSplineModel<iganet::NonUniformBSpline<iganet::real_t, 3, 3>>>(ncoeffs, init);
        else
          return std::make_shared<iganet::webapp::BSplineModel<iganet::   UniformBSpline<iganet::real_t, 3, 3>>>(ncoeffs, init);
      case iganet::webapp::degree::quartic:
        if (nonuniform)
          return std::make_shared<iganet::webapp::BSplineModel<iganet::NonUniformBSpline<iganet::real_t, 3, 4>>>(ncoeffs, init);
        else
          return std::make_shared<iganet::webapp::BSplineModel<iganet::   UniformBSpline<iganet::real_t, 3, 4>>>(ncoeffs, init);
      case iganet::webapp::degree::quintic:
        if (nonuniform)
          return std::make_shared<iganet::webapp::BSplineModel<iganet::NonUniformBSpline<iganet::real_t, 3, 5>>>(ncoeffs, init);
        else
          return std::make_shared<iganet::webapp::BSplineModel<iganet::   UniformBSpline<iganet::real_t, 3, 5>>>(ncoeffs, init);
      default:
        throw std::runtime_error("Invalid degree");
      }
    }
    else
      if (nonuniform)
        return std::make_shared<iganet::webapp::BSplineModel<iganet::NonUniformBSpline<iganet::real_t, 3, 1>>>(ncoeffs, init);
      else
        return std::make_shared<iganet::webapp::BSplineModel<iganet::   UniformBSpline<iganet::real_t, 3, 1>>>(ncoeffs, init);
  }
}
