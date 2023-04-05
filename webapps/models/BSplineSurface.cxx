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
#include <modelmanager.hpp>

#ifdef _WIN32
extern "C" __declspec(dllexport)
#else
extern "C"
#endif
{
  std::shared_ptr<iganet::Model> create(const nlohmann::json& config) {
    enum iganet::webapp::degree degree = iganet::webapp::degree::linear;
    enum iganet::init init = iganet::init::linear;
    std::array<int64_t, 2> ncoeffs = {4,4};
    bool nonuniform = false;

    if (config.contains("data")) {

      if (config["data"].contains("degree"))
        degree = config["data"]["degree"].get<enum iganet::webapp::degree>();

      if (config["data"].contains("init"))
        init = config["data"]["init"].get<enum iganet::init>();

      if (config["data"].contains("ncoeffs"))
        ncoeffs = config["data"]["ncoeffs"].get<std::array<int64_t,2>>();

      if (config["data"].contains("nonuniform"))
        nonuniform = config["data"]["nonuniform"].get<bool>();

      switch (degree) {
      case iganet::webapp::degree::constant:
        if (nonuniform)
          return std::make_shared<iganet::webapp::BSplineModel<iganet::NonUniformBSpline<float, 1, 0,0>>>(ncoeffs, init);
        else
          return std::make_shared<iganet::webapp::BSplineModel<iganet::   UniformBSpline<float, 1, 0,0>>>(ncoeffs, init);
      case iganet::webapp::degree::linear:
        if (nonuniform)
          return std::make_shared<iganet::webapp::BSplineModel<iganet::NonUniformBSpline<float, 1, 1,1>>>(ncoeffs, init);
        else
          return std::make_shared<iganet::webapp::BSplineModel<iganet::   UniformBSpline<float, 1, 1,1>>>(ncoeffs, init);
      case iganet::webapp::degree::quadratic:
        if (nonuniform)
          return std::make_shared<iganet::webapp::BSplineModel<iganet::NonUniformBSpline<float, 1, 2,2>>>(ncoeffs, init);
        else
          return std::make_shared<iganet::webapp::BSplineModel<iganet::   UniformBSpline<float, 1, 2,2>>>(ncoeffs, init);
      case iganet::webapp::degree::cubic:
        if (nonuniform)
          return std::make_shared<iganet::webapp::BSplineModel<iganet::NonUniformBSpline<float, 1, 3,3>>>(ncoeffs, init);
        else
          return std::make_shared<iganet::webapp::BSplineModel<iganet::   UniformBSpline<float, 1, 3,3>>>(ncoeffs, init);
      case iganet::webapp::degree::quartic:
        if (nonuniform)
          return std::make_shared<iganet::webapp::BSplineModel<iganet::NonUniformBSpline<float, 1, 4,4>>>(ncoeffs, init);
        else
          return std::make_shared<iganet::webapp::BSplineModel<iganet::   UniformBSpline<float, 1, 4,4>>>(ncoeffs, init);
      case iganet::webapp::degree::quintic:
        if (nonuniform)
          return std::make_shared<iganet::webapp::BSplineModel<iganet::NonUniformBSpline<float, 1, 5,5>>>(ncoeffs, init);
        else
          return std::make_shared<iganet::webapp::BSplineModel<iganet::   UniformBSpline<float, 1, 5,5>>>(ncoeffs, init);
      default:
        throw std::runtime_error("Invalid degree");
      }
    }
    else
      if (nonuniform)
        return std::make_shared<iganet::webapp::BSplineModel<iganet::NonUniformBSpline<float, 1, 1,1>>>(ncoeffs, init);
      else
        return std::make_shared<iganet::webapp::BSplineModel<iganet::   UniformBSpline<float, 1, 1,1>>>(ncoeffs, init);
  }
}
