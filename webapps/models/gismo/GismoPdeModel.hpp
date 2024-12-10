/**
   @file webapps/models/gismo/GismoPdeModel.hpp

   @brief G+Smo PDE model

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#pragma once

#include <GismoGeometryModel.hpp>

namespace iganet {

namespace webapp {

/// @brief G+Smo PDE model
template <short_t d, class T>
class GismoPdeModel : public GismoGeometryModel<d, T> {

public:
  /// @brief Constructors
  using GismoGeometryModel<d, T>::GismoGeometryModel;

  /// @brief Serializes the model to JSON
  nlohmann::json to_json(const std::string &patch, const std::string &component,
                         const std::string &attribute) const override {

    if (patch == "" && (component == "solution" || component == "") &&
        attribute == "") {

      return utils::to_json(solution_);
    }

    if (patch != "" && component == "solution" && attribute == "") {

      int patchId(0);

      try {
        patchId = stoi(patch);
      } catch (...) {
        // Invalid patchId
        return R"({ INVALID REQUEST })"_json;
      }

      return utils::to_json(solution_.patch(patchId));

    }

    else if (patch != "" && component == "geometry" && attribute != "") {

      int patchId(0);

      try {
        patchId = stoi(patch);
      } catch (...) {
        // Invalid patchId
        return R"({ INVALID REQUEST })"_json;
      }

      nlohmann::json json;

      if (attribute == "degrees") {
        json["degrees"] = nlohmann::json::array();

        for (std::size_t i = 0; i < solution_.patch(patchId).parDim(); ++i)
          json["degrees"].push_back(solution_.patch(patchId).degree(i));
      }

      else if (attribute == "geoDim")
        json["geoDim"] = solution_.patch(patchId).geoDim();

      else if (attribute == "parDim")
        json["parDim"] = solution_.patch(patchId).parDim();

      else if (attribute == "ncoeffs") {
        json["ncoeffs"] = nlohmann::json::array();

        if (auto bspline =
                dynamic_cast<const gsBSpline<T> *>(&solution_.patch(patchId)))
          for (std::size_t i = 0; i < bspline->parDim(); ++i)
            json["ncoeffs"].push_back(bspline->basis().size(i));
        else if (auto bspline = dynamic_cast<const gsTensorBSpline<d, T> *>(
                     &solution_.patch(patchId)))
          for (std::size_t i = 0; i < bspline->parDim(); ++i)
            json["ncoeffs"].push_back(bspline->basis().size(i));
        else
          return R"({ INVALID REQUEST })"_json;

      }

      else if (attribute == "nknots") {
        json["nknots"] = nlohmann::json::array();

        if (auto bspline =
                dynamic_cast<const gsBSpline<T> *>(&solution_.patch(patchId)))
          for (std::size_t i = 0; i < bspline->parDim(); ++i)
            json["nknots"].push_back(bspline->knots(i).size());
        else if (auto bspline = dynamic_cast<const gsTensorBSpline<d, T> *>(
                     &solution_.patch(patchId)))
          for (std::size_t i = 0; i < bspline->parDim(); ++i)
            json["nknots"].push_back(bspline->knots(i).size());
        else
          return R"({ INVALID REQUEST })"_json;
      }

      else if (attribute == "coeffs") {

        if (auto bspline =
                dynamic_cast<const gsBSpline<T> *>(&solution_.patch(patchId)))
          json["coeffs"] = utils::to_json(bspline->coefs());
        else if (auto bspline = dynamic_cast<const gsTensorBSpline<d, T> *>(
                     &solution_.patch(patchId)))
          json["coeffs"] = utils::to_json(bspline->coefs());
        else
          return R"({ INVALID REQUEST })"_json;

      }

      else if (attribute == "knots") {
        json["knots"] = nlohmann::json::array();

        if (auto bspline =
                dynamic_cast<const gsBSpline<T> *>(&solution_.patch(patchId)))
          for (std::size_t i = 0; i < bspline->parDim(); ++i)
            json["knots"].push_back(bspline->knots(i));
        else if (auto bspline = dynamic_cast<const gsTensorBSpline<d, T> *>(
                     &solution_.patch(patchId)))
          for (std::size_t i = 0; i < bspline->parDim(); ++i)
            json["knots"].push_back(bspline->knots(i));
        else
          return R"({ INVALID REQUEST })"_json;
      }

      else
        // Invalid attribute
        return R"({ INVALID REQUEST })"_json;

      return json;
    }

    return GismoGeometryModel<d, T>::to_json(patch, component, attribute);
  }

protected:
  /// @brief Solution
  gsMultiPatch<T> solution_;
};

} // namespace webapp
} // namespace iganet
