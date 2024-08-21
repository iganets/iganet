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
};

} // namespace webapp
} // namespace iganet
