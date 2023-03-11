/**
   @file plugins/BSpline.cxx

   @brief BSpline test pluging

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#ifdef _WIN32
#ifdef __cplusplus
#define EXPORT extern "C" __declspec(dllexport)
#endif
#else
#ifdef __cplusplus
#define EXPORT extern "C"
#endif
#endif

#include <iganet.hpp>

static iganet::UniformBSpline<double,1,1,1> bspline({5,5});
