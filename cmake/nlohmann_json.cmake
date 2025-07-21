########################################################################
# nlohmann_json.cmake
#
# Author: Matthias Moller
# Copyright (C) 2021-2025 by the IgaNet authors
#
# This file is part of the IgaNet project
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
########################################################################

########################################################################
# nlohmann JSON
########################################################################

include(FetchContent)
FetchContent_Declare(nlohmann_json
  URL https://github.com/nlohmann/json/archive/refs/tags/v3.12.0.zip
  )
FetchContent_MakeAvailable(nlohmann_json)
FetchContent_GetProperties(nlohmann_json)
