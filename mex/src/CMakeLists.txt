########################################################################
# CMakeLists.txt
#
# Author: Matthias Moller
# Copyright (C) 2021-2024 by the IgaNet authors
#
# This file is part of the IgaNet project
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
########################################################################

find_package( Matlab COMPONENTS MX_LIBRARY )

if (Matlab_FOUND)
  matlab_add_mex(
    NAME BSpline
    SRC mexbspline.cpp
    )
endif()
