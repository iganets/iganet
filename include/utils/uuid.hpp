/**
   @file include/utils/uuid.hpp

   @brief UUID utility functions

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#pragma once

#include <random>
#include <sstream>

namespace iganet {
  namespace utils {
    
    class uuid {
    public:
    
      /// @brief Generates a uuid string in the form
      /// b9317db-02a2-4882-9b94-d1e1defe8c56
      ///
      /// @result std::string
      static std::string create() {
        std::stringstream hexstream;
        hexstream << uuid::random_hex(4) << "-" << uuid::random_hex(2) << "-"
                  << uuid::random_hex(2) << "-" << uuid::random_hex(2) << "-"
                  << uuid::random_hex(6);
        return hexstream.str();
      }

    private:
    
      /// @brief Generates a string of random hex chars
      ///
      /// @param[in] len Length in bytes
      ///
      /// @result std::string  String random hex chars (2x length of the bytes)
      static std::string random_hex(const unsigned int len) {
        std::stringstream ss;
        for (auto i = 0; i < len; i++) {
          const auto rc = random_char();
          std::stringstream hexstream;
          hexstream << std::hex << rc;
          auto hex = hexstream.str();
          ss << (hex.length() < 2 ? '0' + hex : hex);
        }
        return ss.str();
      }    
    
      /// @brief Generates a safe pseudo-random character
      ///
      /// @result unsigned int
      static unsigned int random_char() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, 255);
        return dis(gen);
      }
    };

  } // namespace utils
} // namespace iganet
