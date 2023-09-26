/**
   @file include/utils/serialize.hpp

   @brief Srialization utility functions

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#pragma once

#include <utils/tensorarray.hpp>

#include <pugixml.hpp>
#include <nlohmann/json.hpp>

#include <torch/torch.h>

namespace iganet {
  namespace utils {

    /// @brief Serialization prototype
    ///
    /// This abstract class defines the functions that must be
    /// implemented to serialize an object
    struct Serializable
    {
      /// @brief Returns the object as JSON object
      virtual nlohmann::json to_json() const = 0;

      /// @brief Returns a string representation of the object
      virtual void pretty_print(std::ostream& os = std::cout) const = 0;
    };

    /// @brief Converts a torch::TensorAccessor object to a JSON object
    template<typename T, std::size_t N>
    auto to_json(const torch::TensorAccessor<T, N>& accessor)
    {
      auto json = nlohmann::json::array();

      if constexpr (N == 1) {
        for (int64_t i = 0; i < accessor.size(0); ++i)
          json.push_back(accessor[i]);
      } else if constexpr (N == 2) {
        for (int64_t i = 0; i < accessor.size(0); ++i)
          for (int64_t j = 0; j < accessor.size(1); ++j)
            json.push_back(accessor[i][j]);
      } else if constexpr (N == 3) {
        for (int64_t i = 0; i < accessor.size(0); ++i)
          for (int64_t j = 0; j < accessor.size(1); ++j)
            for (int64_t k = 0; k < accessor.size(2); ++k)
              json.push_back(accessor[i][j][k]);
      } else if constexpr (N == 4) {
        for (int64_t i = 0; i < accessor.size(0); ++i)
          for (int64_t j = 0; j < accessor.size(1); ++j)
            for (int64_t k = 0; k < accessor.size(2); ++k)
              for (int64_t l = 0; l < accessor.size(3); ++l)
                json.push_back(accessor[i][j][k][l]);
      }

      return json;
    }

    /// @brief Converts a torch::Tensor object to a JSON object
    template<typename T, std::size_t N>
    auto to_json(const torch::Tensor& tensor)
    {
      if (tensor.is_cuda()) {
        auto [tensor_cpu, accessor] = to_tensorAccessor<T, N>(tensor, torch::kCPU);
        return to_json(accessor);
      } else {
        auto accessor = to_tensorAccessor<T, N>(tensor);
        return to_json(accessor);
      }
    }

    /// @brief Converts an std::array of torch::Tensor objects to a JSON
    /// object
    template<typename T, std::size_t N, std::size_t M>
    auto to_json(const std::array<torch::Tensor, M>& tensors)
    {
      auto json = nlohmann::json::array();
      
      for (std::size_t i = 0; i < M; ++i) {
        if (tensors[i].is_cuda()) {
          auto [tensor_cpu, accessor] = to_tensorAccessor<T, N>(tensors[i], torch::kCPU);
          json.push_back(to_json<T, N>(accessor));
        } else {
          auto accessor = to_tensorAccessor<T, N>(tensors[i]);
          json.push_back(to_json<T, N>(accessor));
        }
      }
      
      return json;
    }
    
  } // namespace utils
} // namespace iganet
