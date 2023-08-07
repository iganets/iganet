/**
   @file include/utils/convert.hpp

   @brief Convert utility functions

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#pragma once

#include <array>
#include <initializer_list>
#include <vector>

#include <options.hpp>

#include <torch/torch.h>

namespace iganet {
  namespace utils {

    /// @brief Converts an std::vector object into std::array
    template<std::size_t N, typename T>
    inline std::array<T, N> convert(std::vector<T>&& vector)
    {
      std::array<T, N> array;
      std::move(vector.begin(), vector.end(), array.begin());
      return array;
    }

    /// @brief Converts an std::array object into std::vector
    template<typename T, std::size_t N>
    inline std::vector<T> convert(std::array<T, N>&& array)
    {
      std::vector<T> vector;
      std::move(array.begin(), array.end(), vector.begin());
      return vector;
    }

    /// @brief Converts a list of arguments into std::array
    template<typename...Args>
    auto to_array(Args&&... args)
    {
      return std::array<typename std::common_type<Args...>::type,
                        sizeof...(Args)>{std::move(args)...};
    }

    /// @brief Converts a list of arguments into std::vector
    template<typename...Args>
    auto to_vector(Args&&... args)
    {
      return std::vector<typename std::common_type<Args...>::type>{std::move(args)...};
    }

    /// @brief Converts an std::array to torch::Tensor
    /// @{
    template<typename T, std::size_t N>
    inline auto to_tensor(const std::array<T, N>& array,
                          torch::IntArrayRef sizes = torch::IntArrayRef{-1},
                          const iganet::Options<T>& options = iganet::Options<T>{})
    {
      if (options.device() == torch::kCPU)
        return torch::from_blob(const_cast<T*>(std::data(array)),
                                (sizes == torch::IntArrayRef{-1}) ? array.size() : sizes,
                                options)
          .detach().clone().requires_grad_(options.requires_grad());
      else
        return torch::from_blob(const_cast<T*>(std::data(array)),
                                (sizes == torch::IntArrayRef{-1}) ? array.size() : sizes,
                                options.device(torch::kCPU))
          .detach().clone().to(options.device()).requires_grad_(options.requires_grad());
    }

    template<typename T, std::size_t N>
    inline auto to_tensor(const std::array<T, N>& array,
                          const iganet::Options<T>& options)
    {
      if (options.device() == torch::kCPU)
        return torch::from_blob(const_cast<T*>(std::data(array)),
                                array.size(), options)
          .detach().clone().requires_grad_(options.requires_grad());
      else
        return torch::from_blob(const_cast<T*>(std::data(array)),
                                array.size(),
                                options.device(torch::kCPU))
          .detach().clone().to(options.device()).requires_grad_(options.requires_grad());
    }
    /// @}
    
    /// @brief Converts an std::initializer_list to torch::Tensor
    /// @{
    template<typename T>
    inline auto to_tensor(std::initializer_list<T> list,
                          torch::IntArrayRef sizes = torch::IntArrayRef{-1},
                          const iganet::Options<T>& options = iganet::Options<T>{})
    {
      if (options.device() == torch::kCPU)
        return torch::from_blob(const_cast<T*>(std::data(list)),
                                (sizes == torch::IntArrayRef{-1}) ? list.size() : sizes,
                                options)
          .detach().clone().requires_grad_(options.requires_grad());
      else
        return torch::from_blob(const_cast<T*>(std::data(list)),
                                (sizes == torch::IntArrayRef{-1}) ? list.size() : sizes,
                                options.device(torch::kCPU))
          .detach().clone().to(options.device()).requires_grad_(options.requires_grad());
    }

    template<typename T>
    inline auto to_tensor(std::initializer_list<T>& list,
                          const iganet::Options<T>& options)
    {
      if (options.device() == torch::kCPU)
        return torch::from_blob(const_cast<T*>(std::data(list)),
                                list.size(), options)
          .detach().clone().requires_grad_(options.requires_grad());
      else
        return torch::from_blob(const_cast<T*>(std::data(list)),
                                list.size(),
                                options.device(torch::kCPU))
          .detach().clone().to(options.device()).requires_grad_(options.requires_grad());
    }
    /// @}

    /// @brief Converts an std::vector to torch::Tensor
    /// @{
    template<typename T>
    inline auto to_tensor(const std::vector<T>& vector,
                          torch::IntArrayRef sizes = torch::IntArrayRef{-1},
                          const iganet::Options<T>& options = iganet::Options<T>{})
    {
      if (options.device() == torch::kCPU)
        return torch::from_blob(const_cast<T*>(std::data(vector)),
                                (sizes == torch::IntArrayRef{-1}) ? vector.size() : sizes,
                                options)
          .detach().clone().requires_grad_(options.requires_grad());
      else
        return torch::from_blob(const_cast<T*>(std::data(vector)),
                                (sizes == torch::IntArrayRef{-1}) ? vector.size() : sizes,
                                options.device(torch::kCPU))
          .detach().clone().to(options.device()).requires_grad_(options.requires_grad());
    }

    template<typename T>
    inline auto to_tensor(const std::vector<T>& vector,
                          const iganet::Options<T>& options)
    {
      if (options.device() == torch::kCPU)
        return torch::from_blob(const_cast<T*>(std::data(vector)),
                                vector.size(), options)
          .detach().clone().requires_grad_(options.requires_grad());
      else
        return torch::from_blob(const_cast<T*>(std::data(vector)),
                                vector.size(),
                                options.device(torch::kCPU))
          .detach().clone().to(options.device()).requires_grad_(options.requires_grad());
    }
    /// @}

    

    /// @brief Converts an std::array<int64_t, N> to a at::IntArrayRef object
    template<typename T, std::size_t N>
    auto to_ArrayRef(const std::array<T, N>& array) {
      return at::ArrayRef<T>{array};
    }
    
  } // namespace utils
} // namespace iganet
