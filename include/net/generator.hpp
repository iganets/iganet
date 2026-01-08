/**
   @file net/generator.hpp

   @brief Network generator

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#pragma once

#include <any>
#include <iostream>
#include <vector>

#include <core/core.hpp>
#include <core/options.hpp>
#include <net/layer.hpp>

namespace iganet {

//  clang-format off
/// @brief Enumerator for specifying the initialization of network weights
enum class nn_init : short_t {
  constant = 0,        /*!< initialize weights to constant value                                                               */
  normal = 1,          /*!< initialize weights with values drawn from a normal distribution parameterized by 'mean' and 'std'  */
  uniform = 2,         /*!< initialize weights with values drawn from a uniform distribution parameterized by 'low' and 'high' */
  kaiming_normal = 3,  /*!< initialize weights as proposed by Kaiming He using a normal distribution                           */
  kaiming_uniform = 4, /*!< initialize weights as proposed by Kaiming He using a uniform distribution                          */
  xavier_normal = 5,   /*!< initialize weights as proposed by Xavier Glorot using a normal distribution                        */
  xavier_uniform = 6,  /*!< initialize weights as proposed by Xavier Glorot using a uniform distribution                       */    
};
//  clang-format on
  
/// @brief IgANetGeneratorImpl
///
/// @note Following the discussion of module overship here
///
/// https://pytorch.org/tutorials/advanced/cpp_frontend.html#module-ownership
///
/// we implement a generator implementation class following
///
/// https://pytorch.org/tutorials/advanced/cpp_frontend.html#the-generator-module
template <typename real_t>
class IgANetGeneratorImpl : public torch::nn::Module {
public:
  /// @brief Default constructor
  IgANetGeneratorImpl() = default;

  /// @brief Constructor
  explicit IgANetGeneratorImpl(
      const std::vector<int64_t> &layers,
      const std::vector<std::vector<std::any>> &activations,
      Options<real_t> options = Options<real_t>{}) {
    assert(layers.size() == activations.size() + 1);

    // Generate vector of linear layers and register them as layer[i]
    for (auto i = 0; i < layers.size() - 1; ++i) {
      layers_.emplace_back(
          register_module("layer[" + std::to_string(i) + "]",
                          torch::nn::Linear(layers[i], layers[i + 1])));
      layers_.back()->to(options.device(), options.dtype(), true);

      torch::nn::init::xavier_uniform_(layers_.back()->weight);
      torch::nn::init::constant_(layers_.back()->bias, 0.0);
    }

    // Generate vector of activation functions
    for (const auto &a : activations)
      switch (std::any_cast<activation>(a[0])) {
        // No activation function
      case activation::none:
        switch (a.size()) {
        case 1:
          activations_.emplace_back(new None{});
          break;
        default:
          throw std::runtime_error("Invalid number of parameters");
        }
        break;

        // Batch Normalization
      case activation::batch_norm:
        switch (a.size()) {
        case 8:
          activations_.emplace_back(new BatchNorm{
              std::any_cast<torch::Tensor>(a[1]),
              std::any_cast<torch::Tensor>(a[2]),
              std::any_cast<torch::Tensor>(a[3]),
              std::any_cast<torch::Tensor>(a[4]), std::any_cast<double>(a[5]),
              std::any_cast<double>(a[6]), std::any_cast<bool>(a[7])});
          break;
        case 7:
          activations_.emplace_back(new BatchNorm{
              std::any_cast<torch::Tensor>(a[1]),
              std::any_cast<torch::Tensor>(a[2]),
              std::any_cast<torch::Tensor>(a[3]),
              std::any_cast<torch::Tensor>(a[4]), std::any_cast<double>(a[5]),
              std::any_cast<double>(a[6])});
          break;
        case 4:
          activations_.emplace_back(new BatchNorm{
              std::any_cast<torch::Tensor>(a[1]),
              std::any_cast<torch::Tensor>(a[2]),
              std::any_cast<torch::nn::functional::BatchNormFuncOptions>(
                  a[3])});
          break;
        case 3:
          activations_.emplace_back(
              new BatchNorm{std::any_cast<torch::Tensor>(a[1]),
                            std::any_cast<torch::Tensor>(a[2])});
          break;
        default:
          throw std::runtime_error("Invalid number of parameters");
        }
        break;

        // CELU
      case activation::celu:
        switch (a.size()) {
        case 3:
          activations_.emplace_back(
              new CELU{std::any_cast<double>(a[1]), std::any_cast<bool>(a[2])});
          break;
        case 2:
          try {
            activations_.emplace_back(new CELU{
                std::any_cast<torch::nn::functional::CELUFuncOptions>(a[1])});
          } catch (...) {
            activations_.emplace_back(new CELU{std::any_cast<double>(a[1])});
          }
          break;
        case 1:
          activations_.emplace_back(new CELU{});
          break;
        default:
          throw std::runtime_error("Invalid number of parameters");
        }
        break;

        // ELU
      case activation::elu:
        switch (a.size()) {
        case 3:
          activations_.emplace_back(
              new ELU{std::any_cast<double>(a[1]), std::any_cast<bool>(a[2])});
          break;
        case 2:
          try {
            activations_.emplace_back(new ELU{
                std::any_cast<torch::nn::functional::ELUFuncOptions>(a[1])});
          } catch (...) {
            activations_.emplace_back(new ELU{std::any_cast<double>(a[1])});
          }
          break;
        case 1:
          activations_.emplace_back(new ELU{});
          break;
        default:
          throw std::runtime_error("Invalid number of parameters");
        }
        break;

        // GELU
      case activation::gelu:
        switch (a.size()) {
        case 1:
          activations_.emplace_back(new GELU{});
          break;
        default:
          throw std::runtime_error("Invalid number of parameters");
        }
        break;

        // GLU
      case activation::glu:
        switch (a.size()) {
        case 2:
          try {
            activations_.emplace_back(new GLU{
                std::any_cast<torch::nn::functional::GLUFuncOptions>(a[1])});
          } catch (...) {
            activations_.emplace_back(new GLU{std::any_cast<int64_t>(a[1])});
          }
          break;
        case 1:
          activations_.emplace_back(new GLU{});
          break;
        default:
          throw std::runtime_error("Invalid number of parameters");
        }
        break;

        // Group Normalization
      case activation::group_norm:
        switch (a.size()) {
        case 5:
          activations_.emplace_back(new GroupNorm{
              std::any_cast<int64_t>(a[1]), std::any_cast<torch::Tensor>(a[2]),
              std::any_cast<torch::Tensor>(a[3]), std::any_cast<double>(a[4])});
          break;
        case 2:
          try {
            activations_.emplace_back(new GroupNorm{
                std::any_cast<torch::nn::functional::GroupNormFuncOptions>(
                    a[1])});
          } catch (...) {
            activations_.emplace_back(
                new GroupNorm{std::any_cast<int64_t>(a[1])});
          }
          break;
        default:
          throw std::runtime_error("Invalid number of parameters");
        }
        break;

        // Gumbel-Softmax
      case activation::gumbel_softmax:
        switch (a.size()) {
        case 4:
          activations_.emplace_back(new GumbelSoftmax{
              std::any_cast<double>(a[1]), std::any_cast<int>(a[2]),
              std::any_cast<bool>(a[3])});
          break;
        case 2:
          activations_.emplace_back(new GumbelSoftmax{
              std::any_cast<torch::nn::functional::GumbelSoftmaxFuncOptions>(
                  a[1])});
          break;
        case 1:
          activations_.emplace_back(new GumbelSoftmax{});
          break;
        default:
          throw std::runtime_error("Invalid number of parameters");
        }
        break;

        // Hard shrinkish
      case activation::hardshrink:
        switch (a.size()) {
        case 2:
          try {
            activations_.emplace_back(new Hardshrink{
                std::any_cast<torch::nn::functional::HardshrinkFuncOptions>(
                    a[1])});
          } catch (...) {
            activations_.emplace_back(
                new Hardshrink{std::any_cast<double>(a[1])});
          }
          break;
        case 1:
          activations_.emplace_back(new Hardshrink{});
          break;
        default:
          throw std::runtime_error("Invalid number of parameters");
        }
        break;

        // Hardsigmoid
      case activation::hardsigmoid:
        switch (a.size()) {
        case 1:
          activations_.emplace_back(new Hardsigmoid{});
          break;
        default:
          throw std::runtime_error("Invalid number of parameters");
        }
        break;

        // Hardswish
      case activation::hardswish:
        switch (a.size()) {
        case 1:
          activations_.emplace_back(new Hardswish{});
          break;
        default:
          throw std::runtime_error("Invalid number of parameters");
        }
        break;

        // Hardtanh
      case activation::hardtanh:
        switch (a.size()) {
        case 4:
          activations_.emplace_back(new Hardtanh{std::any_cast<double>(a[1]),
                                                 std::any_cast<double>(a[2]),
                                                 std::any_cast<bool>(a[3])});
          break;
        case 3:
          activations_.emplace_back(new Hardtanh{std::any_cast<double>(a[1]),
                                                 std::any_cast<double>(a[2])});
          break;
        case 2:
          activations_.emplace_back(new Hardtanh{
              std::any_cast<torch::nn::functional::HardtanhFuncOptions>(a[1])});
          break;
        case 1:
          activations_.emplace_back(new Hardtanh{});
          break;
        default:
          throw std::runtime_error("Invalid number of parameters");
        }
        break;

        // Instance Normalization
      case activation::instance_norm:
        switch (a.size()) {
        case 8:
          activations_.emplace_back(new InstanceNorm{
              std::any_cast<torch::Tensor>(a[1]),
              std::any_cast<torch::Tensor>(a[2]),
              std::any_cast<torch::Tensor>(a[3]),
              std::any_cast<torch::Tensor>(a[4]), std::any_cast<double>(a[5]),
              std::any_cast<double>(a[6]), std::any_cast<bool>(a[7])});
          break;
        case 7:
          activations_.emplace_back(new InstanceNorm{
              std::any_cast<torch::Tensor>(a[1]),
              std::any_cast<torch::Tensor>(a[2]),
              std::any_cast<torch::Tensor>(a[3]),
              std::any_cast<torch::Tensor>(a[4]), std::any_cast<double>(a[5]),
              std::any_cast<double>(a[6])});
          break;
        case 2:
          activations_.emplace_back(new InstanceNorm{
              std::any_cast<torch::nn::functional::InstanceNormFuncOptions>(
                  a[1])});
          break;
        case 1:
          activations_.emplace_back(new InstanceNorm{});
          break;
        default:
          throw std::runtime_error("Invalid number of parameters");
        }
        break;

        // Layer Normalization
      case activation::layer_norm:
        switch (a.size()) {
        case 5:
          activations_.emplace_back(new LayerNorm{
              std::any_cast<std::vector<int64_t>>(a[1]),
              std::any_cast<torch::Tensor>(a[2]),
              std::any_cast<torch::Tensor>(a[3]), std::any_cast<double>(a[4])});
          break;
        case 2:
          try {
            activations_.emplace_back(new LayerNorm{
                std::any_cast<torch::nn::functional::LayerNormFuncOptions>(
                    a[1])});
          } catch (...) {
            activations_.emplace_back(
                new LayerNorm{std::any_cast<std::vector<int64_t>>(a[1])});
          }
          break;
        default:
          throw std::runtime_error("Invalid number of parameters");
        }
        break;

        // Leaky ReLU
      case activation::leaky_relu:
        switch (a.size()) {
        case 3:
          activations_.emplace_back(new LeakyReLU{std::any_cast<double>(a[1]),
                                                  std::any_cast<bool>(a[2])});
          break;
        case 2:
          try {
            activations_.emplace_back(new LeakyReLU{
                std::any_cast<torch::nn::functional::LeakyReLUFuncOptions>(
                    a[1])});
          } catch (...) {
            activations_.emplace_back(
                new LeakyReLU{std::any_cast<double>(a[1])});
          }
          break;
        case 1:
          activations_.emplace_back(new LeakyReLU{});
          break;
        default:
          throw std::runtime_error("Invalid number of parameters");
        }
        break;

        // Local response Normalization
      case activation::local_response_norm:
        switch (a.size()) {
        case 5:
          activations_.emplace_back(new LocalResponseNorm{
              std::any_cast<int64_t>(a[1]), std::any_cast<double>(a[2]),
              std::any_cast<double>(a[3]), std::any_cast<double>(a[4])});
          break;
        case 2:
          try {
            activations_.emplace_back(new LocalResponseNorm{std::any_cast<
                torch::nn::functional::LocalResponseNormFuncOptions>(a[1])});
          } catch (...) {
            activations_.emplace_back(
                new LocalResponseNorm{std::any_cast<int64_t>(a[1])});
          }
          break;
        default:
          throw std::runtime_error("Invalid number of parameters");
        }
        break;

        // LogSigmoid
      case activation::logsigmoid:
        switch (a.size()) {
        case 1:
          activations_.emplace_back(new LogSigmoid{});
          break;
        default:
          throw std::runtime_error("Invalid number of parameters");
        }
        break;

        // LogSoftmax
      case activation::logsoftmax:
        switch (a.size()) {
        case 2:
          try {
            activations_.emplace_back(new LogSoftmax{
                std::any_cast<torch::nn::functional::LogSoftmaxFuncOptions>(
                    a[1])});
          } catch (...) {
            activations_.emplace_back(
                new LogSoftmax{std::any_cast<int64_t>(a[1])});
          }
          break;
        default:
          throw std::runtime_error("Invalid number of parameters");
        }
        break;

        // Mish
      case activation::mish:
        switch (a.size()) {
        case 1:
          activations_.emplace_back(new Mish{});
          break;
        default:
          throw std::runtime_error("Invalid number of parameters");
        }
        break;

        // Lp Normalization
      case activation::normalize:
        switch (a.size()) {
        case 4:
          activations_.emplace_back(new Normalize{
              std::any_cast<double>(a[1]), std::any_cast<double>(a[2]),
              std::any_cast<int64_t>(a[3])});
          break;
        case 2:
          activations_.emplace_back(new Normalize{
              std::any_cast<torch::nn::functional::NormalizeFuncOptions>(
                  a[1])});
          break;
        case 1:
          activations_.emplace_back(new Normalize{});
          break;
        default:
          throw std::runtime_error("Invalid number of parameters");
        }
        break;

        // PReLU
      case activation::prelu:
        switch (a.size()) {
        case 2:
          activations_.emplace_back(
              new PReLU{std::any_cast<torch::Tensor>(a[1])});
          break;
        default:
          throw std::runtime_error("Invalid number of parameters");
        }
        break;

        // ReLU
      case activation::relu:
        switch (a.size()) {
        case 2:
          try {
            activations_.emplace_back(new ReLU{
                std::any_cast<torch::nn::functional::ReLUFuncOptions>(a[1])});
          } catch (...) {
            activations_.emplace_back(new ReLU{std::any_cast<bool>(a[1])});
          }
          break;
        case 1:
          activations_.emplace_back(new ReLU{});
          break;
        default:
          throw std::runtime_error("Invalid number of parameters");
        }
        break;

        // Relu6
      case activation::relu6:
        switch (a.size()) {
        case 2:
          try {
            activations_.emplace_back(new ReLU6{
                std::any_cast<torch::nn::functional::ReLU6FuncOptions>(a[1])});
          } catch (...) {
            activations_.emplace_back(new ReLU6{std::any_cast<bool>(a[1])});
          }
          break;
        case 1:
          activations_.emplace_back(new ReLU6{});
          break;
        default:
          throw std::runtime_error("Invalid number of parameters");
        }
        break;

        // Randomized ReLU
      case activation::rrelu:
        switch (a.size()) {
        case 4:
          activations_.emplace_back(new RReLU{std::any_cast<double>(a[1]),
                                              std::any_cast<double>(a[2]),
                                              std::any_cast<bool>(a[3])});
          break;
        case 3:
          activations_.emplace_back(new RReLU{std::any_cast<double>(a[1]),
                                              std::any_cast<double>(a[2])});
          break;
        case 2:
          activations_.emplace_back(new RReLU{
              std::any_cast<torch::nn::functional::RReLUFuncOptions>(a[1])});
          break;
        case 1:
          activations_.emplace_back(new RReLU{});
          break;
        default:
          throw std::runtime_error("Invalid number of parameters");
        }
        break;

        // SELU
      case activation::selu:
        switch (a.size()) {
        case 2:
          try {
            activations_.emplace_back(new SELU{
                std::any_cast<torch::nn::functional::SELUFuncOptions>(a[1])});
          } catch (...) {
            activations_.emplace_back(new SELU{std::any_cast<bool>(a[1])});
          }
          break;
        case 1:
          activations_.emplace_back(new SELU{});
          break;
        default:
          throw std::runtime_error("Invalid number of parameters");
        }
        break;

        // Sigmoid
      case activation::sigmoid:
        switch (a.size()) {
        case 1:
          activations_.emplace_back(new Sigmoid{});
          break;
        default:
          throw std::runtime_error("Invalid number of parameters");
        }
        break;

        // SiLU
      case activation::silu:
        switch (a.size()) {
        case 1:
          activations_.emplace_back(new SiLU{});
          break;
        default:
          throw std::runtime_error("Invalid number of parameters");
        }
        break;

        // Softmax
      case activation::softmax:
        switch (a.size()) {
        case 2:
          try {
            activations_.emplace_back(new Softmax{
                std::any_cast<torch::nn::functional::SoftmaxFuncOptions>(
                    a[1])});
          } catch (...) {
            activations_.emplace_back(
                new Softmax{std::any_cast<int64_t>(a[1])});
          }
          break;
        default:
          throw std::runtime_error("Invalid number of parameters");
        }
        break;

        // Softmin
      case activation::softmin:
        switch (a.size()) {
        case 2:
          try {
            activations_.emplace_back(new Softmin{
                std::any_cast<torch::nn::functional::SoftminFuncOptions>(
                    a[1])});
          } catch (...) {
            activations_.emplace_back(
                new Softmin{std::any_cast<int64_t>(a[1])});
          }
          break;
        default:
          throw std::runtime_error("Invalid number of parameters");
        }
        break;

        // Softplus
      case activation::softplus:
        switch (a.size()) {
        case 3:
          activations_.emplace_back(new Softplus{std::any_cast<double>(a[1]),
                                                 std::any_cast<double>(a[2])});
          break;
        case 2:
          activations_.emplace_back(new Softplus{
              std::any_cast<torch::nn::functional::SoftplusFuncOptions>(a[1])});
          break;
        case 1:
          activations_.emplace_back(new Softplus{});
          break;
        default:
          throw std::runtime_error("Invalid number of parameters");
        }
        break;

        // Softshrink
      case activation::softshrink:
        switch (a.size()) {
        case 2:
          try {
            activations_.emplace_back(new Softshrink{
                std::any_cast<torch::nn::functional::SoftshrinkFuncOptions>(
                    a[1])});
          } catch (...) {
            activations_.emplace_back(
                new Softshrink{std::any_cast<double>(a[1])});
          }
          break;
        case 1:
          activations_.emplace_back(new Softshrink{});
          break;
        default:
          throw std::runtime_error("Invalid number of parameters");
        }
        break;

        // Softsign
      case activation::softsign:
        switch (a.size()) {
        case 1:
          activations_.emplace_back(new Softsign{});
          break;
        default:
          throw std::runtime_error("Invalid number of parameters");
        }
        break;

        // Tanh
      case activation::tanh:
        switch (a.size()) {
        case 1:
          activations_.emplace_back(new Tanh{});
          break;
        default:
          throw std::runtime_error("Invalid number of parameters");
        }
        break;

        // Tanhshrink
      case activation::tanhshrink:
        switch (a.size()) {
        case 1:
          activations_.emplace_back(new Tanhshrink{});
          break;
        default:
          throw std::runtime_error("Invalid number of parameters");
        }
        break;

        // Threshold
      case activation::threshold:
        switch (a.size()) {
        case 4:
          activations_.emplace_back(new Threshold{std::any_cast<double>(a[1]),
                                                  std::any_cast<double>(a[2]),
                                                  std::any_cast<bool>(a[3])});
          break;
        case 3:
          activations_.emplace_back(new Threshold{std::any_cast<double>(a[1]),
                                                  std::any_cast<double>(a[2])});
          break;
        case 2:
          activations_.emplace_back(new Threshold{
              std::any_cast<torch::nn::functional::ThresholdFuncOptions>(
                  a[1])});
          break;
        default:
          throw std::runtime_error("Invalid number of parameters");
        }
        break;

      default:
        throw std::runtime_error("Invalid activation function");
      }
  }

  /// @brief Forward evaluation
  torch::Tensor forward(torch::Tensor x) {
    torch::Tensor x_in = x.clone();

    // Standard feed-forward neural network
    for (auto [layer, activation] : utils::zip(layers_, activations_))
      x = activation->apply(layer->forward(x));

    return x;
  }

  /// @brief Writes the IgANet into a torch::serialize::OutputArchive object
  inline torch::serialize::OutputArchive &
  write(torch::serialize::OutputArchive &archive,
        const std::string &key = "iganet") const {
    assert(layers_.size() == activations_.size());

    archive.write(key + ".layers",
                  torch::full({1}, static_cast<int64_t>(layers_.size())));
    for (std::size_t i = 0; i < layers_.size(); ++i) {
      archive.write(
          key + ".layer[" + std::to_string(i) + "].in_features",
          torch::full({1}, (int64_t)layers_[i]->options.in_features()));
      archive.write(
          key + ".layer[" + std::to_string(i) + "].outputs_features",
          torch::full({1}, (int64_t)layers_[i]->options.out_features()));
      archive.write(key + ".layer[" + std::to_string(i) + "].bias",
                    torch::full({1}, (int64_t)layers_[i]->options.bias()));

      activations_[i]->write(archive, key + ".layer[" + std::to_string(i) +
                                          "].activation");
    }

    return archive;
  }

  /// @brief Reads the IgANet from a torch::serialize::InputArchive object
  inline torch::serialize::InputArchive &
  read(torch::serialize::InputArchive &archive,
       const std::string &key = "iganet") {
    torch::Tensor layers, in_features, outputs_features, bias, activation;

    auto options = iganet::Options<real_t>{};
    
    archive.read(key + ".layers", layers);
    for (int64_t i = 0; i < layers.item<int64_t>(); ++i) {
      archive.read(key + ".layer[" + std::to_string(i) + "].in_features",
                   in_features);
      archive.read(key + ".layer[" + std::to_string(i) + "].outputs_features",
                   outputs_features);
      archive.read(key + ".layer[" + std::to_string(i) + "].bias", bias);
      layers_.emplace_back(register_module(
          "layer[" + std::to_string(i) + "]",
          torch::nn::Linear(
              torch::nn::LinearOptions(in_features.item<int64_t>(),
                                       outputs_features.item<int64_t>())
              .bias(bias.item<bool>()))));
      layers_.back()->to(options.device(), options.dtype(), true);
      
      archive.read(key + ".layer[" + std::to_string(i) + "].activation.type",
                   activation);
      switch (static_cast<enum activation>(activation.item<int64_t>())) {
      case activation::none:
        activations_.emplace_back(new None{});
        break;
      case activation::batch_norm:
        activations_.emplace_back(
            new BatchNorm{torch::Tensor{}, torch::Tensor{}});
        break;
      case activation::celu:
        activations_.emplace_back(new CELU{});
        break;
      case activation::elu:
        activations_.emplace_back(new ELU{});
        break;
      case activation::gelu:
        activations_.emplace_back(new GELU{});
        break;
      case activation::glu:
        activations_.emplace_back(new GLU{});
        break;
      case activation::group_norm:
        activations_.emplace_back(new GroupNorm{0});
        break;
      case activation::gumbel_softmax:
        activations_.emplace_back(new GumbelSoftmax{});
        break;
      case activation::hardshrink:
        activations_.emplace_back(new Hardshrink{});
        break;
      case activation::hardsigmoid:
        activations_.emplace_back(new Hardsigmoid{});
        break;
      case activation::hardswish:
        activations_.emplace_back(new Hardswish{});
        break;
      case activation::hardtanh:
        activations_.emplace_back(new Hardtanh{});
        break;
      case activation::instance_norm:
        activations_.emplace_back(new InstanceNorm{});
        break;
      case activation::layer_norm:
        activations_.emplace_back(new LayerNorm{{}});
        break;
      case activation::leaky_relu:
        activations_.emplace_back(new LeakyReLU{});
        break;
      case activation::local_response_norm:
        activations_.emplace_back(new LocalResponseNorm{0});
        break;
      case activation::logsigmoid:
        activations_.emplace_back(new LogSigmoid{});
        break;
      case activation::logsoftmax:
        activations_.emplace_back(new LogSoftmax{0});
        break;
      case activation::mish:
        activations_.emplace_back(new Mish{});
        break;
      case activation::normalize:
        activations_.emplace_back(new Normalize{0, 0, 0});
        break;
      case activation::prelu:
        activations_.emplace_back(new PReLU{torch::Tensor{}});
        break;
      case activation::relu:
        activations_.emplace_back(new ReLU{});
        break;
      case activation::relu6:
        activations_.emplace_back(new ReLU6{});
        break;
      case activation::rrelu:
        activations_.emplace_back(new RReLU{});
        break;
      case activation::selu:
        activations_.emplace_back(new SELU{});
        break;
      case activation::sigmoid:
        activations_.emplace_back(new Sigmoid{});
        break;
      case activation::silu:
        activations_.emplace_back(new SiLU{});
        break;
      case activation::softmax:
        activations_.emplace_back(new Softmax{0});
        break;
      case activation::softmin:
        activations_.emplace_back(new Softmin{0});
        break;
      case activation::softplus:
        activations_.emplace_back(new Softplus{});
        break;
      case activation::softshrink:
        activations_.emplace_back(new Softshrink{});
        break;
      case activation::softsign:
        activations_.emplace_back(new Softsign{});
        break;
      case activation::tanh:
        activations_.emplace_back(new Tanh{});
        break;
      case activation::tanhshrink:
        activations_.emplace_back(new Tanhshrink{});
        break;
      case activation::threshold:
        activations_.emplace_back(new Threshold{0, 0});
        break;
      default:
        throw std::runtime_error("Invalid activation function");
      }
      activations_.back()->read(archive, key + ".layer[" + std::to_string(i) +
                                             "].activation");
    }
    return archive;
  }

  inline void pretty_print(std::ostream &os) const noexcept override {
    os << "(\n";

    int i = 0;
    for (const auto &activation : activations_)
      os << "activation[" << i++ << "] = " << *activation << "\n";
    os << ")\n";
  }

private:
  /// @brief Vector of linear layers
  std::vector<torch::nn::Linear> layers_;

  /// @brief Vector of activation functions
  std::vector<std::unique_ptr<iganet::ActivationFunction>> activations_;
};

/// @brief IgANetGenerator
///
/// @note: This class is normally generated by the TORCH_MODULE
/// macro. Since the latter cannot handle templated classes
/// correctly, we give the implementation explicitly
template <typename real_t>
class IgANetGenerator
    : public torch::nn::ModuleHolder<IgANetGeneratorImpl<real_t>> {

public:
  using torch::nn::ModuleHolder<IgANetGeneratorImpl<real_t>>::ModuleHolder;
  using Impl = IgANetGeneratorImpl<real_t>;
};

} // namespace iganet
