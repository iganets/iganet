/**
   @file include/iganet.hpp

   @brief Isogeometric analysis network

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#pragma once

#include <any>

#include <boundary.hpp>
#include <functionspace.hpp>
#include <igabase.hpp>
#include <layer.hpp>
#include <optimizer.hpp>
#include <utils/container.hpp>
#include <utils/fqn.hpp>
#include <utils/tuple.hpp>
#include <utils/zip.hpp>

namespace iganet {

/// @brief IgANetOptions
struct IgANetOptions {
  TORCH_ARG(int64_t, max_epoch) = 100;
  TORCH_ARG(int64_t, batch_size) = 1000;
  TORCH_ARG(double, min_loss) = 1e-4;
};

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

    archive.write(key + ".layers", torch::full({1}, (int64_t)layers_.size()));
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

  inline virtual void
  pretty_print(std::ostream &os = Log(log::info)) const noexcept override {
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

/// @brief IgANet
///
/// This class implements the core functionality of IgANets
template <typename Optimizer, typename GeometryMap, typename Variable,
          template <typename, typename> typename IgABase = ::iganet::IgABase>
requires OptimizerType<Optimizer> && FunctionSpaceType<GeometryMap> && FunctionSpaceType<Variable>
class IgANet : public IgABase<GeometryMap, Variable>,
               utils::Serializable,
               private utils::FullQualifiedName {
public:
  /// @brief Base type
  using Base = IgABase<GeometryMap, Variable>;

  /// @brief Type of the optimizer
  using optimizer_type = Optimizer;

  /// @brief Type of the optimizer options
  using optimizer_options_type = typename optimizer_options_type<Optimizer>::type;
  
protected:
  /// @brief IgANet generator
  IgANetGenerator<typename Base::value_type> net_;

  /// @brief Optimizer
  std::unique_ptr<optimizer_type> opt_;

  /// @brief Options
  IgANetOptions options_;

public:
  /// @brief Default constructor
  explicit IgANet(IgANetOptions defaults = {},
                  iganet::Options<typename Base::value_type> options =
                      iganet::Options<typename Base::value_type>{})
    : // Construct the base class
      Base(),
      // Construct the optimizer
      opt_(std::make_unique<optimizer_type>(net_->parameters())),
      // Set options
      options_(defaults) {}

  /// @brief Constructor: number of layers, activation functions, and
  /// number of spline coefficients (same for geometry map and
  /// variables)
  /// @{
  template <std::size_t NumCoeffs>
  IgANet(const std::vector<int64_t> &layers,
         const std::vector<std::vector<std::any>> &activations,
         std::array<int64_t, NumCoeffs> numCoeffs, IgANetOptions defaults = {},
         iganet::Options<typename Base::value_type> options =
             iganet::Options<typename Base::value_type>{})
      : IgANet(layers, activations, std::tuple{numCoeffs}, std::tuple{numCoeffs},
               defaults, options) {}

  template <std::size_t... NumCoeffs>
  IgANet(const std::vector<int64_t> &layers,
         const std::vector<std::vector<std::any>> &activations,
         std::tuple<std::array<int64_t, NumCoeffs>...> numCoeffs,
         IgANetOptions defaults = {},
         iganet::Options<typename Base::value_type> options =
             iganet::Options<typename Base::value_type>{})
      : IgANet(layers, activations, numCoeffs, numCoeffs, defaults, options) {}
  /// @}

  /// @brief Constructor: number of layers, activation functions, and
  /// number of spline coefficients (different for geometry map and
  /// variables)
  /// @{
  template <std::size_t GeometryMapNumCoeffs, std::size_t VariableNumCoeffs>
  IgANet(const std::vector<int64_t> &layers,
         const std::vector<std::vector<std::any>> &activations,
         std::array<int64_t, GeometryMapNumCoeffs> geometryMapNumCoeffs,
         std::array<int64_t, VariableNumCoeffs> variableNumCoeffs,
         IgANetOptions defaults = {},
         iganet::Options<typename Base::value_type> options =
             iganet::Options<typename Base::value_type>{})
      : IgANet(layers, activations, std::tuple{geometryMapNumCoeffs},
               std::tuple{variableNumCoeffs}, defaults, options) {}

  template <std::size_t... GeometryMapNumCoeffs,
            std::size_t... VariableNumCoeffs>
  IgANet(
      const std::vector<int64_t> &layers,
      const std::vector<std::vector<std::any>> &activations,
      std::tuple<std::array<int64_t, GeometryMapNumCoeffs>...>
          geometryMapNumCoeffs,
      std::tuple<std::array<int64_t, VariableNumCoeffs>...> variableNumCoeffs,
      IgANetOptions defaults = {},
      iganet::Options<typename Base::value_type> options =
          iganet::Options<typename Base::value_type>{})
      : // Construct the base class
        Base(geometryMapNumCoeffs, variableNumCoeffs, options),
        // Construct the deep neural network
        net_(utils::concat(std::vector<int64_t>{inputs(/* epoch */ 0).size(0)},
                           layers,
                           std::vector<int64_t>{Base::u_.as_tensor_size()}),
             activations, options),

        // Construct the optimizer
        opt_(std::make_unique<optimizer_type>(net_->parameters())),

        // Set options
        options_(defaults) {}

  /// @brief Returns a constant reference to the IgANet generator
  inline const IgANetGenerator<typename Base::value_type> &net() const {
    return net_;
  }

  /// @brief Returns a non-constant reference to the IgANet generator
  inline IgANetGenerator<typename Base::value_type> &net() { return net_; }

  /// @brief Returns a constant reference to the optimizer
  inline const optimizer_type &optimizer() const { return *opt_; }

  /// @brief Returns a non-constant reference to the optimizer
  inline optimizer_type &optimizer() { return *opt_; }
  
  /// @brief Resets the optimizer
  ///
  /// @param[in] resetOptions Flag to indicate whether the optimizer options should be resetted
  inline void optimizerReset(bool resetOptions = true) {
    if (resetOptions)
      opt_ = std::make_unique<optimizer_type>(net_->parameters());
    else {
      std::vector<optimizer_options_type> options;
      for (auto & group : opt_->param_groups())
        options.push_back(static_cast<optimizer_options_type&>(group.options()));
      opt_ = std::make_unique<optimizer_type>(net_->parameters());
      for (auto [group, options] : utils::zip(opt_->param_groups(), options))
        static_cast<optimizer_options_type&>(group.options()) = options;
    }
  }

  /// @brief Resets the optimizer
  inline void optimizerReset(const optimizer_options_type& optimizerOptions) {
    opt_ = std::make_unique<optimizer_type>(net_->parameters(), optimizerOptions);
  }

  /// @brief Returns a non-constant reference to the optimizer options
  inline optimizer_options_type &optimizerOptions(std::size_t param_group = 0) {
    if (param_group < opt_->param_groups().size())
      return static_cast<optimizer_options_type&>(opt_->param_groups()[param_group].options());
    else
      throw std::runtime_error("Index exceeds number of parameter groups");      
  }

  /// @brief Returns a constant reference to the optimizer options
  inline const optimizer_options_type &optimizerOptions(std::size_t param_group = 0) const {
    if (param_group < opt_->param_groups().size())
      return static_cast<optimizer_options_type&>(opt_->param_groups()[param_group].options());
    else
      throw std::runtime_error("Index exceeds number of parameter groups");      
  }

  /// @brief Resets the optimizer options
  inline void optimizerOptionsReset(const optimizer_options_type& options) {
    for (auto &group : opt_->param_groups())
      static_cast<optimizer_options_type&>(group.options()) = options;
  }

  /// @brief Resets the optimizer options
  inline void optimizerOptionsReset(optimizer_options_type&& options) {
    for (auto &group : opt_->param_groups())
      static_cast<optimizer_options_type&>(group.options()) = options;
  }

  /// @brief Resets the optimizer options
  inline void optimizerOptionsReset(const optimizer_options_type& options, std::size_t param_group) {
    if (param_group < opt_->param_groups().size())
      static_cast<optimizer_options_type&>(opt_->param_group().options()) = options;
    else
      throw std::runtime_error("Index exceeds number of parameter groups");      
  }

  /// @brief Resets the optimizer options
  inline void optimizerOptionsReset(optimizer_options_type&& options, std::size_t param_group) {
    if (param_group < opt_->param_groups().size())
      static_cast<optimizer_options_type&>(opt_->param_group().options()) = options;
    else
      throw std::runtime_error("Index exceeds number of parameter groups");      
  }
  
  /// @brief Returns a constant reference to the options structure
  inline const auto &options() const { return options_; }

  /// @brief Returns a non-constant reference to the options structure
  inline auto &options() { return options_; }

  /// @brief Returns the network inputs
  ///
  /// In the default implementation the inputs are the controll
  /// points of the geometry and the reference spline objects. This
  /// behavior can be changed by overriding this virtual function in
  /// a derived class.
  virtual torch::Tensor inputs(int64_t epoch) const {
    if constexpr (Base::has_GeometryMap && Base::has_RefData)
      return torch::cat({Base::G_.as_tensor(), Base::f_.as_tensor()});
    else if constexpr (Base::has_GeometryMap && !Base::has_RefData)
      return Base::G_.as_tensor();
    else if constexpr (!Base::has_GeometryMap && Base::has_RefData)
      return Base::f_.as_tensor();
    else
      return torch::empty({0});
  }

  /// @brief Initializes epoch
  virtual bool epoch(int64_t) = 0;

  /// @brief Computes the loss function
  virtual torch::Tensor loss(const torch::Tensor &, int64_t) = 0;

  /// @brief Trains the IgANet
  virtual void train(
#ifdef IGANET_WITH_MPI
      c10::intrusive_ptr<c10d::ProcessGroupMPI> pg =
          c10d::ProcessGroupMPI::createProcessGroupMPI()
#endif
  ) {
    torch::Tensor inputs, outputs, loss;
    typename Base::value_type previous_loss(-1.0);

    // Loop over epochs
    for (int64_t epoch = 0; epoch != options_.max_epoch(); ++epoch) {

      // Update epoch and inputs
      if (this->epoch(epoch))
        inputs = this->inputs(epoch);

      auto closure = [&]() {
        // Reset gradients
        net_->zero_grad();

        // Execute the model on the inputs
        outputs = net_->forward(inputs);

        // Compute the loss value
        loss = this->loss(outputs, epoch);

        // Compute gradients of the loss w.r.t. the model parameters
        loss.backward({}, true, false);

        return loss;
      };

#ifdef IGANET_WITH_MPI
      // Averaging the gradients of the parameters in all the processors
      // Note: This may lag behind DistributedDataParallel (DDP) in performance
      // since this synchronizes parameters after backward pass while DDP
      // overlaps synchronizing parameters and computing gradients in backward
      // pass
      std::vector<c10::intrusive_ptr<::c10d::Work>> works;
      for (auto &param : net_->named_parameters()) {
        std::vector<torch::Tensor> tmp = {param.value().grad()};
        works.emplace_back(pg->allreduce(tmp));
      }

      waitWork(pg, works);

      for (auto &param : net_->named_parameters()) {
        param.value().grad().data() =
            param.value().grad().data() / pg->getSize();
      }
#endif

      // Update the parameters based on the calculated gradients
      opt_->step(closure);

      typename Base::value_type current_loss = loss.template item<typename Base::value_type>();
      Log(log::verbose) << "Epoch " << std::to_string(epoch) << ": "
                        << current_loss
                        << std::endl;

      if (current_loss <
          options_.min_loss()) {
        Log(log::info) << "Total epochs: " << epoch << ", loss: "
                       << current_loss
                       << std::endl;
        break;
      }

      if (current_loss == previous_loss || std::abs(current_loss-previous_loss) < previous_loss/10) {
        Log(log::info) << "Total epochs: " << epoch << ", loss: "
                       << current_loss
                       << std::endl;
        break;
      }

      if (loss.isnan().template item<bool>()) {
        Log(log::info) << "Total epochs: " << epoch << ", loss: "
        << current_loss
        << std::endl;
        break;
      }
      previous_loss = current_loss;
    }
  }

  /// @brief Trains the IgANet
  template <typename DataLoader>
  void train(DataLoader &loader
#ifdef IGANET_WITH_MPI
             ,
             c10::intrusive_ptr<c10d::ProcessGroupMPI> pg =
                 c10d::ProcessGroupMPI::createProcessGroupMPI()
#endif
  ) {
    torch::Tensor inputs, outputs, loss;
    typename Base::value_type previous_loss(-1.0);

    // Loop over epochs
    for (int64_t epoch = 0; epoch != options_.max_epoch(); ++epoch) {

      typename Base::value_type Loss(0);

      for (auto &batch : loader) {
        inputs = batch.data;

        if (inputs.dim() > 0) {
          if constexpr (Base::has_GeometryMap && Base::has_RefData) {
            Base::G_.from_tensor(
                inputs.slice(1, 0, Base::G_.as_tensor_size()).t());
            Base::f_.from_tensor(inputs
                                     .slice(1, Base::G_.as_tensor_size(),
                                            Base::G_.as_tensor_size() +
                                                Base::f_.as_tensor_size())
                                     .t());
          } else if constexpr (Base::has_GeometryMap && !Base::has_RefData)
            Base::G_.from_tensor(
                inputs.slice(1, 0, Base::G_.as_tensor_size()).t());
          else if constexpr (!Base::has_GeometryMap && Base::has_RefData)
            Base::f_.from_tensor(
                inputs.slice(1, 0, Base::f_.as_tensor_size()).t());

        } else {
          if constexpr (Base::has_GeometryMap && Base::has_RefData) {
            Base::G_.from_tensor(
                inputs.slice(1, 0, Base::G_.as_tensor_size()).flatten());
            Base::f_.from_tensor(inputs
                                     .slice(1, Base::G_.as_tensor_size(),
                                            Base::G_.as_tensor_size() +
                                                Base::f_.as_tensor_size())
                                     .flatten());
          } else if constexpr (Base::has_GeometryMap && !Base::has_RefData)
            Base::G_.from_tensor(
                inputs.slice(1, 0, Base::G_.as_tensor_size()).flatten());
          else if constexpr (!Base::has_GeometryMap && Base::has_RefData)
            Base::f_.from_tensor(
                inputs.slice(1, 0, Base::f_.as_tensor_size()).flatten());
        }

        this->epoch(epoch);

        auto closure = [&]() {
          // Reset gradients
          net_->zero_grad();

          // Execute the model on the inputs
          outputs = net_->forward(inputs);

          // Compute the loss value
          loss = this->loss(outputs, epoch);

          // Compute gradients of the loss w.r.t. the model parameters
          loss.backward({}, true, false);

          return loss;
        };

        // Update the parameters based on the calculated gradients
        opt_->step(closure);

        Loss += loss.template item<typename Base::value_type>();
      }

      Log(log::verbose) << "Epoch " << std::to_string(epoch) << ": " << Loss
                        << std::endl;

      if (Loss < options_.min_loss()) {
        Log(log::info) << "Total epochs: " << epoch << ", loss: " << Loss
                       << std::endl;
        break;
      }

      if (Loss == previous_loss) {
        Log(log::info) << "Total epochs: " << epoch << ", loss: " << Loss
                       << std::endl;
        break;
      }
      previous_loss = Loss;

      if (epoch == options_.max_epoch() - 1)
        Log(log::warning) << "Total epochs: " << epoch << ", loss: " << Loss
                          << std::endl;
    }
  }

  /// @brief Evaluate IgANet
  void eval() {
    torch::Tensor inputs = this->inputs(0);
    torch::Tensor outputs = net_->forward(inputs);
    Base::u_.from_tensor(outputs);
  }

  /// @brief Returns the IgANet object as JSON object
  inline virtual nlohmann::json to_json() const override {
    return "Not implemented yet";
  }

  /// @brief Returns a constant reference to the parameters of the IgANet object
  inline std::vector<torch::Tensor> parameters() const noexcept {
    return net_->parameters();
  }

  /// @brief Returns a constant reference to the named parameters of the IgANet
  /// object
  inline torch::OrderedDict<std::string, torch::Tensor>
  named_parameters() const noexcept {
    return net_->named_parameters();
  }

  /// @brief Returns the total number of parameters of the IgANet object
  inline std::size_t nparameters() const noexcept {
    std::size_t result = 0;
    for (const auto &param : this->parameters()) {
      result += param.numel();
    }
    return result;
  }

  /// @brief Returns a string representation of the IgANet object
  inline virtual void
  pretty_print(std::ostream &os = Log(log::info)) const noexcept override {
    os << name() << "(\n"
       << "net = " << net_ << "\n";
    if constexpr (Base::has_GeometryMap)
      os << "G = " << Base::G_ << "\n";
    if constexpr (Base::has_RefData)
      os << "f = " << Base::f_ << "\n";
    if constexpr (Base::has_Solution)
      os << "u = " << Base::u_ << "\n)";
  }

  /// @brief Saves the IgANet to file
  inline void save(const std::string &filename,
                   const std::string &key = "iganet") const {
    torch::serialize::OutputArchive archive;
    write(archive, key).save_to(filename);
  }

  /// @brief Loads the IgANet from file
  inline void load(const std::string &filename,
                   const std::string &key = "iganet") {
    torch::serialize::InputArchive archive;
    archive.load_from(filename);
    read(archive, key);
  }

  /// @brief Writes the IgANet into a torch::serialize::OutputArchive object
  inline torch::serialize::OutputArchive &
  write(torch::serialize::OutputArchive &archive,
        const std::string &key = "iganet") const {
    if constexpr (Base::has_GeometryMap)
      Base::G_.write(archive, key + ".geo");
    if constexpr (Base::has_RefData)
      Base::f_.write(archive, key + ".ref");
    if constexpr (Base::has_Solution)
      Base::u_.write(archive, key + ".out");

    net_->write(archive, key + ".net");
    torch::serialize::OutputArchive archive_net;
    net_->save(archive_net);
    archive.write(key + ".net.data", archive_net);

    torch::serialize::OutputArchive archive_opt;
    opt_->save(archive_opt);
    archive.write(key + ".opt", archive_opt);

    return archive;
  }

  /// @brief Loads the IgANet from a torch::serialize::InputArchive object
  inline torch::serialize::InputArchive &
  read(torch::serialize::InputArchive &archive,
       const std::string &key = "iganet") {
    if constexpr (Base::has_GeometryMap)
      Base::G_.read(archive, key + ".geo");
    if constexpr (Base::has_RefData)
      Base::f_.read(archive, key + ".ref");
    if constexpr (Base::has_Solution)
      Base::u_.read(archive, key + ".out");

    net_->read(archive, key + ".net");
    torch::serialize::InputArchive archive_net;
    archive.read(key + ".net.data", archive_net);
    net_->load(archive_net);

    opt_->add_parameters(net_->parameters());
    torch::serialize::InputArchive archive_opt;
    archive.read(key + ".opt", archive_opt);
    opt_->load(archive_opt);

    return archive;
  }

  /// @brief Returns true if both IgANet objects are the same
  bool operator==(const IgANet &other) const {
    bool result(true);

    if constexpr (Base::has_GeometryMap)
      result *= (Base::G_ == other.G());
    if constexpr (Base::has_RefData)
      result *= (Base::f_ == other.f());
    if constexpr (Base::has_Solution)
      result *= (Base::u_ == other.u());

    return result;
  }

  /// @brief Returns true if both IgANet objects are different
  bool operator!=(const IgANet &other) const { return *this != other; }

#ifdef IGANET_WITH_MPI
private:
  /// @brief Waits for all work processes
  static void waitWork(c10::intrusive_ptr<c10d::ProcessGroupMPI> pg,
                       std::vector<c10::intrusive_ptr<c10d::Work>> works) {
    for (auto &work : works) {
      try {
        work->wait();
      } catch (const std::exception &ex) {
        Log(log::error) << "Exception received during waitWork: " << ex.what()
                        << std::endl;
        pg->abort();
      }
    }
  }
#endif
};

/// @brief Print (as string) a IgANet object
  template <typename Optimizer, typename GeometryMap, typename Variable>
  requires OptimizerType<Optimizer> && FunctionSpaceType<GeometryMap> && FunctionSpaceType<Variable>
inline std::ostream &
operator<<(std::ostream &os,
           const IgANet<Optimizer, GeometryMap, Variable> &obj) {
  obj.pretty_print(os);
  return os;
}

/// @brief IgANetCustomizable
///
/// This class implements a customizable variant of IgANets that
/// provides types and attributes for precomputing indices and basis
/// functions
  template <typename GeometryMap, typename Variable>
  requires FunctionSpaceType<GeometryMap> && FunctionSpaceType<Variable>
  class IgANetCustomizable {
public:
  /// @brief Type of the knot indices of the geometry map in the interior
  using geometryMap_interior_knot_indices_type =
      decltype(std::declval<GeometryMap>()
                   .template find_knot_indices<functionspace::interior>(
                       std::declval<typename GeometryMap::eval_type>()));

  /// @brief Type of the knot indices of the geometry map at the boundary
  using geometryMap_boundary_knot_indices_type =
      decltype(std::declval<GeometryMap>()
                   .template find_knot_indices<functionspace::boundary>(
                       std::declval<
                           typename GeometryMap::boundary_eval_type>()));

  /// @brief Type of the knot indices of the variables in the interior
  using variable_interior_knot_indices_type =
      decltype(std::declval<Variable>()
                   .template find_knot_indices<functionspace::interior>(
                       std::declval<typename Variable::eval_type>()));

  /// @brief Type of the knot indices of boundary_eval_type type at the boundary
  using variable_boundary_knot_indices_type =
      decltype(std::declval<Variable>()
                   .template find_knot_indices<functionspace::boundary>(
                       std::declval<typename Variable::boundary_eval_type>()));

  /// @brief Type of the coefficient indices of geometry type in the interior
  using geometryMap_interior_coeff_indices_type =
      decltype(std::declval<GeometryMap>()
                   .template find_coeff_indices<functionspace::interior>(
                       std::declval<typename GeometryMap::eval_type>()));

  /// @brief Type of the coefficient indices of geometry type at the boundary
  using geometryMap_boundary_coeff_indices_type =
      decltype(std::declval<GeometryMap>()
                   .template find_coeff_indices<functionspace::boundary>(
                       std::declval<
                           typename GeometryMap::boundary_eval_type>()));

  /// @brief Type of the coefficient indices of variable type in the interior
  using variable_interior_coeff_indices_type =
      decltype(std::declval<Variable>()
                   .template find_coeff_indices<functionspace::interior>(
                       std::declval<typename Variable::eval_type>()));

  /// @brief Type of the coefficient indices of variable type at the boundary
  using variable_boundary_coeff_indices_type =
      decltype(std::declval<Variable>()
                   .template find_coeff_indices<functionspace::boundary>(
                       std::declval<typename Variable::boundary_eval_type>()));
};

/// @brief IgANet2
///
/// This class implements the core functionality of IgANets
template <typename Optimizer, typename Inputs, typename Outputs, typename CollPts = void>
requires OptimizerType<Optimizer>
class IgANet2 : public IgABase2<Inputs, Outputs, CollPts>,
                utils::Serializable,
                private utils::FullQualifiedName {
public:
  /// @brief Base type
    using Base = IgABase2<Inputs, Outputs, CollPts>;

  /// @brief Type of the optimizer
  using optimizer_type = Optimizer;

  /// @brief Type of the optimizer options
  using optimizer_options_type = typename optimizer_options_type<Optimizer>::type;
  
protected:
  /// @brief IgANet generator
  IgANetGenerator<typename Base::value_type> net_;

  /// @brief Optimizer
  std::unique_ptr<optimizer_type> opt_;

  /// @brief Options
  IgANetOptions options_;

public:
  /// @brief Default constructor
  explicit IgANet2(IgANetOptions defaults = {},
                  iganet::Options<typename Base::value_type> options =
                      iganet::Options<typename Base::value_type>{})
    : // Construct the base class
      Base(),
      // Construct the optimizer
      opt_(std::make_unique<optimizer_type>(net_->parameters())),
      // Set options
      options_(defaults) {}

  /// @brief Constructor: number of layers, activation functions, and
  /// number of spline coefficients (same for all inputs and outputs)
  template <typename NumCoeffs>
  IgANet2(const std::vector<int64_t> &layers,
          const std::vector<std::vector<std::any>> &activations,
          const NumCoeffs &numCoeffs,
          enum init init = init::greville,
          IgANetOptions defaults = {},
          iganet::Options<typename Base::value_type> options =
          iganet::Options<typename Base::value_type>{})
    : IgANet2(layers, activations, numCoeffs, numCoeffs, init, defaults, options)
  {}

  /// @brief Constructor: number of layers, activation functions, and
  /// number of spline coefficients (same for all inputs and outputs)
  template <typename NumCoeffsInputs, typename NumCoeffsOutputs>
  IgANet2(const std::vector<int64_t> &layers,
          const std::vector<std::vector<std::any>> &activations,
          const NumCoeffsInputs &numCoeffsInputs,
          const NumCoeffsOutputs &numCoeffsOutputs,
          enum init init = init::greville,
          IgANetOptions defaults = {},
          iganet::Options<typename Base::value_type> options =
          iganet::Options<typename Base::value_type>{})
    : // Construct the base class
    Base(numCoeffsInputs, numCoeffsOutputs, init, options),
    // Construct the deep neural network
    net_(utils::concat(std::vector<int64_t>{inputs(/* epoch */ 0).size(0)},
                       layers,
                       std::vector<int64_t>{outputs(/* epoch */ 0).size(0)}),
         activations, options),
    
    // Construct the optimizer
    opt_(std::make_unique<optimizer_type>(net_->parameters())),
    
    // Set options
    options_(defaults) {} 
  
  /// @brief Returns a constant reference to the IgANet generator
  inline const IgANetGenerator<typename Base::value_type> &net() const {
    return net_;
  }

  /// @brief Returns a non-constant reference to the IgANet generator
  inline IgANetGenerator<typename Base::value_type> &net() { return net_; }

  /// @brief Returns a constant reference to the optimizer
  inline const optimizer_type &optimizer() const { return *opt_; }

  /// @brief Returns a non-constant reference to the optimizer
  inline optimizer_type &optimizer() { return *opt_; }
  
  /// @brief Resets the optimizer
  ///
  /// @param[in] resetOptions Flag to indicate whether the optimizer options should be resetted
  inline void optimizerReset(bool resetOptions = true) {
    if (resetOptions)
      opt_ = std::make_unique<optimizer_type>(net_->parameters());
    else {
      std::vector<optimizer_options_type> options;
      for (auto & group : opt_->param_groups())
        options.push_back(static_cast<optimizer_options_type&>(group.options()));
      opt_ = std::make_unique<optimizer_type>(net_->parameters());
      for (auto [group, options] : utils::zip(opt_->param_groups(), options))
        static_cast<optimizer_options_type&>(group.options()) = options;
    }
  }

  /// @brief Resets the optimizer
  inline void optimizerReset(const optimizer_options_type& optimizerOptions) {
    opt_ = std::make_unique<optimizer_type>(net_->parameters(), optimizerOptions);
  }

  /// @brief Returns a non-constant reference to the optimizer options
  inline optimizer_options_type &optimizerOptions(std::size_t param_group = 0) {
    if (param_group < opt_->param_groups().size())
      return static_cast<optimizer_options_type&>(opt_->param_groups()[param_group].options());
    else
      throw std::runtime_error("Index exceeds number of parameter groups");      
  }

  /// @brief Returns a constant reference to the optimizer options
  inline const optimizer_options_type &optimizerOptions(std::size_t param_group = 0) const {
    if (param_group < opt_->param_groups().size())
      return static_cast<optimizer_options_type&>(opt_->param_groups()[param_group].options());
    else
      throw std::runtime_error("Index exceeds number of parameter groups");      
  }

  /// @brief Resets the optimizer options
  inline void optimizerOptionsReset(const optimizer_options_type& options) {
    for (auto &group : opt_->param_groups())
      static_cast<optimizer_options_type&>(group.options()) = options;
  }

  /// @brief Resets the optimizer options
  inline void optimizerOptionsReset(optimizer_options_type&& options) {
    for (auto &group : opt_->param_groups())
      static_cast<optimizer_options_type&>(group.options()) = options;
  }

  /// @brief Resets the optimizer options
  inline void optimizerOptionsReset(const optimizer_options_type& options, std::size_t param_group) {
    if (param_group < opt_->param_groups().size())
      static_cast<optimizer_options_type&>(opt_->param_group().options()) = options;
    else
      throw std::runtime_error("Index exceeds number of parameter groups");      
  }

  /// @brief Resets the optimizer options
  inline void optimizerOptionsReset(optimizer_options_type&& options, std::size_t param_group) {
    if (param_group < opt_->param_groups().size())
      static_cast<optimizer_options_type&>(opt_->param_group().options()) = options;
    else
      throw std::runtime_error("Index exceeds number of parameter groups");      
  }
  
  /// @brief Returns a constant reference to the options structure
  inline const auto &options() const { return options_; }

  /// @brief Returns a non-constant reference to the options structure
  inline auto &options() { return options_; }

  /// @brief Returns the network inputs as tensor
  virtual torch::Tensor inputs(int64_t epoch) const {
    return utils::cat_tuple(Base::inputs_, [](const auto& obj){ return obj.as_tensor(); });
  }

  /// @brief Returns the network outputs as tensor
  virtual torch::Tensor outputs(int64_t epoch) const {
    return utils::cat_tuple(Base::outputs_, [](const auto& obj){ return obj.as_tensor(); });
  }

    /// @brief Initializes epoch
  virtual bool epoch(int64_t) = 0;

  /// @brief Computes the loss function
  virtual torch::Tensor loss(const torch::Tensor &, int64_t) = 0;

  /// @brief Trains the IgANet
  virtual void train(
#ifdef IGANET_WITH_MPI
      c10::intrusive_ptr<c10d::ProcessGroupMPI> pg =
          c10d::ProcessGroupMPI::createProcessGroupMPI()
#endif
  ) {
    torch::Tensor inputs, outputs, loss;
    typename Base::value_type previous_loss(-1.0);

    // Loop over epochs
    for (int64_t epoch = 0; epoch != options_.max_epoch(); ++epoch) {

      // Update epoch and inputs
      if (this->epoch(epoch))
        inputs = this->inputs(epoch);

      auto closure = [&]() {
        // Reset gradients
        net_->zero_grad();

        // Execute the model on the inputs
        outputs = net_->forward(inputs);

        // Compute the loss value
        loss = this->loss(outputs, epoch);

        // Compute gradients of the loss w.r.t. the model parameters
        loss.backward({}, true, false);

        return loss;
      };

#ifdef IGANET_WITH_MPI
      // Averaging the gradients of the parameters in all the processors
      // Note: This may lag behind DistributedDataParallel (DDP) in performance
      // since this synchronizes parameters after backward pass while DDP
      // overlaps synchronizing parameters and computing gradients in backward
      // pass
      std::vector<c10::intrusive_ptr<::c10d::Work>> works;
      for (auto &param : net_->named_parameters()) {
        std::vector<torch::Tensor> tmp = {param.value().grad()};
        works.emplace_back(pg->allreduce(tmp));
      }

      waitWork(pg, works);

      for (auto &param : net_->named_parameters()) {
        param.value().grad().data() =
            param.value().grad().data() / pg->getSize();
      }
#endif

      // Update the parameters based on the calculated gradients
      opt_->step(closure);

      typename Base::value_type current_loss = loss.template item<typename Base::value_type>();
      Log(log::verbose) << "Epoch " << std::to_string(epoch) << ": "
                        << current_loss
                        << std::endl;

      if (current_loss <
          options_.min_loss()) {
        Log(log::info) << "Total epochs: " << epoch << ", loss: "
                       << current_loss
                       << std::endl;
        break;
      }

      if (current_loss == previous_loss || std::abs(current_loss-previous_loss) < previous_loss/10) {
        Log(log::info) << "Total epochs: " << epoch << ", loss: "
                       << current_loss
                       << std::endl;
        break;
      }

      if (loss.isnan().template item<bool>()) {
        Log(log::info) << "Total epochs: " << epoch << ", loss: "
        << current_loss
        << std::endl;
        break;
      }
      previous_loss = current_loss;
    }
  }

    /// @brief Trains the IgANet
  template <typename DataLoader>
  void train(DataLoader &loader
#ifdef IGANET_WITH_MPI
             ,
             c10::intrusive_ptr<c10d::ProcessGroupMPI> pg =
                 c10d::ProcessGroupMPI::createProcessGroupMPI()
#endif
  ) {
    torch::Tensor inputs, outputs, loss;
    typename Base::value_type previous_loss(-1.0);

    // Loop over epochs
    for (int64_t epoch = 0; epoch != options_.max_epoch(); ++epoch) {

      typename Base::value_type Loss(0);

      for (auto &batch : loader) {
        inputs = batch.data;

        if (inputs.dim() > 0) {
          // if constexpr (Base::has_GeometryMap && Base::has_RefData) {
          //   Base::G_.from_tensor(
          //       inputs.slice(1, 0, Base::G_.as_tensor_size()).t());
          //   Base::f_.from_tensor(inputs
          //                            .slice(1, Base::G_.as_tensor_size(),
          //                                   Base::G_.as_tensor_size() +
          //                                       Base::f_.as_tensor_size())
          //                            .t());
          // } else if constexpr (Base::has_GeometryMap && !Base::has_RefData)
          //   Base::G_.from_tensor(
          //       inputs.slice(1, 0, Base::G_.as_tensor_size()).t());
          // else if constexpr (!Base::has_GeometryMap && Base::has_RefData)
          //   Base::f_.from_tensor(
          //       inputs.slice(1, 0, Base::f_.as_tensor_size()).t());

        } else {
          // if constexpr (Base::has_GeometryMap && Base::has_RefData) {
          //   Base::G_.from_tensor(
          //       inputs.slice(1, 0, Base::G_.as_tensor_size()).flatten());
          //   Base::f_.from_tensor(inputs
          //                            .slice(1, Base::G_.as_tensor_size(),
          //                                   Base::G_.as_tensor_size() +
          //                                       Base::f_.as_tensor_size())
          //                            .flatten());
          // } else if constexpr (Base::has_GeometryMap && !Base::has_RefData)
          //   Base::G_.from_tensor(
          //       inputs.slice(1, 0, Base::G_.as_tensor_size()).flatten());
          // else if constexpr (!Base::has_GeometryMap && Base::has_RefData)
          //   Base::f_.from_tensor(
          //       inputs.slice(1, 0, Base::f_.as_tensor_size()).flatten());
        }

        this->epoch(epoch);

        auto closure = [&]() {
          // Reset gradients
          net_->zero_grad();

          // Execute the model on the inputs
          outputs = net_->forward(inputs);

          // Compute the loss value
          loss = this->loss(outputs, epoch);

          // Compute gradients of the loss w.r.t. the model parameters
          loss.backward({}, true, false);

          return loss;
        };

        // Update the parameters based on the calculated gradients
        opt_->step(closure);

        Loss += loss.template item<typename Base::value_type>();
      }

      Log(log::verbose) << "Epoch " << std::to_string(epoch) << ": " << Loss
                        << std::endl;

      if (Loss < options_.min_loss()) {
        Log(log::info) << "Total epochs: " << epoch << ", loss: " << Loss
                       << std::endl;
        break;
      }

      if (Loss == previous_loss) {
        Log(log::info) << "Total epochs: " << epoch << ", loss: " << Loss
                       << std::endl;
        break;
      }
      previous_loss = Loss;

      if (epoch == options_.max_epoch() - 1)
        Log(log::warning) << "Total epochs: " << epoch << ", loss: " << Loss
                          << std::endl;
    }
  }

  /// @brief Evaluate IgANet
  void eval() {
    torch::Tensor inputs = this->inputs(0);
    torch::Tensor outputs = net_->forward(inputs);
    Base::outputs_.from_tensor(outputs);
  }

  /// @brief Returns the IgANet object as JSON object
  inline virtual nlohmann::json to_json() const override {
    return "Not implemented yet";
  }

  /// @brief Returns a constant reference to the parameters of the IgANet object
  inline std::vector<torch::Tensor> parameters() const noexcept {
    return net_->parameters();
  }

  /// @brief Returns a constant reference to the named parameters of the IgANet
  /// object
  inline torch::OrderedDict<std::string, torch::Tensor>
  named_parameters() const noexcept {
    return net_->named_parameters();
  }

  /// @brief Returns the total number of parameters of the IgANet object
  inline std::size_t nparameters() const noexcept {
    std::size_t result = 0;
    for (const auto &param : this->parameters()) {
      result += param.numel();
    }
    return result;
  }

  /// @brief Returns a string representation of the IgANet object
  inline virtual void
  pretty_print(std::ostream &os = Log(log::info)) const noexcept override {
    os << name() << "(\n"
       << "net = " << net_ << "\n";
    // if constexpr (Base::has_GeometryMap)
    //   os << "G = " << Base::G_ << "\n";
    // if constexpr (Base::has_RefData)
    //   os << "f = " << Base::f_ << "\n";
    // if constexpr (Base::has_Solution)
    //   os << "u = " << Base::u_ << "\n)";
  }

  /// @brief Saves the IgANet to file
  inline void save(const std::string &filename,
                   const std::string &key = "iganet") const {
    torch::serialize::OutputArchive archive;
    write(archive, key).save_to(filename);
  }

  /// @brief Loads the IgANet from file
  inline void load(const std::string &filename,
                   const std::string &key = "iganet") {
    torch::serialize::InputArchive archive;
    archive.load_from(filename);
    read(archive, key);
  }

  /// @brief Writes the IgANet into a torch::serialize::OutputArchive object
  inline torch::serialize::OutputArchive &
  write(torch::serialize::OutputArchive &archive,
        const std::string &key = "iganet") const {
    // if constexpr (Base::has_GeometryMap)
    //   Base::G_.write(archive, key + ".geo");
    // if constexpr (Base::has_RefData)
    //   Base::f_.write(archive, key + ".ref");
    // if constexpr (Base::has_Solution)
    //   Base::u_.write(archive, key + ".out");

    net_->write(archive, key + ".net");
    torch::serialize::OutputArchive archive_net;
    net_->save(archive_net);
    archive.write(key + ".net.data", archive_net);

    torch::serialize::OutputArchive archive_opt;
    opt_->save(archive_opt);
    archive.write(key + ".opt", archive_opt);

    return archive;
  }

  /// @brief Loads the IgANet from a torch::serialize::InputArchive object
  inline torch::serialize::InputArchive &
  read(torch::serialize::InputArchive &archive,
       const std::string &key = "iganet") {
    // if constexpr (Base::has_GeometryMap)
    //   Base::G_.read(archive, key + ".geo");
    // if constexpr (Base::has_RefData)
    //   Base::f_.read(archive, key + ".ref");
    // if constexpr (Base::has_Solution)
    //   Base::u_.read(archive, key + ".out");

    net_->read(archive, key + ".net");
    torch::serialize::InputArchive archive_net;
    archive.read(key + ".net.data", archive_net);
    net_->load(archive_net);

    opt_->add_parameters(net_->parameters());
    torch::serialize::InputArchive archive_opt;
    archive.read(key + ".opt", archive_opt);
    opt_->load(archive_opt);

    return archive;
  }

  /// @brief Returns true if both IgANet objects are the same
  bool operator==(const IgANet2 &other) const {
    bool result(true);

    // if constexpr (Base::has_GeometryMap)
    //   result *= (Base::G_ == other.G());
    // if constexpr (Base::has_RefData)
    //   result *= (Base::f_ == other.f());
    // if constexpr (Base::has_Solution)
    //   result *= (Base::u_ == other.u());

    return result;
  }

  /// @brief Returns true if both IgANet objects are different
  bool operator!=(const IgANet2 &other) const { return *this != other; }

#ifdef IGANET_WITH_MPI
private:
  /// @brief Waits for all work processes
  static void waitWork(c10::intrusive_ptr<c10d::ProcessGroupMPI> pg,
                       std::vector<c10::intrusive_ptr<c10d::Work>> works) {
    for (auto &work : works) {
      try {
        work->wait();
      } catch (const std::exception &ex) {
        Log(log::error) << "Exception received during waitWork: " << ex.what()
                        << std::endl;
        pg->abort();
      }
    }
  }
#endif
};

/// @brief Print (as string) a IgANet2 object
template <typename Optimizer, typename Inputs, typename Outputs, typename CollPts>
requires OptimizerType<Optimizer>
inline std::ostream &
operator<<(std::ostream &os,
           const IgANet2<Optimizer, Inputs, Outputs, CollPts> &obj) {
  //  obj.pretty_print(os);
  return os;
}

/// @brief IgANetCustomizable2
///
/// This class implements a customizable variant of IgANets2 that
/// provides types and attributes for precomputing indices and basis
/// functions
///
/// @{
template <typename, typename, typename = void>
class IgANetCustomizable2;

template <detail::HasAsTensor... Inputs,
          detail::HasAsTensor... Outputs>
class IgANetCustomizable2<std::tuple<Inputs...>,
                    std::tuple<Outputs...>, void> {
public:
  /// @brief Type of the knot indices of the inputs in the interior
  using inputs_interior_knot_indices_type =
    std::tuple<decltype(std::declval<Inputs>()
                   .template find_knot_indices<functionspace::interior>(
                                                                        std::declval<typename Inputs::eval_type>()))...>;

  /// @brief Type of the knot indices of the inputs at the boundary
  using inputs_boundary_knot_indices_type =
    std::tuple<decltype(std::declval<Inputs>()
                        .template find_knot_indices<functionspace::boundary>(
                                                                             std::declval<
                                                                             typename Inputs::boundary_eval_type>()))...>;

  /// @brief Type of the knot indices of the outputs in the interior
  using outputs_interior_knot_indices_type =
    std::tuple<decltype(std::declval<Outputs>()
                   .template find_knot_indices<functionspace::interior>(
                                                                        std::declval<typename Outputs::eval_type>()))...>;

  /// @brief Type of the knot indices of the outputs at the boundary
  using outputs_boundary_knot_indices_type =
    std::tuple<decltype(std::declval<Outputs>()
                        .template find_knot_indices<functionspace::boundary>(
                                                                             std::declval<
                                                                             typename Outputs::boundary_eval_type>()))...>;

  /// @brief Type of the coefficient indices of the inputs in the interior
  using inputs_interior_coeff_indices_type =
    std::tuple<decltype(std::declval<Inputs>()
                        .template find_coeff_indices<functionspace::interior>(
                                                                              std::declval<typename Inputs::eval_type>()))...>;

  /// @brief Type of the coefficient indices of the inputs at the boundary
  using inputs_boundary_coeff_indices_type =
    std::tuple<decltype(std::declval<Inputs>()
                        .template find_coeff_indices<functionspace::boundary>(
                                                                              std::declval<
                                                                              typename Inputs::boundary_eval_type>()))...>;

  /// @brief Type of the coefficient indices of the outputs in the interior
  using outputs_interior_coeff_indices_type =
    std::tuple<decltype(std::declval<Outputs>()
                        .template find_coeff_indices<functionspace::interior>(
                                                                              std::declval<typename Outputs::eval_type>()))...>;

  /// @brief Type of the coefficient indices of the outputs at the boundary
  using outputs_boundary_coeff_indices_type =
    std::tuple<decltype(std::declval<Outputs>()
                        .template find_coeff_indices<functionspace::boundary>(
                                                                              std::declval<
                                                                              typename Outputs::boundary_eval_type>()))...>;
};

template <detail::HasAsTensor... Inputs,
          detail::HasAsTensor... Outputs,
          detail::HasAsTensor... CollPts>
class IgANetCustomizable2<std::tuple<Inputs...>,
                    std::tuple<Outputs...>,
                          std::tuple<CollPts...>> : public IgANetCustomizable2<std::tuple<Inputs...>,
                                                                               std::tuple<Outputs...>, void> {
public:
  /// @brief Type of the knot indices of the collocation points objects in the interior
  using collPts_interior_knot_indices_type =
    std::tuple<decltype(std::declval<CollPts>()
                   .template find_knot_indices<functionspace::interior>(
                                                                        std::declval<typename CollPts::eval_type>()))...>;

  /// @brief Type of the knot indices of the collocation points objects at the boundary
  using collPts_boundary_knot_indices_type =
    std::tuple<decltype(std::declval<CollPts>()
                        .template find_knot_indices<functionspace::boundary>(
                                                                             std::declval<
                                                                             typename CollPts::boundary_eval_type>()))...>;

  /// @brief Type of the coefficient indices of the collocation points objects in the interior
  using collPts_interior_coeff_indices_type =
    std::tuple<decltype(std::declval<CollPts>()
                        .template find_coeff_indices<functionspace::interior>(
                                                                              std::declval<typename CollPts::eval_type>()))...>;

  /// @brief Type of the coefficient indices of the collocation points objects at the boundary
  using collPts_boundary_coeff_indices_type =
    std::tuple<decltype(std::declval<CollPts>()
                        .template find_coeff_indices<functionspace::boundary>(
                                                                              std::declval<
                                                                              typename CollPts::boundary_eval_type>()))...>; 
};  

/// @}
  
} // namespace iganet
