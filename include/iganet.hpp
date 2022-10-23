/**
   @file include/iganet.hpp

   @brief Isogeometric analysis network

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include <boundary.hpp>
#include <bspline.hpp>
#include <layer.hpp>
#include <zip.hpp>
#include <any>

#pragma once

namespace iganet {

  struct IgANetOptions
  {
    TORCH_ARG(int64_t, max_epoch)  = 100;
    TORCH_ARG(int64_t, batch_size) = 1000;
    TORCH_ARG(double,  min_loss)   = 1e-4;
  };
  
  /**
   * IgANetGeneratorImpl
   *
   * @note Following the discussion of module overship here
   *
   * https://pytorch.org/tutorials/advanced/cpp_frontend.html#module-ownership
   *
   * we implement a generator implementation class following
   *
   * https://pytorch.org/tutorials/advanced/cpp_frontend.html#the-generator-module
   */
  template<typename real_t>
  class IgANetGeneratorImpl :
    public torch::nn::Module
  {
  public:
    /// Default constructor
    IgANetGeneratorImpl() = default;

    /// Constructor
    explicit IgANetGeneratorImpl(const std::vector<int64_t>& layers,
                                 const std::vector<std::vector<std::any>>& activations)
    {
      assert(layers.size() == activations.size()+1);
      
      // Generate vector of linear layers and register them as layer[i]
      for (auto i=0; i<layers.size()-1; ++i)
        {
          layers_.emplace_back(register_module("layer["+std::to_string(i)+"]",
                                               torch::nn::Linear(layers[i], layers[i+1])));
        }

      // Generate vector of activation functions
      for (const auto& a : activations)
        switch (std::any_cast<activation>(a[0]))
          {
            // No activation function
          case activation::none:
            switch (a.size()) {
            case 1:
              activations_.emplace_back( new None{} );
              break;
            default:
              throw std::runtime_error("Invalid number of parameters");
            }
            break;

            // Batch Normalization
          case activation::batch_norm:
            switch (a.size()) {
            case 8:
              activations_.emplace_back( new BatchNorm{ std::any_cast<torch::Tensor>(a[1]),
                                                        std::any_cast<torch::Tensor>(a[2]),
                                                        std::any_cast<torch::Tensor>(a[3]),
                                                        std::any_cast<torch::Tensor>(a[4]),
                                                        std::any_cast<double>(a[5]),
                                                        std::any_cast<double>(a[6]),
                                                        std::any_cast<bool>(a[7]) } );
              break;
            case 7:
              activations_.emplace_back( new BatchNorm{ std::any_cast<torch::Tensor>(a[1]),
                                                        std::any_cast<torch::Tensor>(a[2]),
                                                        std::any_cast<torch::Tensor>(a[3]),
                                                        std::any_cast<torch::Tensor>(a[4]),
                                                        std::any_cast<double>(a[5]),
                                                        std::any_cast<double>(a[6]) } );
              break;
            case 4:
              activations_.emplace_back( new BatchNorm{ std::any_cast<torch::Tensor>(a[1]),
                                                        std::any_cast<torch::Tensor>(a[2]),
                                                        std::any_cast<torch::nn::functional::BatchNormFuncOptions>(a[3]) } );
              break;
            case 3:
              activations_.emplace_back( new BatchNorm{ std::any_cast<torch::Tensor>(a[1]),
                                                        std::any_cast<torch::Tensor>(a[2]) } );
              break;
            default:
              throw std::runtime_error("Invalid number of parameters");
            }
            break;

            // CELU
          case activation::celu:
            switch (a.size()) {
            case 3:
              activations_.emplace_back( new CELU{ std::any_cast<double>(a[1]),
                                                   std::any_cast<bool>(a[2])} );
              break;
            case 2:
              try {
                activations_.emplace_back( new CELU{ std::any_cast<torch::nn::functional::CELUFuncOptions>(a[1]) } );
              } catch(...) {
                activations_.emplace_back( new CELU{ std::any_cast<double>(a[1]) } );
              }
              break;
            case 1:
              activations_.emplace_back( new CELU{} );
              break;
            default:
              throw std::runtime_error("Invalid number of parameters");
            }
            break;
            
            // ELU
          case activation::elu:
            switch (a.size()) {
            case 3:
              activations_.emplace_back( new ELU{ std::any_cast<double>(a[1]),
                                                   std::any_cast<bool>(a[2])} );
              break;
            case 2:
              try {
                activations_.emplace_back( new ELU{ std::any_cast<torch::nn::functional::ELUFuncOptions>(a[1]) } );
              } catch(...) {
                activations_.emplace_back( new ELU{ std::any_cast<double>(a[1]) } );
              }
              break;
            case 1:
              activations_.emplace_back( new ELU{} );
              break;
            default:
              throw std::runtime_error("Invalid number of parameters");
            }
            break;

            // GELU
          case activation::gelu:
            switch (a.size()) {
            case 1:
              activations_.emplace_back( new GELU{} );
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
                activations_.emplace_back( new GLU{ std::any_cast<torch::nn::functional::GLUFuncOptions>(a[1]) } );
              } catch(...) {
                activations_.emplace_back( new GLU{ std::any_cast<int64_t>(a[1]) } );
              }
              break;
            case 1:
              activations_.emplace_back( new GLU{} );
              break;
            default:
              throw std::runtime_error("Invalid number of parameters");
            }
            break;

            // Group Normalization
          case activation::group_norm:
            switch (a.size()) {
            case 5:
              activations_.emplace_back( new GroupNorm{ std::any_cast<int64_t>(a[1]),
                                                        std::any_cast<torch::Tensor>(a[2]),
                                                        std::any_cast<torch::Tensor>(a[3]),
                                                        std::any_cast<double>(a[4]) } );
              break;
            case 2:
              try {
                activations_.emplace_back( new GroupNorm{ std::any_cast<torch::nn::functional::GroupNormFuncOptions>(a[1]) } );
              } catch(...) {
                activations_.emplace_back( new GroupNorm{ std::any_cast<int64_t>(a[1]) } );
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
              activations_.emplace_back( new GumbelSoftmax{ std::any_cast<double>(a[1]),
                                                            std::any_cast<int>(a[2]),
                                                            std::any_cast<bool>(a[3]) } );
              break;
            case 2:
              activations_.emplace_back( new GumbelSoftmax{ std::any_cast<torch::nn::functional::GumbelSoftmaxFuncOptions>(a[1]) } );
              break;
            case 1:
              activations_.emplace_back( new GumbelSoftmax{} );
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
                activations_.emplace_back( new Hardshrink{ std::any_cast<torch::nn::functional::HardshrinkFuncOptions>(a[1]) } );
              } catch(...) {
                activations_.emplace_back( new Hardshrink{ std::any_cast<double>(a[1]) } );
              }
              break;
            case 1:
              activations_.emplace_back( new Hardshrink{} );
              break;
            default:
              throw std::runtime_error("Invalid number of parameters");
            }
            break;

            // Hardsigmoid
          case activation::hardsigmoid:
            switch (a.size()) {
            case 1:
              activations_.emplace_back( new Hardsigmoid{} );
              break;
            default:
              throw std::runtime_error("Invalid number of parameters");
            }
            break;
            
            // Hardswish
          case activation::hardswish:
            switch (a.size()) {
            case 1:
              activations_.emplace_back( new Hardswish{} );
              break;
            default:
              throw std::runtime_error("Invalid number of parameters");
            }
            break;

            // Hardtanh
          case activation::hardtanh:
            switch (a.size()) {
            case 4:
              activations_.emplace_back( new Hardtanh{ std::any_cast<double>(a[1]),
                                                       std::any_cast<double>(a[2]),
                                                       std::any_cast<bool>(a[3]) } );
              break;
            case 3:
              activations_.emplace_back( new Hardtanh{ std::any_cast<double>(a[1]),
                                                       std::any_cast<double>(a[2]) } );
              break;
            case 2:
              activations_.emplace_back( new Hardtanh{ std::any_cast<torch::nn::functional::HardtanhFuncOptions>(a[1]) } );
              break;
            case 1:
              activations_.emplace_back( new Hardtanh{} );
              break;
            default:
              throw std::runtime_error("Invalid number of parameters");
            }
            break;

            // Instance Normalization
          case activation::instance_norm:
            switch (a.size()) {
            case 8:
              activations_.emplace_back( new InstanceNorm{ std::any_cast<torch::Tensor>(a[1]),
                                                           std::any_cast<torch::Tensor>(a[2]),
                                                           std::any_cast<torch::Tensor>(a[3]),
                                                           std::any_cast<torch::Tensor>(a[4]),
                                                           std::any_cast<double>(a[5]),
                                                           std::any_cast<double>(a[6]),
                                                           std::any_cast<bool>(a[7]) } );
              break;
            case 7:
              activations_.emplace_back( new InstanceNorm{ std::any_cast<torch::Tensor>(a[1]),
                                                           std::any_cast<torch::Tensor>(a[2]),
                                                           std::any_cast<torch::Tensor>(a[3]),
                                                           std::any_cast<torch::Tensor>(a[4]),
                                                           std::any_cast<double>(a[5]),
                                                           std::any_cast<double>(a[6]) } );
              break;
            case 2:
              activations_.emplace_back( new InstanceNorm{ std::any_cast<torch::nn::functional::InstanceNormFuncOptions>(a[1]) } );
              break;
            case 1:
              activations_.emplace_back( new InstanceNorm{} );
              break;
            default:
              throw std::runtime_error("Invalid number of parameters");
            }
            break;

            // Layer Normalization
          case activation::layer_norm:
            switch (a.size()) {
            case 5:
              activations_.emplace_back( new LayerNorm{ std::any_cast<std::vector<int64_t>>(a[1]),
                                                        std::any_cast<torch::Tensor>(a[2]),
                                                        std::any_cast<torch::Tensor>(a[3]),
                                                        std::any_cast<double>(a[4]) } );
              break;
            case 2:
              try {
                activations_.emplace_back( new LayerNorm{ std::any_cast<torch::nn::functional::LayerNormFuncOptions>(a[1]) } );
              } catch(...) {
                activations_.emplace_back( new LayerNorm{ std::any_cast<std::vector<int64_t>>(a[1]) } );
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
              activations_.emplace_back( new LeakyReLU{ std::any_cast<double>(a[1]),
                                                        std::any_cast<bool>(a[2]) } );
              break;
            case 2:
              try {
                activations_.emplace_back( new LeakyReLU{ std::any_cast<torch::nn::functional::LeakyReLUFuncOptions>(a[1]) } );
              } catch(...) {
                activations_.emplace_back( new LeakyReLU{ std::any_cast<double>(a[1]) } );
              }
              break;
            case 1:
              activations_.emplace_back( new LeakyReLU{} );
              break;
            default:
              throw std::runtime_error("Invalid number of parameters");
            }
            break;

            // Local response Normalization
          case activation::local_response_norm:
            switch (a.size()) {
            case 5:
              activations_.emplace_back( new LocalResponseNorm{ std::any_cast<int64_t>(a[1]),
                                                                std::any_cast<double>(a[2]),
                                                                std::any_cast<double>(a[3]),
                                                                std::any_cast<double>(a[4]) } );
              break;
            case 2:
              try {
                activations_.emplace_back( new LocalResponseNorm{ std::any_cast<torch::nn::functional::LocalResponseNormFuncOptions>(a[1]) } );
              } catch(...) {
                activations_.emplace_back( new LocalResponseNorm{ std::any_cast<int64_t>(a[1]) } );
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
              activations_.emplace_back( new LogSigmoid{} );
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
                activations_.emplace_back( new LogSoftmax{ std::any_cast<torch::nn::functional::LogSoftmaxFuncOptions>(a[1]) } );
              } catch(...) {
                activations_.emplace_back( new LogSoftmax{ std::any_cast<int64_t>(a[1]) } );
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
              activations_.emplace_back( new Mish{} );
              break;
            default:
              throw std::runtime_error("Invalid number of parameters");
            }
            break;
            
            // Lp Normalization
          case activation::normalize:
            switch (a.size()) {
            case 4:
              activations_.emplace_back( new Normalize{ std::any_cast<double>(a[1]),
                                                                std::any_cast<double>(a[2]),
                                                                std::any_cast<int64_t>(a[3]) } );
              break;
            case 2:
              activations_.emplace_back( new Normalize{ std::any_cast<torch::nn::functional::NormalizeFuncOptions>(a[1]) } );
              break;
            case 1:
              activations_.emplace_back( new Normalize{} );
              break;
            default:
              throw std::runtime_error("Invalid number of parameters");
            }
            break;

            // PReLU
          case activation::prelu:
            switch (a.size()) {
            case 2:
              activations_.emplace_back( new PReLU{ std::any_cast<torch::Tensor>(a[1]) } );
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
                activations_.emplace_back( new ReLU{ std::any_cast<torch::nn::functional::ReLUFuncOptions>(a[1]) } );
              } catch(...) {
                activations_.emplace_back( new ReLU{ std::any_cast<bool>(a[1]) } );
              }
              break;
            case 1:
              activations_.emplace_back( new ReLU{} );
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
                activations_.emplace_back( new ReLU6{ std::any_cast<torch::nn::functional::ReLU6FuncOptions>(a[1]) } );
              } catch(...) {
                activations_.emplace_back( new ReLU6{ std::any_cast<bool>(a[1]) } );
              }
              break;
            case 1:
              activations_.emplace_back( new ReLU6{} );
              break;
            default:
              throw std::runtime_error("Invalid number of parameters");
            }
            break;

            // Randomized ReLU
          case activation::rrelu:
            switch (a.size()) {
            case 4:
              activations_.emplace_back( new RReLU{ std::any_cast<double>(a[1]),
                                                    std::any_cast<double>(a[2]),
                                                    std::any_cast<bool>(a[3]) } );
              break;
            case 3:
              activations_.emplace_back( new RReLU{ std::any_cast<double>(a[1]),
                                                    std::any_cast<double>(a[2]) } );
              break;
            case 2:
              activations_.emplace_back( new RReLU{ std::any_cast<torch::nn::functional::RReLUFuncOptions>(a[1]) } );
              break;
            case 1:
              activations_.emplace_back( new RReLU{} );
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
                activations_.emplace_back( new SELU{ std::any_cast<torch::nn::functional::SELUFuncOptions>(a[1]) } );
              } catch(...) {
                activations_.emplace_back( new SELU{ std::any_cast<bool>(a[1]) } );
              }
              break;
            case 1:
              activations_.emplace_back( new SELU{} );
              break;
            default:
              throw std::runtime_error("Invalid number of parameters");
            }
            break;

            // Sigmoid
          case activation::sigmoid:
            switch (a.size()) {
            case 1:
              activations_.emplace_back( new Sigmoid{} );
              break;
            default:
              throw std::runtime_error("Invalid number of parameters");
            }
            break;

            // SiLU
          case activation::silu:
            switch (a.size()) {
            case 1:
              activations_.emplace_back( new SiLU{} );
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
                activations_.emplace_back( new Softmax{ std::any_cast<torch::nn::functional::SoftmaxFuncOptions>(a[1]) } );
              } catch(...) {
                activations_.emplace_back( new Softmax{ std::any_cast<bool>(a[1]) } );
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
                activations_.emplace_back( new Softmin{ std::any_cast<torch::nn::functional::SoftminFuncOptions>(a[1]) } );
              } catch(...) {
                activations_.emplace_back( new Softmin{ std::any_cast<bool>(a[1]) } );
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
              activations_.emplace_back( new Softplus{ std::any_cast<double>(a[1]),
                                                       std::any_cast<double>(a[2]) } );
              break;
            case 2:
              activations_.emplace_back( new Softplus{ std::any_cast<torch::nn::functional::SoftplusFuncOptions>(a[1]) } );
              break;
            case 1:
              activations_.emplace_back( new Softplus{} );
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
                activations_.emplace_back( new Softshrink{ std::any_cast<torch::nn::functional::SoftshrinkFuncOptions>(a[1]) } );  
              } catch(...) {
                activations_.emplace_back( new Softshrink{ std::any_cast<double>(a[1]) } );
              }
              break;
            case 1:
              activations_.emplace_back( new Softshrink{} );
              break;
            default:
              throw std::runtime_error("Invalid number of parameters");
            }
            break;

            // Softsign
          case activation::softsign:
            switch (a.size()) {
            case 1:
              activations_.emplace_back( new Softsign{} );
              break;
            default:
              throw std::runtime_error("Invalid number of parameters");
            }
            break;

            // Tanh
          case activation::tanh:
            switch (a.size()) {
            case 1:
              activations_.emplace_back( new Tanh{} );
              break;
            default:
              throw std::runtime_error("Invalid number of parameters");
            }
            break;

            // Tanhshrink
          case activation::tanhshrink:
            switch (a.size()) {
            case 1:
              activations_.emplace_back( new Tanhshrink{} );
              break;
            default:
              throw std::runtime_error("Invalid number of parameters");
            }
            break;

            // Threshold
          case activation::threshold:
            switch (a.size()) {
            case 4:
              activations_.emplace_back( new Threshold{ std::any_cast<double>(a[1]),
                                                        std::any_cast<double>(a[2]),
                                                        std::any_cast<bool>(a[3]) } );
              break;
            case 3:
              activations_.emplace_back( new Threshold{ std::any_cast<double>(a[1]),
                                                        std::any_cast<double>(a[2]) } );
              break;              
            case 2:
              activations_.emplace_back( new Threshold{ std::any_cast<torch::nn::functional::ThresholdFuncOptions>(a[1]) } );
              break;
            default:
              throw std::runtime_error("Invalid number of parameters");
            }
            break;
            
          default:
            throw std::runtime_error("Invalid activation function");
          }
    }
    
    /// Forward evaluation
    torch::Tensor forward(torch::Tensor x)
    {            
      // Standard feed-forward neural network
      for (auto layer : zip(layers_, activations_))
        x = std::get<1>(layer)->apply(std::get<0>(layer)->forward(x));
      return x;
    }

    /// Writes the IgANet into a torch::serialize::OutputArchive object
    inline torch::serialize::OutputArchive& write(torch::serialize::OutputArchive& archive,
                                                  const std::string& key="iganet") const
    {
      assert(layers_.size() == activations_.size());
      
      archive.write(key+".layers", torch::full({1}, (int64_t)layers_.size()));
      for (std::size_t i=0; i<layers_.size(); ++i) {
        archive.write(key+".layer["+std::to_string(i)+"].in_features",
                      torch::full({1}, (int64_t)layers_[i]->options.in_features()));
        archive.write(key+".layer["+std::to_string(i)+"].out_features",
                      torch::full({1}, (int64_t)layers_[i]->options.out_features()));
        archive.write(key+".layer["+std::to_string(i)+"].bias",
                      torch::full({1}, (int64_t)layers_[i]->options.bias()));
        
        activations_[i]->write(archive, key+".layer["+std::to_string(i)+"].activation");
      }
      
      return archive;
    }

    /// Reads the IgANet from a torch::serialize::InputArchive object
    inline torch::serialize::InputArchive& read(torch::serialize::InputArchive& archive,
                                                const std::string& key="iganet")
    {
      torch::Tensor layers, in_features, out_features, bias, activation;
      
      archive.read(key+".layers", layers);
      for (int64_t i=0; i<layers.item<int64_t>(); ++i) {
        archive.read(key+".layer["+std::to_string(i)+"].in_features", in_features);
        archive.read(key+".layer["+std::to_string(i)+"].out_features", out_features);
        archive.read(key+".layer["+std::to_string(i)+"].bias", bias);
        layers_.emplace_back(register_module("layer["+std::to_string(i)+"]",
                                             torch::nn::Linear(
                                                               torch::nn::LinearOptions(in_features.item<int64_t>(),
                                                                                        out_features.item<int64_t>()
                                                                                        ).bias(bias.item<bool>())
                                                               )));
        
        archive.read(key+".layer["+std::to_string(i)+"].activation.type", activation);
        switch (static_cast<enum activation>(activation.item<int64_t>())) {
        case activation::none:
          activations_.emplace_back( new None{} );
          break;
        case activation::batch_norm:
          activations_.emplace_back( new BatchNorm{ torch::Tensor{}, torch::Tensor{} } );
          break;
        case activation::celu:
          activations_.emplace_back( new CELU{} );
          break;
        case activation::elu:
          activations_.emplace_back( new ELU{} );
          break;
        case activation::gelu:
          activations_.emplace_back( new GELU{} );
          break;
        case activation::glu:
          activations_.emplace_back( new GLU{} );
          break;
        case activation::group_norm:
          activations_.emplace_back( new GroupNorm{ 0 } );
          break;
        case activation::gumbel_softmax:
          activations_.emplace_back( new GumbelSoftmax{} );
          break;
        case activation::hardshrink:
          activations_.emplace_back( new Hardshrink{} );
          break;
        case activation::hardsigmoid:
          activations_.emplace_back( new Hardsigmoid{} );
          break;
        case activation::hardswish:
          activations_.emplace_back( new Hardswish{} );
          break;
        case activation::hardtanh:
          activations_.emplace_back( new Hardtanh{} );
          break;
        case activation::instance_norm:
          activations_.emplace_back( new InstanceNorm{} );
          break;
        case activation::layer_norm:
          activations_.emplace_back( new LayerNorm{ {} } );
          break;
        case activation::leaky_relu:
          activations_.emplace_back( new LeakyReLU{} );
          break;
        case activation::local_response_norm:
          activations_.emplace_back( new LocalResponseNorm{ 0 } );
          break;
        case activation::logsigmoid:
          activations_.emplace_back( new LogSigmoid{} );
          break;
        case activation::logsoftmax:
          activations_.emplace_back( new LogSoftmax{ 0 } );
          break;
        case activation::mish:
          activations_.emplace_back( new Mish{} );
          break;
        case activation::normalize:
          activations_.emplace_back( new Normalize{ 0, 0, 0 } );
          break;
        case activation::prelu:
          activations_.emplace_back( new PReLU{ torch::Tensor{} } );
          break;
        case activation::relu:
          activations_.emplace_back( new ReLU{} );
          break;
        case activation::relu6:
          activations_.emplace_back( new ReLU6{} );
          break;
        case activation::rrelu:
          activations_.emplace_back( new RReLU{} );
          break;
        case activation::selu:
          activations_.emplace_back( new SELU{} );
          break;
        case activation::sigmoid:
          activations_.emplace_back( new Sigmoid{} );
          break;
        case activation::silu:
          activations_.emplace_back( new SiLU{} );
          break;
        case activation::softmax:
          activations_.emplace_back( new Softmax{ 0 } );
          break;
        case activation::softmin:
          activations_.emplace_back( new Softmin{ 0 } );
          break;
        case activation::softplus:
          activations_.emplace_back( new Softplus{} );
          break;
        case activation::softshrink:
          activations_.emplace_back( new Softshrink{} );
          break;
        case activation::softsign:
          activations_.emplace_back( new Softsign{} );
          break;
        case activation::tanh:
          activations_.emplace_back( new Tanh{} );
          break;
        case activation::tanhshrink:
          activations_.emplace_back( new Tanhshrink{} );
          break;
        case activation::threshold:
          activations_.emplace_back( new Threshold{ 0, 0 } );
          break;
        default:
          throw std::runtime_error("Invalid activation function");
        }

        activations_.back()->read(archive, key+".layer["+std::to_string(i)+"].activation");
        
      }
      return archive;
    }

    inline void pretty_print(std::ostream& os = std::cout) const override final
    {
      os << "(\n";

      int i=0;
      for (const auto& activation : activations_)
        os << "activation[" << i++ << "]: " << *activation << "\n";
      os << ")\n";
    }
    
  private:
    /// Vector of linear layers
    std::vector<torch::nn::Linear> layers_;

    /// Vector of activation functions
    std::vector<std::unique_ptr<iganet::ActivationFunction>> activations_;
  };
  
  /**
   * IgANetGenerator
   *
   * @note: This class is normally generated by the TORCH_MODULE
   * macro. Since the latter cannot handle templated classes
   * correctly, we give the implementation explicitly
   */
  template<typename real_t>
  class IgANetGenerator :
    public torch::nn::ModuleHolder<IgANetGeneratorImpl<real_t>> {

  public:
    using torch::nn::ModuleHolder<IgANetGeneratorImpl<real_t>>::ModuleHolder;
    using Impl = IgANetGeneratorImpl<real_t>;
  };

  /**
   * IgANet
   */
  template<typename real_t,
           typename optimizer_t,
           template<typename, short_t, short_t...> class bspline_t,
           short_t... Degrees>
  class IgANet : public core<real_t>
  {
  private:
    /// Dimension of the differential equation
    static constexpr const short_t dim_ = sizeof...(Degrees);

    /// B-spline representation of the geometry
    bspline_t<real_t, dim_, Degrees...> geo_;

    /// B-spline representation of the right-hand side
    bspline_t<real_t, 1, Degrees...> rhs_;

    /// B-spline representation of the solution
    bspline_t<real_t, 1, Degrees...> sol_;

    /// B-spline representation of the network output
    bspline_t<real_t, 1, Degrees...> out_;
    
    /// B-spline representation of the boundary conditions
    Boundary<bspline_t,real_t, 1, Degrees...> bdr_;

    /// IgANet generator
    IgANetGenerator<real_t> net_;

    /// Optimizer
    optimizer_t opt_;

    /// Options
    IgANetOptions options_;
    
  public:
    /// Default constructor
    explicit IgANet(IgANetOptions defaults = {})
      : core<real_t>(),
        geo_(),
        rhs_(),
        sol_(),
        out_(),
        bdr_(),
        opt_(net_->parameters()),
        options_(defaults)
    {}

    /// Constructor: layers + bspline (same for all)
    IgANet(const std::vector<int64_t>& layers,
           const std::vector<std::vector<std::any>>& activations,
           const std::array<int64_t,dim_>& bspline_ncoeffs,
           IgANetOptions defaults = {})
      : IgANet(layers, activations, bspline_ncoeffs, bspline_ncoeffs, bspline_ncoeffs, defaults)
    {
    }

    /// Constructor: layers + geo-bspline + rhs-bspline + sol-bspline
    IgANet(const std::vector<int64_t>& layers,
           const std::vector<std::vector<std::any>>& activations,
           const std::array<int64_t,dim_>& geo_bspline_ncoeffs,
           const std::array<int64_t,dim_>& rhs_bspline_ncoeffs,
           const std::array<int64_t,dim_>& sol_bspline_ncoeffs,
           IgANetOptions defaults = {})
      : core<real_t>(),

        // Construct the different B-Spline objects individually
        geo_(geo_bspline_ncoeffs, BSplineInit::greville),
        rhs_(rhs_bspline_ncoeffs, BSplineInit::zeros),
        bdr_(sol_bspline_ncoeffs, BSplineInit::zeros),
        sol_(sol_bspline_ncoeffs, BSplineInit::random),
        out_(sol_bspline_ncoeffs, BSplineInit::random),

        // Construct the deep neural network with the large tensor as
        // input and the coefficient vector of the solution's BSpline
        // object as output
        net_(concat(std::vector<int64_t>{geo_.ncoeffs()+rhs_.ncoeffs()+bdr_.ncoeffs() /*+dim_*/ },
                    layers,
                    std::vector<int64_t>{sol_.ncoeffs()}
                    ),
             activations),

        // Construct the optimizer
        opt_(net_->parameters()),

        // Set options
        options_(defaults)
    {
    }

    /// Constructor: layers + bspline (same for all)
    IgANet(const std::vector<int64_t>& layers,
           const std::vector<std::vector<std::any>>& activations,
           const std::array<std::vector<real_t>,dim_>& bspline_kv,
           IgANetOptions defaults = {})
      : IgANet(layers, activations, bspline_kv, bspline_kv, bspline_kv, defaults)
    {
    }

    /// Constructor: layers + geo-bspline + rhs-bspline + sol-bspline
    IgANet(const std::vector<int64_t>& layers,
           const std::vector<std::vector<std::any>>& activations,
           const std::array<std::vector<real_t>,dim_>& geo_bspline_kv,
           const std::array<std::vector<real_t>,dim_>& rhs_bspline_kv,
           const std::array<std::vector<real_t>,dim_>& sol_bspline_kv,
           IgANetOptions defaults = {})
      : core<real_t>(),

        // Construct the different B-Spline objects individually
        geo_(geo_bspline_kv, BSplineInit::greville),
        rhs_(rhs_bspline_kv, BSplineInit::zeros),
        bdr_(sol_bspline_kv, BSplineInit::zeros),
        sol_(sol_bspline_kv, BSplineInit::random),
        out_(sol_bspline_kv, BSplineInit::random),

        // Construct the deep neural network with the large tensor as
        // input and the coefficient vector of the solution's BSpline
        // object as output
        net_(concat(std::vector<int64_t>{geo_.ncoeffs()+rhs_.ncoeffs()+bdr_.ncoeffs() /*+dim_*/ },
                    layers,
                    std::vector<int64_t>{sol_.ncoeffs()}),
             activations),
        
        // Construct the optimizer
        opt_(net_->parameters()),
        
        // Set options
        options_(defaults)
    {
    }

    /// Returns a constant reference to the IgANet generator
    inline const IgANetGenerator<real_t>& net() const
    {
      return net_;
    }

    /// Returns a non-constant reference to the IgANet generator
    inline IgANetGenerator<real_t>& net()
    {
      return net_;
    }

    /// Returns a constant reference to the optimizer
    inline const optimizer_t& opt() const
    {
      return opt_;
    }

    /// Returns a non-constant reference to the optimizer
    inline optimizer_t& opt()
    {
      return opt_;
    }
    
    /// Returns a constant reference to the B-spline representation of the geometry
    inline const bspline_t<real_t, dim_, Degrees...>& geo() const
    {
      return geo_;
    }

    /// Returns a non-constant reference to the B-spline representation of the geometry
    inline bspline_t<real_t, dim_, Degrees...>& geo()
    {
      return geo_;
    }

    /// Returns a constant reference to the B-spline representation of the right-hand side
    inline const bspline_t<real_t, 1, Degrees...>& rhs() const
    {
      return rhs_;
    }

    /// Returns a non-constant reference to the B-spline representation of the right-hand side
    inline bspline_t<real_t, 1, Degrees...>& rhs()
    {
      return rhs_;
    }

    /// Returns a constant reference to the B-spline representation of the solution
    inline const bspline_t<real_t, 1, Degrees...>& sol() const
    {
      return sol_;
    }

    /// Returns a non-constant reference to the B-spline representation of the solution
    inline bspline_t<real_t, 1, Degrees...>& sol()
    {
      return sol_;
    }

    /// Returns a constant reference to the B-spline representation of the boundary contitions
    inline const auto& bdr() const
    {
      return bdr_;      
    }

    /// Returns a non-constant reference to the B-spline representation of the boundary conditions
    inline auto& bdr()
    {
      return bdr_;      
    }

    /// Returns the dimension
    inline constexpr short_t dim() const
    {
      return dim_;
    }

    /// Returns a constant reference to the options structure
    inline const auto& options() const
    {
      return options_;
    }

    /// Returns a non-constant reference to the options structure
    inline auto& options()
    {
      return options_;
    }

    /// Trains the IgANet
    inline void train()
    {
      // Get Greville points
      auto sample_points = sol_.greville();

      // Evaluate right-hand side
      auto rhs = rhs_.eval( sample_points );
      
      // Construct full sample set
      auto samples = torch::cat(
                                { geo_.coeffs()[0].view({1,-1}).repeat({1,1}),
                                  rhs_.coeffs()[0].view({1,-1}).repeat({1,1}),
                                  bdr_.coeffs()[0].view({1,-1}).repeat({1,1}),
                                  bdr_.coeffs()[1].view({1,-1}).repeat({1,1})
                                }, 1
                                );
      
      for (int64_t epoch = 0; epoch != options_.max_epoch(); ++epoch)
        {
          std::cout << "Epoch " << std::to_string(epoch) << ": ";

          // Reset gradients
          net_->zero_grad();    

          // Execute the model on the input data
          auto pred = net_->forward(samples);

          // Evaluate solution
          auto out = out_.eval( sample_points );

          std::cout << pred << std::endl;
          std::cout << out << std::endl;
          
          // Compute the loss value
          auto loss_pde = torch::mse_loss( pred , rhs );
          std::cout << "loss = " << loss_pde.template item<real_t>() << std::endl;

          // Compute gradients of the loss w.r.t. the model parameters
          loss_pde.backward({}, true, false);

          // Update the parameters based on the calculated gradients
          opt_.step();

          if (loss_pde.template item<real_t>() < options_.min_loss())
            break;
        }


    }
    
    /// Returns a string representation of the IgANet object
    inline void pretty_print(std::ostream& os = std::cout) const
    {
      os << core<real_t>::name()
         << "(\n"
         << "net = " << net_ << "\n"
         << "geo = " << geo_ << "\n"
         << "rhs = " << rhs_ << "\n"
         << "bdr = " << bdr_ << "\n"
         << "sol = " << sol_
         << "\n)";
    }

    /// Plots the B-Spline geometry
    inline auto plot_geo(int64_t xres=10, int64_t yres=10, int64_t zres=10) const
    {
      return geo_.plot(xres, yres, zres);
    }

    /// Plots the B-Spline right-hand side
    inline auto plot_rhs(int64_t xres=10, int64_t yres=10, int64_t zres=10) const
    {
      return rhs_.plot(rhs_, xres, yres, zres);
    }

    /// Plots the B-Spline solution
    inline auto plot_sol(int64_t xres=10, int64_t yres=10, int64_t zres=10) const
    {
      return rhs_.plot(sol_, xres, yres, zres);
    }

    /// Saves the IgANet to file
    inline void save(const std::string& filename,
                     const std::string& key="iganet") const
    {
      torch::serialize::OutputArchive archive;
      write(archive, key).save_to(filename);
    }

    /// Loads the IgANet from file
    inline void load(const std::string& filename,
                     const std::string& key="iganet")
    {
      torch::serialize::InputArchive archive;
      archive.load_from(filename);      
      read(archive, key);
    }

    /// Writes the IgANet into a torch::serialize::OutputArchive object
    inline torch::serialize::OutputArchive& write(torch::serialize::OutputArchive& archive,
                                                  const std::string& key="iganet") const
    {
      archive.write(key+".dim", torch::full({1}, dim_));

      geo_.write(archive, key+".geo");
      rhs_.write(archive, key+".rhs");
      sol_.write(archive, key+".sol");
      
      net_->write(archive, key+".net");      
      torch::serialize::OutputArchive archive_net;
      net_->save(archive_net);
      archive.write(key+".net.data", archive_net);

      torch::serialize::OutputArchive archive_opt;
      opt_.save(archive_opt);
      archive.write(key+".opt", archive_opt);

      return archive;
    }

    /// Loads the IgANet from a torch::serialize::InputArchive object
    inline torch::serialize::InputArchive& read(torch::serialize::InputArchive& archive,
                                                const std::string& key="iganet")
    {
      torch::Tensor tensor;

      archive.read(key+".dim", tensor);
      if (tensor.item<int64_t>() != dim_)
        throw std::runtime_error("dim mismatch");

      geo_.read(archive, key+".geo");
      rhs_.read(archive, key+".rhs");
      sol_.read(archive, key+".sol");

      net_->read(archive, key+".net");      
      torch::serialize::InputArchive archive_net;
      archive.read(key+".net.data", archive_net);
      net_->load(archive_net);
      
      torch::serialize::InputArchive archive_opt;
      archive.read(key+".opt", archive_opt);            
      opt_.load(archive_opt);

      for (const auto& k : archive_opt.keys())
        std::cout << k << std::endl;
      
      return archive;
    }

    /// Returns true if both IgANet objects are the same
    bool operator==(const IgANet& other) const
    {
      bool result(true);

      result *= (dim_ == other.dim());
      result *= (geo_ == other.geo());
      result *= (rhs_ == other.rhs());
      result *= (sol_ == other.sol());

      return result;
    }

    /// Returns true if both IgANet objects are different
    bool operator!=(const IgANet& other) const
    {
      return *this != other;
    }
  };

  /// Print (as string) a IgANet object
  template<typename real_t,
           typename optimizer_t,
           template<typename, short_t, short_t...> class bspline_t,
           short_t... Degrees>
  inline std::ostream& operator<<(std::ostream& os,
                                  const IgANet<real_t, optimizer_t, bspline_t, Degrees...>& obj)
  {
    obj.pretty_print(os);
    return os;
  }

} // namespace iganet
