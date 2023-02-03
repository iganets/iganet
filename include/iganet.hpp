/**
   @file include/iganet.hpp

   @brief Isogeometric analysis network

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include <any>

#include <boundary.hpp>
#include <creator.hpp>
#include <functionspace.hpp>
#include <layer.hpp>
#include <zip.hpp>

#pragma once

namespace iganet {

  /// @brief Enumerator for the status of the various data
  enum class status : short_t
    {
      bdr  = 1<<0, /*!< boundary data needs update */
      geo  = 1<<1, /*!< geometry data needs update */
      rhs  = 1<<2  /*!< right-hand side data needs update */
    };

  /// @brief Returns the sum of two status objects
  status operator+(status lhs, status rhs)
  {
    return status(static_cast<short_t>(lhs)+static_cast<short_t>(rhs));
  }
  
  /// @brief IgANetOptions
  struct IgANetOptions
  {
    TORCH_ARG(int64_t, max_epoch)  = 100;
    TORCH_ARG(int64_t, batch_size) = 1000;
    TORCH_ARG(double,  min_loss)   = 1e-4;
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
  template<typename real_t>
  class IgANetGeneratorImpl :
    public torch::nn::Module
  {
  public:
    /// @brief Default constructor
    IgANetGeneratorImpl() = default;

    /// @brief Constructor
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
    
    /// @brief Forward evaluation
    torch::Tensor forward(torch::Tensor x)
    {            
      // Standard feed-forward neural network
      for (auto layer : zip(layers_, activations_))
        x = std::get<1>(layer)->apply(std::get<0>(layer)->forward(x));
      return x;
    }

    /// @brief Writes the IgANet into a torch::serialize::OutputArchive object
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

    /// @brief Reads the IgANet from a torch::serialize::InputArchive object
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
  template<typename real_t>
  class IgANetGenerator :
    public torch::nn::ModuleHolder<IgANetGeneratorImpl<real_t>> {

  public:
    using torch::nn::ModuleHolder<IgANetGeneratorImpl<real_t>>::ModuleHolder;
    using Impl = IgANetGeneratorImpl<real_t>;
  };

  /// @brief IgANet
  template<typename real_t,
           typename optimizer_t,
           short_t GeoDim,
           short_t PdeDim,
           template<typename, short_t, short_t...> class bspline_t,
           short_t... Degrees>
  class IgANet : public core<real_t>
  {
  protected:
    /// @brief Number of parametric dimensions
    static constexpr const short_t parDim_ = sizeof...(Degrees);

    /// @brief Type of the geometry B-spline object
    using Geometry_t = bspline_t<real_t, GeoDim, Degrees...>;

    /// @brief Type of the variable B-spline object
    using Variable_t = bspline_t<real_t, PdeDim, Degrees...>;

    /// @brief Type of the boundary B-spline object
    using Boundary_t = Boundary<Variable_t>;
    
    /// @brief B-spline representation of the geometry
    Geometry_t geo_;

    /// @brief B-spline representation of the right-hand side
    Variable_t rhs_;

    /// @brief B-spline representation of the solution
    Variable_t sol_;

    /// @brief B-spline representation of the network output
    Variable_t out_;
    
    /// @brief B-spline representation of the boundary conditions
    Boundary_t bdr_;

    /// @brief IgANet generator
    IgANetGenerator<real_t> net_;

    /// @brief Optimizer
    optimizer_t opt_;

    /// @brief Options
    IgANetOptions options_;

  public:
    /// @brief Default constructor
    explicit IgANet(IgANetOptions defaults = {},
                    iganet::core<real_t> core = iganet::core<real_t>{})
      : iganet::core<real_t>(core),
        geo_(),
        rhs_(),
        sol_(),
        out_(),
        bdr_(),
        opt_(net_->parameters()),
        options_(defaults)
    {}

    /// @brief Constructor: layers + bspline (same for all)
    IgANet(const std::vector<int64_t>& layers,
           const std::vector<std::vector<std::any>>& activations,
           const std::array<int64_t,parDim_>& bspline_ncoeffs,
           IgANetOptions defaults = {},
           iganet::core<real_t> core = iganet::core<real_t>{})
      : IgANet(layers, activations, bspline_ncoeffs,
               bspline_ncoeffs, bspline_ncoeffs, defaults, core)
    {
    }

    /// @brief Constructor: layers + geo-bspline + rhs-bspline + sol-bspline
    IgANet(const std::vector<int64_t>& layers,
           const std::vector<std::vector<std::any>>& activations,
           const std::array<int64_t,parDim_>& geo_bspline_ncoeffs,
           const std::array<int64_t,parDim_>& rhs_bspline_ncoeffs,
           const std::array<int64_t,parDim_>& sol_bspline_ncoeffs,
           IgANetOptions defaults = {},
           iganet::core<real_t> core = iganet::core<real_t>{})
      : iganet::core<real_t>(core),

        // Construct the different B-Spline objects individually
        geo_(geo_bspline_ncoeffs, init::greville),
        rhs_(rhs_bspline_ncoeffs, init::zeros),
        bdr_(sol_bspline_ncoeffs, init::zeros),
        sol_(sol_bspline_ncoeffs, init::random),
        out_(sol_bspline_ncoeffs, init::random),

        // Construct the deep neural network with the large tensor as
        // input and the coefficient vector of the solution's BSpline
        // object as output
        net_(concat(std::vector<int64_t>{GeoDim*geo_.ncoeffs()+PdeDim*rhs_.ncoeffs()+PdeDim*bdr_.ncoeffs()},
                    layers,
                    std::vector<int64_t>{PdeDim*sol_.ncoeffs()}
                    ),
             activations),

        // Construct the optimizer
        opt_(net_->parameters()),

        // Set options
        options_(defaults)
    {
    }

    /// @brief Constructor: layers + bspline (same for all)
    IgANet(const std::vector<int64_t>& layers,
           const std::vector<std::vector<std::any>>& activations,
           const std::array<std::vector<real_t>,parDim_>& bspline_kv,
           IgANetOptions defaults = {},
           iganet::core<real_t> core = iganet::core<real_t>{})
      : IgANet(layers, activations, bspline_kv, bspline_kv, bspline_kv, defaults, core)
    {
    }

    /// @brief Constructor: layers + geo-bspline + rhs-bspline + sol-bspline
    IgANet(const std::vector<int64_t>& layers,
           const std::vector<std::vector<std::any>>& activations,
           const std::array<std::vector<real_t>,parDim_>& geo_bspline_kv,
           const std::array<std::vector<real_t>,parDim_>& rhs_bspline_kv,
           const std::array<std::vector<real_t>,parDim_>& sol_bspline_kv,
           IgANetOptions defaults = {},
           iganet::core<real_t> core = iganet::core<real_t>{})
      : iganet::core<real_t>(core),

        // Construct the different B-Spline objects individually
        geo_(geo_bspline_kv, init::greville),
        rhs_(rhs_bspline_kv, init::zeros),
        bdr_(sol_bspline_kv, init::zeros),
        sol_(sol_bspline_kv, init::random),
        out_(sol_bspline_kv, init::random),

        // Construct the deep neural network with the large tensor as
        // input and the coefficient vector of the solution's BSpline
        // object as output
        net_(concat(std::vector<int64_t>{GeoDim*geo_.ncoeffs()+PdeDim*rhs_.ncoeffs()+PdeDim*bdr_.ncoeffs()},
                    layers,
                    std::vector<int64_t>{PdeDim*sol_.ncoeffs()}),
             activations),
        
        // Construct the optimizer
        opt_(net_->parameters()),
        
        // Set options
        options_(defaults)
    {
    }

    /// @brief Returns a constant reference to the IgANet generator
    inline const IgANetGenerator<real_t>& net() const
    {
      return net_;
    }

    /// @brief Returns a non-constant reference to the IgANet generator
    inline IgANetGenerator<real_t>& net()
    {
      return net_;
    }

    /// @brief Returns a constant reference to the optimizer
    inline const optimizer_t& opt() const
    {
      return opt_;
    }

    /// @brief Returns a non-constant reference to the optimizer
    inline optimizer_t& opt()
    {
      return opt_;
    }
    
    /// @brief Returns a constant reference to the B-spline
    /// representation of the geometry
    inline const Geometry_t& geo() const
    {
      return geo_;
    }

    /// @brief Returns a non-constant reference to the B-spline
    /// representation of the geometry
    inline Geometry_t& geo()
    {
      return geo_;
    }

    /// @brief Returns a constant reference to the B-spline
    /// representation of the right-hand side
    inline const Variable_t& rhs() const
    {
      return rhs_;
    }

    /// @brief Returns a non-constant reference to the B-spline
    /// representation of the right-hand side
    inline Variable_t& rhs()
    {
      return rhs_;
    }

    /// @brief Returns a constant reference to the B-spline
    /// representation of the solution
    inline const Variable_t& sol() const
    {
      return sol_;
    }

    /// @brief Returns a non-constant reference to the B-spline
    /// representation of the solution
    inline Variable_t& sol()
    {
      return sol_;
    }

    /// @brief Returns a constant reference to the B-spline
    /// representation of the boundary contitions
    inline const auto& bdr() const
    {
      return bdr_;      
    }

    /// @brief Returns a non-constant reference to the B-spline
    /// representation of the boundary conditions
    inline auto& bdr()
    {
      return bdr_;      
    }

    /// @brief Returns a constant reference to the options structure
    inline const auto& options() const
    {
      return options_;
    }

    /// @brief Returns a non-constant reference to the options structure
    inline auto& options()
    {
      return options_;
    }

    /// @brief Updates the samples
    ///
    /// In the default implementation the samples are the Greville
    /// abscissae in the interior of the domain and on the boundary
    /// faces. This behavior can be changed by overriding this virtual
    /// function in a derived class.
    virtual std::pair<std::array<torch::Tensor, parDim_>,
                      typename std::conditional<parDim_ == 1,
                                                std::array<torch::Tensor, Boundary_t::sides()>,
                                                std::array<std::array<torch::Tensor, parDim_-1>,Boundary_t::sides()>>::type>
    get_samples() const
    {      
      std::pair<std::array<torch::Tensor, parDim_>,
                typename std::conditional<parDim_ == 1,
                                          std::array<torch::Tensor, Boundary_t::sides()>,
                                          std::array<std::array<torch::Tensor, parDim_-1>,Boundary_t::sides()>>::type>
        result;
      
      // Get Greville abscissae inside the domain
      result.first = sol_.greville();

      // Get Greville abscissae at the boundary
      if constexpr (Boundary_t::sides() == side::east)
        {
          // 1D
          result.second[0] = bdr_.template side<side::west>().greville();
          result.second[1] = bdr_.template side<side::east>().greville();
        }
      else if constexpr (Boundary_t::sides() == side::north)
        {
          // 2D
          result.second[0] = bdr_.template side<side::west>().greville();
          result.second[1] = bdr_.template side<side::east>().greville();
          result.second[2] = bdr_.template side<side::south>().greville();
          result.second[3] = bdr_.template side<side::north>().greville();
        }
      else if constexpr (Boundary_t::sides() == side::back)
        {
          // 3D
          result.second[0] = bdr_.template side<side::west>().greville();
          result.second[1] = bdr_.template side<side::east>().greville();
          result.second[2] = bdr_.template side<side::south>().greville();
          result.second[3] = bdr_.template side<side::north>().greville();
          result.second[4] = bdr_.template side<side::front>().greville();
          result.second[5] = bdr_.template side<side::back>().greville();
        }
      else if constexpr (Boundary_t::sides() == side::etime)
        {
          // 4D
          result.second[0] = bdr_.template side<side::west>().greville();
          result.second[1] = bdr_.template side<side::east>().greville();
          result.second[2] = bdr_.template side<side::south>().greville();
          result.second[3] = bdr_.template side<side::north>().greville();
          result.second[4] = bdr_.template side<side::front>().greville();
          result.second[5] = bdr_.template side<side::back>().greville();
          result.second[6] = bdr_.template side<side::stime>().greville();
          result.second[7] = bdr_.template side<side::etime>().greville();
        }
      else
        {
          throw std::runtime_error("Unsupported boundary");
        }
      
      return result;
    }

  private:
    /// @brief Updates the network inputs
    template<std::size_t... geoDimIs, std::size_t... pdeDimIs, std::size_t... bdrIs>
    inline auto get_inputs(std::index_sequence<geoDimIs...>,
                           std::index_sequence<pdeDimIs...>,
                           std::index_sequence<bdrIs...>) const
    {      
      if constexpr (PdeDim == 1)
        return torch::cat({
            geo_.coeffs()[geoDimIs]...,
            rhs_.coeffs()[pdeDimIs]...,
            std::get<bdrIs>(bdr_.coeffs()).coeffs(0)...          
          });
      else if constexpr (PdeDim == 2)
        return torch::cat({
            geo_.coeffs()[geoDimIs]...,
            rhs_.coeffs()[pdeDimIs]...,
            std::get<bdrIs>(bdr_.coeffs()).coeffs(0)...,
            std::get<bdrIs>(bdr_.coeffs()).coeffs(1)...          
          });
      else if constexpr (PdeDim == 3)
        return torch::cat({
            geo_.coeffs()[geoDimIs]...,
            rhs_.coeffs()[pdeDimIs]...,
            std::get<bdrIs>(bdr_.coeffs()).coeffs(0)...,
            std::get<bdrIs>(bdr_.coeffs()).coeffs(1)...,
            std::get<bdrIs>(bdr_.coeffs()).coeffs(2)...
          });
      else if constexpr (PdeDim == 4)
        return torch::cat({
            geo_.coeffs()[geoDimIs]...,
            rhs_.coeffs()[pdeDimIs]...,
            std::get<bdrIs>(bdr_.coeffs()).coeffs(0)...,
            std::get<bdrIs>(bdr_.coeffs()).coeffs(1)...,
            std::get<bdrIs>(bdr_.coeffs()).coeffs(2)...,
            std::get<bdrIs>(bdr_.coeffs()).coeffs(3)...
          });
      else
        throw std::runtime_error("Unsupported PDE dimension");
    }

  public:
    /// @brief Updates the network inputs
    virtual torch::Tensor get_inputs() const
    {
      return get_inputs(std::make_index_sequence<GeoDim>(),
                        std::make_index_sequence<PdeDim>(),
                        std::make_index_sequence<Boundary_t::sides()>());     
    }
    
    /// @brief Updates object at the beginning of each epoch
    ///
    /// @note This functions is decleared pure virtual and needs to be
    /// overridden by the user in the derived problem-specific class
    virtual status get_epoch(int64_t epoch) const
    {
      return (epoch == 0
              ? status::bdr + status::geo + status::rhs
              : status(0));
    }
    
    /// @brief Trains the IgANet
    virtual void train()
    {
      // Get inputs
      auto inputs = get_inputs();
      
      // Get samples
      auto samples = get_samples();

      // Evaluate right-hand side
      auto rhs = rhs_.eval( samples.first );
      
      for (int64_t epoch = 0; epoch != options_.max_epoch(); ++epoch)
        {
          // Update from user-defined callback function
          auto status = get_epoch(epoch);
          
          // Reset gradients
          net_->zero_grad();    

          // Execute the model on the inputs
          auto pred = net_->forward(inputs);

          // Evaluate solution
          auto out = out_.eval( samples.first );
          
          // Compute the loss value
          auto loss_pde = torch::mse_loss( pred , rhs(0) );
          std::cout << "loss = " << loss_pde.template item<real_t>() << std::endl;
          
          // Compute gradients of the loss w.r.t. the model parameters
          loss_pde.backward({}, true, false);
          
          // Update the parameters based on the calculated gradients
          opt_.step();
          
          if (loss_pde.template item<real_t>() < options_.min_loss())
            break;
        }
    }
    
    /// @brief Returns a string representation of the IgANet object
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

    /// @brief Plots the B-Spline geometry
    inline auto plot_geo(int64_t xres=10, int64_t yres=10, int64_t zres=10) const
    {
      return geo_.plot(xres, yres, zres);
    }

    /// @brief Plots the B-Spline right-hand side
    inline auto plot_rhs(int64_t xres=10, int64_t yres=10, int64_t zres=10) const
    {
      return rhs_.plot(rhs_, xres, yres, zres);
    }

    /// @brief Plots the B-Spline solution
    inline auto plot_sol(int64_t xres=10, int64_t yres=10, int64_t zres=10) const
    {
      return rhs_.plot(sol_, xres, yres, zres);
    }

    /// @brief Saves the IgANet to file
    inline void save(const std::string& filename,
                     const std::string& key="iganet") const
    {
      torch::serialize::OutputArchive archive;
      write(archive, key).save_to(filename);
    }

    /// @brief Loads the IgANet from file
    inline void load(const std::string& filename,
                     const std::string& key="iganet")
    {
      torch::serialize::InputArchive archive;
      archive.load_from(filename);      
      read(archive, key);
    }

    /// @brief Writes the IgANet into a torch::serialize::OutputArchive object
    inline torch::serialize::OutputArchive& write(torch::serialize::OutputArchive& archive,
                                                  const std::string& key="iganet") const
    {
      archive.write(key+".GeoDim", torch::full({1}, GeoDim));
      archive.write(key+".PdeDim", torch::full({1}, PdeDim));

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

    /// @brief Loads the IgANet from a torch::serialize::InputArchive object
    inline torch::serialize::InputArchive& read(torch::serialize::InputArchive& archive,
                                                const std::string& key="iganet")
    {
      torch::Tensor tensor;

      archive.read(key+".GeoDim", tensor);
      if (tensor.item<int64_t>() != GeoDim)
        throw std::runtime_error("GeoDim mismatch");

      archive.read(key+".PdeDim", tensor);
      if (tensor.item<int64_t>() != PdeDim)
        throw std::runtime_error("PdeDim mismatch");

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

    /// @brief Returns true if both IgANet objects are the same
    bool operator==(const IgANet& other) const
    {
      bool result(true);

      result *= (geo_ == other.geo());
      result *= (rhs_ == other.rhs());
      result *= (sol_ == other.sol());

      return result;
    }

    /// @brief Returns true if both IgANet objects are different
    bool operator!=(const IgANet& other) const
    {
      return *this != other;
    }
  };

  /// @brief Print (as string) a IgANet object
  template<typename real_t,
           typename optimizer_t,
           short_t GeoDim,
           short_t PdeDim,
           template<typename, short_t, short_t...> class bspline_t,
           short_t... Degrees>
  inline std::ostream& operator<<(std::ostream& os,
                                  const IgANet<real_t, optimizer_t, GeoDim, PdeDim,
                                               bspline_t, Degrees...>& obj)
  {
    obj.pretty_print(os);
    return os;
  }

} // namespace iganet
