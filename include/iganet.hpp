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
#include <creator.hpp>
#include <functionspace.hpp>
#include <layer.hpp>
#include <zip.hpp>

namespace iganet {

  /// @brief Enumerator for the status of the various data
  enum class status : short_t
    {
      inputs  = 1<<0,          /*!< inputs need update  */
      geometry_samples = 1<<1, /*!< geometry samples need update */
      variable_samples = 1<<2  /*!< variable samples need update */
    };

  /// @brief Returns the sum of two status objects
  status operator+(enum status lhs, enum status rhs)
  {
    return status(static_cast<short_t>(lhs)+static_cast<short_t>(rhs));
  }

  /// @brief Returns true if flag is set in the given status
  bool operator&(enum status status, enum status flag)
  {
    return bool(static_cast<short_t>(status) & static_cast<short_t>(flag));
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
  class IgANetGeneratorImpl : public torch::nn::Module
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
          layers_.back()->to(dtype<real_t>());
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
                activations_.emplace_back( new Softmax{ std::any_cast<int64_t>(a[1]) } );
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
                activations_.emplace_back( new Softmin{ std::any_cast<int64_t>(a[1]) } );
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
      for (auto [layer, activation] : zip(layers_, activations_))
        x = activation->apply(layer->forward(x));
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
        archive.write(key+".layer["+std::to_string(i)+"].outputs_features",
                      torch::full({1}, (int64_t)layers_[i]->options.outputs_features()));
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
      torch::Tensor layers, in_features, outputs_features, bias, activation;
      
      archive.read(key+".layers", layers);
      for (int64_t i=0; i<layers.item<int64_t>(); ++i) {
        archive.read(key+".layer["+std::to_string(i)+"].in_features", in_features);
        archive.read(key+".layer["+std::to_string(i)+"].outputs_features", outputs_features);
        archive.read(key+".layer["+std::to_string(i)+"].bias", bias);
        layers_.emplace_back(register_module("layer["+std::to_string(i)+"]",
                                             torch::nn::Linear(
                                                               torch::nn::LinearOptions(in_features.item<int64_t>(),
                                                                                        outputs_features.item<int64_t>()
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

    inline virtual void pretty_print(std::ostream& os = std::cout) const override final
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
  ///
  /// This class implements the core functionality of IgANets
  template<typename optimizer_t,
           typename geometry_t,
           typename variable_t>
  class IgANet : public core<typename std::common_type<typename geometry_t::value_type,
                                                       typename variable_t::value_type>::type>
  {
  public:
    /// @brief Value type
    using value_type = typename std::common_type<typename geometry_t::value_type,
                                                 typename variable_t::value_type>::type;

    /// @brief Type of the geometry samples spline objects
    using geometry_samples_t = std::pair<typename geometry_t::eval_t,
                                         typename geometry_t::boundary_eval_t>;
    
    /// @brief Type of the variable samples spline objects
    using variable_samples_t = std::pair<typename variable_t::eval_t,
                                         typename variable_t::boundary_eval_t>;

  protected:
    /// @brief Spline representation of the geometry
    geometry_t geometry_;

    /// @brief Spline representation of the reference data
    variable_t variable_;
    
    /// @brief Spline representation of the network output
    variable_t outputs_;
    
    /// @brief IgANet generator
    IgANetGenerator<value_type> net_;

    /// @brief Optimizer
    optimizer_t opt_;

    /// @brief Options
    IgANetOptions options_;
    
    /// @brief Constructor: number of layers, activation functions,
    /// and number of spline coefficients (different for geometry_t
    /// and variable_t types)
    template<typename... geometry_spline_t, size_t... Is,
             typename... variable_spline_t, size_t... Js>
    IgANet(const std::vector<int64_t>&                layers,
           const std::vector<std::vector<std::any>>&  activations,
           std::tuple<geometry_spline_t...>           geometry_splines,
           std::index_sequence<Is...>,
           std::tuple<variable_spline_t...>           variable_splines,
           std::index_sequence<Js...>,
           IgANetOptions                              defaults = {},
           iganet::core<value_type>                   core = iganet::core<value_type>{})
    : iganet::core<value_type>(core),
      
      // Construct the different spline objects individually
      geometry_(std::get<Is>(geometry_splines)..., init::greville, core),
      variable_(std::get<Js>(variable_splines)..., init::zeros,    core),
      
      outputs_(std::get<Js>(variable_splines)..., init::random,   core),

      
      // Construct the deep neural network
      net_(concat(std::vector<int64_t>{inputs(0).size(0)},
                  layers,
                  std::vector<int64_t>{outputs_.as_tensor().size(0)}),
           activations),

      // Construct the optimizer
      opt_(net_->parameters()),

      // Set options
      options_(defaults)
    {}
    
  public:
    /// @brief Default constructor
    explicit IgANet(IgANetOptions defaults = {},
                    iganet::core<value_type> core = iganet::core<value_type>{})
      : iganet::core<value_type>(core),
        geometry_(),
        variable_(),
        outputs_(),
        opt_(net_->parameters()),
        options_(defaults)
    {}
    
    /// @brief Constructor: number of layers, activation functions,
    /// and number of spline coefficients (same for geometry_t and
    /// variable_t types)
    template<typename... spline_t>
    IgANet(const std::vector<int64_t>&               layers,
           const std::vector<std::vector<std::any>>& activations,
           std::tuple<spline_t...>                   splines,
           IgANetOptions                             defaults = {},
           iganet::core<value_type>                  core = iganet::core<value_type>{})
      : IgANet(layers, activations, splines, splines, defaults, core)
    {}

    /// @brief Constructor: number of layers, activation functions,
    /// and number of spline coefficients (different for geometry_t
    /// and variable_t types)
    template<typename... geometry_splines_t,
             typename... variable_splines_t>
    IgANet(const std::vector<int64_t>&               layers,
           const std::vector<std::vector<std::any>>& activations,
           std::tuple<geometry_splines_t...>         geometry_splines,
           std::tuple<variable_splines_t...>         variable_splines,
           IgANetOptions                             defaults = {},
           iganet::core<value_type>                  core = iganet::core<value_type>{})
      : IgANet(layers, activations,
               geometry_splines, std::make_index_sequence<sizeof...(geometry_splines_t)>{},
               variable_splines, std::make_index_sequence<sizeof...(variable_splines_t)>{},
               defaults, core)
    {}
    
    /// @brief Returns a constant reference to the IgANet generator
    inline const IgANetGenerator<value_type>& net() const
    {
      return net_;
    }

    /// @brief Returns a non-constant reference to the IgANet generator
    inline IgANetGenerator<value_type>& net()
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
    
    /// @brief Returns a constant reference to the spline
    /// representation of the geometry
    inline const geometry_t& geo() const
    {
      return geometry_;
    }

    /// @brief Returns a non-constant reference to the spline
    /// representation of the geometry
    inline geometry_t& geo()
    {
      return geometry_;
    }

    /// @brief Returns a constant reference to the spline
    /// representation of the variables
    inline const variable_t& variable() const
    {
      return variable_;
    }

    /// @brief Returns a non-constant reference to the spline
    /// representation of the variables
    inline variable_t& variable()
    {
      return variable_;
    }

    /// @brief Returns a constant reference to the spline
    /// representation of the network's output
    inline const variable_t& outputs() const
    {
      return outputs_;
    }

    /// @brief Returns a non-constant reference to the spline
    /// representation of the network's output
    inline variable_t& outputs()
    {
      return outputs_;
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

  private:
    /// @brief Returns the geometry samples
    ///
    /// In the default implementation the samples are the Greville
    /// abscissae in the interior of the domain and on the boundary
    /// faces. This behavior can be changed by overriding this virtual
    /// function in a derived class.
    template<size_t... Is>
    geometry_samples_t geometry_samples(std::index_sequence<Is...>) const
    {      
      geometry_samples_t samples_;
      
      // Get Greville abscissae inside the domain
      ((std::get<Is>(samples_.first) = std::get<Is>(geometry_).greville()), ...);
      
      // Get Greville abscissae at the domain
      ((std::get<Is>(samples_.second) = std::get<Is>(geometry_.boundary()).greville()), ...);
      
      return samples_;
    }

    /// @brief Returns the variable samples
    ///
    /// In the default implementation the samples are the Greville
    /// abscissae in the interior of the domain and on the boundary
    /// faces. This behavior can be changed by overriding this virtual
    /// function in a derived class.
    template<size_t... Is>
    variable_samples_t variable_samples(std::index_sequence<Is...>) const
    {      
      variable_samples_t samples_;
      
      // Get Greville abscissae inside the domain
      ((std::get<Is>(samples_.first) = std::get<Is>(variable_).greville()), ...);
      
      // Get Greville abscissae at the domain
      ((std::get<Is>(samples_.second) = std::get<Is>(variable_.boundary()).greville()), ...);
      
      return samples_;
    }

  public:
    /// @brief Returns the geometry samples
    ///
    /// In the default implementation the samples are the Greville
    /// abscissae in the interior of the domain and on the boundary
    /// faces. This behavior can be changed by overriding this virtual
    /// function in a derived class.
    virtual geometry_samples_t geometry_samples(int64_t epoch) const
    {
      if constexpr (geometry_t::dim() == 1)
        return {geometry_.greville(), geometry_.boundary().greville()};
      else
        return geometry_samples(std::make_index_sequence<geometry_t::dim()>{});
    }

    /// @brief Returns the variable samples
    ///
    /// In the default implementation the samples are the Greville
    /// abscissae in the interior of the domain and on the boundary
    /// faces. This behavior can be changed by overriding this virtual
    /// function in a derived class.
    virtual variable_samples_t variable_samples(int64_t epoch) const
    {
      if constexpr (variable_t::dim() == 1)
        return {variable_.greville(), variable_.boundary().greville()};
      else
        return variable_samples(std::make_index_sequence<variable_t::dim()>{});
    }
    
    /// @brief Returns the network inputs
    ///
    /// In the default implementation the inputs are the controll
    /// points of the geometry and the reference spline objects. This
    /// behavior can be changed by overriding this virtual function in
    /// a derived class.
    virtual torch::Tensor inputs(int64_t epoch) const
    {
      return torch::cat({geometry_.as_tensor(), variable_.as_tensor()});
    }
    
    /// @brief Initializes epoch
    virtual enum status epoch(int64_t) = 0;

    /// @brief Computes the loss function
    virtual torch::Tensor loss(const torch::Tensor&,
                               const geometry_samples_t&,
                               const variable_samples_t&,
                               int64_t, enum status) = 0;
    
    /// @brief Trains the IgANet
    virtual void train()
    {
      torch::Tensor inputs, outputs, loss;
      geometry_samples_t geometry_samples;
      variable_samples_t variable_samples;
      status status;
      
      // Loop over epochs
      for (int64_t epoch = 0; epoch != options_.max_epoch(); ++epoch)
        {
          // Update from user-defined callback function
          status = this->epoch(epoch);
          
          if (status & status::inputs)
            inputs = this->inputs(epoch);
          
          if (status & status::geometry_samples)
            geometry_samples = this->geometry_samples(epoch);

          if (status & status::variable_samples)
            variable_samples = this->variable_samples(epoch);
          
          // Reset gradients
          net_->zero_grad();    
          
          // Execute the model on the inputs
          outputs = net_->forward(inputs);
          
          // Compute the loss value
          loss = this->loss(outputs, geometry_samples, variable_samples, epoch, status);
          
          // Compute gradients of the loss w.r.t. the model parameters
          loss.backward({}, true, false);
          
          // Update the parameters based on the calculated gradients
          opt_.step();
          
          if (loss.template item<value_type>() < options_.min_loss())
            break;
        }
    }

    /// @brief Returns a string representation of the IgANet object
    inline virtual void pretty_print(std::ostream& os = std::cout) const
    {
      os << core<value_type>::name()
         << "(\n"
         << "net = " << net_ << "\n"
         << "geo = " << geometry_ << "\n"
         << "ref = " << variable_ << "\n"
         << "out = " << outputs_
         << "\n)";
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
      geometry_.write(archive, key+".geo");
      variable_.write(archive, key+".ref");
      outputs_.write(archive, key+".out");
      
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
      geometry_.read(archive, key+".geo");
      variable_.read(archive, key+".ref");
      outputs_.read(archive, key+".out");

      net_->read(archive, key+".net");      
      torch::serialize::InputArchive archive_net;      
      archive.read(key+".net.data", archive_net);
      net_->load(archive_net);

      opt_.add_parameters(net_->parameters());      
      torch::serialize::InputArchive archive_opt;
      archive.read(key+".opt", archive_opt);            
      opt_.load(archive_opt);
      
      return archive;
    }

    /// @brief Returns true if both IgANet objects are the same
    bool operator==(const IgANet& other) const
    {
      bool result(true);

      result *= (geometry_ == other.geometry());
      result *= (variable_ == other.variable());
      result *= (outputs_  == other.outputs());

      return result;
    }

    /// @brief Returns true if both IgANet objects are different
    bool operator!=(const IgANet& other) const
    {
      return *this != other;
    }
  };

  /// @brief Print (as string) a IgANet object
  template<typename optimizer_t,
           typename geometry_t,
           typename variable_t>
  inline std::ostream& operator<<(std::ostream& os,
                                  const IgANet<optimizer_t, geometry_t, variable_t>& obj)
  {
    obj.pretty_print(os);
    return os;
  }

  /// @brief IgANetCustomizable
  ///
  /// This class implements a customizable variant of IgANets that
  /// provides types and attributes for precomputing indices and basis
  /// functions
  template<typename optimizer_t,
           typename geometry_t,
           typename variable_t>
  class IgANetCustomizable
  {    
  public:
    /// @brief Type of the knot indices of geometry_t type in the interior
    using geometry_interior_knot_indices_t =
      decltype(std::declval<geometry_t>().template find_knot_indices<functionspace::interior>(std::declval<typename geometry_t::eval_t>()));

    /// @brief Type of the knot indices of geometry_t type at the boundary
    using geometry_boundary_knot_indices_t =
      decltype(std::declval<geometry_t>().template find_knot_indices<functionspace::boundary>(std::declval<typename geometry_t::boundary_eval_t>()));
  
    /// @brief Type of the knot indices of variable_t type in the interior
    using variable_interior_knot_indices_t =
      decltype(std::declval<variable_t>().template find_knot_indices<functionspace::interior>(std::declval<typename variable_t::eval_t>()));
  
    /// @brief Type of the knot indices of boundary_t type at the boundary
    using variable_boundary_knot_indices_t =
      decltype(std::declval<variable_t>().template find_knot_indices<functionspace::boundary>(std::declval<typename variable_t::boundary_eval_t>()));

  protected:
    /// @brief Knot indices of geometry_t type in the interior
    geometry_interior_knot_indices_t geometry_interior_knot_indices_;

    /// @brief Knot indices of geometry_t type at the boundary
    geometry_boundary_knot_indices_t geometry_boundary_knot_indices_;
  
    /// @brief Knot indices of variable_t type in the interior
    variable_interior_knot_indices_t variable_interior_knot_indices_;

    /// @brief Knot indices of variable_t type at the boundary
    variable_boundary_knot_indices_t variable_boundary_knot_indices_;

  public:
    /// @brief Type of the coefficient indices of geometry_t type in the interior
    using geometry_interior_coeff_indices_t =
      decltype(std::declval<geometry_t>().template find_coeff_indices<functionspace::interior>(std::declval<typename geometry_t::eval_t>()));

    /// @brief Type of the coefficient indices of geometry_t type at the boundary
    using geometry_boundary_coeff_indices_t =
      decltype(std::declval<geometry_t>().template find_coeff_indices<functionspace::boundary>(std::declval<typename geometry_t::boundary_eval_t>()));
  
    /// @brief Type of the coefficient indices of variable_t type in the interior
    using variable_interior_coeff_indices_t =
      decltype(std::declval<variable_t>().template find_coeff_indices<functionspace::interior>(std::declval<typename variable_t::eval_t>()));
  
    /// @brief Type of the coefficient indices of variable_t type at the boundary
    using variable_boundary_coeff_indices_t =
      decltype(std::declval<variable_t>().template find_coeff_indices<functionspace::boundary>(std::declval<typename variable_t::boundary_eval_t>()));

  protected:
    /// @brief Coefficient indices of geometry_t type in the interior
    geometry_interior_coeff_indices_t geometry_interior_coeff_indices_;

    /// @brief Coefficient indices of geometry_t type at the vboundary
    geometry_boundary_coeff_indices_t geometry_boundary_coeff_indices_;
  
    /// @brief Coefficient indices of variable_t type in the interior
    variable_interior_coeff_indices_t variable_interior_coeff_indices_;

    /// @brief Coefficient indices of variable_t type at the boundary
    variable_boundary_coeff_indices_t variable_boundary_coeff_indices_;
  };
  
} // namespace iganet
