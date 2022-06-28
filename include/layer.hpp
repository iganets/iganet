/**
   @file include/layer.hpp

   @brief Network layer

   @author Matthias Moller

   @copyright This file is part of the IgaNet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include <core.hpp>

#pragma once

namespace iganet {

  /// Enumerator for nonlinear activation functions
  enum class activation : short_t
    {
      none                         =  0,
      batch_norm                   =  1,
      celu                         =  2,
      elu                          =  3,
      gelu                         =  4,
      glu                          =  5,
      group_norm                   =  6,
      gumble_softmax               =  7,
      hardshrink                   =  9,      
      hardsigmoid                  =  8,      
      hardswish                    = 10,      
      hardtanh                     = 11,      
      instance_norm                = 12,      
      layer_norm                   = 13,
      leaky_relu                   = 14,      
      local_response_norm          = 15,
      logsigmoid                   = 16,      
      logsoftmax                   = 17,      
      mish                         = 18,      
      normalize                    = 19,      
      prelu                        = 20,      
      relu                         = 21,      
      relu6                        = 22,      
      rrelu                        = 23,      
      selu                         = 24,      
      sigmoid                      = 25,      
      silu                         = 26,      
      softmax                      = 27,
      softmin                      = 28,      
      softplus                     = 29,      
      softshrink                   = 30,      
      softsign                     = 31,      
      tanh                         = 32,
      tanhshrink                   = 33,
      threshold                    = 34
    };

  /// Abstract activation function structure
  class ActivationFunction
  {
  public:
    virtual torch::Tensor apply(const torch::Tensor&) const = 0;
    virtual ~ActivationFunction() = default;
    virtual void pretty_print(std::ostream& os = std::cout) const = 0;
  };

  /// Print (as string) an ActivationFunction object
  inline std::ostream& operator<<(std::ostream& os,
                                  const ActivationFunction& obj)
  {
    obj.pretty_print(os);
    return os;
  }
  
  /// No-op activation function
  class None : public ActivationFunction
  {
  public:
    inline virtual torch::Tensor apply(const torch::Tensor& input) const override
    {
      return input;
    }
    
    inline void pretty_print(std::ostream& os = std::cout) const override
    {
      os << "none";
    }
  };

  /// Batch Normalization as described in the paper
  ///
  /// Batch Normalization: Accelerating Deep Network Training by
  /// Reducing Internal Covariate Shift,
  /// https://arxiv.org/abs/1502.03167
  class BatchNorm : public ActivationFunction
  {
  public:
    explicit BatchNorm(std::function<const torch::Tensor&()> running_mean,
                       std::function<const torch::Tensor&()> running_var,
                       const torch::nn::functional::BatchNormFuncOptions& options = {})
      : running_mean(running_mean),
        running_var(running_var),
        options_(options) {}

    explicit BatchNorm(std::function<const torch::Tensor&()> running_mean,
                       std::function<const torch::Tensor&()> running_var,
                       const torch::Tensor& weight,
                       const torch::Tensor& bias,
                       double eps, double momentum,
                       bool training=false)
      : running_mean(running_mean),
        running_var(running_var),
        options_(torch::nn::functional::BatchNormFuncOptions()
                 .weight(weight)
                 .bias(bias)
                 .eps(eps)
                 .momentum(momentum)
                 .training(training)) {}

    ~BatchNorm() = default;
    
    inline virtual torch::Tensor apply(const torch::Tensor& input) const override
    {
      return torch::nn::functional::batch_norm(input, running_mean(), running_var(), options_);
    }
    
    const torch::nn::functional::BatchNormFuncOptions& options() const
    {
      return options_;
    }

    torch::nn::functional::BatchNormFuncOptions& options()
    {
      return options_;
    }

    inline void pretty_print(std::ostream& os = std::cout) const override
    {
      os << "BatchNorm"
         << "(\n  eps=" << options_.eps()
         << ", momentum=" << options_.momentum().value()
         << ", training=" << options_.training();

      if (is_verbose(os)) {
        os << "\n  running_mean = " << running_mean()
           << "\n  running_var = " << running_var()
           << "\n  weight = " << options_.weight()
           << "\n  bias = " << options_.bias();
      }
      
      os << "\n)";
    }
    
  private:
    torch::nn::functional::BatchNormFuncOptions options_;
    std::function<torch::Tensor()> running_mean, running_var;
  };
  
  /// Continuously Differentiable Exponential Linear Units activation function
  ///
  /// \f[
  ///     \text{CELU}(x) = \max(0,x) + \min(0, \alpha * (\exp(x/\alpha)-1))
  /// \f]
  class CELU : public ActivationFunction
  {
  public:
    explicit CELU(const torch::nn::functional::CELUFuncOptions& options = {})
      : options_(options) {}

    explicit CELU(double alpha, bool inplace=false)
      : options_(torch::nn::functional::CELUFuncOptions()
                 .alpha(alpha)
                 .inplace(inplace)) {}

    ~CELU() = default;
    
    inline virtual torch::Tensor apply(const torch::Tensor& input) const override
    {
      return torch::nn::functional::celu(input, options_);
    }
    
    const torch::nn::functional::CELUFuncOptions& options() const
    {
      return options_;
    }

    torch::nn::functional::CELUFuncOptions& options()
    {
      return options_;
    }

    inline void pretty_print(std::ostream& os = std::cout) const override
    {
      os << "CELU"
         << "(\n  alpha=" << options_.alpha()
         << ", inplace=" << options_.inplace()
         << "\n)";
    }
    
  private:
    torch::nn::functional::CELUFuncOptions options_;
  };

  /// Exponential Linear Units activation function
  ///
  /// \f[
  ///     \text{ELU}(x) =
  ///       \begin{cases}
  ///         x                    & \text{ if } x > 0
  ///         \alpha * (\exp(x)-1) & \text{ if } x \le 0
  ///       \end{cases}
  /// \f]
  class ELU : public ActivationFunction
  {
  public:
    explicit ELU(const torch::nn::functional::ELUFuncOptions& options = {})
      : options_(options) {}

    explicit ELU(double alpha, bool inplace=false)
      : options_(torch::nn::functional::ELUFuncOptions()
                 .alpha(alpha)
                 .inplace(inplace)) {}

    ~ELU() = default;
    
    inline virtual torch::Tensor apply(const torch::Tensor& input) const override
    {
      return torch::nn::functional::elu(input, options_);
    }
    
    const torch::nn::functional::ELUFuncOptions& options() const
    {
      return options_;
    }

    torch::nn::functional::ELUFuncOptions& options()
    {
      return options_;
    }

    inline void pretty_print(std::ostream& os = std::cout) const override
    {
      os << "ELU"
         << "(\n  alpha=" << options_.alpha()
         << ", inplace=" << options_.inplace()
         << "\n)";
    }
    
  private:
    torch::nn::functional::ELUFuncOptions options_;
  };

  /// Gaussian Error Linear Units activation function
  ///
  /// \f[
  ///     \text{GELU}(x) = x * \Psi(x),
  /// \f]
  ///
  /// where \f$\Psi(x)$\f is the Cumulative Distribution Function for
  /// Gaussian Distribution
  class GELU : public ActivationFunction
  {
  public:
    explicit GELU() = default;

    ~GELU() = default;
    
    inline virtual torch::Tensor apply(const torch::Tensor& input) const override
    {
      return torch::gelu(input);
    }

    inline void pretty_print(std::ostream& os = std::cout) const override
    {
      os << "GELU";
    }
  };

  /// Grated Linear Units activation function
  ///
  /// \f[
  ///     \text{GLU}(a,b) = a \otimes \sigma(b),
  /// \f]
  ///
  /// where input is split in half along dim to form \f$ a \f$ and \f$
  /// b \f$,
  /// \f$ \sigma \f$ is the sigmoid function and \f$ \otimes \f$
  /// is the element-wise product between matrices.
  class GLU : public ActivationFunction
  {
  public:
    explicit GLU(const torch::nn::functional::GLUFuncOptions& options = {})
      : options_(options) {}

    explicit GLU(int dim)
      : options_(torch::nn::functional::GLUFuncOptions()
                 .dim(dim)) {}

    ~GLU() = default;
    
    inline virtual torch::Tensor apply(const torch::Tensor& input) const override
    {
      return torch::nn::functional::glu(input, options_);
    }
    
    const torch::nn::functional::GLUFuncOptions& options() const
    {
      return options_;
    }

    torch::nn::functional::GLUFuncOptions& options()
    {
      return options_;
    }

    inline void pretty_print(std::ostream& os = std::cout) const override
    {
      os << "GLU"
         << "(\n  dim=" << options_.dim()
         << "\n)";
    }
    
  private:
    torch::nn::functional::GLUFuncOptions options_;
  };

  /// Group Normalization over a mini-batch of inputs as described in
  /// the paper Group Normalization, https://arxiv.org/abs/1803.08494
  class GroupNorm : public ActivationFunction
  {
  public:
    explicit GroupNorm(int64_t num_groups)
      : options_(torch::nn::functional::GroupNormFuncOptions(num_groups)) {}
    
    explicit GroupNorm(const torch::nn::functional::GroupNormFuncOptions& options)
      : options_(options) {}

    explicit GroupNorm(int64_t num_groups,
                       const torch::Tensor& weight,
                       const torch::Tensor& bias,
                       double eps)
      : options_(torch::nn::functional::GroupNormFuncOptions(num_groups)
                 .weight(weight)
                 .bias(bias)
                 .eps(eps)) {}

    ~GroupNorm() = default;
    
    inline virtual torch::Tensor apply(const torch::Tensor& input) const override
    {
      return torch::nn::functional::group_norm(input, options_);
    }
    
    const torch::nn::functional::GroupNormFuncOptions& options() const
    {
      return options_;
    }

    torch::nn::functional::GroupNormFuncOptions& options()
    {
      return options_;
    }

    inline void pretty_print(std::ostream& os = std::cout) const override
    {
      os << "GroupNorm"
         << "(\n  eps=" << options_.eps();
      
      if (is_verbose(os)) {
        os << "\n  weight = " << options_.weight()
           << "\n  bias = " << options_.bias();
      }
      
      os << "\n)";
    }
    
  private:
    torch::nn::functional::GroupNormFuncOptions options_;
  };

  /// Gumbel-Softmax distribution activation function
  class GumbelSoftmax : public ActivationFunction
  {
  public:
    explicit GumbelSoftmax(const torch::nn::functional::GumbelSoftmaxFuncOptions& options = {})
      : options_(options) {}

    explicit GumbelSoftmax(double tau, int dim, bool hard)
      : options_(torch::nn::functional::GumbelSoftmaxFuncOptions()
                 .tau(tau)
                 .dim(dim)
                 .hard(hard)) {}

    ~GumbelSoftmax() = default;
    
    inline virtual torch::Tensor apply(const torch::Tensor& input) const override
    {
      return torch::nn::functional::gumbel_softmax(input, options_);
    }
    
    const torch::nn::functional::GumbelSoftmaxFuncOptions& options() const
    {
      return options_;
    }

    torch::nn::functional::GumbelSoftmaxFuncOptions& options()
    {
      return options_;
    }

    inline void pretty_print(std::ostream& os = std::cout) const override
    {
      os << "GumbelSoftmax"
         << "(\n  tau=" << options_.tau()
         << ", dim=" << options_.dim()
         << ", hard=" << options_.hard()
         << "\n)";
    }
    
  private:
    torch::nn::functional::GumbelSoftmaxFuncOptions options_;
  };

  /// Hard shrinkish activation function
  class HardShrink : public ActivationFunction
  {
  public:
    explicit HardShrink(const torch::nn::functional::HardshrinkFuncOptions& options = {})
      : options_(options) {}

    explicit HardShrink(double lambda)
      : options_(torch::nn::functional::HardshrinkFuncOptions()
                 .lambda(lambda)) {}

    ~HardShrink() = default;
    
    inline virtual torch::Tensor apply(const torch::Tensor& input) const override
    {
      return torch::nn::functional::hardshrink(input, options_);
    }
    
    const torch::nn::functional::HardshrinkFuncOptions& options() const
    {
      return options_;
    }

    torch::nn::functional::HardshrinkFuncOptions& options()
    {
      return options_;
    }

    inline void pretty_print(std::ostream& os = std::cout) const override
    {
      os << "HardShrink"
         << "(\n  lambda=" << options_.lambda()
         << "\n)";
    }
    
  private:
    torch::nn::functional::HardshrinkFuncOptions options_;
  };

  /// HardSigmoid activation function
  ///
  /// \f[
  ///     \text{Hardsigmoid}(x) =
  ///       \begin{cases}
  ///         0         & \text{ if } x \le -3
  ///         1         & \text{ if } x \ge +3
  ///         x/6 + 1/2 & \text{ otherwise }
  ///       \end{cases}
  /// \f]
  class HardSigmoid : public ActivationFunction
  {
  public:
    explicit HardSigmoid() = default;

    ~HardSigmoid() = default;
    
    inline virtual torch::Tensor apply(const torch::Tensor& input) const override
    {
      return torch::hardsigmoid(input);
    }

    inline void pretty_print(std::ostream& os = std::cout) const override
    {
      os << "HardSigmoid";
    }
  };

  /// HardSwish activation function
  ///
  /// \f[
  ///     \text{Hardswish}(x) =
  ///       \begin{cases}
  ///         0         & \text{ if } x \le -3
  ///         x         & \text{ if } x \ge +3
  ///         x*(x+3)/6 & \text{ otherwise }
  ///       \end{cases}
  /// \f]
  class HardSwish : public ActivationFunction
  {
  public:
    explicit HardSwish() = default;

    ~HardSwish() = default;
    
    inline virtual torch::Tensor apply(const torch::Tensor& input) const override
    {
      return torch::hardswish(input);
    }

    inline void pretty_print(std::ostream& os = std::cout) const override
    {
      os << "HardSwish";
    }
  };
  
  /// HardTanh activation function
  ///
  /// \f[
  ///     \text{HardTanh}(x) =
  ///       \begin{cases}
  ///         +1 & \text{ if } x > +1
  ///         -1 & \text{ if } x < -1
  ///          x & \text{ otherwise }
  ///       \end{cases}
  /// \f]
  class HardTanh : public ActivationFunction
  {
  public:
    explicit HardTanh(const torch::nn::functional::HardtanhFuncOptions& options = {})
      : options_(options) {}

    explicit HardTanh(double min_val, double max_val, bool inplace=false)
      : options_(torch::nn::functional::HardtanhFuncOptions()
                 .min_val(min_val)
                 .max_val(max_val)
                 .inplace(inplace)) {}

    ~HardTanh() = default;
    
    inline virtual torch::Tensor apply(const torch::Tensor& input) const override
    {
      return torch::nn::functional::hardtanh(input, options_);
    }
    
    const torch::nn::functional::HardtanhFuncOptions& options() const
    {
      return options_;
    }

    torch::nn::functional::HardtanhFuncOptions& options()
    {
      return options_;
    }

    inline void pretty_print(std::ostream& os = std::cout) const override
    {
      os << "HardTanh"
         << "(\n  min_val=" << options_.min_val()
         << ", max_val="  << options_.max_val()
         << ", inplace="  << options_.inplace()
         << "\n)";
    }
    
  private:
    torch::nn::functional::HardtanhFuncOptions options_;
  };

  /// Instance Normalization as described in the paper
  ///
  /// Instance Normalization: The Missing Ingredient for Fast
  /// Stylization, https://arxiv.org/abs/1607.08022
  class InstanceNorm : public ActivationFunction
  {
  public:
    explicit InstanceNorm(const torch::nn::functional::InstanceNormFuncOptions& options = {})
      : options_(options) {}

    explicit InstanceNorm(const torch::Tensor& running_mean,
                          const torch::Tensor& running_var,
                          const torch::Tensor& weight,
                          const torch::Tensor& bias,
                          double eps, double momentum,
                          bool use_input_stats=true)
      : options_(torch::nn::functional::InstanceNormFuncOptions()
                 .running_mean(running_mean)
                 .running_var(running_var)
                 .weight(weight)
                 .bias(bias)
                 .eps(eps)
                 .momentum(momentum)
                 .use_input_stats(use_input_stats)) {}

    ~InstanceNorm() = default;
    
    inline virtual torch::Tensor apply(const torch::Tensor& input) const override
    {
      return torch::nn::functional::instance_norm(input, options_);
    }
    
    const torch::nn::functional::InstanceNormFuncOptions& options() const
    {
      return options_;
    }

    torch::nn::functional::InstanceNormFuncOptions& options()
    {
      return options_;
    }

    inline void pretty_print(std::ostream& os = std::cout) const override
    {
      os << "InstanceNorm"
         << "(\n  eps=" << options_.eps()
         << ", momentum=" << options_.momentum()
         << ", use_input_stats=" << options_.use_input_stats();

      if (is_verbose(os)) {
        os << "\n  running_mean = " << options_.running_mean()
           << "\n  running_var = " << options_.running_var()
           << "\n  weight = " << options_.weight()
           << "\n  bias = " << options_.bias();
      }
      
      os << "\n)";
    }
    
  private:
    torch::nn::functional::InstanceNormFuncOptions options_;
  };

  /// Layer Normalization as described in the paper
  ///
  /// Layer Normalization, https://arxiv.org/abs/1607.06450
  class LayerNorm : public ActivationFunction
  {
  public:
    explicit LayerNorm(std::vector<int64_t> normalized_shape)
      : options_(torch::nn::functional::LayerNormFuncOptions(normalized_shape)) {}
    
    explicit LayerNorm(const torch::nn::functional::LayerNormFuncOptions& options)
      : options_(options) {}
    
    explicit LayerNorm(std::vector<int64_t> normalized_shape,
                       const torch::Tensor& weight,
                       const torch::Tensor& bias,
                       double eps)
      : options_(torch::nn::functional::LayerNormFuncOptions(normalized_shape)
                 .weight(weight)
                 .bias(bias)
                 .eps(eps)) {}

    ~LayerNorm() = default;
    
    inline virtual torch::Tensor apply(const torch::Tensor& input) const override
    {
      return torch::nn::functional::layer_norm(input, options_);
    }
    
    const torch::nn::functional::LayerNormFuncOptions& options() const
    {
      return options_;
    }

    torch::nn::functional::LayerNormFuncOptions& options()
    {
      return options_;
    }

    inline void pretty_print(std::ostream& os = std::cout) const override
    {
      os << "LayerNorm"
         << "(\n  eps=" << options_.eps();

      if (is_verbose(os)) {
        os << "\n  normalized_shape = " << options_.normalized_shape()
           << "\n  weight = " << options_.weight()
           << "\n  bias = " << options_.bias();
      }
      
      os << "\n)";
    }
    
  private:
    torch::nn::functional::LayerNormFuncOptions options_;
  };
  
  /// Leaky ReLU activation function
  ///
  /// \f[
  ///     \text{LeakyReLU}(x) =
  ///       \begin{cases}
  ///         x                         & \text{ if } x \ge 0
  ///         \text{negative_slope} * x & \text{ otherwise }
  ///       \end{cases}
  /// \f]
  class LeakyReLU : public ActivationFunction
  {
  public:
    explicit LeakyReLU(const torch::nn::functional::LeakyReLUFuncOptions& options = {})
      : options_(options) {}

    explicit LeakyReLU(double negative_slope, bool inplace=false)
      : options_(torch::nn::functional::LeakyReLUFuncOptions()
                 .negative_slope(negative_slope)
                 .inplace(inplace)) {}

    ~LeakyReLU() = default;
    
    inline virtual torch::Tensor apply(const torch::Tensor& input) const override
    {
      return torch::nn::functional::leaky_relu(input, options_);
    }
    
    const torch::nn::functional::LeakyReLUFuncOptions& options() const
    {
      return options_;
    }

    torch::nn::functional::LeakyReLUFuncOptions& options()
    {
      return options_;
    }

    inline void pretty_print(std::ostream& os = std::cout) const override
    {
      os << "LeakyReLU"
         << "(\n  negative_slope=" << options_.negative_slope()
         << ", inplace="  << options_.inplace()
         << "\n)";
    }
    
  private:
    torch::nn::functional::LeakyReLUFuncOptions options_;
  };

  /// Local response Normalization
  class LocalResponseNorm : public ActivationFunction
  {
  public:
    explicit LocalResponseNorm(int64_t size)
      : options_(torch::nn::functional::LocalResponseNormFuncOptions(size)) {}
    
    explicit LocalResponseNorm(const torch::nn::functional::LocalResponseNormFuncOptions& options)
      : options_(options) {}
    
    explicit LocalResponseNorm(int64_t size,
                               double alpha, double beta, double k)
      : options_(torch::nn::functional::LocalResponseNormFuncOptions(size)
                 .alpha(alpha)
                 .beta(beta)
                 .k(k)) {}

    ~LocalResponseNorm() = default;
    
    inline virtual torch::Tensor apply(const torch::Tensor& input) const override
    {
      return torch::nn::functional::local_response_norm(input, options_);
    }
    
    const torch::nn::functional::LocalResponseNormFuncOptions& options() const
    {
      return options_;
    }

    torch::nn::functional::LocalResponseNormFuncOptions& options()
    {
      return options_;
    }

    inline void pretty_print(std::ostream& os = std::cout) const override
    {
      os << "LocalResponseNorm"
         << "(\n  size=" << options_.size()
         << ", alpha="  << options_.alpha()
         << ", beta="  << options_.beta()
         << ", k="  << options_.k()
         << "\n)";
    }
    
  private:
    torch::nn::functional::LocalResponseNormFuncOptions options_;
  };

  /// LogSigmoid activation function
  ///
  /// \f[
  ///     \text{LogSigmoid}(x) = \log \left( \frac{1}{1+\exp(-x)} \right)
  /// \f]
  class LogSigmoid : public ActivationFunction
  {
  public:
    explicit LogSigmoid() = default;

    ~LogSigmoid() = default;
    
    inline virtual torch::Tensor apply(const torch::Tensor& input) const override
    {
      return torch::log_sigmoid(input);
    }

    inline void pretty_print(std::ostream& os = std::cout) const override
    {
      os << "LogSigmoid";
    }
  };

  /// LogSoftmax activation function
  ///
  /// \f[
  ///     \text{LogSigmoid}(x_i) = \log \left( \frac{\exp(x_i)}{\sum_j \exp(x_j)} \right)
  /// \f]
  ///
  /// where \f$ x \f$ is an \f$n\f$-dimensional input tensor
  class LogSoftmax : public ActivationFunction
  {
  public:
    explicit LogSoftmax(int64_t dim)
      : options_(torch::nn::functional::LogSoftmaxFuncOptions(dim)) {}
    
    explicit LogSoftmax(const torch::nn::functional::LogSoftmaxFuncOptions& options)
      : options_(options) {}

    ~LogSoftmax() = default;
    
    inline virtual torch::Tensor apply(const torch::Tensor& input) const override
    {
      return torch::nn::functional::log_softmax(input, options_);
    }

    const torch::nn::functional::LogSoftmaxFuncOptions& options() const
    {
      return options_;
    }

    torch::nn::functional::LogSoftmaxFuncOptions& options()
    {
      return options_;
    }

    inline void pretty_print(std::ostream& os = std::cout) const override
    {
      os << "LogSoftmax"
         << "(\n  dim=" << options_.dim()
         << "\n)";
    }
    
  private:
    torch::nn::functional::LogSoftmaxFuncOptions options_;
  };

  /// Mish activation function
  ///
  /// \f[
  ///     \text{Mish}(x) = x * \tanh(\text{Softplus}(x))
  /// \f]
  class Mish : public ActivationFunction
  {
  public:
    explicit Mish() = default;

    ~Mish() = default;
    
    inline virtual torch::Tensor apply(const torch::Tensor& input) const override
    {
      return torch::mish(input);
    }

    inline void pretty_print(std::ostream& os = std::cout) const override
    {
      os << "Mish";
    }
  };

  /// \f$L_p\f$ Normalization
  class Normalize : public ActivationFunction
  {
  public:
    explicit Normalize(const torch::nn::functional::NormalizeFuncOptions& options = {})
      : options_(options) {}

    explicit Normalize(double p, double eps, int64_t dim)
      : options_(torch::nn::functional::NormalizeFuncOptions()
                 .p(p)
                 .eps(eps)
                 .dim(dim)) {}

    ~Normalize() = default;
    
    inline virtual torch::Tensor apply(const torch::Tensor& input) const override
    {
      return torch::nn::functional::normalize(input, options_);
    }
    
    const torch::nn::functional::NormalizeFuncOptions& options() const
    {
      return options_;
    }

    torch::nn::functional::NormalizeFuncOptions& options()
    {
      return options_;
    }

    inline void pretty_print(std::ostream& os = std::cout) const override
    {
      os << "Normalize"
         << "(\n  eps=" << options_.eps()
         << "(\n  p=" << options_.p()
         << "(\n  dim=" << options_.dim()
         << "\n)";
    }
    
  private:
    torch::nn::functional::NormalizeFuncOptions options_;
  };

  /// PReLU activation function
  class PReLU : public ActivationFunction
  {
  public:
    explicit PReLU(std::function<const torch::Tensor&()> weight)
      : weight(weight) {}

    ~PReLU() = default;
    
    inline virtual torch::Tensor apply(const torch::Tensor& input) const override
    {
      return torch::nn::functional::prelu(input, weight());
    }

    inline void pretty_print(std::ostream& os = std::cout) const override
    {
      os << "PReLU";
      
      if (is_verbose(os))
        os << "(\n  weight = " << weight() << "\n)";
    }
    
  private:
    std::function<torch::Tensor()> weight;
  };

  /// ReLU activation function
  ///
  /// \f[
  ///     \text{ReLU}(x) = \max(0,x)
  /// \f]
  class ReLU : public ActivationFunction
  {
  public:
    explicit ReLU(const torch::nn::functional::ReLUFuncOptions& options = {})
      : options_(options) {}

    explicit ReLU(bool inplace)
      : options_(torch::nn::functional::ReLUFuncOptions()
                 .inplace(inplace)) {}

    ~ReLU() = default;
    
    inline virtual torch::Tensor apply(const torch::Tensor& input) const override
    {
      return torch::nn::functional::relu(input, options_);
    }
    
    const torch::nn::functional::ReLUFuncOptions& options() const
    {
      return options_;
    }

    torch::nn::functional::ReLUFuncOptions& options()
    {
      return options_;
    }

    inline void pretty_print(std::ostream& os = std::cout) const override
    {
      os << "ReLU"
         << "(\n  inplace=" << options_.inplace()
         << "\n)";
    }
    
  private:
    torch::nn::functional::ReLUFuncOptions options_;
  };

  /// ReLU6 activation function
  ///
  /// \f[
  ///     \text{ReLU6}(x) = \min(\max(0,x),6)
  /// \f]
  class ReLU6 : public ActivationFunction
  {
  public:
    explicit ReLU6(const torch::nn::functional::ReLU6FuncOptions& options = {})
      : options_(options) {}

    explicit ReLU6(bool inplace)
      : options_(torch::nn::functional::ReLU6FuncOptions()
                 .inplace(inplace)) {}

    ~ReLU6() = default;
    
    inline virtual torch::Tensor apply(const torch::Tensor& input) const override
    {
      return torch::nn::functional::relu6(input, options_);
    }
    
    const torch::nn::functional::ReLU6FuncOptions& options() const
    {
      return options_;
    }

    torch::nn::functional::ReLU6FuncOptions& options()
    {
      return options_;
    }

    inline void pretty_print(std::ostream& os = std::cout) const override
    {
      os << "ReLU6"
         << "(\n  inplace=" << options_.inplace()
         << "\n)";
    }
    
  private:
    torch::nn::functional::ReLU6FuncOptions options_;
  };

  /// Randomized ReLU activation function
  ///
  /// \f[
  ///     \text{RReLU}(x) =
  ///     \begin{cases}
  ///           x & \text{ if } x \ge 0
  ///       a * x & \text{ otherwise }
  ///     \end{cases}
  /// \f]
  class RReLU : public ActivationFunction
  {
  public:
    explicit RReLU(const torch::nn::functional::RReLUFuncOptions& options = {})
      : options_(options) {}

    explicit RReLU(double lower, double upper, bool inplace=false)
      : options_(torch::nn::functional::RReLUFuncOptions()
                 .lower(lower)
                 .upper(upper)
                 .inplace(inplace)) {}

    ~RReLU() = default;
    
    inline virtual torch::Tensor apply(const torch::Tensor& input) const override
    {
      return torch::nn::functional::rrelu(input, options_);
    }
    
    const torch::nn::functional::RReLUFuncOptions& options() const
    {
      return options_;
    }

    torch::nn::functional::RReLUFuncOptions& options()
    {
      return options_;
    }

    inline void pretty_print(std::ostream& os = std::cout) const override
    {
      os << "PReLU"
         << "(\n  lower=" << options_.lower()
         << ",  upper=" << options_.upper()
         << ",  inplace=" << options_.inplace()
         << "\n)";
    }
    
  private:
    torch::nn::functional::RReLUFuncOptions options_;
  };

  /// SELU activation function
  ///
  /// \f[
  ///     \text{SELU}(x) = s * ( \max(0,x) + \min(0, \alpha*(\exp(x)-1 ) ) )
  /// \f]
  ///
  /// with \f$ s = 1.0507009873554804934193349852946 \f$ and
  /// \f$ \alpha = 1.6732632423543772848170429916717 \f$.  
  class SELU : public ActivationFunction
  {
  public:
    explicit SELU(const torch::nn::functional::SELUFuncOptions& options = {})
      : options_(options) {}

    explicit SELU(bool inplace)
      : options_(torch::nn::functional::SELUFuncOptions()
                 .inplace(inplace)) {}

    ~SELU() = default;
    
    inline virtual torch::Tensor apply(const torch::Tensor& input) const override
    {
      return torch::nn::functional::selu(input, options_);
    }
    
    const torch::nn::functional::SELUFuncOptions& options() const
    {
      return options_;
    }

    torch::nn::functional::SELUFuncOptions& options()
    {
      return options_;
    }

    inline void pretty_print(std::ostream& os = std::cout) const override
    {
      os << "SELU"
        << "(\n  inplace=" << options_.inplace()
         << "\n)";
    }
    
  private:
    torch::nn::functional::SELUFuncOptions options_;
  };

  /// Sigmoid activation function
  ///
  /// \f[
  ///    \text{Sigmoid}(x) = \sigma(x) = \frac{1}{1+\exp(-x)}
  /// \f]
  class Sigmoid : public ActivationFunction
  {
  public:    
    inline virtual torch::Tensor apply(const torch::Tensor& input) const override
    {
      return torch::sigmoid(input);
    }

    inline void pretty_print(std::ostream& os = std::cout) const override
    {
      os << "Sigmoid";
    }
    
  };

  /// Sigmoid Linear Unit activation function
  ///
  /// \f[
  ///    \text{SiLU}(x) = x * \sigma(x)
  /// \f]
  class SiLU : public ActivationFunction
  {
  public:    
    inline virtual torch::Tensor apply(const torch::Tensor& input) const override
    {
      return torch::silu(input);
    }

    inline void pretty_print(std::ostream& os = std::cout) const override
    {
      os << "SiLU";
    }
  };

  /// Softmax activation function
  ///
  /// \f[
  ///     \text{Softmax}(x_i) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}
  /// \f]
  ///
  /// where \f$ x \f$ is an \f$n\f$-dimensional input tensor
  class Softmax : public ActivationFunction
  {
  public:
    explicit Softmax(int64_t dim)
      : options_(torch::nn::functional::SoftmaxFuncOptions(dim)) {}
    
    explicit Softmax(const torch::nn::functional::SoftmaxFuncOptions& options)
      : options_(options) {}   
    
    ~Softmax() = default;
    
    inline virtual torch::Tensor apply(const torch::Tensor& input) const override
    {
      return torch::nn::functional::softmax(input, options_);
    }
    
    const torch::nn::functional::SoftmaxFuncOptions& options() const
    {
      return options_;
    }

    torch::nn::functional::SoftmaxFuncOptions& options()
    {
      return options_;
    }

    inline void pretty_print(std::ostream& os = std::cout) const override
    {
      os << "Softmax"
         << "(\n  dim=" << options_.dim()
         << "\n)";
    }
    
  private:
    torch::nn::functional::SoftmaxFuncOptions options_;
  };

  /// Softmin activation function
  ///
  /// \f[
  ///     \text{Softmin}(x) = \text{Softmax}(-x)
  /// \f]
  class Softmin : public ActivationFunction
  {
  public:
    explicit Softmin(int64_t dim)
      : options_(torch::nn::functional::SoftminFuncOptions(dim)) {}
    
    explicit Softmin(const torch::nn::functional::SoftminFuncOptions& options)
      : options_(options) {}   
    
    ~Softmin() = default;
    
    inline virtual torch::Tensor apply(const torch::Tensor& input) const override
    {
      return torch::nn::functional::softmin(input, options_);
    }
    
    const torch::nn::functional::SoftminFuncOptions& options() const
    {
      return options_;
    }

    torch::nn::functional::SoftminFuncOptions& options()
    {
      return options_;
    }

    inline void pretty_print(std::ostream& os = std::cout) const override
    {
      os << "Softmin"
         << "(\n  dim=" << options_.dim()
         << "\n)";
    }
    
  private:
    torch::nn::functional::SoftminFuncOptions options_;
  };

  /// Softplus activation function
  ///
  /// \f[
  ///     \text{Softplus}(x) = \frac{1}{\beta} * \log( 1+\exp(\beta * x) )
  /// \f]
  class Softplus : public ActivationFunction
  {
  public:    
    explicit Softplus(const torch::nn::functional::SoftplusFuncOptions& options = {})
      : options_(options) {}   
    
    explicit Softplus(double beta, double threshold)
      : options_(torch::nn::functional::SoftplusFuncOptions()
                 .beta(beta)
                 .threshold(threshold)) {}
    
    ~Softplus() = default;
    
    inline virtual torch::Tensor apply(const torch::Tensor& input) const override
    {
      return torch::nn::functional::softplus(input, options_);
    }
    
    const torch::nn::functional::SoftplusFuncOptions& options() const
    {
      return options_;
    }

    torch::nn::functional::SoftplusFuncOptions& options()
    {
      return options_;
    }

    inline void pretty_print(std::ostream& os = std::cout) const override
    {
      os << "Softplus"
         << "(\n  beta=" << options_.beta()
         << ",  theshold=" << options_.threshold()
         << "\n)";
    }
    
  private:
    torch::nn::functional::SoftplusFuncOptions options_;
  };
  
  /// Softshrink activation function
  ///
  /// \f[
  ///     \text{Softshrink}(x) =
  ///     \begin{cases}
  ///       x - \lambda & \text{ if } x > \lambda
  ///       x + \lambda & \text{ if } x < \lambda
  ///       0           & \text{ otherwise }
  ///     \end{cases}
  /// \f]
  class Softshrink : public ActivationFunction
  {
  public:    
    explicit Softshrink(const torch::nn::functional::SoftshrinkFuncOptions& options = {})
      : options_(options) {}   
    
    explicit Softshrink(double lambda)
      : options_(torch::nn::functional::SoftshrinkFuncOptions()
                 .lambda(lambda)) {}
    
    ~Softshrink() = default;
    
    inline virtual torch::Tensor apply(const torch::Tensor& input) const override
    {
      return torch::nn::functional::softshrink(input, options_);
    }
    
    const torch::nn::functional::SoftshrinkFuncOptions& options() const
    {
      return options_;
    }

    torch::nn::functional::SoftshrinkFuncOptions& options()
    {
      return options_;
    }

    inline void pretty_print(std::ostream& os = std::cout) const override
    {
      os << "Softshrink"
         << "(\n  lambda=" << options_.lambda()
         << "\n)";
    }
    
  private:
    torch::nn::functional::SoftshrinkFuncOptions options_;
  };

  /// Softsign activation function
  ///
  /// \f[
  ///    \text{Softsign}(x) = \frac{x}{1+\abs{x}}
  /// \f]
  class Softsign : public ActivationFunction
  {
  public:    
    inline virtual torch::Tensor apply(const torch::Tensor& input) const override
    {
      return torch::nn::functional::softsign(input);
    }

    inline void pretty_print(std::ostream& os = std::cout) const override
    {
      os << "Softsign";
    }    
  };

  /// Tanh activation function
  ///
  /// \f[
  ///    \text{Tanh}(x) = \frac{\exp(x)-\exp(-x)}{\exp(x)+\exp(-x)}
  /// \f]
  class Tanh : public ActivationFunction
  {
  public:    
    inline virtual torch::Tensor apply(const torch::Tensor& input) const override
    {
      return torch::tanh(input);
    }

    inline void pretty_print(std::ostream& os = std::cout) const override
    {
      os << "Tanh";
    }
  };

  /// Tanhshrink activation function
  ///
  /// \f[
  ///    \text{Tanhshrink}(x) = x - \text{Tanh}(x)
  /// \f]
  class Tanhshrink : public ActivationFunction
  {
  public:    
    inline virtual torch::Tensor apply(const torch::Tensor& input) const override
    {
      return torch::nn::functional::tanhshrink(input);
    }

    inline void pretty_print(std::ostream& os = std::cout) const override
    {
      os << "Tanhshrink";
    }
  };
  
  /// Threshold activation function
  ///
  /// \f[
  ///     \text{Threshold}(x) =
  ///     \begin{cases}
  ///       x     & \text{ if } x > threshold
  ///       value & \text{ otherwise }
  ///     \end{cases}
  /// \f]
  class Threshold : public ActivationFunction
  {
  public:    
    explicit Threshold(const torch::nn::functional::ThresholdFuncOptions& options)
      : options_(options) {}   
    
    explicit Threshold(double threshold, double value, bool inplace=false)
      : options_(torch::nn::functional::ThresholdFuncOptions(threshold, value)
                 .inplace(inplace)) {}
    
    ~Threshold() = default;
    
    inline virtual torch::Tensor apply(const torch::Tensor& input) const override
    {
      return torch::nn::functional::threshold(input, options_);
    }
    
    const torch::nn::functional::ThresholdFuncOptions& options() const
    {
      return options_;
    }

    torch::nn::functional::ThresholdFuncOptions& options()
    {
      return options_;
    }

    inline void pretty_print(std::ostream& os = std::cout) const override
    {
      os << "Threshold"
         << "(\n  threshold=" << options_.threshold()
         << ",  value=" << options_.value()
         << ",  inplace=" << options_.inplace()
         << "\n)";
    }
    
  private:
    torch::nn::functional::ThresholdFuncOptions options_;
  };  
  
} // namespace iganet
