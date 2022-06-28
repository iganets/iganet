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
  };

  /// No-op activation function
  class None : public ActivationFunction
  {
  public:
    inline virtual torch::Tensor apply(const torch::Tensor& input) const override
    {
      std::cout << "apply: none\n";
      return input;
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
      std::cout << "apply: batch_norm\n";
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
      std::cout << "apply: celu\n";
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
      std::cout << "apply: elu\n";
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
      std::cout << "apply: gelu\n";
      return torch::gelu(input);
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
      std::cout << "apply: glu\n";
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
      std::cout << "apply: group_norm\n";
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
      std::cout << "apply: gumbel_softmax\n";
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
      std::cout << "apply: hardshrink\n";
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
      std::cout << "apply: hardsigmoid\n";
      return torch::hardsigmoid(input);
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
      std::cout << "apply: hardswish\n";
      return torch::hardswish(input);
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
      std::cout << "apply: hardtanh\n";
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
      std::cout << "apply: instance_norm\n";
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
      std::cout << "apply: layer_norm\n";
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
      std::cout << "apply: leaky_relu\n";
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
      std::cout << "apply: local_response_norm\n";
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
      std::cout << "apply: logsigmoid\n";
      return torch::log_sigmoid(input);
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
      std::cout << "apply: log_softmax\n";
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
      std::cout << "apply: mish\n";
      return torch::mish(input);
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
      std::cout << "apply: normalize\n";
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
      std::cout << "apply: prelu\n";
      return torch::nn::functional::prelu(input, weight());
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
      std::cout << "apply: relu\n";
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
      std::cout << "apply: relu6\n";
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
      std::cout << "apply: rrelu\n";
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
      std::cout << "apply: selu\n";
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
      std::cout << "apply: sigmoid\n";
      return torch::sigmoid(input);
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
      std::cout << "apply: silu\n";
      return torch::silu(input);
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
      std::cout << "apply: softmax\n";
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
      std::cout << "apply: softmin\n";
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
      std::cout << "apply: softplus";
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
      std::cout << "apply: softshrink\n";
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
      std::cout << "apply: softsign\n";
      return torch::nn::functional::softsign(input);
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
      std::cout << "apply: tanh\n";
      return torch::tanh(input);
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
      std::cout << "apply: tanhshrink\n";
      return torch::nn::functional::tanhshrink(input);
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
      std::cout << "apply: threshold\n";
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

  private:
    torch::nn::functional::ThresholdFuncOptions options_;
  };  
  
} // namespace iganet
