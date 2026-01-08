/**
   @file net/iganet.hpp

   @brief Isogeometric analysis network

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#pragma once

#include <core/options.hpp>
#include <net/generator.hpp>
#include <net/collocation.hpp>
#include <net/optimizer.hpp>
#include <splines/boundary.hpp>
#include <splines/functionspace.hpp>
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
  TORCH_ARG(double, min_loss_change) = 0;
  TORCH_ARG(double, min_loss_rel_change) = 1e-3;
};

/// @brief IgA base class
///
/// This class implements the base functionality of IgANets
/// @{
template <typename, typename, typename = void> class IgABase;

template <detail::HasAsTensor... Inputs, detail::HasAsTensor... Outputs,
          detail::HasAsTensor... CollPts>
class IgABase<std::tuple<Inputs...>, std::tuple<Outputs...>,
               std::tuple<CollPts...>> {
public:
  /// @brief Value type
  using value_type = std::common_type_t<typename Inputs::value_type...,
                                        typename Outputs::value_type...>;

  /// @brief Type of the inputs
  using inputs_type = std::tuple<Inputs...>;

  /// @brief Type of the outputs
  using outputs_type = std::tuple<Outputs...>;

  /// @brief Type of the collocation points
  using collPts_type = std::tuple<typename CollPtsHelper<CollPts...>::type>;

protected:
  /// @brief Inputs
  inputs_type inputs_;

  /// @brief Outputs
  outputs_type outputs_;

  /// @brief Outputs
  collPts_type collPts_;

private:
  /// @brief Constructs a tuple from arrays
  ///
  /// @{
  template <typename... Objs, std::size_t... NumCoeffs, std::size_t... Is>
  auto construct_tuple_from_arrays_impl(
      const std::tuple<std::array<int64_t, NumCoeffs>...> &numCoeffs,
      enum init init, iganet::Options<value_type> options,
      std::index_sequence<Is...>) {
    static_assert(sizeof...(Objs) == sizeof...(NumCoeffs));
    return std::make_tuple(std::apply(
        [&]<typename... Args>(Args &&...args) {
          return Objs(std::forward<Args>(args)..., init, options);
        },
        std::get<Is>(numCoeffs))...);
  }

  template <typename... Objs, std::size_t... NumCoeffs>
  auto construct_tuple_from_arrays(
      const std::tuple<std::array<int64_t, NumCoeffs>...> &numCoeffs,
      enum init init, iganet::Options<value_type> options) {
    return construct_tuple_from_arrays_impl<Objs...>(
        numCoeffs, init, options, std::index_sequence_for<Objs...>{});
  }
  /// @}

  /// @brief Constructs a tuple from tuples
  ///
  /// @{
  template <typename... Objs, typename... NumCoeffsTuples, std::size_t... Is>
  auto construct_tuple_from_tuples_impl(
      const std::tuple<NumCoeffsTuples...> &numCoeffs, enum init init,
      iganet::Options<value_type> options, std::index_sequence<Is...>) {
    static_assert(sizeof...(Objs) == sizeof...(NumCoeffsTuples));
    return std::make_tuple(std::apply(
        [&]<typename... Args>(Args &&...args) {
          return Objs(std::forward<Args>(args)..., init, options);
        },
        std::get<Is>(numCoeffs))...);
  }

  template <typename... Objs, typename... NumCoeffsTuples>
  auto
  construct_tuple_from_tuples(const std::tuple<NumCoeffsTuples...> &numCoeffs,
                              enum init init,
                              iganet::Options<value_type> options) {
    return construct_tuple_from_tuples_impl<Objs...>(
        numCoeffs, init, options, std::index_sequence_for<Objs...>{});
  }
  /// @}

public:
  /// @brief Default constructor
  explicit IgABase(
      iganet::Options<value_type> options = iganet::Options<value_type>{})
      : inputs_(), outputs_(), collPts_() {}

  /// @brief Constructor
  ///
  /// Number of spline coefficients is the same for all spaces in the
  /// input, output and collocation points objects
  template <std::size_t NumCoeffs>
  explicit IgABase(
      const std::array<int64_t, NumCoeffs> &ncoeffs,
      enum init init = init::greville,
      iganet::Options<value_type> options = iganet::Options<value_type>{})
      : IgABase(std::tuple{ncoeffs}, std::tuple{ncoeffs}, std::tuple{ncoeffs},
                 init, options) {}

  /// @brief Constructor
  ///
  /// Number of spline coefficients is the same for all spaces in the
  /// input, output and collocation points objects, respectively
  template <std::size_t NumCoeffsInputs, std::size_t NumCoeffsOutputs,
            std::size_t NumCoeffsCollPts>
  IgABase(const std::array<int64_t, NumCoeffsInputs> &ncoeffsInputs,
           const std::array<int64_t, NumCoeffsOutputs> &ncoeffsOutputs,
           const std::array<int64_t, NumCoeffsCollPts> &ncoeffsCollPts,
           enum init init = init::greville,
           iganet::Options<value_type> options = iganet::Options<value_type>{})
      : IgABase(std::tuple{ncoeffsInputs}, std::tuple{ncoeffsOutputs},
                 std::tuple{ncoeffsCollPts}, init, options) {}

  /// @brief Constructor
  ///
  /// Number of spline coefficients is different for the different
  /// spaces of the inputs, outputs, and collocation points,
  /// but the same for inputs, outputs and collocation points objects
  template <std::size_t... NumCoeffs>
  explicit IgABase(
      const std::tuple<std::array<int64_t, NumCoeffs>...> &ncoeffs,
      enum init init = init::greville,
      iganet::Options<value_type> options = iganet::Options<value_type>{})
      : IgABase(ncoeffs, ncoeffs, ncoeffs, init, options) {}

  /// @brief Constructor
  ///
  /// Number of spline coefficients is different all inputs, outputs,
  /// and collocation points objects
  template <std::size_t... NumCoeffsInputs, std::size_t... NumCoeffsOutputs,
            std::size_t... NumCoeffsCollPts>
  IgABase(
      const std::tuple<std::array<int64_t, NumCoeffsInputs>...> &ncoeffsInputs,
      const std::tuple<std::array<int64_t, NumCoeffsOutputs>...>
          &ncoeffsOutputs,
      const std::tuple<std::array<int64_t, NumCoeffsCollPts>...>
          &ncoeffsCollPts,
      enum init init = init::greville,
      iganet::Options<value_type> options = iganet::Options<value_type>{})
      : inputs_(construct_tuple_from_arrays<Inputs...>(ncoeffsInputs, init,
                                                       options)),
        outputs_(construct_tuple_from_arrays<Outputs...>(ncoeffsOutputs, init,
                                                         options)),
        collPts_(construct_tuple_from_arrays<Outputs...>(ncoeffsCollPts, init,
                                                         options)) {}

  /// @brief Constructor
  ///
  /// Number of coefficients is different for all inputs, outputs, and
  /// collocation points objects and passed as a tuple of tuples of
  /// arrays of different sizes
  template <typename... CoeffsInputs, typename... CoeffsOutputs,
            typename... CoeffsCollPts>
  IgABase(const std::tuple<CoeffsInputs...> &coeffsInputs,
           const std::tuple<CoeffsOutputs...> &coeffsOutputs,
           const std::tuple<CoeffsCollPts...> &coeffsCollPts,
           enum init init = init::greville,
           iganet::Options<value_type> options = iganet::Options<value_type>{})
      : inputs_(construct_tuple_from_tuples<Inputs...>(coeffsInputs, init,
                                                       options)),
        outputs_(construct_tuple_from_tuples<Outputs...>(coeffsOutputs, init,
                                                         options)),
        collPts_(construct_tuple_from_tuples<CollPts...>(coeffsCollPts, init,
                                                         options)) {}

  /// @brief Returns the number of elements in the tuple of input objects
  inline static constexpr std::size_t ninputs() noexcept {
    return sizeof...(Inputs);
  }

  /// @brief Returns a constant reference to the tuple of input objects
  inline constexpr const auto &inputs() const { return inputs_; }

  /// @brief Returns a non-constant reference to the tuple of input objects
  inline constexpr auto &inputs() { return inputs_; }

  /// @brief Returns a constant reference to the index-th input object
  template <std::size_t index> inline constexpr const auto &input() const {
    static_assert(index < sizeof...(Inputs));
    return std::get<index>(inputs_);
  }

  /// @brief Returns a non-constant reference to the index-th input object
  template <std::size_t index> inline constexpr auto &input() {
    static_assert(index < sizeof...(Inputs));
    return std::get<index>(inputs_);
  }

  /// @brief Returns the number of elements in the tuple of output objects
  inline static constexpr std::size_t noutputs() noexcept {
    return sizeof...(Outputs);
  }

  /// @brief Returns a constant reference to the tuple of output objects
  inline constexpr const auto &outputs() const { return outputs_; }

  /// @brief Returns a non-constant reference to the tuple of output objects
  inline constexpr auto &outputs() { return outputs_; }

  /// @brief Returns a constant reference to the index-th output object
  template <std::size_t index> inline constexpr const auto &output() const {
    static_assert(index < sizeof...(Outputs));
    return std::get<index>(outputs_);
  }

  /// @brief Returns a non-constant reference to the index-th output object
  template <std::size_t index> inline constexpr auto &output() {
    static_assert(index < sizeof...(Outputs));
    return std::get<index>(outputs_);
  }

  /// @brief Returns the number of elements in the tuple of collocation points
  /// objects
  inline static constexpr std::size_t ncollPts() noexcept {
    return sizeof...(CollPts);
  }

  /// @brief Returns a constant reference to the tuple of collocation points
  /// objects
  inline constexpr const auto &collPts() const { return collPts_; }

  /// @brief Returns a non-constant reference to the tuple of collocation points
  /// objects
  inline constexpr auto &collPts() { return collPts_; }

  /// @brief Returns the collocation points of the index-th function spaces
  ///
  /// In the default implementation the collocation points are the Greville
  /// abscissae in the interior of the domain and on the boundary
  /// faces. This behavior can be changed by overriding this virtual
  /// function in a derived class.
  template <std::size_t index>
  std::tuple_element_t<index, collPts_type>
  collPts(enum collPts collPts) const {
    return CollPtsHelper<std::tuple_element_t<index, collPts_type>>::collPts(collPts, std::get<index>(collPts_));
  }
};

template <detail::HasAsTensor... Inputs, detail::HasAsTensor... Outputs>
class IgABase<std::tuple<Inputs...>, std::tuple<Outputs...>, void> {
public:
  /// @brief Value type
  using value_type = std::common_type_t<typename Inputs::value_type...,
                                        typename Outputs::value_type...>;

  /// @brief Type of the inputs
  using inputs_type = std::tuple<Inputs...>;

  /// @brief Type alias for the type of the index-th inputs object
  template <std::size_t index>
  using input_t = std::tuple_element_t<index, inputs_type>;

  /// @brief Type of the outputs
  using outputs_type = std::tuple<Outputs...>;

  /// @brief Type alias for the type of the index-th outputs object
  template <std::size_t index>
  using output_t = std::tuple_element_t<index, outputs_type>;

  /// @brief Type of the collocation points
  using collPts_type = std::tuple<typename CollPtsHelper<Outputs...>::type>;

  /// @brief Type alias for the type of the index-th collocation points object
  template <std::size_t index>
  using collPts_t = std::tuple_element_t<index, collPts_type>;

protected:
  /// @brief Inputs
  inputs_type inputs_;

  /// @brief Outputs
  outputs_type outputs_;

private:
  /// @brief Constructs a tuple from arrays
  ///
  /// @{
  template <typename... Objs, std::size_t... NumCoeffs, std::size_t... Is>
  auto construct_tuple_from_arrays_impl(
      const std::tuple<std::array<int64_t, NumCoeffs>...> &numCoeffs,
      enum init init, iganet::Options<value_type> options,
      std::index_sequence<Is...>) {
    static_assert(sizeof...(Objs) == sizeof...(NumCoeffs));
    return std::make_tuple(Objs(std::get<Is>(numCoeffs), init, options)...);
  }

  template <typename... Objs, std::size_t... NumCoeffs>
  auto construct_tuple_from_arrays(
      const std::tuple<std::array<int64_t, NumCoeffs>...> &numCoeffs,
      enum init init, iganet::Options<value_type> options) {
    return construct_tuple_from_arrays_impl<Objs...>(
        numCoeffs, init, options, std::index_sequence_for<Objs...>{});
  }
  /// @}

  /// @brief Constructs a tuple from tuples
  ///
  /// @{
  template <typename... Objs, typename... NumCoeffs, std::size_t... Is>
  auto construct_tuple_from_tuples_impl(
      const std::tuple<NumCoeffs...> &numCoeffs, enum init init,
      iganet::Options<value_type> options, std::index_sequence<Is...>) {
    static_assert(sizeof...(Objs) == sizeof...(NumCoeffs));
    return std::make_tuple(std::apply(
        [&]<typename... Args>(Args &&...args) {
          return Objs(std::forward<Args>(args)..., init, options);
        },
        std::get<Is>(numCoeffs))...);
  }

  template <typename... Objs, typename... NumCoeffsTuples>
  auto
  construct_tuple_from_tuples(const std::tuple<NumCoeffsTuples...> &numCoeffs,
                              enum init init,
                              iganet::Options<value_type> options) {
    return construct_tuple_from_tuples_impl<Objs...>(
        numCoeffs, init, options, std::index_sequence_for<Objs...>{});
  }
  /// @}

public:
  /// @brief Default constructor
  explicit IgABase(
      iganet::Options<value_type> options = iganet::Options<value_type>{})
      : inputs_(), outputs_() {}

  /// @brief Constructor
  ///
  /// Number of spline coefficients is the same for all spaces in the
  /// input and output objects
  template <std::size_t NumCoeffs>
  explicit IgABase(
      const std::array<int64_t, NumCoeffs> &ncoeffs,
      enum init init = init::greville,
      iganet::Options<value_type> options = iganet::Options<value_type>{})
      : IgABase(std::tuple{ncoeffs}, std::tuple{ncoeffs}, init, options) {}

  /// @brief Constructor
  ///
  /// Number of spline coefficients is the same for all spaces in the
  /// input and output objects, respectively
  template <std::size_t NumCoeffsInputs, std::size_t NumCoeffsOutputs>
  IgABase(const std::array<int64_t, NumCoeffsInputs> &ncoeffsInputs,
           const std::array<int64_t, NumCoeffsOutputs> &ncoeffsOutputs,
           enum init init = init::greville,
           iganet::Options<value_type> options = iganet::Options<value_type>{})
      : IgABase(std::tuple{ncoeffsInputs}, std::tuple{ncoeffsOutputs}, init,
                 options) {}

  /// @brief Constructor
  ///
  /// Number of spline coefficients is different for the different
  /// spaces of the inputs and outputs, but the same for input and
  /// output objects
  template <std::size_t... NumCoeffs>
  explicit IgABase(
      const std::tuple<std::array<int64_t, NumCoeffs>...> &ncoeffs,
      enum init init = init::greville,
      iganet::Options<value_type> options = iganet::Options<value_type>{})
      : IgABase(ncoeffs, ncoeffs, init, options) {}

  /// @brief Constructor
  ///
  /// Number of spline coefficients is different all inputs and
  /// outputs, respectively
  template <std::size_t... NumCoeffsInputs, std::size_t... NumCoeffsOutputs>
  IgABase(
      const std::tuple<std::array<int64_t, NumCoeffsInputs>...> &ncoeffsInputs,
      const std::tuple<std::array<int64_t, NumCoeffsOutputs>...>
          &ncoeffsOutputs,
      enum init init = init::greville,
      iganet::Options<value_type> options = iganet::Options<value_type>{})
      : inputs_(construct_tuple_from_arrays<Inputs...>(ncoeffsInputs, init,
                                                       options)),
        outputs_(construct_tuple_from_arrays<Outputs...>(ncoeffsOutputs, init,
                                                         options)) {}

  /// @brief Constructor
  ///
  /// Number of coefficients is different for all inputs and outputs
  /// and passed as a tuple of tuples of arrays of different sizes
  template <typename... CoeffsInputs, typename... CoeffsOutputs>
  IgABase(const std::tuple<CoeffsInputs...> &coeffsInputs,
           const std::tuple<CoeffsOutputs...> &coeffsOutputs,
           enum init init = init::greville,
           iganet::Options<value_type> options = iganet::Options<value_type>{})
      : inputs_(construct_tuple_from_tuples<Inputs...>(coeffsInputs, init,
                                                       options)),
        outputs_(construct_tuple_from_tuples<Outputs...>(coeffsOutputs, init,
                                                         options)) {}

  /// @brief Returns the number of elements in the tuple of input objects
  inline static constexpr std::size_t ninputs() noexcept {
    return sizeof...(Inputs);
  }

  /// @brief Returns a constant reference to the tuple of input objects
  inline constexpr const auto &inputs() const { return inputs_; }

  /// @brief Returns a non-constant reference to the tuple of input objects
  inline constexpr auto &inputs() { return inputs_; }

  /// @brief Returns a constant reference to the index-th input object
  template <std::size_t index> inline constexpr const auto &input() const {
    static_assert(index < sizeof...(Inputs));
    return std::get<index>(inputs_);
  }

  /// @brief Returns a non-constant reference to the index-th input object
  template <std::size_t index> inline constexpr auto &input() {
    static_assert(index < sizeof...(Inputs));
    return std::get<index>(inputs_);
  }

  /// @brief Returns the number of elements in the tuple of output objects
  inline static constexpr std::size_t noutputs() noexcept {
    return sizeof...(Outputs);
  }

  /// @brief Returns a constant reference to the tuple of output objects
  inline constexpr const auto &outputs() const { return outputs_; }

  /// @brief Returns a non-constant reference to the tuple of output objects
  inline constexpr auto &outputs() { return outputs_; }

  /// @brief Returns a constant reference to the index-th output object
  template <std::size_t index> inline constexpr const auto &output() const {
    static_assert(index < sizeof...(Outputs));
    return std::get<index>(outputs_);
  }

  /// @brief Returns a non-constant reference to the index-th output object
  template <std::size_t index> inline constexpr auto &output() {
    static_assert(index < sizeof...(Outputs));
    return std::get<index>(outputs_);
  }

  /// @brief Returns the number of elements in the tuple of collocation points
  /// objects
  inline static constexpr std::size_t ncollPts() noexcept {
    return sizeof...(Outputs);
  }

  /// @brief Returns a constant reference to the tuple of collocation points
  /// objects
  inline constexpr const auto &collPts() const { return outputs_; }

  /// @brief Returns a non-constant reference to the tuple of collocation points
  /// objects
  inline constexpr auto &collPts() { return outputs_; }

  /// @brief Returns the collocation points of the index-th function spaces
  ///
  /// In the default implementation the collocation points are the Greville
  /// abscissae in the interior of the domain and on the boundary
  /// faces. This behavior can be changed by overriding this virtual
  /// function in a derived class.
  template <std::size_t index>
  std::tuple_element_t<index, collPts_type>
  collPts(enum collPts collPts) const {
    return CollPtsHelper<std::tuple_element_t<index, outputs_type>>::collPts(collPts, std::get<index>(outputs_));
  }
};
/// @}
  
/// @brief IgANet
///
/// This class implements the core functionality of IgANets
template <typename Optimizer, typename Inputs, typename Outputs,
          typename CollPts = void>
  requires OptimizerType<Optimizer>
class IgANet : public IgABase<Inputs, Outputs, CollPts>,
                utils::Serializable,
                private utils::FullQualifiedName {
public:
  /// @brief Base type
  using Base = IgABase<Inputs, Outputs, CollPts>;

  /// @brief Value type
  using value_type = Base::value_type;

  /// @brief Type of the optimizer
  using optimizer_type = Optimizer;

  /// @brief Type of the optimizer options
  using optimizer_options_type = optimizer_options_type<Optimizer>::type;

protected:
  /// @brief IgANet generator
  IgANetGenerator<typename Base::value_type> net_;

  /// @brief Optimizer
  std::unique_ptr<optimizer_type> opt_;

  /// @brief Options
  IgANetOptions options_;

public:
  /// @brief Default constructor
  explicit IgANet(const IgANetOptions &defaults = {},
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
  IgANet(const std::vector<int64_t> &layers,
          const std::vector<std::vector<std::any>> &activations,
          const NumCoeffs &numCoeffs, enum init init = init::greville,
          IgANetOptions defaults = {},
          iganet::Options<typename Base::value_type> options =
              iganet::Options<typename Base::value_type>{})
      : IgANet(layers, activations, numCoeffs, numCoeffs, init, defaults,
                options) {}

  /// @brief Constructor: number of layers, activation functions, and
  /// number of spline coefficients (same for all inputs and outputs)
  template <typename NumCoeffsInputs, typename NumCoeffsOutputs>
  IgANet(const std::vector<int64_t> &layers,
          const std::vector<std::vector<std::any>> &activations,
          const NumCoeffsInputs &numCoeffsInputs,
          const NumCoeffsOutputs &numCoeffsOutputs,
          enum init init = init::greville, IgANetOptions defaults = {},
          iganet::Options<typename Base::value_type> options =
              iganet::Options<typename Base::value_type>{})
      : // Construct the base class
        Base(numCoeffsInputs, numCoeffsOutputs, init, options),
        // Construct the deep neural network
        net_(utils::concat(
                 std::vector<int64_t>{inputs(/* epoch */ 0).size(0)}, layers,
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
  /// @param[in] resetOptions Flag to indicate whether the optimizer options
  /// should be resetted
  inline void optimizerReset(bool resetOptions = true) {
    if (resetOptions)
      opt_ = std::make_unique<optimizer_type>(net_->parameters());
    else {
      std::vector<optimizer_options_type> options;
      for (auto &group : opt_->param_groups())
        options.push_back(
            static_cast<optimizer_options_type &>(group.options()));
      opt_ = std::make_unique<optimizer_type>(net_->parameters());
      for (auto [group, options] : utils::zip(opt_->param_groups(), options))
        static_cast<optimizer_options_type &>(group.options()) = options;
    }
  }

  /// @brief Resets the optimizer
  inline void optimizerReset(const optimizer_options_type &optimizerOptions) {
    opt_ =
        std::make_unique<optimizer_type>(net_->parameters(), optimizerOptions);
  }

  /// @brief Returns a non-constant reference to the optimizer options
  inline optimizer_options_type &optimizerOptions(std::size_t param_group = 0) {
    if (param_group < opt_->param_groups().size())
      return static_cast<optimizer_options_type &>(
          opt_->param_groups()[param_group].options());
    else
      throw std::runtime_error("Index exceeds number of parameter groups");
  }

  /// @brief Returns a constant reference to the optimizer options
  inline const optimizer_options_type &
  optimizerOptions(std::size_t param_group = 0) const {
    if (param_group < opt_->param_groups().size())
      return static_cast<optimizer_options_type &>(
          opt_->param_groups()[param_group].options());
    else
      throw std::runtime_error("Index exceeds number of parameter groups");
  }

  /// @brief Resets the optimizer options
  inline void optimizerOptionsReset(const optimizer_options_type &options) {
    for (auto &group : opt_->param_groups())
      static_cast<optimizer_options_type &>(group.options()) = options;
  }

  /// @brief Resets the optimizer options
  inline void optimizerOptionsReset(optimizer_options_type &&options) {
    for (auto &group : opt_->param_groups())
      static_cast<optimizer_options_type &>(group.options()) = options;
  }

  /// @brief Resets the optimizer options
  inline void optimizerOptionsReset(const optimizer_options_type &options,
                                    std::size_t param_group) {
    if (param_group < opt_->param_groups().size())
      static_cast<optimizer_options_type &>(opt_->param_group().options()) =
          options;
    else
      throw std::runtime_error("Index exceeds number of parameter groups");
  }

  /// @brief Resets the optimizer options
  inline void optimizerOptionsReset(optimizer_options_type &&options,
                                    std::size_t param_group) {
    if (param_group < opt_->param_groups().size())
      static_cast<optimizer_options_type &>(opt_->param_group().options()) =
          options;
    else
      throw std::runtime_error("Index exceeds number of parameter groups");
  }

  /// @brief Returns a constant reference to the options structure
  inline const auto &options() const { return options_; }

  /// @brief Returns a non-constant reference to the options structure
  inline auto &options() { return options_; }

  /// @brief Returns a constant reference to the tuple of input objects
  inline constexpr const auto &inputs() const { return Base::inputs(); }

  /// @brief Returns a non-constant reference to the tuple of input objects
  inline constexpr auto &inputs() { return Base::inputs(); }

  /// @brief Returns a constant reference to the tuple of output objects
  inline constexpr const auto &outputs() const { return Base::outputs(); }

  /// @brief Returns a non-constant reference to the tuple of output objects
  inline constexpr auto &outputs() { return Base::outputs(); }

  /// @brief Returns the network inputs as tensor
  virtual torch::Tensor inputs(int64_t epoch) const {
    return utils::cat_tuple_into_tensor(
        Base::inputs_, [](const auto &obj) { return obj.as_tensor(); });
  }

  /// @brief Returns the network outputs as tensor
  virtual torch::Tensor outputs(int64_t epoch) const {
    return utils::cat_tuple_into_tensor(
        Base::outputs_, [](const auto &obj) { return obj.as_tensor(); });
  }

  /// @brief Attaches the given tensor to the inputs
  virtual void inputs(const torch::Tensor &tensor) {
    utils::slice_tensor_into_tuple(
        Base::inputs_, tensor,
        [](const auto &obj) { return obj.as_tensor_size(); },
        [](auto &obj, const auto &tensor) { return obj.from_tensor(tensor); });
  }

  /// @brief Attaches the given tensor to the outputs
  virtual void outputs(const torch::Tensor &tensor) {
    utils::slice_tensor_into_tuple(
        Base::outputs_, tensor,
        [](const auto &obj) { return obj.as_tensor_size(); },
        [](auto &obj, const auto &tensor) { return obj.from_tensor(tensor); });
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

      typename Base::value_type current_loss =
          loss.item<typename Base::value_type>();
      Log(log::verbose) << "Epoch " << std::to_string(epoch) << ": "
                        << current_loss << std::endl;

      if (current_loss < options_.min_loss() ||
          std::abs(current_loss - previous_loss) < options_.min_loss_change() ||
          std::abs(current_loss - previous_loss) / current_loss <
              options_.min_loss_rel_change() ||
          loss.isnan().item<bool>()) {
        Log(log::info) << "Total epochs: " << epoch
                       << ", loss: " << current_loss << std::endl;
        return;
      }
      previous_loss = current_loss;
    }
    Log(log::info) << "Max epochs reached: " << options_.max_epoch()
                   << ", loss: " << previous_loss << std::endl;
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

      typename Base::value_type current_loss(0);

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

        current_loss += loss.item<typename Base::value_type>();
      }
      Log(log::verbose) << "Epoch " << std::to_string(epoch) << ": "
                        << current_loss << std::endl;

      if (current_loss < options_.min_loss() ||
          std::abs(current_loss - previous_loss) < options_.min_loss_change() ||
          std::abs(current_loss - previous_loss) / current_loss <
              options_.min_loss_rel_change() ||
          loss.isnan().item<bool>()) {
        Log(log::info) << "Total epochs: " << epoch
                       << ", loss: " << current_loss << std::endl;
        return;
      }
      previous_loss = current_loss;
    }
    Log(log::info) << "Max epochs reached: " << options_.max_epoch()
                   << ", loss: " << previous_loss << std::endl;
  }

  /// @brief Evaluate IgANet
  void eval() {
    torch::Tensor inputs = this->inputs(0);
    torch::Tensor outputs = net_->forward(inputs);
    this->outputs(outputs);
  }

  /// @brief Returns the IgANet object as JSON object
  inline nlohmann::json to_json() const override {
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

  /// @brief Registers a parameter
  torch::Tensor& register_parameter(std::string name, torch::Tensor tensor, bool requires_grad = true) {
    return net_->register_parameter(name, tensor, requires_grad);
  }
  
  /// @brief Returns a string representation of the IgANet object
  inline void pretty_print(std::ostream &os) const noexcept override {
    os << name() << "(\n"
       << "net = " << net_ << "\n";

    os << "inputs[" << Base::ninputs() << "] = (";
    std::apply([&os](const auto &...elems) { ((os << elems << "\n"), ...); },
               Base::inputs());
    os << ")";

    os << "outputs [" << Base::noutputs() << "]= (";
    std::apply([&os](const auto &...elems) { ((os << elems << "\n"), ...); },
               Base::inputs());
    os << ")";

    os << "collPts [" << Base::ncollPts() << "]= (";
    std::apply([&os](const auto &...elems) { ((os << elems << "\n"), ...); },
               Base::collPts());
    os << ")";
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

    std::apply(
        [&](auto &&...elems) {
          std::size_t counter = 0;
          (elems.write(archive,
                       key + ".input[" + std::to_string(counter++) + "]"),
           ...);
        },
        Base::inputs());

    std::apply(
        [&](auto &&...elems) {
          std::size_t counter = 0;
          (elems.write(archive,
                       key + ".output[" + std::to_string(counter++) + "]"),
           ...);
        },
        Base::outputs());

    if constexpr (!std::is_void_v<CollPts>) {
      std::apply(
          [&](auto &&...elems) {
            std::size_t counter = 0;
            (elems.write(archive,
                         key + ".collpts[" + std::to_string(counter++) + "]"),
             ...);
          },
          Base::collPts());
    }

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

    std::apply(
        [&](auto &&...elems) {
          std::size_t counter = 0;
          (elems.read(archive,
                      key + ".input[" + std::to_string(counter++) + "]"),
           ...);
        },
        Base::inputs());

    std::apply(
        [&](auto &&...elems) {
          std::size_t counter = 0;
          (elems.read(archive,
                      key + ".output[" + std::to_string(counter++) + "]"),
           ...);
        },
        Base::outputs());

    if constexpr (!std::is_void_v<CollPts>) {
      std::apply(
          [&](auto &&...elems) {
            std::size_t counter = 0;
            (elems.read(archive,
                        key + ".collpts[" + std::to_string(counter++) + "]"),
             ...);
          },
          Base::collPts());
    }

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

    result *= std::apply(
        [&](auto &&...elemsThis) {
          return std::apply(
              [&](auto &&...elemsOther) {
                return ((elemsThis == elemsOther) && ...);
              },
              other.inputs());
        },
        Base::inputs());

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
template <typename Optimizer, typename Inputs, typename Outputs,
          typename CollPts>
  requires OptimizerType<Optimizer>
inline std::ostream &
operator<<(std::ostream &os,
           const IgANet<Optimizer, Inputs, Outputs, CollPts> &obj) {
  obj.pretty_print(os);
  return os;
}

/// @brief IgANetCustomizable
///
/// This class implements a customizable variant of IgANets that
/// provides types and attributes for precomputing indices and basis
/// functions
///
/// @{
template <typename, typename, typename = void> class IgANetCustomizable;

template <detail::HasAsTensor... Inputs, detail::HasAsTensor... Outputs>
class IgANetCustomizable<std::tuple<Inputs...>, std::tuple<Outputs...>, void> {
private:
  /// @brief Returns the interior knot indices of all tuple elements
  static auto find_interior_knot_indices(auto &&tuple) {
    return std::apply(
        []<typename... Elems>(Elems &&...elems) {
          return std::make_tuple(([&] {
            using T = std::decay_t<Elems>;
            if constexpr (detail::HasTemplatedFindKnotIndices<T>)
              return elems.template find_knot_indices<functionspace::interior>(
                                                                               typename T::eval_type{});
            else if constexpr (detail::HasFindKnotIndices<T>)
              // Note that this is a fake call here
              return elems.find_knot_indices(typename T::eval_type{});
          })()...);
        },
        tuple);
  }

  /// @brief Returns the boundary knot indices of all tuple elements
  static auto find_boundary_knot_indices(auto &&tuple) {
    return std::apply(
        []<typename... Elems>(Elems &&...elems) {
          return std::make_tuple(([&] {
            using T = std::decay_t<Elems>;
            if constexpr (detail::HasTemplatedFindKnotIndices<T>)
              return elems.template find_knot_indices<functionspace::boundary>(
                                                                               typename T::boundary_eval_type{});
            else if constexpr (detail::HasFindKnotIndices<T>)
              // Note that this is a fake call here
              return elems.find_knot_indices(typename T::eval_type{});
          })()...);
        },
        tuple);
  }

  /// @brief Returns the interior coeff indices of all tuple elements
  static auto find_interior_coeff_indices(auto &&tuple) {
    return std::apply(
        []<typename... Elems>(Elems &&...elems) {
          return std::make_tuple(([&] {
            using T = std::decay_t<Elems>;
            if constexpr (detail::HasTemplatedFindCoeffIndices<T>)
              return elems.template find_coeff_indices<functionspace::interior>(
                                                                                typename T::eval_type{});
            else if constexpr (detail::HasFindCoeffIndices<T>)
              // Note that this is a fake call here              
              return elems.find_coeff_indices(typename T::eval_type{});
          })()...);
        },
        tuple);
  }

  /// @brief Returns the boundary coeff indices of all tuple elements
  static auto find_boundary_coeff_indices(auto &&tuple) {
    return std::apply(
        []<typename... Elems>(Elems &&...elems) {
          return std::make_tuple(([&] {
            using T = std::decay_t<Elems>;
            if constexpr (detail::HasTemplatedFindCoeffIndices<T>)
              return elems.template find_coeff_indices<functionspace::boundary>(
                                                                                typename T::boundary_eval_type{});
            else if constexpr (detail::HasFindCoeffIndices<T>)
              // Note that this is a fake call here
              return elems.find_coeff_indices(typename T::eval_type{});
          })()...);
        },
        tuple);
  }

public:
  /// @brief Type of the knot indices of the inputs in the interior
  using inputs_interior_knot_indices_type = decltype(find_interior_knot_indices(
      std::declval<std::tuple<Inputs...>>()));

  /// @brief Type alias for the type of the index-th knot indices of the inputs
  /// in the interior
  template <std::size_t index>
  using input_interior_knot_indices_t =
      std::tuple_element_t<index, inputs_interior_knot_indices_type>;

  /// @brief Type of the knot indices of the inputs at the boundary
  using inputs_boundary_knot_indices_type = decltype(find_boundary_knot_indices(
      std::declval<std::tuple<Inputs...>>()));

  /// @brief Type alias for the type of the index-th knot indices of the inputs
  /// at the boundary
  template <std::size_t index>
  using input_boundary_knot_indices_t =
      std::tuple_element_t<index, inputs_boundary_knot_indices_type>;

  /// @brief Type of the knot indices of the outputs in the interior
  using outputs_interior_knot_indices_type =
      decltype(find_interior_knot_indices(
          std::declval<std::tuple<Outputs...>>()));

  /// @brief Type alias for the type of the index-th knot indices of the outputs
  /// in the interior
  template <std::size_t index>
  using output_interior_knot_indices_t =
      std::tuple_element_t<index, outputs_interior_knot_indices_type>;

  /// @brief Type of the knot indices of the outputs at the boundary
  using outputs_boundary_knot_indices_type =
      decltype(find_boundary_knot_indices(
          std::declval<std::tuple<Outputs...>>()));

  /// @brief Type alias for the type of the index-th knot indices of the outputs
  /// at the boundary
  template <std::size_t index>
  using output_boundary_knot_indices_t =
      std::tuple_element_t<index, outputs_boundary_knot_indices_type>;

  /// @brief Type of the coefficient indices of the inputs in the interior
  using inputs_interior_coeff_indices_type =
      decltype(find_interior_coeff_indices(
          std::declval<std::tuple<Inputs...>>()));

  /// @brief Type alias for the type of the index-th coefficient indices of the
  /// inputs in the interior
  template <std::size_t index>
  using input_interior_coeff_indices_t =
      std::tuple_element_t<index, inputs_interior_coeff_indices_type>;

  /// @brief Type of the coefficient indices of the inputs at the boundary
  using inputs_boundary_coeff_indices_type =
      decltype(find_boundary_coeff_indices(
          std::declval<std::tuple<Inputs...>>()));

  /// @brief Type alias for the type of the index-th coefficient indices of the
  /// inputs at the boundary
  template <std::size_t index>
  using input_boundary_coeff_indices_t =
      std::tuple_element_t<index, inputs_boundary_coeff_indices_type>;

  /// @brief Type of the coefficient indices of the outputs in the interior
  using outputs_interior_coeff_indices_type =
      decltype(find_interior_coeff_indices(
          std::declval<std::tuple<Outputs...>>()));

  /// @brief Type alias for the type of the index-th coefficient indices of the
  /// outputs in the interior
  template <std::size_t index>
  using output_interior_coeff_indices_t =
      std::tuple_element_t<index, outputs_interior_coeff_indices_type>;

  /// @brief Type of the coefficient indices of the outputs at the boundary
  using outputs_boundary_coeff_indices_type =
      decltype(find_boundary_coeff_indices(
          std::declval<std::tuple<Outputs...>>()));

  /// @brief Type alias for the type of the index-th coefficient indices of the
  /// outputs at the boundary
  template <std::size_t index>
  using output_boundary_coeff_indices_t =
      std::tuple_element_t<index, outputs_boundary_coeff_indices_type>;
};

template <detail::HasAsTensor... Inputs, detail::HasAsTensor... Outputs,
          detail::HasAsTensor... CollPts>
class IgANetCustomizable<std::tuple<Inputs...>, std::tuple<Outputs...>,
                          std::tuple<CollPts...>>
    : public IgANetCustomizable<std::tuple<Inputs...>, std::tuple<Outputs...>,
                                 void> {
public:
  /// @brief Type of the knot indices of the collocation points objects in the
  /// interior
  using collPts_interior_knot_indices_type = std::tuple<
      decltype(std::declval<CollPts>()
                   .template find_knot_indices<functionspace::interior>(
                       std::declval<typename CollPts::eval_type>()))...>;

  /// @brief Type of the knot indices of the collocation points objects at the
  /// boundary
  using collPts_boundary_knot_indices_type = std::tuple<
      decltype(std::declval<CollPts>()
                   .template find_knot_indices<functionspace::boundary>(
                       std::declval<
                           typename CollPts::boundary_eval_type>()))...>;

  /// @brief Type of the coefficient indices of the collocation points objects
  /// in the interior
  using collPts_interior_coeff_indices_type = std::tuple<
      decltype(std::declval<CollPts>()
                   .template find_coeff_indices<functionspace::interior>(
                       std::declval<typename CollPts::eval_type>()))...>;

  /// @brief Type of the coefficient indices of the collocation points objects
  /// at the boundary
  using collPts_boundary_coeff_indices_type = std::tuple<
      decltype(std::declval<CollPts>()
                   .template find_coeff_indices<functionspace::boundary>(
                       std::declval<
                           typename CollPts::boundary_eval_type>()))...>;
};

/// @}
  
} // namespace iganet
