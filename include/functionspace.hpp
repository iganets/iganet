/**
   @file include/functionspace.hpp

   @brief Function spaces

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#pragma once

#include <boost/preprocessor/seq/for_each.hpp>

#include <bspline.hpp>
#include <utils/zip.hpp>

namespace iganet {

  /// @brief Enumerator for the function space component
  enum class functionspace : short_t
    {
      interior = 0, /*!< interior component */
      boundary = 1  /*!< boundary component */
    };
  
#define TUPLE_WRAPPER(FunctionSpace)                                    \
  namespace detail {                                                    \
    template<typename T>                                                \
    struct tuple<FunctionSpace<T>>                                      \
    {                                                                   \
      using type = typename tuple<typename FunctionSpace<T>::Base>::type; \
    };                                                                  \
  }

  namespace detail {

    // Forward declaration
    template<typename... spline_t>
    class FunctionSpace;

    /// @brief Tuple wrapper
    /// @{
    template<typename... Ts>
    struct tuple
    {
      using type = std::tuple<Ts...>;
    };

    template<typename... Ts>
    struct tuple<std::tuple<Ts...>>
    {
      using type = typename tuple<Ts...>::type;
    };

    template<typename... Ts>
    struct tuple<FunctionSpace<Ts...>>
    {
      using type = typename tuple<Ts...>::type;
    };
    /// @}

    /// @brief Function space type dispatcher
    /// @{
    template<typename... Ts>
    struct FunctionSpace_dispatch;

    template<typename... Ts>
    struct FunctionSpace_dispatch<std::tuple<Ts...>> {
      using type = FunctionSpace<Ts...>;
    };
    /// @}

    /// @brief Function space type
    template<typename... Ts>
    using FunctionSpace_t =
      typename FunctionSpace_dispatch<decltype(std::tuple_cat(std::declval<typename tuple<Ts>::type>()...))>::type;

    /// @brief Tensor-product function space
    ///
    /// @note This class is not meant for direct use in
    /// applications. Instead use S1, S2, S3, S4, TH1, TH2, TH3, TH4,
    /// NE1, NE2, NE3, NE, RT1, RT2, RT3, or RT4.
    template<typename... spline_t>
    class FunctionSpace
      : public std::tuple<spline_t...>, public utils::Serializable, private utils::FullQualifiedName
    {
    public:
      /// @brief Boundary spline objects type
      using boundary_t = std::tuple<Boundary<spline_t>...>;

      /// @brief Boundary spline objects evaluation type
      using boundary_eval_t = std::tuple<typename Boundary<spline_t>::eval_t...>;
      
    protected:
      /// @brief Boundary spline objects
      boundary_t boundary_;

    public:
      /// @brief Base class
      using Base = std::tuple<spline_t...>;

      /// @brief Value type
      using value_type = typename std::common_type<typename spline_t::value_type...>::type;

      /// @brief Evaluation type
      using eval_t = std::tuple<std::array<torch::Tensor, spline_t::parDim()>...>;
      
      /// @brief Default constructor
      FunctionSpace() = default;

      /// @brief Copy constructor
      FunctionSpace(const FunctionSpace&) = default;

      /// @brief Move constructor
      FunctionSpace(FunctionSpace&&) = default;

      /// @brief Constructor
      /// @{
      FunctionSpace(const std::array<int64_t, spline_t::parDim()>&... ncoeffs,
                    enum init init = init::zeros,
                    Options<value_type> options = iganet::Options<value_type>{})
        : Base({ncoeffs, init, options}...),
          boundary_({ncoeffs, init, options}...)
      {}

      FunctionSpace(const std::array<std::vector<typename spline_t::value_type>,
                    spline_t::parDim()>&... kv,
                    enum init init = init::zeros,
                    Options<value_type> options = iganet::Options<value_type>{})
        : Base({kv, init, options}...),
          boundary_({kv, init, options}...)
      {}      
      /// @}

      /// @brief Returns the dimension
      inline static constexpr short_t dim()
      {
        return sizeof...(spline_t);
      }
      
    private:
      /// @brief Returns the coefficients of all spaces as a single tensor
      template<size_t... Is>
      inline torch::Tensor as_tensor_(std::index_sequence<Is...>,
                                      bool boundary = true) const
      {
        if (boundary)
          return torch::cat({std::get<Is>(*this).as_tensor()...,
              std::get<Is>(boundary_).as_tensor()...});
        else
          return torch::cat({std::get<Is>(*this).as_tensor()...});
      }

      /// @brief Returns the size of the single tensor representation of all spaces
      template<size_t... Is>
      inline int64_t as_tensor_size_(std::index_sequence<Is...>,
                                     bool boundary = true) const
      {
        if (boundary)
          return std::apply([]( auto... v ){ return ( v + ... ); },
                            std::make_tuple(std::get<Is>(*this).as_tensor_size()...))
            +    std::apply([]( auto... v ){ return ( v + ... ); },
                            std::make_tuple(std::get<Is>(boundary_).as_tensor_size()...));
        else
          return  std::apply([]( auto... v ){ return ( v + ... ); },
                             std::make_tuple(std::get<Is>(*this).as_tensor_size()...));
      }
      
      /// @brief Sets the coefficients of all spaces from a single tensor
      template<size_t... Is>
      inline auto& from_tensor_(std::index_sequence<Is...>,
                                const torch::Tensor& coeffs,
                                bool boundary = true)
      {
        throw std::runtime_error("from_tensor is not implemented yet");
        return *this;
      }
      
    public:
      /// @brief Returns the coefficients of all spaces as a single tensor
      inline torch::Tensor as_tensor(bool boundary = true) const
      {
        return as_tensor_(std::make_index_sequence<sizeof...(spline_t)>{}, boundary);
      }

      /// @brief Returns the size of the single tensor representation of all spaces
      inline int64_t as_tensor_size(bool boundary = true) const
      {
        return as_tensor_size_(std::make_index_sequence<sizeof...(spline_t)>{}, boundary);
      }

      /// @brief Sets the coefficients of all spaces from a single tensor
      inline auto& from_tensor(const torch::Tensor& coeffs, bool boundary = true)
      {
        return from_tensor_(std::make_index_sequence<sizeof...(spline_t)>{}, coeffs, boundary);
      }
      
      /// @brief Returns a constant reference to the boundary spline object
      inline const auto& boundary() const
      {
        return boundary_;
      }

      /// @brief Returns a non-constant reference to the boundary spline object
      inline auto& boundary()
      {
        return boundary_;
      }
      
    private:
      /// @brief Returns the dimension of all bases
      template<functionspace comp = functionspace::interior,
               size_t... Is>
      int64_t basisDim_(std::index_sequence<Is...>) const
      {
        if constexpr (comp == functionspace::interior)
          return (std::get<Is>(*this).ncumcoeffs() + ...);
        else if constexpr (comp == functionspace::boundary)
          return (std::get<Is>(boundary_).ncumcoeffs() + ...);
      }

      /// @brief Serialization to JSON
      template<size_t... Is>
      nlohmann::json to_json_(std::index_sequence<Is...>) const
      {
        auto json_this = nlohmann::json::array();
        auto json_boundary = nlohmann::json::array();
        (json_this.push_back(std::get<Is>(*this).to_json()), ...);
        (json_boundary.push_back(std::get<Is>(boundary_).to_json()), ...);

        auto json = nlohmann::json::array();
        for (auto [t,b] : utils::zip(json_this, json_boundary)) {
          auto json_inner = nlohmann::json::array();
          json_inner.push_back(t);
          json_inner.push_back(b);
          json.push_back(json_inner);
        }        
        
        return json;
      }
        
      /// @brief Returns the values of the spline objects in the points `xi`
      /// @{
      template<functionspace comp = functionspace::interior,
               deriv deriv = deriv::func,
               bool memory_optimized = false,
               size_t... Is, typename... Xi>
      inline auto eval_(std::index_sequence<Is...>,
                        const std::tuple<Xi...>& xi) const
      {
        if constexpr (comp == functionspace::interior)
          return std::tuple(std::get<Is>(*this).template eval<deriv, memory_optimized>(std::get<Is>(xi))...);
        else if constexpr (comp == functionspace::boundary)
          return std::tuple(std::get<Is>(boundary_).template eval<deriv, memory_optimized>(std::get<Is>(xi))...);
      }

      template<functionspace comp = functionspace::interior,
               deriv deriv = deriv::func,
               bool memory_optimized = false,
               size_t... Is, typename... Xi, typename... Indices>
      inline auto eval_(std::index_sequence<Is...>,
                        const std::tuple<Xi...>& xi,
                        const std::tuple<Indices...>& indices) const
      {
        if constexpr (comp == functionspace::interior)
          return std::tuple(std::get<Is>(*this).template eval<deriv, memory_optimized>(std::get<Is>(xi),
                                                                                       std::get<Is>(indices))...);
        else if constexpr (comp == functionspace::boundary)
          return std::tuple(std::get<Is>(boundary_).template eval<deriv, memory_optimized>(std::get<Is>(xi),
                                                                                           std::get<Is>(indices))...);
      }

      template<functionspace comp = functionspace::interior,
               deriv deriv = deriv::func,
               bool memory_optimized = false,
               size_t... Is, typename... Xi, typename... Indices, typename... Coeff_Indices>
      inline auto eval_(std::index_sequence<Is...>,
                        const std::tuple<Xi...>& xi,
                        const std::tuple<Indices...>& indices,
                        const std::tuple<Coeff_Indices...>& coeff_indices) const
      {
        if constexpr (comp == functionspace::interior)
          return std::tuple(std::get<Is>(*this).template eval<deriv, memory_optimized>(std::get<Is>(xi),
                                                                                       std::get<Is>(indices),
                                                                                       std::get<Is>(coeff_indices))...);
        else if constexpr (comp == functionspace::boundary)
          return std::tuple(std::get<Is>(boundary_).template eval<deriv, memory_optimized>(std::get<Is>(xi),
                                                                                           std::get<Is>(indices),
                                                                                           std::get<Is>(coeff_indices))...);
      }
      /// @}

      /// @brief Returns the value of the spline objects from
      /// precomputed basis function
      /// @{
      template<functionspace comp = functionspace::interior,
               size_t... Is,
               typename... Basfunc, typename... Coeff_Indices,
               typename... Numeval, typename... Sizes>
      inline auto eval_from_precomputed_(std::index_sequence<Is...>,
                                         const std::tuple<Basfunc...>& basfunc,
                                         const std::tuple<Coeff_Indices...>& coeff_indices,
                                         const std::tuple<Numeval...>& numeval,
                                         const std::tuple<Sizes...>& sizes) const
      {
        if constexpr (comp == functionspace::interior)
          return std::tuple(std::get<Is>(*this).eval_from_precomputed(std::get<Is>(basfunc),
                                                                      std::get<Is>(coeff_indices),
                                                                      std::get<Is>(numeval),
                                                                      std::get<Is>(sizes))...);
        else if constexpr (comp == functionspace::boundary)
          return std::tuple(std::get<Is>(boundary_).eval_from_precomputed(std::get<Is>(basfunc),
                                                                          std::get<Is>(coeff_indices),
                                                                          std::get<Is>(numeval),
                                                                          std::get<Is>(sizes))...);
      }

      template<functionspace comp = functionspace::interior,
               size_t... Is,
               typename... Basfunc, typename... Coeff_Indices, typename... Xi>
      inline auto eval_from_precomputed_(std::index_sequence<Is...>,
                                         const std::tuple<Basfunc...>& basfunc,
                                         const std::tuple<Coeff_Indices...>& coeff_indices,
                                         const std::tuple<Xi...>& xi) const
      {
        if constexpr (comp == functionspace::interior)        
          return std::tuple(std::get<Is>(*this).eval_from_precomputed(std::get<Is>(basfunc),
                                                                      std::get<Is>(coeff_indices),
                                                                      std::get<Is>(xi)[0].numel(),
                                                                      std::get<Is>(xi)[0].sizes())...);
        else if constexpr (comp == functionspace::boundary)
          return std::tuple(std::get<Is>(boundary_).eval_from_precomputed(std::get<Is>(basfunc),
                                                                          std::get<Is>(coeff_indices),
                                                                          std::get<Is>(xi))...);
      }
      /// @}

      /// @brief Returns the knot indicies of knot spans containing `xi`
      template<functionspace comp = functionspace::interior,
               size_t... Is, typename... Xi>
      inline auto find_knot_indices_(std::index_sequence<Is...>,
                                     const std::tuple<Xi...>& xi) const
      {
        if constexpr (comp == functionspace::interior)
          return std::tuple(std::get<Is>(*this).find_knot_indices(std::get<Is>(xi))...);
        else if constexpr (comp == functionspace::boundary)
          return std::tuple(std::get<Is>(boundary_).find_knot_indices(std::get<Is>(xi))...);
      }

      /// @brief Returns the values of the spline objects' basis functions in the points `xi`
      /// @{
      template<functionspace comp = functionspace::interior,
               deriv deriv = deriv::func,
               bool memory_optimized = false,
               size_t... Is, typename... Xi>
      inline auto eval_basfunc_(std::index_sequence<Is...>,
                                const std::tuple<Xi...>& xi) const
      {
        if constexpr (comp == functionspace::interior)
          return std::tuple(std::get<Is>(*this).template eval_basfunc<deriv, memory_optimized>(std::get<Is>(xi))...);
        else if constexpr (comp == functionspace::boundary)
          return std::tuple(std::get<Is>(boundary_).template eval_basfunc<deriv, memory_optimized>(std::get<Is>(xi))...);
      }
      
      template<functionspace comp = functionspace::interior,
               deriv deriv = deriv::func,
               bool memory_optimized = false,
               size_t... Is, typename... Xi, typename... Indices>
      inline auto eval_basfunc_(std::index_sequence<Is...>,
                                const std::tuple<Xi...>& xi,
                                const std::tuple<Indices...>& indices) const
      {
        if constexpr (comp == functionspace::interior)
          return std::tuple(std::get<Is>(*this).template eval_basfunc<deriv, memory_optimized>(std::get<Is>(xi),
                                                                                               std::get<Is>(indices))...);
        else if constexpr (comp == functionspace::boundary)
          return std::tuple(std::get<Is>(boundary_).template eval_basfunc<deriv, memory_optimized>(std::get<Is>(xi),
                                                                                                   std::get<Is>(indices))...);
      }
      /// @}

      /// @brief Returns the indices of the spline objects'
      /// coefficients corresponding to the knot indices `indices`
      template<functionspace comp = functionspace::interior,
               bool memory_optimized = false,
               size_t... Is, typename... Indices>
      inline auto find_coeff_indices_(std::index_sequence<Is...>,
                                      const std::tuple<Indices...>& indices) const
      {
        if constexpr (comp == functionspace::interior)
          return std::tuple(std::get<Is>(*this).template find_coeff_indices<memory_optimized>(std::get<Is>(indices))...);
        else if constexpr (comp == functionspace::boundary)
          return std::tuple(std::get<Is>(boundary_).template find_coeff_indices<memory_optimized>(std::get<Is>(indices))...);
      }
      
      /// @brief Returns the spline objects with uniformly refined
      /// knot and coefficient vectors
      template<size_t... Is>
      inline auto& uniform_refine_(std::index_sequence<Is...>,
                                   int numRefine = 1, int dim = -1)
      {
        (std::get<Is>(*this).uniform_refine(numRefine, dim), ...);
        (std::get<Is>(boundary_).uniform_refine(numRefine, dim), ...);
        return *this;
      }
      
      /// @brief Writes the function space object into a
      /// torch::serialize::OutputArchive object
      template<size_t... Is>
      inline torch::serialize::OutputArchive& write_(std::index_sequence<Is...>,
                                                     torch::serialize::OutputArchive& archive,
                                                     const std::string& key="functionspace") const
      {
        (std::get<Is>(*this).write(archive, key+".fspace["+std::to_string(Is)+"].interior"), ...);
        (std::get<Is>(boundary_).write(archive, key+".fspace["+std::to_string(Is)+"].boundary"), ...);
        return archive;
      }
      
      /// @brief Loads the function space object from a
      /// torch::serialize::InputArchive object
      template<size_t... Is>
      inline torch::serialize::InputArchive& read_(std::index_sequence<Is...>,
                                                   torch::serialize::InputArchive& archive,
                                                   const std::string& key="functionspace")
      {
        (std::get<Is>(*this).read(archive, key+".fspace["+std::to_string(Is)+"].interior"), ...);
        (std::get<Is>(boundary_).read(archive, key+".fspace["+std::to_string(Is)+"].boundary"), ...);
        return archive;
      }
      
    public:      
      /// @brief Returns the dimension of all bases
      template<functionspace comp = functionspace::interior>
      int64_t basisDim() const
      {
        return basisDim_<comp>(std::make_index_sequence<sizeof...(spline_t)>{});
      }

      /// @brief Serialization to JSON
      nlohmann::json to_json() const override
      {
        return to_json_(std::make_index_sequence<sizeof...(spline_t)>{});
      }

      /// @brief Returns the values of the spline objects in the points `xi`
      /// @{
      template<functionspace comp = functionspace::interior,
               deriv deriv = deriv::func,
               bool memory_optimized = false,
               typename... Xi>
      inline auto eval(const std::tuple<Xi...>& xi) const
      {
        return eval_<comp, deriv, memory_optimized>(std::make_index_sequence<sizeof...(spline_t)>{}, xi);
      }

      template<functionspace comp = functionspace::interior,
               deriv deriv = deriv::func,
               bool memory_optimized = false,
               typename... Xi, typename... Indices>
      inline auto eval(const std::tuple<Xi...>& xi,
                       const std::tuple<Indices...>& indices) const
      {
        return eval_<comp, deriv, memory_optimized>(std::make_index_sequence<sizeof...(spline_t)>{}, xi, indices);
      }

      template<functionspace comp = functionspace::interior,
               deriv deriv = deriv::func,
               bool memory_optimized = false,
               typename... Xi, typename... Indices, typename... Coeff_Indices>
      inline auto eval(const std::tuple<Xi...>& xi,
                       const std::tuple<Indices...>& indices,
                       const std::tuple<Coeff_Indices...>& coeff_indices) const
      {
        return eval_<comp, deriv, memory_optimized>(std::make_index_sequence<sizeof...(spline_t)>{}, xi, indices, coeff_indices);
      }
      /// @}

      /// @brief Returns the value of the spline objects from
      /// precomputed basis function
      /// @{
      template<functionspace comp = functionspace::interior,
               typename... Basfunc, typename... Coeff_Indices,
               typename... Numeval, typename... Sizes>
      inline auto eval_from_precomputed(const std::tuple<Basfunc...>& basfunc,
                                        const std::tuple<Coeff_Indices...>& coeff_indices,
                                        const std::tuple<Numeval...>& numeval,
                                        const std::tuple<Sizes...>& sizes) const
      {
        return eval_from_precomputed_<comp>(std::make_index_sequence<sizeof...(spline_t)>{},
                                            basfunc, coeff_indices, numeval, sizes);
      }

      template<functionspace comp = functionspace::interior,
               typename... Basfunc, typename... Coeff_Indices, typename... Xi>
      inline auto eval_from_precomputed(const std::tuple<Basfunc...>& basfunc,
                                        const std::tuple<Coeff_Indices...>& coeff_indices,
                                        const std::tuple<Xi...>& xi) const
      {
        return eval_from_precomputed_<comp>(std::make_index_sequence<sizeof...(spline_t)>{},
                                            basfunc, coeff_indices, xi);
      }
      /// @}

      /// @brief Returns the knot indicies of knot spans containing `xi`
      template<functionspace comp = functionspace::interior,
               typename... Xi>
      inline auto find_knot_indices(const std::tuple<Xi...>& xi) const
      {
        return find_knot_indices_<comp>(std::make_index_sequence<sizeof...(spline_t)>{}, xi);
      }

      /// @brief Returns the values of the spline objects' basis
      /// functions in the points `xi` @{
      template<functionspace comp = functionspace::interior,
               deriv deriv = deriv::func,
               bool memory_optimized = false,
               typename... Xi>
      inline auto eval_basfunc(const std::tuple<Xi...>& xi) const
      {
        return eval_basfunc_<comp, deriv, memory_optimized>(std::make_index_sequence<sizeof...(spline_t)>{}, xi);
      }

      template<functionspace comp = functionspace::interior,
               deriv deriv = deriv::func,
               bool memory_optimized = false,
               typename... Xi, typename... Indices>
      inline auto eval_basfunc(const std::tuple<Xi...>& xi,
                               const std::tuple<Indices...>& indices) const
      {
        return eval_basfunc_<comp, deriv, memory_optimized>(std::make_index_sequence<sizeof...(spline_t)>{}, xi, indices);
      }
      /// @}

      /// @brief Returns the indices of the spline objects'
      /// coefficients corresponding to the knot indices `indices`
      template<functionspace comp = functionspace::interior,
               bool memory_optimized = false,
               typename... Indices>
      inline auto find_coeff_indices(const std::tuple<Indices...>& indices) const
      {
        return find_coeff_indices_<comp, memory_optimized>(std::make_index_sequence<sizeof...(spline_t)>{}, indices);
      }

      /// @brief Returns the spline objects with uniformly refined
      /// knot and coefficient vectors
      inline auto& uniform_refine(int numRefine = 1, int dim = -1)
      {
        uniform_refine_(std::make_index_sequence<sizeof...(spline_t)>{}, numRefine, dim);
        return *this;
      }

      /// @brief Writes the function space object into a
      /// torch::serialize::OutputArchive object
      inline torch::serialize::OutputArchive& write(torch::serialize::OutputArchive& archive,
                                                    const std::string& key="functionspace") const
      {
        write_(std::make_index_sequence<sizeof...(spline_t)>{}, archive, key);
        return archive;
      }
      
      /// @brief Loads the function space object from a
      /// torch::serialize::InputArchive object
      inline torch::serialize::InputArchive& read(torch::serialize::InputArchive& archive,
                                                  const std::string& key="functionspace")
      {
        read_(std::make_index_sequence<sizeof...(spline_t)>{}, archive, key);
        return archive;
      }

      /// @brief Returns a string representation of the function space object
      inline virtual void pretty_print(std::ostream& os = std::cout) const noexcept override
      {
        os << *this;
      }
    };

    /// @brief Function space
    ///
    /// @note This class is not meant for direct use in
    /// applications. Instead use S1, S2, S3, S4, TH1, TH2, TH3, TH4,
    /// NE1, NE2, NE3, NE, RT1, RT2, RT3, or RT4.
    template<typename spline_t>
    class FunctionSpace<spline_t>
      : public spline_t
    {
    public:
      /// @brief Boundary spline objects type
      using boundary_t = Boundary<spline_t>;

      /// @brief Boundary spline objects evaluation type
      using boundary_eval_t = typename Boundary<spline_t>::eval_t;
      
    protected:
      /// @brief Boundary spline objects
      boundary_t boundary_;

    public:
      /// @brief Base class
      using Base = spline_t;

      /// @brief Value type
      using value_type = typename spline_t::value_type;

      /// @brief Evaluation type
      using eval_t = std::array<torch::Tensor, spline_t::parDim()>;
      
      /// @brief Default constructor
      FunctionSpace() = default;

      /// @brief Copy constructor
      FunctionSpace(const FunctionSpace&) = default;

      /// @brief Move constructor
      FunctionSpace(FunctionSpace&&) = default;

      /// @brief Constructor
      /// @{
      FunctionSpace(const std::array<int64_t, spline_t::parDim()>& ncoeffs,
                    enum init init = init::zeros,
                    //bool shared_boundary = false,
                    Options<value_type> options = iganet::Options<value_type>{})
        : Base(ncoeffs, init, options),
          boundary_(ncoeffs, init, options)
      {}

      FunctionSpace(std::array<std::vector<typename spline_t::value_type>,
                    spline_t::parDim()> kv,
                    enum init init = init::zeros,
                    //bool shared_boundary = false,
                    Options<value_type> options = iganet::Options<value_type>{})
        : Base(kv, init, options),
          boundary_(kv, init, options)
      {}
      /// @}
      
      /// @brief Returns the dimension
      inline static constexpr short_t dim()
      {
        return 1;
      }     

      /// @brief Returns the coefficients of all spaces as a single tensor
      inline torch::Tensor as_tensor(bool boundary = true) const
      {
        if (boundary)
          return torch::cat({Base::as_tensor(), boundary_.as_tensor()});
        else
          return Base::as_tensor();
      }

      /// @brief Returns the size of the single tensor representation of all spaces
      inline int64_t as_tensor_size(bool boundary = true) const
      {
        if (boundary)
          return Base::as_tensor_size() + boundary_.as_tensor_size();
        else
          return Base::as_tensor_size();
      }

      /// @brief Sets the coefficients of all spaces from a single tensor
      inline auto& from_tensor(const torch::Tensor& coeffs, bool boundary = true)
      {
        Base::from_tensor(coeffs.index({torch::indexing::Slice(0, Base::as_tensor_size())}));
        
        if (boundary)
          boundary_.from_tensor(coeffs.index({torch::indexing::Slice(Base::as_tensor_size(), torch::indexing::None)}));
        
        return *this;
      }

      /// @brief Returns a constant reference to the boundary spline object
      inline const auto& boundary() const
      {
        return boundary_;
      }

      /// @brief Returns a non-constant reference to the boundary spline object
      inline auto& boundary()
      {
        return boundary_;
      }

      /// @brief Returns the dimension of the basis
      template<functionspace comp = functionspace::interior>
      int64_t basisDim() const
      {
        if constexpr (comp == functionspace::interior)
          return spline_t::ncumcoeffs();
        else if constexpr (comp == functionspace::boundary)
          return boundary_.ncumcoeffs();
      }

      /// @brief Serialization to JSON
      nlohmann::json to_json() const override
      {
        auto json = nlohmann::json::array();
        json.push_back(Base::to_json());
        json.push_back(boundary_.to_json());
        return json;
      }

      /// @brief Returns the values of the spline object in the points `xi`
      template<functionspace comp = functionspace::interior,
               deriv deriv = deriv::func,
               bool memory_optimized = false, 
               typename... Args>
      inline auto eval(const Args&... args) const
      {
        if constexpr (comp == functionspace::interior)
          return spline_t::template eval<deriv, memory_optimized>(args...);
        else if constexpr (comp == functionspace::boundary)
          return boundary_.template eval<deriv, memory_optimized>(args...);
      }
      
#define GENERATE_EXPR_MACRO(r, data, name)                              \
      template<functionspace comp = functionspace::interior,            \
               bool memory_optimized = false,                           \
               typename... Args>                                        \
      inline auto name(const Args&... args) const                       \
      {                                                                 \
        if constexpr (comp == functionspace::interior)                  \
          return spline_t::template name<memory_optimized>(args...);    \
        else if constexpr (comp == functionspace::boundary)             \
          return boundary_.template name<memory_optimized>(args...);    \
      }

      BOOST_PP_SEQ_FOR_EACH(GENERATE_EXPR_MACRO, _, GENERATE_EXPR_SEQ)
#undef GENERATE_EXPR_MACRO
      
#define GENERATE_IEXPR_MACRO(r, data, name)                             \
      template<functionspace comp = functionspace::interior,            \
               bool memory_optimized = false,                           \
               typename Geometry_t,                                     \
               typename... Args>                                        \
      inline auto name(const Geometry_t& G,                             \
                       const Args&... args) const                       \
      {                                                                 \
        if constexpr (comp == functionspace::interior)                  \
          return spline_t::template name<memory_optimized>              \
            (static_cast<typename Geometry_t::Base::Base>(G), args...); \
        else if constexpr (comp == functionspace::boundary)             \
          return boundary_.template name<memory_optimized>              \
            (static_cast<typename Geometry_t::Base::Base>(G), args...); \
      }
      
      BOOST_PP_SEQ_FOR_EACH(GENERATE_IEXPR_MACRO, _, GENERATE_IEXPR_SEQ)
#undef GENERATE_IEXPR_MACRO
      
      /// @brief Returns the value of the spline object from
      /// precomputed basis function
      template<functionspace comp = functionspace::interior,
               typename... Args>
      inline auto eval_from_precomputed(const Args&... args) const
      {
        if constexpr (comp == functionspace::interior)
          return spline_t::eval_from_precomputed(args...);
        else if constexpr (comp == functionspace::boundary)
          return boundary_.eval_from_precomputed(args...);
      }

      /// @brief Returns the knot indicies of knot spans containing `xi`
      template<functionspace comp = functionspace::interior, typename Xi>
      inline auto find_knot_indices(const Xi& xi) const
      {
        if constexpr (comp == functionspace::interior)
          return spline_t::find_knot_indices(xi);
        else if constexpr (comp == functionspace::boundary)
          return boundary_.find_knot_indices(xi);
      }

      /// @brief Returns the values of the spline objects' basis
      /// functions in the points `xi`
      template<functionspace comp = functionspace::interior,
               deriv deriv = deriv::func,
               bool memory_optimized = false,
               typename... Args>
      inline auto eval_basfunc(const Args&... args) const
      {
        if constexpr (comp == functionspace::interior)
          return spline_t::template eval_basfunc<deriv, memory_optimized>(args...);
        else if constexpr (comp == functionspace::boundary)
          return boundary_.template eval_basfunc<deriv, memory_optimized>(args...);
      }

      /// @brief Returns the indices of the spline objects'
      /// coefficients corresponding to the knot indices `indices`
      template<functionspace comp = functionspace::interior,
               bool memory_optimized = false, 
               typename Indices>
      inline auto find_coeff_indices(const Indices& indices) const
      {
        if constexpr (comp == functionspace::interior)
          return spline_t::template find_coeff_indices<memory_optimized>(indices);
        else if constexpr (comp == functionspace::boundary)
          return boundary_.template find_coeff_indices<memory_optimized>(indices);
      }

      /// @brief Returns the spline objects with uniformly refined
      /// knot and coefficient vectors
      inline auto& uniform_refine(int numRefine = 1, int dim = -1)
      {
        spline_t::uniform_refine(numRefine, dim);
        boundary_.uniform_refine(numRefine, dim);
        return *this;
      }

      /// @brief Writes the function space object into a
      /// torch::serialize::OutputArchive object
      inline torch::serialize::OutputArchive& write(torch::serialize::OutputArchive& archive,
                                                    const std::string& key="functionspace") const
      {
        spline_t::write(archive, key);
        boundary_.write(archive, key);
        return archive;
      }
      
      /// @brief Loads the function space object from a
      /// torch::serialize::InputArchive object
      inline torch::serialize::InputArchive& read(torch::serialize::InputArchive& archive,
                                                  const std::string& key="functionspace")
      {
        spline_t::read(archive, key);
        boundary_.read(archive, key);
        return archive;
      }

      /// @brief Returns a string representation of the function space object
      inline virtual void pretty_print(std::ostream& os = std::cout) const noexcept override
      {
        os << spline_t::name()
           << "(\ninterior = ";
        Base::pretty_print(os);
        os << "\nboundary = ";
        boundary_.pretty_print(os);
        os << "\n)";
      }
    };
  } // namespace detail

  template<typename T, typename... Ts>
  using FunctionSpace = detail::FunctionSpace_t<T, Ts...>;

  /// @brief Print (as string) a function space object
  template<typename T, typename... Ts>
  inline std::ostream& operator<<(std::ostream& os,
                                  const FunctionSpace<T, Ts...>& obj)
  {
    obj.pretty_print(os);
    return os;
  }
  
  /// @brief Spline function space \f$ S_{p}^{p-1} \f$
  template<typename spline_t, short_t... Cs>
  class S1
    : public FunctionSpace<typename spline_t::template
                           derived_self_type_t<typename spline_t::value_type,
                                               spline_t::geoDim(),
                                               spline_t::degree(0)>>
  {
  public:
    /// @brief Base type
    using Base = FunctionSpace<typename spline_t::template
                               derived_self_type_t<typename spline_t::value_type,
                                                   spline_t::geoDim(),
                                                   spline_t::degree(0)>>;

    /// @brief Constructor
    /// @{
    S1() = default;
    S1(S1&&) = default;
    S1(const S1&) = default;

    S1(const std::array<int64_t, spline_t::parDim()>& ncoeffs,
       enum init init = init::zeros,
       Options<typename spline_t::value_type> options = iganet::Options<typename spline_t::value_type>{})
      : Base(ncoeffs, init, options)
    {
      if constexpr (sizeof...(Cs) == 1) {
        if constexpr (const auto c = std::get<0>(std::make_tuple(Cs...));
                      c != spline_t::degree(0)-1)
          Base::uniform_refine(spline_t::degree(0)-1-c, 0);
      } else
        static_assert(sizeof...(Cs) == 0, "Dimensions mismatch");
    }

    S1(std::array<std::vector<typename spline_t::value_type>,
       spline_t::parDim()> kv,
       enum init init = init::zeros,
       Options<typename spline_t::value_type> options = iganet::Options<typename spline_t::value_type>{})
      : Base(kv, init, options)
    {
      if constexpr (sizeof...(Cs) == 1) {
        if constexpr (const auto c = std::get<0>(std::make_tuple(Cs...));
                      c != spline_t::degree(0)-1)
          Base::uniform_refine(spline_t::degree(0)-1-c, 0);
      } else
        static_assert(sizeof...(Cs) == 0, "Dimensions mismatch");
    }
    /// @}
  };

  TUPLE_WRAPPER(S1);

  template<typename spline_t, short_t... Cs>
  class S2
    : public FunctionSpace<typename spline_t::template
                           derived_self_type_t<typename spline_t::value_type,
                                               spline_t::geoDim(),
                                               spline_t::degree(0),
                                               spline_t::degree(1)>>
  {
  public:
    /// @brief Base type
    using Base = FunctionSpace<typename spline_t::template
                               derived_self_type_t<typename spline_t::value_type,
                                                   spline_t::geoDim(),
                                                   spline_t::degree(0),
                                                   spline_t::degree(1)>>;

    /// @brief Constructor
    /// @{
    S2() = default;
    S2(S2&&) = default;
    S2(const S2&) = default;

    S2(const std::array<int64_t, spline_t::parDim()>& ncoeffs,
       enum init init = init::zeros,
       Options<typename spline_t::value_type> options = iganet::Options<typename spline_t::value_type>{})
      : Base(ncoeffs, init, options)
    {
      if constexpr (sizeof...(Cs) == 2) {
        if constexpr (const auto c = std::get<0>(std::make_tuple(Cs...));
                      c != spline_t::degree(0)-1)
          Base::uniform_refine(spline_t::degree(0)-1-c, 0);
        if constexpr (const auto c = std::get<1>(std::make_tuple(Cs...));
                      c != spline_t::degree(1)-1)
          Base::uniform_refine(spline_t::degree(1)-1-c, 1);
      } else
        static_assert(sizeof...(Cs) == 0, "Dimensions mismatch");
    }

    S2(std::array<std::vector<typename spline_t::value_type>,
       spline_t::parDim()> kv,
       enum init init = init::zeros,
       Options<typename spline_t::value_type> options = iganet::Options<typename spline_t::value_type>{})
      : Base(kv, init, options)
    {
      if constexpr (sizeof...(Cs) == 2) {
        if constexpr (const auto c = std::get<0>(std::make_tuple(Cs...));
                      c != spline_t::degree(0)-1)
          Base::uniform_refine(spline_t::degree(0)-1-c, 0);
        if constexpr (const auto c = std::get<1>(std::make_tuple(Cs...));
                      c != spline_t::degree(1)-1)
          Base::uniform_refine(spline_t::degree(1)-1-c, 1);
      } else
        static_assert(sizeof...(Cs) == 0, "Dimensions mismatch");
    }
    /// @}
  };

  TUPLE_WRAPPER(S2);

  template<typename spline_t, short_t... Cs>
  class S3
    : public FunctionSpace<typename spline_t::template
                           derived_self_type_t<typename spline_t::value_type,
                                               spline_t::geoDim(),
                                               spline_t::degree(0),
                                               spline_t::degree(1),
                                               spline_t::degree(2)>>
  {
  public:
    /// @brief Base type
    using Base = FunctionSpace<typename spline_t::template
                               derived_self_type_t<typename spline_t::value_type,
                                                   spline_t::geoDim(),
                                                   spline_t::degree(0),
                                                   spline_t::degree(1),
                                                   spline_t::degree(2)>>;

    /// @brief Constructor
    /// @{
    S3() = default;
    S3(S3&&) = default;
    S3(const S3&) = default;

    S3(const std::array<int64_t, spline_t::parDim()>& ncoeffs,
       enum init init = init::zeros,
       Options<typename spline_t::value_type> options = iganet::Options<typename spline_t::value_type>{})
      : Base(ncoeffs, init, options)
    {
      if constexpr (sizeof...(Cs) == 3) {
        if constexpr (const auto c = std::get<0>(std::make_tuple(Cs...));
                      c != spline_t::degree(0)-1)
          Base::uniform_refine(spline_t::degree(0)-1-c, 0);
        if constexpr (const auto c = std::get<1>(std::make_tuple(Cs...));
                      c != spline_t::degree(1)-1)
          Base::uniform_refine(spline_t::degree(1)-1-c, 1);
        if constexpr (const auto c = std::get<2>(std::make_tuple(Cs...));
                      c != spline_t::degree(2)-1)
          Base::uniform_refine(spline_t::degree(2)-1-c, 2);
      } else
        static_assert(sizeof...(Cs) == 0, "Dimensions mismatch");
    }

    S3(std::array<std::vector<typename spline_t::value_type>,
       spline_t::parDim()> kv,
       enum init init = init::zeros,
       Options<typename spline_t::value_type> options = iganet::Options<typename spline_t::value_type>{})
      : Base(kv, init, options)
    {
      if constexpr (sizeof...(Cs) == 3) {
        if constexpr (const auto c = std::get<0>(std::make_tuple(Cs...));
                      c != spline_t::degree(0)-1)
          Base::uniform_refine(spline_t::degree(0)-1-c, 0);
        if constexpr (const auto c = std::get<1>(std::make_tuple(Cs...));
                      c != spline_t::degree(1)-1)
          Base::uniform_refine(spline_t::degree(1)-1-c, 1);
        if constexpr (const auto c = std::get<2>(std::make_tuple(Cs...));
                      c != spline_t::degree(2)-1)
          Base::uniform_refine(spline_t::degree(2)-1-c, 2);
      } else
        static_assert(sizeof...(Cs) == 0, "Dimensions mismatch");
    }
    /// @}
  };

  TUPLE_WRAPPER(S3);

  template<typename spline_t, short_t... Cs>
  class S4
    : public FunctionSpace<typename spline_t::template
                           derived_self_type_t<typename spline_t::value_type,
                                               spline_t::geoDim(),
                                               spline_t::degree(0),
                                               spline_t::degree(1),
                                               spline_t::degree(2),
                                               spline_t::degree(3)>>
  {
  public:
    /// @brief Base type
    using Base = FunctionSpace<typename spline_t::template
                               derived_self_type_t<typename spline_t::value_type,
                                                   spline_t::geoDim(),
                                                   spline_t::degree(0),
                                                   spline_t::degree(1),
                                                   spline_t::degree(2),
                                                   spline_t::degree(3)>>;

    /// @brief Constructor
    /// @{
    S4() = default;
    S4(S4&&) = default;
    S4(const S4&) = default;

    S4(const std::array<int64_t, spline_t::parDim()>& ncoeffs,
       enum init init = init::zeros,
       Options<typename spline_t::value_type> options = iganet::Options<typename spline_t::value_type>{})
      : Base(ncoeffs, init, options)
    {
      if constexpr (sizeof...(Cs) == 4) {
        if constexpr (const auto c = std::get<0>(std::make_tuple(Cs...));
                      c != spline_t::degree(0)-1)
          Base::uniform_refine(spline_t::degree(0)-1-c, 0);
        if constexpr (const auto c = std::get<1>(std::make_tuple(Cs...));
                      c != spline_t::degree(1)-1)
          Base::uniform_refine(spline_t::degree(1)-1-c, 1);
        if constexpr (const auto c = std::get<2>(std::make_tuple(Cs...));
                      c != spline_t::degree(2)-1)
          Base::uniform_refine(spline_t::degree(2)-1-c, 2);
        if constexpr (const auto c = std::get<3>(std::make_tuple(Cs...));
                      c != spline_t::degree(3)-1)
          Base::uniform_refine(spline_t::degree(3)-1-c, 3);
      } else
        static_assert(sizeof...(Cs) == 0, "Dimensions mismatch");
    }

    S4(std::array<std::vector<typename spline_t::value_type>,
       spline_t::parDim()> kv,
       enum init init = init::zeros,
       Options<typename spline_t::value_type> options = iganet::Options<typename spline_t::value_type>{})
      : Base(kv, init, options)
    {
      if constexpr (sizeof...(Cs) == 4) {
        if constexpr (const auto c = std::get<0>(std::make_tuple(Cs...));
                      c != spline_t::degree(0)-1)
          std::get<0>(*this).uniform_refine(spline_t::degree(0)-1-c, 0);
        if constexpr (const auto c = std::get<1>(std::make_tuple(Cs...));
                      c != spline_t::degree(1)-1)
          std::get<1>(*this).uniform_refine(spline_t::degree(1)-1-c, 1);
        if constexpr (const auto c = std::get<2>(std::make_tuple(Cs...));
                      c != spline_t::degree(2)-1)
          std::get<2>(*this).uniform_refine(spline_t::degree(2)-1-c, 2);
        if constexpr (const auto c = std::get<3>(std::make_tuple(Cs...));
                      c != spline_t::degree(3)-1)
          std::get<3>(*this).uniform_refine(spline_t::degree(3)-1-c, 3);
      } else
        static_assert(sizeof...(Cs) == 0, "Dimensions mismatch");
    }
    /// @}
  };

  TUPLE_WRAPPER(S4);

  /// @brief Taylor-Hood like function space
  /// \f$ S_{p+1}^{p-1} \otimes S_{p}^{p-1} \f$
  template<typename spline_t>
  class TH1
    : public FunctionSpace<S1<typename spline_t::template
                              derived_self_type_t<typename spline_t::value_type,
                                                  spline_t::geoDim(),
                                                  spline_t::degree(0)+1>>,
                           S1<typename spline_t::template
                              derived_self_type_t<typename spline_t::value_type,
                                                  spline_t::geoDim(),
                                                  spline_t::degree(0)>>>
  {
  public:
    /// @brief Base type
    using Base = FunctionSpace<S1<typename spline_t::template
                                  derived_self_type_t<typename spline_t::value_type,
                                                      spline_t::geoDim(),
                                                      spline_t::degree(0)+1>>,
                               S1<typename spline_t::template
                                  derived_self_type_t<typename spline_t::value_type,
                                                      spline_t::geoDim(),
                                                      spline_t::degree(0)>>>;

    /// @brief Constructor
    /// @{
    TH1() = default;
    TH1(TH1&&) = default;
    TH1(const TH1&) = default;

    TH1(const std::array<int64_t, spline_t::parDim()>& ncoeffs,
        enum init init = init::zeros,
        Options<typename spline_t::value_type> options = iganet::Options<typename spline_t::value_type>{})
      : Base(ncoeffs, ncoeffs, init, options)
    {
      std::get<0>(*this).uniform_refine();
    }
    
    TH1(const std::array<std::vector<typename spline_t::value_type>,
        spline_t::parDim()>& kv,
        enum init init = init::zeros,
        Options<typename spline_t::value_type> options = iganet::Options<typename spline_t::value_type>{})
      : Base(kv, kv, init, options)
    {
      std::get<0>(*this).uniform_refine();
    }
    /// @}
  };

  TUPLE_WRAPPER(TH1);

  /// @brief Alias for Taylor-Hood like function space \f$
  /// S_{p+1,p+1}^{p-1,p-1} \otimes
  /// S_{p+1,p+1}^{p-1,p-1} \otimes
  /// S_{p,p}^{p-1,p-1} \f$
  template<typename spline_t>
  class TH2
    : public FunctionSpace<S2<typename spline_t::template
                              derived_self_type_t<typename spline_t::value_type,
                                                  spline_t::geoDim(),
                                                  spline_t::degree(0)+1,
                                                  spline_t::degree(1)+1>>,
                           S2<typename spline_t::template
                              derived_self_type_t<typename spline_t::value_type,
                                                  spline_t::geoDim(),
                                                  spline_t::degree(0)+1,
                                                  spline_t::degree(1)+1>>,
                           S2<typename spline_t::template
                              derived_self_type_t<typename spline_t::value_type,
                                                  spline_t::geoDim(),
                                                  spline_t::degree(0),
                                                  spline_t::degree(1)>>>
  {
  public:
    /// @brief Base type
    using Base = FunctionSpace<S2<typename spline_t::template
                                  derived_self_type_t<typename spline_t::value_type,
                                                      spline_t::geoDim(),
                                                      spline_t::degree(0)+1,
                                                      spline_t::degree(1)+1>>,
                               S2<typename spline_t::template
                                  derived_self_type_t<typename spline_t::value_type,
                                                      spline_t::geoDim(),
                                                      spline_t::degree(0)+1,
                                                      spline_t::degree(1)+1>>,
                               S2<typename spline_t::template
                                  derived_self_type_t<typename spline_t::value_type,
                                                      spline_t::geoDim(),
                                                      spline_t::degree(0),
                                                      spline_t::degree(1)>>>;

    /// @brief Constructor
    /// @{
    TH2() = default;
    TH2(TH2&&) = default;
    TH2(const TH2&) = default;

    TH2(const std::array<int64_t, spline_t::parDim()>& ncoeffs,
        enum init init = init::zeros,
        Options<typename spline_t::value_type> options = iganet::Options<typename spline_t::value_type>{})
      : Base(ncoeffs, ncoeffs, ncoeffs, init, options)
    {
      std::get<0>(*this).uniform_refine();
      std::get<1>(*this).uniform_refine();
    }

    TH2(const std::array<std::vector<typename spline_t::value_type>,
        spline_t::parDim()>& kv,
        enum init init = init::zeros,
        Options<typename spline_t::value_type> options = iganet::Options<typename spline_t::value_type>{})
      : Base(kv, kv, kv, init, options)
    {
      std::get<0>(*this).uniform_refine();
      std::get<1>(*this).uniform_refine();
    }
    /// @}
  };

  TUPLE_WRAPPER(TH2);

  /// @brief Alias for Taylor-Hood like function space \f$
  /// S_{p+1,p+1,p+1}^{p-1,p-1,p-1} \otimes
  /// S_{p+1,p+1,p+1}^{p-1,p-1,p-1} \otimes
  /// S_{p+1,p+1,p+1}^{p-1,p-1,p-1} \otimes
  /// S_{p,p,p}^{p-1,p-1,p-1} \f$
  template<typename spline_t>
  class TH3
    : public FunctionSpace<S3<typename spline_t::template
                              derived_self_type_t<typename spline_t::value_type,
                                                  spline_t::geoDim(),
                                                  spline_t::degree(0)+1,
                                                  spline_t::degree(1)+1,
                                                  spline_t::degree(2)+1>>,
                           S3<typename spline_t::template
                              derived_self_type_t<typename spline_t::value_type,
                                                  spline_t::geoDim(),
                                                  spline_t::degree(0)+1,
                                                  spline_t::degree(1)+1,
                                                  spline_t::degree(2)+1>>,
                           S3<typename spline_t::template
                              derived_self_type_t<typename spline_t::value_type,
                                                  spline_t::geoDim(),
                                                  spline_t::degree(0)+1,
                                                  spline_t::degree(1)+1,
                                                  spline_t::degree(2)+1>>,
                           S3<typename spline_t::template
                              derived_self_type_t<typename spline_t::value_type,
                                                  spline_t::geoDim(),
                                                  spline_t::degree(0),
                                                  spline_t::degree(1),
                                                  spline_t::degree(2)>>>
  {
  public:
    /// @brief Base type
    using Base = FunctionSpace<S3<typename spline_t::template
                                  derived_self_type_t<typename spline_t::value_type,
                                                      spline_t::geoDim(),
                                                      spline_t::degree(0)+1,
                                                      spline_t::degree(1)+1,
                                                      spline_t::degree(2)+1>>,
                               S3<typename spline_t::template
                                  derived_self_type_t<typename spline_t::value_type,
                                                      spline_t::geoDim(),
                                                      spline_t::degree(0)+1,
                                                      spline_t::degree(1)+1,
                                                      spline_t::degree(2)+1>>,
                               S3<typename spline_t::template
                                  derived_self_type_t<typename spline_t::value_type,
                                                      spline_t::geoDim(),
                                                      spline_t::degree(0)+1,
                                                      spline_t::degree(1)+1,
                                                      spline_t::degree(2)+1>>,
                               S3<typename spline_t::template
                                  derived_self_type_t<typename spline_t::value_type,
                                                      spline_t::geoDim(),
                                                      spline_t::degree(0),
                                                      spline_t::degree(1),
                                                      spline_t::degree(2)>>>;

    /// @brief Constructor
    /// @{
    TH3() = default;
    TH3(TH3&&) = default;
    TH3(const TH3&) = default;

    TH3(const std::array<int64_t, spline_t::parDim()>& ncoeffs,
        enum init init = init::zeros,
        Options<typename spline_t::value_type> options = iganet::Options<typename spline_t::value_type>{})
      : Base(ncoeffs, ncoeffs, ncoeffs, ncoeffs, init, options)
    {
      std::get<0>(*this).uniform_refine();
      std::get<1>(*this).uniform_refine();
      std::get<2>(*this).uniform_refine();
    }

    TH3(const std::array<std::vector<typename spline_t::value_type>,
        spline_t::parDim()>& kv,
        enum init init = init::zeros,
        Options<typename spline_t::value_type> options = iganet::Options<typename spline_t::value_type>{})
      : Base(kv, kv, kv, kv, init, options)
    {
      std::get<0>(*this).uniform_refine();
      std::get<1>(*this).uniform_refine();
      std::get<2>(*this).uniform_refine();
    }
    /// @}
  };

  TUPLE_WRAPPER(TH3);

  /// @brief Alias for Taylor-Hood like function space \f$
  /// S_{p+1,p+1,p+1,p+1}^{p-1,p-1,p-1,p-1} \otimes
  /// S_{p+1,p+1,p+1,p+1}^{p-1,p-1,p-1,p-1} \otimes
  /// S_{p+1,p+1,p+1,p+1}^{p-1,p-1,p-1,p-1} \otimes
  /// S_{p+1,p+1,p+1,p+1}^{p-1,p-1,p-1,p-1} \otimes
  /// S_{p,p,p,p}^{p-1,p-1,p-1,p-1} \f$
  template<typename spline_t>
  class TH4
    : public FunctionSpace<S4<typename spline_t::template
                              derived_self_type_t<typename spline_t::value_type,
                                                  spline_t::geoDim(),
                                                  spline_t::degree(0)+1,
                                                  spline_t::degree(1)+1,
                                                  spline_t::degree(2)+1,
                                                  spline_t::degree(3)+1>>,
                           S4<typename spline_t::template
                              derived_self_type_t<typename spline_t::value_type,
                                                  spline_t::geoDim(),
                                                  spline_t::degree(0)+1,
                                                  spline_t::degree(1)+1,
                                                  spline_t::degree(2)+1,
                                                  spline_t::degree(3)+1>>,
                           S4<typename spline_t::template
                              derived_self_type_t<typename spline_t::value_type,
                                                  spline_t::geoDim(),
                                                  spline_t::degree(0)+1,
                                                  spline_t::degree(1)+1,
                                                  spline_t::degree(2)+1,
                                                  spline_t::degree(3)+1>>,
                           S4<typename spline_t::template
                              derived_self_type_t<typename spline_t::value_type,
                                                  spline_t::geoDim(),
                                                  spline_t::degree(0)+1,
                                                  spline_t::degree(1)+1,
                                                  spline_t::degree(2)+1,
                                                  spline_t::degree(3)+1>>,
                           S4<typename spline_t::template
                              derived_self_type_t<typename spline_t::value_type,
                                                  spline_t::geoDim(),
                                                  spline_t::degree(0),
                                                  spline_t::degree(1),
                                                  spline_t::degree(2),
                                                  spline_t::degree(3)>>>
  {
  public:
    /// @brief Base type
    using Base = FunctionSpace<S4<typename spline_t::template
                                  derived_self_type_t<typename spline_t::value_type,
                                                      spline_t::geoDim(),
                                                      spline_t::degree(0)+1,
                                                      spline_t::degree(1)+1,
                                                      spline_t::degree(2)+1,
                                                      spline_t::degree(3)+1>>,
                               S4<typename spline_t::template
                                  derived_self_type_t<typename spline_t::value_type,
                                                      spline_t::geoDim(),
                                                      spline_t::degree(0)+1,
                                                      spline_t::degree(1)+1,
                                                      spline_t::degree(2)+1,
                                                      spline_t::degree(3)+1>>,
                               S4<typename spline_t::template
                                  derived_self_type_t<typename spline_t::value_type,
                                                      spline_t::geoDim(),
                                                      spline_t::degree(0)+1,
                                                      spline_t::degree(1)+1,
                                                      spline_t::degree(2)+1,
                                                      spline_t::degree(3)+1>>,
                               S4<typename spline_t::template
                                  derived_self_type_t<typename spline_t::value_type,
                                                      spline_t::geoDim(),
                                                      spline_t::degree(0)+1,
                                                      spline_t::degree(1)+1,
                                                      spline_t::degree(2)+1,
                                                      spline_t::degree(3)+1>>,
                               S4<typename spline_t::template
                                  derived_self_type_t<typename spline_t::value_type,
                                                      spline_t::geoDim(),
                                                      spline_t::degree(0),
                                                      spline_t::degree(1),
                                                      spline_t::degree(2),
                                                      spline_t::degree(3)>>>;

    /// @brief Constructor
    /// @{
    TH4() = default;
    TH4(TH4&&) = default;
    TH4(const TH4&) = default;

    TH4(const std::array<int64_t, spline_t::parDim()>& ncoeffs,
        enum init init = init::zeros,
        Options<typename spline_t::value_type> options = iganet::Options<typename spline_t::value_type>{})
      : Base(ncoeffs, ncoeffs, ncoeffs, ncoeffs, ncoeffs, init, options)
    {
      std::get<0>(*this).uniform_refine();
      std::get<1>(*this).uniform_refine();
      std::get<2>(*this).uniform_refine();
      std::get<3>(*this).uniform_refine();
    }

    TH4(const std::array<std::vector<typename spline_t::value_type>,
        spline_t::parDim()>& kv,
        enum init init = init::zeros,
        Options<typename spline_t::value_type> options = iganet::Options<typename spline_t::value_type>{})
      : Base(kv, kv, kv, kv, kv, init, options)
    {
      std::get<0>(*this).uniform_refine();
      std::get<1>(*this).uniform_refine();
      std::get<2>(*this).uniform_refine();
      std::get<3>(*this).uniform_refine();
    }
    /// @}
  };

  TUPLE_WRAPPER(TH4);

  /// @brief Alias for Nedelec like function space
  /// \f$ S_{p+1}^{p-1} \otimes S_{p}^{p-1} \f$
  template<typename spline_t>
  class NE1
    : public FunctionSpace<S1<typename spline_t::template
                              derived_self_type_t<typename spline_t::value_type,
                                                  spline_t::geoDim(),
                                                  spline_t::degree(0)+1>>,
                           S1<typename spline_t::template
                              derived_self_type_t<typename spline_t::value_type,
                                                  spline_t::geoDim(),
                                                  spline_t::degree(0)>>>
  {
  public:
    /// @brief Base type
    using Base = FunctionSpace<S1<typename spline_t::template
                                  derived_self_type_t<typename spline_t::value_type,
                                                      spline_t::geoDim(),
                                                      spline_t::degree(0)+1>>,
                               S1<typename spline_t::template
                                  derived_self_type_t<typename spline_t::value_type,
                                                      spline_t::geoDim(),
                                                      spline_t::degree(0)>>>;

    /// @brief Constructor
    /// @{
    NE1() = default;
    NE1(NE1&&) = default;
    NE1(const NE1&) = default;

    NE1(const std::array<int64_t, spline_t::parDim()>& ncoeffs,
        enum init init = init::zeros,
        Options<typename spline_t::value_type> options = iganet::Options<typename spline_t::value_type>{})
      : Base(ncoeffs, ncoeffs, init, options)
    {
      std::get<0>(*this).uniform_refine();
    }

    NE1(const std::array<std::vector<typename spline_t::value_type>,
        spline_t::parDim()>& kv,
        enum init init = init::zeros,
        Options<typename spline_t::value_type> options = iganet::Options<typename spline_t::value_type>{})
      : Base(kv, kv, init, options)
    {
      std::get<0>(*this).uniform_refine();
    }
    /// @}
  };

  TUPLE_WRAPPER(NE1);

  /// @brief Alias for Nedelec like function space \f$
  /// S_{p+1,p+1}^{p,p-1} \otimes
  /// S_{p+1,p+1}^{p-1,p} \otimes
  /// S_{p,p}^{p-1,p-1} \f$
  template<typename spline_t>
  class NE2
    : public FunctionSpace<S2<typename spline_t::template
                              derived_self_type_t<typename spline_t::value_type,
                                                  spline_t::geoDim(),
                                                  spline_t::degree(0)+1,
                                                  spline_t::degree(1)+1>>,
                           S2<typename spline_t::template
                              derived_self_type_t<typename spline_t::value_type,
                                                  spline_t::geoDim(),
                                                  spline_t::degree(0)+1,
                                                  spline_t::degree(1)+1>>,
                           S2<typename spline_t::template
                              derived_self_type_t<typename spline_t::value_type,
                                                  spline_t::geoDim(),
                                                  spline_t::degree(0),
                                                  spline_t::degree(1)>>>
  {
  public:
    /// @brief Base type
    using Base = FunctionSpace<S2<typename spline_t::template
                                  derived_self_type_t<typename spline_t::value_type,
                                                      spline_t::geoDim(),
                                                      spline_t::degree(0)+1,
                                                      spline_t::degree(1)+1>>,
                               S2<typename spline_t::template
                                  derived_self_type_t<typename spline_t::value_type,
                                                      spline_t::geoDim(),
                                                      spline_t::degree(0)+1,
                                                      spline_t::degree(1)+1>>,
                               S2<typename spline_t::template
                                  derived_self_type_t<typename spline_t::value_type,
                                                      spline_t::geoDim(),
                                                      spline_t::degree(0),
                                                      spline_t::degree(1)>>>;

    /// @brief Constructor
    /// @{
    NE2() = default;
    NE2(NE2&&) = default;
    NE2(const NE2&) = default;

    NE2(const std::array<int64_t, spline_t::parDim()>& ncoeffs,
        enum init init = init::zeros,
        Options<typename spline_t::value_type> options = iganet::Options<typename spline_t::value_type>{})
      : Base(ncoeffs, ncoeffs, ncoeffs, init, options)
    {
      std::get<0>(*this).uniform_refine(1, 1);
      std::get<1>(*this).uniform_refine(1, 0);
    }

    NE2(const std::array<std::vector<typename spline_t::value_type>,
        spline_t::parDim()>& kv,
        enum init init = init::zeros,
        Options<typename spline_t::value_type> options = iganet::Options<typename spline_t::value_type>{})
      : Base(kv, kv, kv, init, options)
    {
      std::get<0>(*this).uniform_refine(1, 1);
      std::get<1>(*this).uniform_refine(1, 0);
    }
    /// @}
  };

  TUPLE_WRAPPER(NE2);

  /// @brief Alias for Nedelec like function space \f$
  /// S_{p+1,p+1,p+1}^{p,p-1,p-1} \otimes
  /// S_{p+1,p+1,p+1}^{p-1,p,p-1} \otimes
  /// S_{p+1,p+1,p+1}^{p-1,p-1,p} \otimes
  /// S_{p,p,p}^{p-1,p-1,p-1} \f$
  template<typename spline_t>
  class NE3
    : public FunctionSpace<S3<typename spline_t::template
                              derived_self_type_t<typename spline_t::value_type,
                                                  spline_t::geoDim(),
                                                  spline_t::degree(0)+1,
                                                  spline_t::degree(1)+1,
                                                  spline_t::degree(2)+1>>,
                           S3<typename spline_t::template
                              derived_self_type_t<typename spline_t::value_type,
                                                  spline_t::geoDim(),
                                                  spline_t::degree(0)+1,
                                                  spline_t::degree(1)+1,
                                                  spline_t::degree(2)+1>>,
                           S3<typename spline_t::template
                              derived_self_type_t<typename spline_t::value_type,
                                                  spline_t::geoDim(),
                                                  spline_t::degree(0)+1,
                                                  spline_t::degree(1)+1,
                                                  spline_t::degree(2)+1>>,
                           S3<typename spline_t::template
                              derived_self_type_t<typename spline_t::value_type,
                                                  spline_t::geoDim(),
                                                  spline_t::degree(0),
                                                  spline_t::degree(1),
                                                  spline_t::degree(2)>>>
  {
  public:
    /// @brief Base type
    using Base = FunctionSpace<S3<typename spline_t::template
                                  derived_self_type_t<typename spline_t::value_type,
                                                      spline_t::geoDim(),
                                                      spline_t::degree(0)+1,
                                                      spline_t::degree(1)+1,
                                                      spline_t::degree(2)+1>>,
                               S3<typename spline_t::template
                                  derived_self_type_t<typename spline_t::value_type,
                                                      spline_t::geoDim(),
                                                      spline_t::degree(0)+1,
                                                      spline_t::degree(1)+1,
                                                      spline_t::degree(2)+1>>,
                               S3<typename spline_t::template
                                  derived_self_type_t<typename spline_t::value_type,
                                                      spline_t::geoDim(),
                                                      spline_t::degree(0)+1,
                                                      spline_t::degree(1)+1,
                                                      spline_t::degree(2)+1>>,
                               S3<typename spline_t::template
                                  derived_self_type_t<typename spline_t::value_type,
                                                      spline_t::geoDim(),
                                                      spline_t::degree(0),
                                                      spline_t::degree(1),
                                                      spline_t::degree(2)>>>;

    /// @brief Constructor
    /// @{
    NE3() = default;
    NE3(NE3&&) = default;
    NE3(const NE3&) = default;

    NE3(const std::array<int64_t, spline_t::parDim()>& ncoeffs,
        enum init init = init::zeros,
        Options<typename spline_t::value_type> options = iganet::Options<typename spline_t::value_type>{})
      : Base(ncoeffs, ncoeffs, ncoeffs, ncoeffs, init, options)
    {
      std::get<0>(*this).uniform_refine(1, 1).uniform_refine(1, 2);
      std::get<1>(*this).uniform_refine(1, 0).uniform_refine(1, 2);
      std::get<2>(*this).uniform_refine(1, 0).uniform_refine(1, 1);
    }

    NE3(const std::array<std::vector<typename spline_t::value_type>,
        spline_t::parDim()>& kv,
        enum init init = init::zeros,
        Options<typename spline_t::value_type> options = iganet::Options<typename spline_t::value_type>{})
      : Base(kv, kv, kv, kv, init, options)
    {
      std::get<0>(*this).uniform_refine(1, 1).uniform_refine(1, 2);
      std::get<1>(*this).uniform_refine(1, 0).uniform_refine(1, 2);
      std::get<2>(*this).uniform_refine(1, 0).uniform_refine(1, 1);
    }
    /// @}
  };

  TUPLE_WRAPPER(NE3);

  /// @brief Alias for Nedelec like function space \f$
  /// S_{p+1,p+1,p+1,p+1}^{p,p-1,p-1,p-1} \otimes
  /// S_{p+1,p+1,p+1,p+1}^{p-1,p,p-1,p-1} \otimes
  /// S_{p+1,p+1,p+1,p+1}^{p-1,p-1,p,p-1} \otimes
  /// S_{p+1,p+1,p+1,p+1}^{p-1,p-1,p-1,p} \otimes
  /// S_{p,p,p,p}^{p-1,p-1,p-1,p-1} \f$
  template<typename spline_t>
  class NE4
    : public FunctionSpace<S4<typename spline_t::template
                              derived_self_type_t<typename spline_t::value_type,
                                                  spline_t::geoDim(),
                                                  spline_t::degree(0)+1,
                                                  spline_t::degree(1)+1,
                                                  spline_t::degree(2)+1,
                                                  spline_t::degree(3)+1>>,
                           S4<typename spline_t::template
                              derived_self_type_t<typename spline_t::value_type,
                                                  spline_t::geoDim(),
                                                  spline_t::degree(0)+1,
                                                  spline_t::degree(1)+1,
                                                  spline_t::degree(2)+1,
                                                  spline_t::degree(3)+1>>,
                           S4<typename spline_t::template
                              derived_self_type_t<typename spline_t::value_type,
                                                  spline_t::geoDim(),
                                                  spline_t::degree(0)+1,
                                                  spline_t::degree(1)+1,
                                                  spline_t::degree(2)+1,
                                                  spline_t::degree(3)+1>>,
                           S4<typename spline_t::template
                              derived_self_type_t<typename spline_t::value_type,
                                                  spline_t::geoDim(),
                                                  spline_t::degree(0)+1,
                                                  spline_t::degree(1)+1,
                                                  spline_t::degree(2)+1,
                                                  spline_t::degree(3)+1>>,
                           S4<typename spline_t::template
                              derived_self_type_t<typename spline_t::value_type,
                                                  spline_t::geoDim(),
                                                  spline_t::degree(0),
                                                  spline_t::degree(1),
                                                  spline_t::degree(2),
                                                  spline_t::degree(3)>>>
  {
  public:
    /// @brief Base type
    using Base = FunctionSpace<S4<typename spline_t::template
                                  derived_self_type_t<typename spline_t::value_type,
                                                      spline_t::geoDim(),
                                                      spline_t::degree(0)+1,
                                                      spline_t::degree(1)+1,
                                                      spline_t::degree(2)+1,
                                                      spline_t::degree(3)+1>>,
                               S4<typename spline_t::template
                                  derived_self_type_t<typename spline_t::value_type,
                                                      spline_t::geoDim(),
                                                      spline_t::degree(0)+1,
                                                      spline_t::degree(1)+1,
                                                      spline_t::degree(2)+1,
                                                      spline_t::degree(3)+1>>,
                               S4<typename spline_t::template
                                  derived_self_type_t<typename spline_t::value_type,
                                                      spline_t::geoDim(),
                                                      spline_t::degree(0)+1,
                                                      spline_t::degree(1)+1,
                                                      spline_t::degree(2)+1,
                                                      spline_t::degree(3)+1>>,
                               S4<typename spline_t::template
                                  derived_self_type_t<typename spline_t::value_type,
                                                      spline_t::geoDim(),
                                                      spline_t::degree(0)+1,
                                                      spline_t::degree(1)+1,
                                                      spline_t::degree(2)+1,
                                                      spline_t::degree(3)+1>>,
                               S4<typename spline_t::template
                                  derived_self_type_t<typename spline_t::value_type,
                                                      spline_t::geoDim(),
                                                      spline_t::degree(0),
                                                      spline_t::degree(1),
                                                      spline_t::degree(2),
                                                      spline_t::degree(3)>>>;

    /// @brief Constructor
    /// @{
    NE4() = default;
    NE4(NE4&&) = default;
    NE4(const NE4&) = default;

    NE4(const std::array<int64_t, spline_t::parDim()>& ncoeffs,
        enum init init = init::zeros,
        Options<typename spline_t::value_type> options = iganet::Options<typename spline_t::value_type>{})
      : Base(ncoeffs, ncoeffs, ncoeffs, ncoeffs, ncoeffs, init, options)
    {
      std::get<0>(*this).uniform_refine(1, 1).uniform_refine(1, 2).uniform_refine(1, 3);
      std::get<1>(*this).uniform_refine(1, 0).uniform_refine(1, 2).uniform_refine(1, 3);
      std::get<2>(*this).uniform_refine(1, 0).uniform_refine(1, 1).uniform_refine(1, 3);
      std::get<3>(*this).uniform_refine(1, 0).uniform_refine(1, 1).uniform_refine(1, 2);
    }

    NE4(const std::array<std::vector<typename spline_t::value_type>,
        spline_t::parDim()>& kv,
        enum init init = init::zeros,
        Options<typename spline_t::value_type> options = iganet::Options<typename spline_t::value_type>{})
      : Base(kv, kv, kv, kv, kv, init, options)
    {
      std::get<0>(*this).uniform_refine(1, 1).uniform_refine(1, 2).uniform_refine(1, 3);
      std::get<1>(*this).uniform_refine(1, 0).uniform_refine(1, 2).uniform_refine(1, 3);
      std::get<2>(*this).uniform_refine(1, 0).uniform_refine(1, 1).uniform_refine(1, 3);
      std::get<3>(*this).uniform_refine(1, 0).uniform_refine(1, 1).uniform_refine(1, 2);
    }
    /// @}
  };

  TUPLE_WRAPPER(NE4);

  /// @brief Alias for Raviart-Thomas like function space
  /// \f$ S_{p+1}^{p-1} \otimes S_{p}^{p-1} \f$
  template<typename spline_t>
  class RT1
    : public FunctionSpace<S1<typename spline_t::template
                              derived_self_type_t<typename spline_t::value_type,
                                                  spline_t::geoDim(),
                                                  spline_t::degree(0)+1>>,
                           S1<typename spline_t::template
                              derived_self_type_t<typename spline_t::value_type,
                                                  spline_t::geoDim(),
                                                  spline_t::degree(0)>>>
  {
  public:
    /// @brief Base type
    using Base = FunctionSpace<S1<typename spline_t::template
                                  derived_self_type_t<typename spline_t::value_type,
                                                      spline_t::geoDim(),
                                                      spline_t::degree(0)+1>>,
                               S1<typename spline_t::template
                                  derived_self_type_t<typename spline_t::value_type,
                                                      spline_t::geoDim(),
                                                      spline_t::degree(0)>>>;

    /// @brief Constructor
    /// @{
    RT1() = default;
    RT1(RT1&&) = default;
    RT1(const RT1&) = default;

    RT1(const std::array<int64_t, spline_t::parDim()>& ncoeffs,
        enum init init = init::zeros,
        Options<typename spline_t::value_type> options = iganet::Options<typename spline_t::value_type>{})
      : Base(ncoeffs, ncoeffs, init, options)
    {
      std::get<0>(*this).uniform_refine();
    }

    RT1(const std::array<std::vector<typename spline_t::value_type>,
        spline_t::parDim()>& kv,
        enum init init = init::zeros,
        Options<typename spline_t::value_type> options = iganet::Options<typename spline_t::value_type>{})
      : Base(kv, kv, init, options)
    {
      std::get<0>(*this).uniform_refine();
    }
    /// @}
  };

  TUPLE_WRAPPER(RT1);

  /// @brief Alias for Raviart-Thomas like function space \f$
  /// S_{p+1,p}^{p,p-1} \otimes
  /// S_{p,p+1}^{p-1,p} \otimes
  /// S_{p,p}^{p-1,p-1} \f$
  template<typename spline_t>
  class RT2
    : public FunctionSpace<S2<typename spline_t::template
                              derived_self_type_t<typename spline_t::value_type,
                                                  spline_t::geoDim(),
                                                  spline_t::degree(0)+1,
                                                  spline_t::degree(1)>>,
                           S2<typename spline_t::template
                              derived_self_type_t<typename spline_t::value_type,
                                                  spline_t::geoDim(),
                                                  spline_t::degree(0),
                                                  spline_t::degree(1)+1>>,
                           S2<typename spline_t::template
                              derived_self_type_t<typename spline_t::value_type,
                                                  spline_t::geoDim(),
                                                  spline_t::degree(0),
                                                  spline_t::degree(1)>>>
  {
  public:
    using Base = FunctionSpace<S2<typename spline_t::template
                                  derived_self_type_t<typename spline_t::value_type,
                                                      spline_t::geoDim(),
                                                      spline_t::degree(0)+1,
                                                      spline_t::degree(1)>>,
                               S2<typename spline_t::template
                                  derived_self_type_t<typename spline_t::value_type,
                                                      spline_t::geoDim(),
                                                      spline_t::degree(0),
                                                      spline_t::degree(1)+1>>,
                               S2<typename spline_t::template
                                  derived_self_type_t<typename spline_t::value_type,
                                                      spline_t::geoDim(),
                                                      spline_t::degree(0),
                                                      spline_t::degree(1)>>>;

    /// @brief Constructor
    /// @{
    RT2() = default;
    RT2(RT2&&) = default;
    RT2(const RT2&) = default;

    RT2(const std::array<int64_t, spline_t::parDim()>& ncoeffs,
        enum init init = init::zeros,
        Options<typename spline_t::value_type> options = iganet::Options<typename spline_t::value_type>{})
      : Base(ncoeffs, ncoeffs, ncoeffs, init, options)
    {
    }

    RT2(const std::array<std::vector<typename spline_t::value_type>,
        spline_t::parDim()>& kv,
        enum init init = init::zeros,
        Options<typename spline_t::value_type> options = iganet::Options<typename spline_t::value_type>{})
      : Base(kv, kv, kv, init, options)
    {
    }
    /// @}
  };

  TUPLE_WRAPPER(RT2);

  /// @brief Alias for Raviart-Thomas like function space \f$
  /// S_{p+1,p,p}^{p,p-1,p-1} \otimes
  /// S_{p,p+1,p}^{p-1,p,p-1} \otimes
  /// S_{p,p,p+1}^{p-1,p-1,p} \otimes
  /// S_{p,p,p}^{p-1,p-1,p-1} \f$
  template<typename spline_t>
  class RT3
    : public FunctionSpace<S3<typename spline_t::template
                              derived_self_type_t<typename spline_t::value_type,
                                                  spline_t::geoDim(),
                                                  spline_t::degree(0)+1,
                                                  spline_t::degree(1),
                                                  spline_t::degree(2)>>,
                           S3<typename spline_t::template
                              derived_self_type_t<typename spline_t::value_type,
                                                  spline_t::geoDim(),
                                                  spline_t::degree(0),
                                                  spline_t::degree(1)+1,
                                                  spline_t::degree(2)>>,
                           S3<typename spline_t::template
                              derived_self_type_t<typename spline_t::value_type,
                                                  spline_t::geoDim(),
                                                  spline_t::degree(0),
                                                  spline_t::degree(1),
                                                  spline_t::degree(2)+1>>,
                           S3<typename spline_t::template
                              derived_self_type_t<typename spline_t::value_type,
                                                  spline_t::geoDim(),
                                                  spline_t::degree(0),
                                                  spline_t::degree(1),
                                                  spline_t::degree(2)>>>
  {
  public:
    /// @brief Base type
    using Base = FunctionSpace<S3<typename spline_t::template
                                  derived_self_type_t<typename spline_t::value_type,
                                                      spline_t::geoDim(),
                                                      spline_t::degree(0)+1,
                                                      spline_t::degree(1),
                                                      spline_t::degree(2)>>,
                               S3<typename spline_t::template
                                  derived_self_type_t<typename spline_t::value_type,
                                                      spline_t::geoDim(),
                                                      spline_t::degree(0),
                                                      spline_t::degree(1)+1,
                                                      spline_t::degree(2)>>,
                               S3<typename spline_t::template
                                  derived_self_type_t<typename spline_t::value_type,
                                                      spline_t::geoDim(),
                                                      spline_t::degree(0),
                                                      spline_t::degree(1),
                                                      spline_t::degree(2)+1>>,
                               S3<typename spline_t::template
                                  derived_self_type_t<typename spline_t::value_type,
                                                      spline_t::geoDim(),
                                                      spline_t::degree(0),
                                                      spline_t::degree(1),
                                                      spline_t::degree(2)>>>;

    /// @brief Constructor
    /// @{
    RT3() = default;
    RT3(RT3&&) = default;
    RT3(const RT3&) = default;

    RT3(const std::array<int64_t, spline_t::parDim()>& ncoeffs,
        enum init init = init::zeros,
        Options<typename spline_t::value_type> options = iganet::Options<typename spline_t::value_type>{})
      : Base(ncoeffs, ncoeffs, ncoeffs, ncoeffs, init, options)
    {
    }

    RT3(const std::array<std::vector<typename spline_t::value_type>,
        spline_t::parDim()>& kv,
        enum init init = init::zeros,
        Options<typename spline_t::value_type> options = iganet::Options<typename spline_t::value_type>{})
      : Base(kv, kv, kv, kv, init, options)
    {
    }
    /// @}
  };

  TUPLE_WRAPPER(RT3);

  /// @brief Alias for Raviart-Thomas like function space \f$
  /// S_{p+1,p,p,p}^{p,p-1,p-1,p-1} \otimes
  /// S_{p,p+1,p,p}^{p-1,p,p-1,p-1} \otimes
  /// S_{p,p,p+1,p}^{p-1,p-1,p,p-1} \otimes
  /// S_{p,p,p,p+1}^{p-1,p-1,p-1,p} \otimes
  /// S_{p,p,p,p}^{p-1,p-1,p-1,p-1} \f$
  template<typename spline_t>
  class RT4
    : public FunctionSpace<S4<typename spline_t::template
                              derived_self_type_t<typename spline_t::value_type,
                                                  spline_t::geoDim(),
                                                  spline_t::degree(0)+1,
                                                  spline_t::degree(1),
                                                  spline_t::degree(2),
                                                  spline_t::degree(3)>>,
                           S4<typename spline_t::template
                              derived_self_type_t<typename spline_t::value_type,
                                                  spline_t::geoDim(),
                                                  spline_t::degree(0),
                                                  spline_t::degree(1)+1,
                                                  spline_t::degree(2),
                                                  spline_t::degree(3)>>,
                           S4<typename spline_t::template
                              derived_self_type_t<typename spline_t::value_type,
                                                  spline_t::geoDim(),
                                                  spline_t::degree(0),
                                                  spline_t::degree(1),
                                                  spline_t::degree(2)+1,
                                                  spline_t::degree(3)>>,
                           S4<typename spline_t::template
                              derived_self_type_t<typename spline_t::value_type,
                                                  spline_t::geoDim(),
                                                  spline_t::degree(0),
                                                  spline_t::degree(1),
                                                  spline_t::degree(2),
                                                  spline_t::degree(3)+1>>,
                           S4<typename spline_t::template
                              derived_self_type_t<typename spline_t::value_type,
                                                  spline_t::geoDim(),
                                                  spline_t::degree(0),
                                                  spline_t::degree(1),
                                                  spline_t::degree(2),
                                                  spline_t::degree(3)>>>
  {
  public:
    /// @brief Base type
    using Base = FunctionSpace<S4<typename spline_t::template
                                  derived_self_type_t<typename spline_t::value_type,
                                                      spline_t::geoDim(),
                                                      spline_t::degree(0)+1,
                                                      spline_t::degree(1),
                                                      spline_t::degree(2),
                                                      spline_t::degree(3)>>,
                               S4<typename spline_t::template
                                  derived_self_type_t<typename spline_t::value_type,
                                                      spline_t::geoDim(),
                                                      spline_t::degree(0),
                                                      spline_t::degree(1)+1,
                                                      spline_t::degree(2),
                                                      spline_t::degree(3)>>,
                               S4<typename spline_t::template
                                  derived_self_type_t<typename spline_t::value_type,
                                                      spline_t::geoDim(),
                                                      spline_t::degree(0),
                                                      spline_t::degree(1),
                                                      spline_t::degree(2)+1,
                                                      spline_t::degree(3)>>,
                               S4<typename spline_t::template
                                  derived_self_type_t<typename spline_t::value_type,
                                                      spline_t::geoDim(),
                                                      spline_t::degree(0),
                                                      spline_t::degree(1),
                                                      spline_t::degree(2),
                                                      spline_t::degree(3)+1>>,
                               S4<typename spline_t::template
                                  derived_self_type_t<typename spline_t::value_type,
                                                      spline_t::geoDim(),
                                                      spline_t::degree(0),
                                                      spline_t::degree(1),
                                                      spline_t::degree(2),
                                                      spline_t::degree(3)>>>;

    /// @brief Constructor
    /// @{
    RT4() = default;
    RT4(RT4&&) = default;
    RT4(const RT4&) = default;

    RT4(const std::array<int64_t, spline_t::parDim()>& ncoeffs,
        enum init init = init::zeros,
        Options<typename spline_t::value_type> options = iganet::Options<typename spline_t::value_type>{})
      : Base(ncoeffs, ncoeffs, ncoeffs, ncoeffs, ncoeffs, init, options)
    {
    }

    RT4(const std::array<std::vector<typename spline_t::value_type>,
        spline_t::parDim()>& kv,
        enum init init = init::zeros,
        Options<typename spline_t::value_type> options = iganet::Options<typename spline_t::value_type>{})
      : Base(kv, kv, kv, kv, kv, init, options)
    {
    }
    /// @}
  };

  TUPLE_WRAPPER(RT4);

} // namespace iganet
