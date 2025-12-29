/**
   @file optimizier.hpp

   @brief Optimizier type traits

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#pragma once

namespace iganet {

/// @brief Concept to identify template parameters that are derived from
/// torch::optim::Optimizer
template <typename T>
concept OptimizerType = std::is_base_of_v<torch::optim::Optimizer, T>;

/// @brief Type trait for the optimizer options type
/// @{
template <typename Optimizer>
  requires OptimizerType<Optimizer>
struct optimizer_options_type;

template <> struct optimizer_options_type<torch::optim::Adagrad> {
  using type = torch::optim::AdagradOptions;
};

template <> struct optimizer_options_type<torch::optim::Adam> {
  using type = torch::optim::AdamOptions;
};

template <> struct optimizer_options_type<torch::optim::AdamW> {
  using type = torch::optim::AdamWOptions;
};

template <> struct optimizer_options_type<torch::optim::LBFGS> {
  using type = torch::optim::LBFGSOptions;
};

template <> struct optimizer_options_type<torch::optim::SGD> {
  using type = torch::optim::SGDOptions;
};

template <> struct optimizer_options_type<torch::optim::RMSprop> {
  using type = torch::optim::RMSpropOptions;
};
/// @}

} // namespace iganet

namespace torch {
namespace optim {

/// Print (as string) a torch::optim::AdagradOptions object
inline std::ostream &operator<<(std::ostream &os,
                                const torch::optim::AdagradOptions &obj) {
  at::optional<std::string> name_ = c10::demangle(typeid(obj).name());

#if defined(_WIN32)
  // Windows adds "struct" or "class" as a prefix.
  if (name_->find("struct ") == 0) {
    name_->erase(name_->begin(), name_->begin() + 7);
  } else if (name_->find("class ") == 0) {
    name_->erase(name_->begin(), name_->begin() + 6);
  }
#endif // defined(_WIN32)

  os << *name_ << "(\nlr = " << obj.lr() << ", lr_decay = " << obj.lr_decay()
     << ", weight_decay = " << obj.weight_decay()
     << ", initial_accumulator_value = " << obj.initial_accumulator_value()
     << ", eps = " << obj.eps() << "\n)";

  return os;
}

/// Print (as string) a torch::optim::AdamOptions object
inline std::ostream &operator<<(std::ostream &os,
                                const torch::optim::AdamOptions &obj) {
  at::optional<std::string> name_ = c10::demangle(typeid(obj).name());

#if defined(_WIN32)
  // Windows adds "struct" or "class" as a prefix.
  if (name_->find("struct ") == 0) {
    name_->erase(name_->begin(), name_->begin() + 7);
  } else if (name_->find("class ") == 0) {
    name_->erase(name_->begin(), name_->begin() + 6);
  }
#endif // defined(_WIN32)

  os << *name_ << "(\nlr = " << obj.lr() << ", betas = ["
     << std::get<0>(obj.betas()) << ", " << std::get<1>(obj.betas()) << "]"
     << ", weight_decay = " << obj.weight_decay() << ", eps = " << obj.eps()
     << ", amsgrad = " << obj.amsgrad() << "\n)";

  return os;
}

/// Print (as string) a torch::optim::AdamWOptions object
inline std::ostream &operator<<(std::ostream &os,
                                const torch::optim::AdamWOptions &obj) {
  at::optional<std::string> name_ = c10::demangle(typeid(obj).name());

#if defined(_WIN32)
  // Windows adds "struct" or "class" as a prefix.
  if (name_->find("struct ") == 0) {
    name_->erase(name_->begin(), name_->begin() + 7);
  } else if (name_->find("class ") == 0) {
    name_->erase(name_->begin(), name_->begin() + 6);
  }
#endif // defined(_WIN32)

  os << *name_ << "(\nlr = " << obj.lr() << ", betas = ["
     << std::get<0>(obj.betas()) << ", " << std::get<1>(obj.betas()) << "]"
     << ", weight_decay = " << obj.weight_decay() << ", eps = " << obj.eps()
     << ", amsgrad = " << obj.amsgrad() << "\n)";

  return os;
}

/// Print (as string) a torch::optim::LBFGSOptions object
inline std::ostream &operator<<(std::ostream &os,
                                const torch::optim::LBFGSOptions &obj) {
  at::optional<std::string> name_ = c10::demangle(typeid(obj).name());

#if defined(_WIN32)
  // Windows adds "struct" or "class" as a prefix.
  if (name_->find("struct ") == 0) {
    name_->erase(name_->begin(), name_->begin() + 7);
  } else if (name_->find("class ") == 0) {
    name_->erase(name_->begin(), name_->begin() + 6);
  }
#endif // defined(_WIN32)

  os << *name_ << "(\nlr = " << obj.lr() << ", max_iter = " << obj.max_iter()
     << ", max_eval = "
     << (obj.max_eval().has_value() ? std::to_string(*obj.max_eval())
                                    : "undefined")
     << ", tolerance_grad = " << obj.tolerance_grad()
     << ", tolerance_change = " << obj.tolerance_change()
     << ", history_size = " << obj.history_size() << ", line_search_fn = "
     << (obj.line_search_fn().has_value() ? *obj.line_search_fn() : "undefined")
     << "\n)";

  return os;
}

/// Print (as string) a torch::optim::RMSpropOptions object
inline std::ostream &operator<<(std::ostream &os,
                                const torch::optim::RMSpropOptions &obj) {
  at::optional<std::string> name_ = c10::demangle(typeid(obj).name());

#if defined(_WIN32)
  // Windows adds "struct" or "class" as a prefix.
  if (name_->find("struct ") == 0) {
    name_->erase(name_->begin(), name_->begin() + 7);
  } else if (name_->find("class ") == 0) {
    name_->erase(name_->begin(), name_->begin() + 6);
  }
#endif // defined(_WIN32)

  os << *name_ << "(\nlr = " << obj.lr() << ", alpha = " << obj.alpha()
     << ", eps = " << obj.eps() << ", weight_decay = " << obj.weight_decay()
     << ", momentum = " << obj.momentum() << ", centered = " << obj.centered()
     << "\n)";

  return os;
}

/// Print (as string) a torch::optim::SGDOptions object
inline std::ostream &operator<<(std::ostream &os,
                                const torch::optim::SGDOptions &obj) {
  at::optional<std::string> name_ = c10::demangle(typeid(obj).name());

#if defined(_WIN32)
  // Windows adds "struct" or "class" as a prefix.
  if (name_->find("struct ") == 0) {
    name_->erase(name_->begin(), name_->begin() + 7);
  } else if (name_->find("class ") == 0) {
    name_->erase(name_->begin(), name_->begin() + 6);
  }
#endif // defined(_WIN32)

  os << *name_ << "(\nlr = " << obj.lr() << ", momentum = " << obj.momentum()
     << ", dampening = " << obj.dampening()
     << ", weight_decay = " << obj.weight_decay()
     << ", nesterov = " << obj.nesterov() << "\n)";

  return os;
}

} // namespace optim
} // namespace torch
