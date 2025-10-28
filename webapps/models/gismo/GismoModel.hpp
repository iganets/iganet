/**
   @file webapps/models/gismo/GismoModel.hpp

   @brief G+Smo model

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#pragma once

#include <iganet.h>
#include <model.hpp>

namespace iganet {

namespace webapp {

/// @brief G+Smo boundary sides
/// @{
template <short_t>
const std::initializer_list<gismo::boundary::side> GismoBoundarySides;

template <>
const std::initializer_list<gismo::boundary::side> GismoBoundarySides<1>{
    gismo::boundary::side::west, gismo::boundary::side::east};

template <>
const std::initializer_list<gismo::boundary::side> GismoBoundarySides<2>{
    gismo::boundary::side::west, gismo::boundary::side::east,
    gismo::boundary::side::south, gismo::boundary::side::north};

template <>
const std::initializer_list<gismo::boundary::side> GismoBoundarySides<3>{
    gismo::boundary::side::west,  gismo::boundary::side::east,
    gismo::boundary::side::south, gismo::boundary::side::north,
    gismo::boundary::side::front, gismo::boundary::side::back};

template <>
const std::initializer_list<gismo::boundary::side> GismoBoundarySides<4>{
    gismo::boundary::side::west,  gismo::boundary::side::east,
    gismo::boundary::side::south, gismo::boundary::side::north,
    gismo::boundary::side::front, gismo::boundary::side::back,
    gismo::boundary::side::stime, gismo::boundary::side::etime};
/// @}

/// @brief G+Smo boundary side names
/// @{
template <short_t>
const std::initializer_list<std::string> GismoBoundarySideStrings;

template <>
const std::initializer_list<std::string> GismoBoundarySideStrings<1>{"west",
                                                                     "east"};

template <>
const std::initializer_list<std::string> GismoBoundarySideStrings<2>{
    "west", "east", "south", "north"};

template <>
const std::initializer_list<std::string> GismoBoundarySideStrings<3>{
    "west", "east", "south", "north", "front", "back"};

template <>
const std::initializer_list<std::string> GismoBoundarySideStrings<4>{
    "west", "east", "south", "north", "front", "back", "stime", "etime"};
/// @}

/// @brief G+Smo base model
template <class T> class GismoModel : public Model<T> {

public:
  /// @brief Default constructor
  GismoModel() {}

  /// @brief Destructor
  ~GismoModel() {}

  /// @brief Serializes the model to JSON
  virtual nlohmann::json to_json(const std::string &patch,
                                 const std::string &component,
                                 const std::string &attribute) const override {

    return R"({ INVALID REQUEST })"_json;
  }

  /// @brief Updates the attributes of the model
  virtual nlohmann::json updateAttribute(const std::string &patch,
                                         const std::string &component,
                                         const std::string &attribute,
                                         const nlohmann::json &json) override {

    return R"({ INVALID REQUEST })"_json;
  }
};

} // namespace webapp
} // namespace iganet
