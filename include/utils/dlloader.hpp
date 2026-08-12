/**
   @file utils/dlloader.hpp

   @brief Dynamic library loader

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#pragma once

#include <exception>
#include <filesystem>
#include <map>

#if defined(_WIN32)
#include <direct.h>
#include <windows.h>
#else
#include <dlfcn.h>
#include <sys/stat.h>
#endif

namespace iganet {

/// @brief Dynamic library handler
///
/// This class implements the dynamic library handler
class DLHandler {
public:
  /// @brief Default constructor deleted
  DLHandler() = delete;
  DLHandler(DLHandler &&) = delete;
  DLHandler(const DLHandler &) = delete;

  /// @brief Constructor from file
  /// @{
  explicit DLHandler(const char *filename, int flags = RTLD_NOW) {
#if defined(_WIN32)
    static_cast<void>(flags);
    HMODULE dl = LoadLibrary(filename);
    if (!dl)
      throw std::runtime_error("LoadLibrary - error: " + GetLastError());
    handle.reset(dl, FreeLibrary);
#elif defined(__APPLE__) || defined(__linux__) || defined(__unix)
    void *dl = ::dlopen(filename, flags);
    if (!dl)
      throw std::runtime_error(::dlerror());
    handle.reset(dl, ::dlclose);
#else
#error("Unsupported operating system")
#endif
  }

  explicit DLHandler(const std::string &filename, int flags = RTLD_NOW)
      : DLHandler(filename.c_str(), flags) {}

  explicit DLHandler(const std::filesystem::path &filename,
                     int flags = RTLD_NOW)
      : DLHandler(filename.string(), flags) {}
  /// @}

  /// @brief Gets symbol from dynamic library
  void *getSymbol(const char *name) const {
    if (!handle)
      throw std::runtime_error(
          "An error occured while accessing the dynamic library");

    void *symbol = NULL;
#if defined(_WIN32)
    *(void **)(&symbol) = (void *)GetProcAddress(handle.get(), name);
#elif defined(__APPLE__) || defined(__linux__) || defined(__unix)
    *(void **)(&symbol) = ::dlsym(handle.get(), name);
#endif
    if (!symbol)
      throw std::runtime_error(
          "An error occured while getting the symbol from the dynamic library");

    return symbol;
  }

  /// @brief Checks if handle is assigned
  operator bool() const { return (bool)handle; }

private:
  /// @brief Handle to dynamic library object
#if defined(_WIN32)
  std::shared_ptr<std::remove_pointer<HMODULE>::type> handle;
#elif defined(__APPLE__) || defined(__linux__) || defined(__unix)
  std::shared_ptr<void> handle;
#endif
};

/// @brief Dynamic library loader
///
/// This class implements the dynamic library loader
class DLLoader : protected utils::FullQualifiedName {
private:
  /// @brief List of dynamic libraries
  std::map<std::string, std::shared_ptr<DLHandler>> libraries;

public:
  // /// @brief Default constructor deleted
  // DLLoader() = delete;

  // /// @brief Constructor from filesystem
  // /// @{
  // DLLoader(const std::string &path)
  //     : DLLoader(std::vector<std::string>({path})) {}

  // DLLoader(const std::vector<std::string> &paths) { addLibraryPath(paths); }
  // /// @}

  // /// @brief Adds librariess from given path
  // inline void addLibraryPath(const std::string &path) {
  //   addLibraryPath(std::vector<std::string>({path}));
  // }

  // /// @brief Adds librariess from list of directories
  // inline void addLibraryPath(const std::vector<std::string> &paths) {
  //   for (const auto &path : paths) {
  //     try {
  //       const std::filesystem::path fspath{path};
  //       if (std::filesystem::exists(fspath)) {
  //         for (const auto &entry :
  //              std::filesystem::directory_iterator{fspath}) {
  //           if (entry.path().extension() == ".dll" ||
  //               entry.path().extension() == ".dylib" ||
  //               entry.path().extension() == ".so") {
  //             try {
  //               auto handler =
  //                   std::make_shared<DLHandler>(entry.path().c_str());
  //               std::shared_ptr<Model<iganet::real_t>> (*create)(
  //                   const nlohmann::json &);
  //               create =
  //                   reinterpret_cast<std::shared_ptr<Model<iganet::real_t>> (*)(
  //                       const nlohmann::json &)>(handler->getSymbol("create"));
  //               models[create({})->getName()] = handler;
  //             } catch (...) {
  //               std::clog << "Unable to create model " << entry << std::endl;
  //             }
  //           }
  //         }
  //       }
  //     } catch (...) {
  //       std::clog << "Unable to open path " << path << std::endl;
  //     }
  //   }
  // }

  // /// @brief Returns a new instance of the requested model and
  // /// throws an exception if model cannot be found
  // inline std::shared_ptr<Model<iganet::real_t>>
  // create(const std::string &name, const nlohmann::json &json = NULL) const {
  //   try {
  //     auto model = models.find(name);
  //     if (model == models.end())
  //       throw InvalidModelException();
  //     std::shared_ptr<Model<iganet::real_t>> (*create)(const nlohmann::json &);
  //     create = reinterpret_cast<std::shared_ptr<Model<iganet::real_t>> (*)(
  //         const nlohmann::json &)>(model->second->getSymbol("create"));
  //     return create(json);
  //   } catch (...) {
  //     throw InvalidModelException();
  //   }
  // }

  // /// @brief Returns a new model instance from binary data stream
  // /// throws an exception if model cannot be created
  // inline std::shared_ptr<Model<iganet::real_t>>
  // load(const nlohmann::json &json) const {

  //   for (auto &model : models) {
  //     try {
  //       std::shared_ptr<Model<iganet::real_t>> (*load)(const nlohmann::json &);
  //       load = reinterpret_cast<std::shared_ptr<Model<iganet::real_t>> (*)(
  //           const nlohmann::json &)>(model.second->getSymbol("load"));
  //       return load(json);
  //     } catch (...) { /* try next model */
  //     }
  //   }

  //   // No working model found, through exception and quit load request
  //   throw InvalidModelException();
  // }

  // /// @brief Serializes the list of models to JSON
  // inline nlohmann::json getModels() const {
  //   auto data = nlohmann::json::array();
  //   for (auto const &model : models)
  //     data.push_back(create(model.first)->getModel());
  //   return data;
  // }

  // /// @brief Returns a string representation of the model manager object
  // inline virtual void
  // pretty_print(std::ostream &os = std::cout) const noexcept override {
  //   os << name();
  // }
};

// /// @brief Print (as string) a model manager object
// inline std::ostream &operator<<(std::ostream &os, const DLLoader &obj) {
//   obj.pretty_print(os);
//   return os;
// }

} // namespace iganet
