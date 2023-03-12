/**
   @file webapps/pluginmanager.hpp

   @brief Plugin manager

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

  /// @brief Plugin interface
  class Plugin {
  public:    
    /// @brief Destructor
    virtual ~Plugin() {};
    
    /// @brief Returns the plugin's name
    virtual std::string getName() const = 0;

    /// @brief Returns the plugin's description
    virtual std::string getDescription() const = 0;

    /// @brief Returns the plugin's options
    virtual std::string getOptions() const = 0;

    /// @brief Returns the plugin's JSON serialization
    virtual nlohmann::json getPlugin() const {
      return nlohmann::json::parse(std::string("{ \"name\" : \"") + getName() + "\"," +
                                   "\"description\" : \"" + getDescription() + "\"," +
                                   "\"options\" : " + getOptions() + " }");
    }
    
    /// @brief Serializes the plugin to JSON
    virtual nlohmann::json to_json() const = 0;    
  };
  
  /// @brief InvalidPlugin exception
  struct InvalidPluginException : public std::exception {  
    const char * what() const throw() {  
      return "Invalid plugin name";  
    }
  };

  /// @brief Plugin evaluator
  template<short_t geoDim_, short_t parDim_>
  class PluginEval : public Plugin {
  public:
    /// @brief Evaluate plugin
    virtual BlockTensor<torch::Tensor, 1, geoDim_> eval(const nlohmann::json& config) const = 0;
  };
  
  /// @brief Plugin manager
  ///
  /// This class implements the plugin manager
  class PluginManager {
  private:
    
    /// @brief PluginHandler
    class PluginHandler {
    public:
      
      /// @brief Default constructor deleted
      PluginHandler() = delete;
      PluginHandler(PluginHandler&&) = delete;
      PluginHandler(const PluginHandler&) = delete;
      
      /// @brief Constructor from file
      PluginHandler(const char * filename, int flags = RTLD_NOW) {
#if defined(_WIN32)
        static_cast<void>(flags);
        HMODULE dl = LoadLibrary(filename);
        if (!dl)
          throw std::runtime_error( "LoadLibrary - error: " + GetLastError() );
        handle.reset(dl, FreeLibrary);
#elif defined(__APPLE__) || defined(__linux__) || defined(__unix)      
        void * dl = ::dlopen(filename, flags);
        if (!dl)
          throw std::runtime_error( ::dlerror() );
        handle.reset(dl, ::dlclose);
#else
#error("Unsupported operating system")
#endif
      }

      /// @brief Gets symbol from dynamic library
      void* getSymbol(const char* name) const
      {
        if (!handle)
          throw std::runtime_error("An error occured while accessing the dynamic library");
        
        void *symbol = NULL;
#if defined(_WIN32)
        *(void **)(&symbol) = (void*)GetProcAddress(handle.get(), name );
#elif defined(__APPLE__) || defined(__linux__) || defined(__unix)
        *(void **)(&symbol) = ::dlsym( handle.get(), name );
#endif
        if (!symbol)
          throw std::runtime_error("An error occured while getting the symbol from the dynamic library");
        
        return symbol;
      }
      
      /// @brief Checks if handle is assigned
      operator bool() const { return (bool)handle; }
      
    private:
      
      /// @brief Handle to dynamic library object
#if defined(_WIN32)
      std::shared_ptr< std::remove_pointer<HMODULE>::type > handle;
#elif defined(__APPLE__) || defined(__linux__) || defined(__unix)
      std::shared_ptr<void> handle;
#endif
    };

    /// @brief List of plugins
    std::map<std::string, std::shared_ptr<PluginHandler>> plugins;
    
  public:
    /// @brief Default constructor deleted
    PluginManager() = delete;
    
    /// @brief Constructor from filesystem
    /// @{
    PluginManager(const std::string& path)
      : PluginManager(std::vector<std::string>({path}))
    {}
    
    PluginManager(const std::vector<std::string>& paths) {
      for (const auto& path : paths) {
        const std::filesystem::path fspath{path};
        for (const auto& entry : std::filesystem::directory_iterator{fspath}) {
          if (entry.path().extension() == ".dylib" ||
              entry.path().extension() == ".so") {
            auto handler = std::make_shared<PluginHandler>(entry.path().c_str());
            std::shared_ptr<Plugin> (*create)();
            create = reinterpret_cast<std::shared_ptr<Plugin> (*)()> (handler->getSymbol("create"));
            plugins[create()->getName()] = std::move(handler);
          }
        }
      }
    }    
    /// @}

    /// @brief Returns a new instance of the requested plugin and
    /// throws an exception if plugin cannot be found
    std::shared_ptr<Plugin> create(const std::string& name,
                                   const nlohmann::json& config = NULL) const {
      try {
        auto it = plugins.find(name);
        if (it == plugins.end())
          throw InvalidPluginException();
        
        std::shared_ptr<Plugin> (*create)(const nlohmann::json&);
        create = reinterpret_cast<std::shared_ptr<Plugin> (*)(const nlohmann::json&)> (it->second->getSymbol("create"));
        return create(config);
      } catch(...) {
        throw InvalidPluginException();
      }
    }

    /// @brief Serializes the list of plugins to JSON
    nlohmann::json getPlugins() const {
      auto data = nlohmann::json::array();
      for (auto const& plugin : plugins)
        data.push_back(create(plugin.first)->getPlugin());
      return data;
    }
  };

} // namespace iganet
    
