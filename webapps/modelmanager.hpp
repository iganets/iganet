/**
   @file webapps/modelmanager.hpp

   @brief Model manager

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

  /// @brief Model interface
  class Model {
  public:    
    /// @brief Destructor
    virtual ~Model() {};
    
    /// @brief Returns the model's name
    virtual std::string getName() const = 0;

    /// @brief Returns the model's description
    virtual std::string getDescription() const = 0;

    /// @brief Returns the model's options
    virtual std::string getOptions() const = 0;

    /// @brief Returns the model's JSON serialization
    virtual nlohmann::json getModel() const {     
      return nlohmann::json::parse(std::string("{ \"name\" : \"") + getName() + "\"," +
                                   "\"description\" : \"" + getDescription() + "\"," +
                                   "\"options\" : " + getOptions() + " }");
    }
    
    /// @brief Serializes the model to JSON
    virtual nlohmann::json to_json(const std::string& attribute = "") const = 0;    
  };
  
  /// @brief InvalidModel exception
  struct InvalidModelException : public std::exception {  
    const char * what() const throw() {  
      return "Invalid model name";  
    }
  };

  /// @brief Model evaluator
  template<short_t Dim>
  class ModelEval {
  public:
    /// @brief Evaluate model
    virtual BlockTensor<torch::Tensor, 1, Dim> eval(const nlohmann::json& config) const = 0;
  };

  /// @brief Model refinement
  class ModelRefine {
  public:
    /// @brief Refine model
    virtual void refine(const nlohmann::json& config) const = 0;
  };
  
  /// @brief Model manager
  ///
  /// This class implements the model manager
  class ModelManager {
  private:
    
    /// @brief ModelHandler
    class ModelHandler {
    public:
      
      /// @brief Default constructor deleted
      ModelHandler() = delete;
      ModelHandler(ModelHandler&&) = delete;
      ModelHandler(const ModelHandler&) = delete;
      
      /// @brief Constructor from file
      ModelHandler(const char * filename, int flags = RTLD_NOW) {
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

    /// @brief List of models
    std::map<std::string, std::shared_ptr<ModelHandler>> models;
    
  public:
    /// @brief Default constructor deleted
    ModelManager() = delete;
    
    /// @brief Constructor from filesystem
    /// @{
    ModelManager(const std::string& path)
      : ModelManager(std::vector<std::string>({path}))
    {}
    
    ModelManager(const std::vector<std::string>& paths) {
      for (const auto& path : paths) {
        const std::filesystem::path fspath{path};
        for (const auto& entry : std::filesystem::directory_iterator{fspath}) {
          if (entry.path().extension() == ".dylib" ||
              entry.path().extension() == ".so") {
            auto handler = std::make_shared<ModelHandler>(entry.path().c_str());
            std::shared_ptr<Model> (*create)();
            create = reinterpret_cast<std::shared_ptr<Model> (*)()> (handler->getSymbol("create"));
            models[create()->getName()] = std::move(handler);
          }
        }
      }
    }    
    /// @}

    /// @brief Returns a new instance of the requested model and
    /// throws an exception if model cannot be found
    std::shared_ptr<Model> create(const std::string& name,
                                   const nlohmann::json& config = NULL) const {
      try {
        auto it = models.find(name);
        if (it == models.end())
          throw InvalidModelException();
        
        std::shared_ptr<Model> (*create)(const nlohmann::json&);
        create = reinterpret_cast<std::shared_ptr<Model> (*)(const nlohmann::json&)> (it->second->getSymbol("create"));
        return create(config);
      } catch(...) {
        throw InvalidModelException();
      }
    }

    /// @brief Serializes the list of models to JSON
    nlohmann::json getModels() const {
      auto data = nlohmann::json::array();
      for (auto const& model : models)
        data.push_back(create(model.first)->getModel());
      return data;
    }
  };

} // namespace iganet
    
