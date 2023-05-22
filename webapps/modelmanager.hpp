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

  /// @brief Enumerator for specifying the capabilities
  enum class capability
    {
      eval      =   0, /*!< evaluates object */
      refine    =   1, /*!< h-refines object */
      elevate   =   2, /*!< p-refines object */
            
      /*!< Model loading/saving */
      load      = 101, /*!< loads model from PyTorch file    */
      save      = 102, /*!< saves model to PyTorch file    */

      /*!< Model import/export */
      importXML = 201, /*!< imports object from G+Smo XML file */     
      exportXML = 202, /*!< exports object to G+Smo XML file */

      /*!< Error computation */
      computeL1error = 301, /*!< computes model's L1-error */
      computeL2error = 302, /*!< computes model's L2-error */
      computeH1error = 303  /*!< computes model's H1-error */      
    };

  /// @brief Enumerator for specifying the output type
  enum class io
    {
      scalar               = 0, /*!< scalar value */
      scalarfield          = 1, /*!< scalar field */
      vectorfield          = 2, /*!< vector field */
      scalarfield_boundary = 3, /*!< scalar field at the boundary */
      vectorfield_boundary = 4  /*!< vector field at the boundary */
    };

  /// @brief Model elevation
  class ModelElevate {
  public:
    /// @brief elevates model
    virtual void elevate(const nlohmann::json& json) = 0;

    // @brief Returns model capabilities
    std::vector<std::string> getCapabilities() const {
      return std::vector{std::string("elevate")};
    }
  };
  
  /// @brief Model evaluator
  class ModelEval {
  public:
    /// @brief Evaluates model
    virtual nlohmann::json eval(const std::string& component,
                                const nlohmann::json& json) const = 0;

    // @brief Returns model capabilities
    std::vector<std::string> getCapabilities() const {
      return std::vector{std::string("eval")};
    }
  };

  /// @brief Model refinement
  class ModelRefine {
  public:
    /// @brief Refines model
    virtual void refine(const nlohmann::json& json) = 0;

    // @brief Returns model capabilities
    std::vector<std::string> getCapabilities() const {
      return std::vector{std::string("refine")};
    }
  };
  
  /// @brief Model XML serialization
  class ModelXML {
  public:
    /// @brief Imports model from XML
    virtual void importXML(const nlohmann::json& json,
                           const std::string& component,
                           std::size_t id) = 0;
    
    /// @brief Exports model to XML
    virtual nlohmann::json exportXML(const std::string& component,
                                     std::size_t id) = 0;

    // @brief Returns model capabilities
    std::vector<std::string> getCapabilities() const {
      return std::vector{std::string("exportxml"),
                         std::string("importxml")};
    }
  };
  
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

    /// @brief Returns the model's inputs
    virtual std::string getInputs() const = 0;

    /// @brief Returns the model's outputs
    virtual std::string getOutputs() const = 0;

    /// @brief Returns the model's JSON serialization
    virtual nlohmann::json getModel() const {
      return nlohmann::json::parse(std::string("{ \"name\" : \"") + getName() + "\"," +
                                   "\"description\" : \"" + getDescription() + "\"," +
                                   "\"options\" : " + getOptions() + "," +
                                   "\"capabilities\" : " + getCapabilities().dump() + "," +
                                   "\"inputs\" : " + getInputs() + "," +
                                   "\"outputs\" : " + getOutputs() + " }");
    }

    /// @brief Returns the model's capabilities
    virtual nlohmann::json getCapabilities() const {

      std::vector<std::string> capabilities;

      if (auto m = dynamic_cast<const ModelElevate*>(this))
        for (auto const& capability : m->getCapabilities())
          capabilities.push_back(capability);
      
      if (auto m = dynamic_cast<const ModelEval*>(this))
        for (auto const& capability : m->getCapabilities())
          capabilities.push_back(capability);
      
      if (auto m = dynamic_cast<const ModelRefine*>(this))
        for (auto const& capability : m->getCapabilities())
          capabilities.push_back(capability);

      if (auto m = dynamic_cast<const ModelXML*>(this))
        for (auto const& capability : m->getCapabilities())
          capabilities.push_back(capability);
      
      auto data = nlohmann::json::array();
      for (auto const& capability : capabilities)
        data.push_back("\""+capability+"\"");

      nlohmann::json json;
      json["capability"] = data;
      
      return json;
    }
    
    /// @brief Serializes the model to JSON
    virtual nlohmann::json to_json(const std::string& attribute = "") const = 0;

    /// @brief Updates the attributes of the model
    virtual nlohmann::json updateAttribute(const std::string& attribute,
                                           const nlohmann::json& json) = 0;
  };

  /// @brief IndexOutOfBounds exception
  struct IndexOutOfBoundsException : public std::exception {
    const char * what() const throw() {
      return "Index is out of bounds";
    }
  };

  /// @brief InvalidModel exception
  struct InvalidModelException : public std::exception {
    const char * what() const throw() {
      return "Invalid model name";
    }
  };

  /// @brief InvalidModelAttribute exception
  struct InvalidModelAttributeException : public std::exception {
    const char * what() const throw() {
      return "Invalid model attribute";
    }
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
      addModelPath(paths);
    }
    /// @}
    
    /// @brief Adds models from given path
    inline void addModelPath(const std::string& path) {
      addModelPath(std::vector<std::string>({path}));
    }

    /// @brief Adds models from list of directories
    inline void addModelPath(const std::vector<std::string>& paths) {
      for (const auto& path : paths) {
        const std::filesystem::path fspath{path};
        for (const auto& entry : std::filesystem::directory_iterator{fspath}) {
          if (entry.path().extension() == ".dylib" ||
              entry.path().extension() == ".so") {
            auto handler = std::make_shared<ModelHandler>(entry.path().c_str());
            std::shared_ptr<Model> (*create)(const nlohmann::json&);
            create = reinterpret_cast<std::shared_ptr<Model> (*)(const nlohmann::json&)> (handler->getSymbol("create"));
            models[create({})->getName()] = handler;
          }
        }
      }
    }
    
    /// @brief Returns a new instance of the requested model and
    /// throws an exception if model cannot be found
    inline std::shared_ptr<Model> create(const std::string& name,
                                         const nlohmann::json& json = NULL) const {
      try {
        auto it = models.find(name);
        if (it == models.end())
          throw InvalidModelException();

        std::shared_ptr<Model> (*create)(const nlohmann::json&);
        create = reinterpret_cast<std::shared_ptr<Model> (*)(const nlohmann::json&)> (it->second->getSymbol("create"));
        return create(json);
      } catch(...) {
        throw InvalidModelException();
      }
    }

    /// @brief Serializes the list of models to JSON
    inline nlohmann::json getModels() const {
      auto data = nlohmann::json::array();
      for (auto const& model : models)
        data.push_back(create(model.first)->getModel());
      return data;
    }
  };

} // namespace iganet

