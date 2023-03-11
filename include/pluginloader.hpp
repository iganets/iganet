/**
   @file include/pluginloader.hpp

   @brief Plugin loader

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#pragma once

#include <filesystem>
#include <fstream>

#if defined(_WIN32)
#include <direct.h>
#include <windows.h>
#else
#include <dlfcn.h>
#include <sys/stat.h>
#endif

namespace iganet {

  /// @brief Plugin loader
  ///
  /// This class implements the plugin loader framework
  class PluginLoader
  {
  public:
    
    /// @brief Default constructor deleted
    PluginLoader() = delete;
    
    /// @brief Constructor from file
    PluginLoader(const std::string& filename) {
#if defined(_WIN32)
      static_cast<void>(0);
      HMODULE dl = LoadLibrary(filename.c_str());
      if (!dl)
        throw std::runtime_error( "LoadLibrary - error: " + GetLastError() );
      handle.reset(dl, FreeLibrary);
#elif defined(__APPLE__) || defined(__linux__) || defined(__unix)
      
      void * dl = ::dlopen(filename.c_str(), RTLD_LAZY
#ifdef RTLD_DEEPBIND
                                           | RTLD_DEEPBIND
#endif
                           );
      if (!dl)
        throw std::runtime_error( ::dlerror() );
      handle.reset(dl, ::dlclose);
#else
#error("Unsupported operating system")
#endif
    }
        
    /// @brief Gets symbol from dynamic library
    template<class T>
    T* getSymbol(const char* name) const
    {
      if (!handle)
        throw std::runtime_error("An error occured while accessing the dynamic library");
      
      T *symbol = NULL;
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
} // namespace iganet
