/**
   @file webapps/models/gismo/GismoPoisson2d.cxx

   @brief G+Smo Poisson solver in 2d

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include <jit.hpp>
#include <modelmanager.hpp>

#include <GismoPoissonModel.hpp>

#ifdef _WIN32
extern "C" __declspec(dllexport)
#else
extern "C"
#endif
{
  // @brief List of JIT-compiled model handlers
  static std::map<std::string, std::shared_ptr<iganet::ModelHandler>> models;

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wreturn-type-c-linkage"

  /// @brief Create a G+Smo Poisson solver
  std::shared_ptr<iganet::Model> create(const nlohmann::json &json) {
    std::array<iganet::short_t, 2> degrees = {1, 1};
    std::array<int64_t, 2> ncoeffs = {4, 4};
    std::array<int64_t, 2> npatches = {1, 1};

    if (json.contains("data")) {

      if (json["data"].contains("degrees"))
        degrees = json["data"]["degrees"].get<std::array<iganet::short_t, 2>>();

      if (json["data"].contains("ncoeffs"))
        ncoeffs = json["data"]["ncoeffs"].get<std::array<int64_t, 2>>();

      if (json["data"].contains("npatches"))
        npatches = json["data"]["npatches"].get<std::array<int64_t, 2>>();

      try {
        // generate list of include files
        std::string includes =
            "#include <GismoPoissonModel.hpp>\n"
            "#pragma GCC diagnostic push\n"
            "#pragma GCC diagnostic ignored \"-Wreturn-type-c-linkage\"\n";

        // generate source code
        std::string src = "std::shared_ptr<iganet::Model> create("
                          "const std::array<iganet::short_t, 2>& degrees, "
                          "const std::array<int64_t, 2>& ncoeffs, "
                          "const std::array<int64_t, 2>& npatches)\n{\n";

        src.append("return "
                   "std::make_shared<iganet::webapp::GismoPoissonModel<2,"
                   "iganet::real_t>>(degrees, ncoeffs, npatches);\n}\n#pragma "
                   "GCC diagnostic pop\n");

        // compile dynamic library
        auto libname = iganet::jit{}.compile(includes, src, "GismoPoisson2d");

        // search for library name
        auto model = models.find(libname);
        if (model == models.end()) {
          models[libname] =
              std::make_shared<iganet::ModelHandler>(libname.c_str());
          model = models.find(libname);
        }

        // create model instance
        std::shared_ptr<iganet::Model> (*create)(
            const std::array<iganet::short_t, 2> &,
            const std::array<int64_t, 2> &, const std::array<int64_t, 2> &);
        create = reinterpret_cast<std::shared_ptr<iganet::Model> (*)(
            const std::array<iganet::short_t, 2> &,
            const std::array<int64_t, 2> &, const std::array<int64_t, 2> &)>(
            model->second->getSymbol("create"));
        return create(degrees, ncoeffs, npatches);
      } catch (...) {
        throw iganet::InvalidModelException();
      }
    } else
      return std::make_shared<
          iganet::webapp::GismoPoissonModel<2, iganet::real_t>>(
          degrees, ncoeffs, npatches);
  }

#pragma GCC diagnostic pop
}
