/**
   @file webapps/server.cxx

   @brief Demonstration of a server application

   @author Matthias Moller

   @copyright This file is part of the IgANet project

   This Source Code Form is subject to the terms of the Mozilla Public
   License, v. 2.0. If a copy of the MPL was not distributed with this
   file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include <App.h>
#include <iganet.hpp>
#include <jit.hpp>
#include <popl.hpp>
#include <iostream>
#include <tuple>
#include <vector>

namespace iganet { namespace webapp {

    /// @brief InvalidSessionId exception
    struct InvalidSessionIdException : public std::exception {  
      const char * what() const throw() {  
        return "Invalid session id";  
      }
    };
    
    /// @brief InvalidModelId exception
    struct InvalidModelIdException : public std::exception {  
      const char * what() const throw() {  
        return "Invalid model id";  
      }
    };

    /// @brief InvalidModelType exception
    struct InvalidModelTypeException : public std::exception {  
      const char * what() const throw() {  
        return "Invalid model type";  
      }
    }; 
    
    /// @brief Tokenize the input string
    auto tokenize(std::string str) {
      std::vector<std::string> tokens;
      for (auto i = strtok(&str[0], "/"); i != NULL; i = strtok(NULL, "/"))        
        tokens.push_back(i);
      return tokens;
    }

    /// @brief Session
    template<typename T>
    struct Session {
    private:
      /// @brief Session UUID
      const std::string uuid;
            
    public:
      /// @brief Default constructor
      Session() : uuid(iganet::uuid::create())
      {}

      /// @brief Returns the UUID
      const std::string& getUUID() const {
        return uuid;
      }
      
      /// Returns the requested model or throws an exception
      std::shared_ptr<iganet::core<T>> getModel(int64_t id) {
        auto it = models.find(id);
        if (it == models.end())
          throw InvalidModelIdException();
        else
          return it->second;
      }

      /// Returns the model and removes it from the list of models
      std::shared_ptr<iganet::core<T>> removeModel(int64_t id) {
        auto it = models.find(id);
        if (it == models.end())
          throw InvalidModelIdException();
        else {
          auto model = it->second;
          models.erase(it);
          return model;
        }
      }
      
      /// @brief List of models
      std::map<int64_t, std::shared_ptr<iganet::core<T>>> models;
    };
    
    /// @brief Sessions structure
    template<typename T>
    struct Sessions {
    public:     
      /// Returns the requested session model or throws an exception
      std::shared_ptr<Session<T>> getSession(std::string uuid) {
        auto it = sessions.find(uuid);
        if (it == sessions.end())
          throw InvalidSessionIdException();
        else
          return it->second;
      }

      /// Returns the session and removes it from the list of sessions
      std::shared_ptr<Session<T>> removeSession(std::string uuid) {
        auto it = sessions.find(uuid);
        if (it == sessions.end())
          throw InvalidSessionIdException();
        else {
          auto session = it->second;
          sessions.erase(it);
          return session;
        }
      }
      
      /// Static list of sessions shared between all sockets
      inline static std::map<std::string, std::shared_ptr<Session<T>>> sessions;
    };
    
}} // namespace iganet::webapp


int main(int argc, char const* argv[])
{
  // iganet::JITCompiler compiler;
  // compiler << "#include <iganet.hpp>\n"
  //          << "namespace iganet {\n"
  //          << "EXPORT std::shared_ptr<iganet::core<double>> create()\n"
  //          << "{ return std::make_shared<iganet::core<double>>(iganet::UniformBSpline<double,1,1>({5})); }"
  //          << "} // namespace iganet\n";

  // try {
  //   iganet::DynamicLibrary dl = compiler.build();
  //   try {
  //     typedef std::shared_ptr<iganet::core<double>> (dl_expr)();
  //     dl_expr * e = dl.getSymbol<dl_expr>("create");
  //     std::cout << e() << std::endl;
  //   } catch (...) {
  //     throw std::runtime_error("An error occured while loading the dynamic library");
  //   }      
  // } catch(...) {
  //   throw std::runtime_error("An error occured while compiling the dynamic library");
  // }

  // exit(0);
  
  using PerSocketData = iganet::webapp::Sessions<double>;
  
  popl::OptionParser op("Allowed options");
  auto help_option = op.add<popl::Switch>("h", "help", "print help message");
  auto port_option = op.add<popl::Value<int>>("p", "port", "TCP port of the server", 3000);
  op.parse(argc, argv);

  // Print auto-generated help message
  if (help_option->count() == 1)
    std::cout << op << std::endl;
  else if (help_option->count() == 2)
    std::cout << op.help(popl::Attribute::advanced) << std::endl;
  else if (help_option->count() > 2)
    std::cout << op.help(popl::Attribute::expert) << std::endl;  
  
  // Create WebSocket application
  uWS::App().ws<PerSocketData>("/*", {
      /* Settings */
      .compression = uWS::CompressOptions(uWS::DEDICATED_COMPRESSOR_4KB |
                                          uWS::DEDICATED_DECOMPRESSOR),
        .maxPayloadLength = 100 * 1024 * 1024,
        .idleTimeout = 16,
        .maxBackpressure = 100 * 1024 * 1024,
        .closeOnBackpressureLimit = false,
        .resetIdleTimeoutOnSend = false,
        .sendPingsAutomatically = true,
        /* Handlers */
        .upgrade = nullptr,
        .open = [](auto *ws) {
          std::clog << "Connection has been opened\n";          
        },
        .message = [](auto *ws, std::string_view message, uWS::OpCode opCode) {          
          try {
            // Tokenize request
            auto request = nlohmann::json::parse(message);
            auto tokens  = iganet::webapp::tokenize(request["request"].get<std::string>());

            // Prepare response
            nlohmann::json response;
            response["request"] = request["id"];
            response["status"]  = 0;

            // Dispatch request
            if (tokens[0] == "get") {
              //
              // request: get/*
              //
              
              if (tokens.size() == 1) {
                //
                // request: get
                //
                
                // Get list of all active sessions
                std::vector<std::string> ids;
                for (const auto& session : ws->getUserData()->sessions)
                  ids.push_back(session.first);
                response["data"] = ids;
                ws->send(response.dump(), uWS::OpCode::TEXT, true);
              }

              else if (tokens.size() == 2) {
                //
                // request: get/<session-uuid>
                //

                // Get session
                auto session = ws->getUserData()->getSession(tokens[1]);
                //session->authenticate(request["token"]);

                // Get list of all active models in session
                std::vector<int64_t> ids;
                for (const auto& model : session->models)
                  ids.push_back(model.first);
                response["data"] = ids;
                ws->send(response.dump(), uWS::OpCode::TEXT, true);
              }

              else if (tokens.size() == 3) {
                //
                // request: get/<session-uuid>/<model-id>
                //

                // Get session
                auto session = ws->getUserData()->getSession(tokens[1]);
                //session->authenticate(request["token"]);

                // Get model
                auto model = session->getModel(stoi(tokens[2]));
                
                // Serialize model to JSON
                response["data"] = model->to_json();
                ws->send(response.dump(), uWS::OpCode::TEXT, true);
              }

              else if (tokens.size() == 4) {
                //
                // request: get/<session-uuid>/<model-id>/<attribute>
                //

                // Get session
                auto session = ws->getUserData()->getSession(tokens[1]);
                //session->authenticate(request["token"]);

                // Get model
                auto model = session->getModel(stoi(tokens[2]));

                // // Get attribute
                // if (tokens[3] == "geoDim") {
                //   data["geoDim"]  = model->geoDim();
                // }
                // else if (tokens[3] == "parDim") {
                //   data["parDim"]  = model->parDim();
                // }
                // else if (tokens[3] == "degrees") {
                //   data["degrees"] = model()->degrees();
                // }
                // else if (tokens[3] == "ncoeffs") {
                //   data["ncoeffs"] = model->ncoeffs();
                // }
                // else if (tokens[3] == "nknots") {
                //   data["nknots"]  = model->nknots();
                // }
                // else if (tokens[3] == "coeffs") {
                // }
                // else if (tokens[3] == "knots") {
                // }
                // else {
                //   response["status"] = 1;
                //   response["reason"] = "Invalid attribute. Valid attributes are \"geoDim\", \"parDim\", \"degrees\", \"ncoeffs\", \"ncoeffs\", \"nknots\", \"coeffs\", and \"knots\"";
                // ws->send(response.dump(), uWS::OpCode::TEXT, true);                
                // }
                  
                // Serialize model to JSON
                response["data"] = "Not implemented yet (get/<session-uuid>/<model-id>/<attribute>)";
                ws->send(response.dump(), uWS::OpCode::TEXT, true);
              }

              else {
                response["status"] = 1;
                response["reason"] = "Invalid get request. Valid requests are \"get\", \"get/<session-uuid>\", \"get/<session-uuid>/<model-id>\", and \"get/<session-uuid>/<model-id>/<attribute>\"";
                ws->send(response.dump(), uWS::OpCode::TEXT, true);                
              }
                
            }
            
            else if (tokens[0] == "put") {
              //
              // request: put/*
              //

              if (tokens.size() == 3) {
                //
                // request: put/<session-uuid>/<model-id>/<attribute>
                //

                // Get session
                auto session = ws->getUserData()->getSession(tokens[1]);
                //session->authenticate(request["token"]);

                // Get model
                auto model = session->getModel(stoi(tokens[2]));

                response["data"] = "Not implemented yet (put/<session-uuid>/<model-id>/<attribute>)";
                ws->send(response.dump(), uWS::OpCode::TEXT, true);
              }
              
              else {
                response["status"] = 1;
                response["reason"] = "Invalid put request. Valid requests are \"put/<session-uuid>/<model-id>/<attribute>\"";
                ws->send(response.dump(), uWS::OpCode::TEXT, true);                
              }
              
            }
            
            else if (tokens[0] == "create") {
              //
              // request: create/*
              //
              
              if (tokens.size() == 2 && tokens[1] == "session") {
                //
                // request: create/session
                //
                
                // Create a new session
                auto session = std::make_shared<iganet::webapp::Session<double>>();
                std::string uuid = session->getUUID();
                ws->getUserData()->sessions[uuid] = session; 
                response["data"]["uuid"] = uuid;
                ws->send(response.dump(), uWS::OpCode::TEXT, true);                
              }
              
              else if (tokens.size() == 3) {
                //
                // request: create/<session-uuid>/<model-type>
                //

                // Get session
                auto session = ws->getUserData()->getSession(tokens[1]);
                //session->authenticate(request["token"]);
                
                // Create new model
                int64_t id = (session->models.size() > 0 ?
                              session->models.crbegin()->first+1 : 0);
                
                // Create a new model
                if (tokens[2] == "uniformBSpline") {
                  // Create a new uniform B-Spline model

                  try {
                    // nlohmann::json data = request["data"];
                    // short_t geoDim = data["geoDim"];
                    // short_t init   = data["init"];
                    // std::vector<short_t> parDim  = data["parDim"];
                    // std::vector<int64_t> ncoeffs = data["ncoeffs"];

                    session->models[id] = std::make_shared<iganet::UniformBSpline<double,1,1,1>>(iganet::UniformBSpline<double,1,1,1>({5,6}));
                    response["data"]["id"] = std::to_string(id);
                    ws->send(response.dump(), uWS::OpCode::TEXT, true);
                    
                  } catch(...) {
                    response["status"] = 1;
                    response["reason"] = "Malformed create request";
                    ws->send(response.dump(), uWS::OpCode::TEXT, true);
                  }
                }
                else if (tokens[2] == "nonuniformBSpline") {
                  // Create a new non-uniform B-Spline model

                  try {
                    // nlohmann::json data = request["data"];
                    // short_t geoDim = data["geoDim"];
                    // short_t init   = data["init"];
                    // std::vector<short_t> parDim  = data["parDim"];
                    // std::vector<int64_t> ncoeffs = data["ncoeffs"];
                    
                    session->models[id] = std::make_shared<iganet::UniformBSpline<double,1,1,1>>(iganet::UniformBSpline<double,1,1,1>({5,6}));
                    response["data"]["id"] = std::to_string(id);
                    ws->send(response.dump(), uWS::OpCode::TEXT, true);
                    
                  } catch(...) {
                    response["status"] = 1;
                    response["reason"] = "Malformed create request";
                    ws->send(response.dump(), uWS::OpCode::TEXT, true);
                  }
                }
                else {
                  response["status"] = 1;
                  response["reason"] = "\"" + tokens[2] + "\" is not a valid model type";
                  ws->send(response.dump(), uWS::OpCode::TEXT, true);
                }
              }
              
              else {
                response["status"] = 1;
                response["reason"] = "Invalid create request. Valid requests are \"create/session\" and \"create/<session-uuid>/<model-type>\"";
                ws->send(response.dump(), uWS::OpCode::TEXT, true);
              }                             
            }
            
            else if (tokens[0] == "remove") {
              //
              // request: remove/*
              //

              if (tokens.size() == 2) {
                //
                // request: remove/<session-uuid>
                //

                // Remove session
                auto session = ws->getUserData()->removeSession(tokens[1]);
                ws->send(response.dump(), uWS::OpCode::TEXT, true);
              }

              else if (tokens.size() == 3) {
                //
                // request: remove/<session-uuid>/<model-id>
                //

                // Get session
                auto session = ws->getUserData()->getSession(tokens[1]);
                //session->authenticate(request["token"]);

                // Remove model
                auto model = session->removeModel(stoi(tokens[2]));
                ws->send(response.dump(), uWS::OpCode::TEXT, true);
              }

              else {
                response["status"] = 1;
                response["reason"] = "Invalid remove request. Valid requests are \"remove/<session-uuid>\" and \"remove/<session-uuid>/<model-id>\"";
                ws->send(response.dump(), uWS::OpCode::TEXT, true);
              }
                            
            }
            
            else if (tokens[0] == "connect") {
              //
              // request: connect/*
              //

              if (tokens.size() == 2) {
                //
                // request: connect/<session-uuid>
                //

                // Get session
                auto session = ws->getUserData()->getSession(tokens[1]);
                //session->authenticate(request["token"]);

                // Connect to an existing session

                // TODO
              }

              else {
                response["status"] = 1;
                response["reason"] = "Invalid connect request. Valid requests are \"connect/<session-uuid>\"";
                ws->send(response.dump(), uWS::OpCode::TEXT, true);
              }              
            }
            
            else if (tokens[0] == "disconnect") {
              //
              // request: diconnect/*
              //

              if (tokens.size() == 2) {
                //
                // request: diconnect/<session-uuid>
                //

                // Get session
                auto session = ws->getUserData()->getSession(tokens[1]);
                //session->authenticate(request["token"]);

                // Disconnect from an existing session

                // TODO
              }

              else {
                response["status"] = 1;
                response["reason"] = "Invalid disconnect request. Valid requests are \"diconnect/<session-uuid>\"";
                ws->send(response.dump(), uWS::OpCode::TEXT, true);
              }              
            }
            
            else if (tokens[0] == "eval") {
              //
              // request: eval/*
              //

              if (tokens.size() == 3) {
                //
                // request: eval/<session-uuid>/<model-id>
                //

                // Get session
                auto session = ws->getUserData()->getSession(tokens[1]);
                //session->authenticate(request["token"]);

                // Get model
                auto model = session->getModel(stoi(tokens[2]));
                
                // Evaluate an existing model
                if (auto spline = std::dynamic_pointer_cast<iganet::UniformBSpline<double,1,1,1>>(model)) {
                  switch (spline->parDim()) {
                  // case 1:
                  //   iganet::TensorArray1 xi1 = {torch::linspace(0,1,100)};
                  //   auto data1 = spline->eval(xi1);
                  //   response["data"] = ::iganet::to_json<double,1>(*data1[0]);
                  //   ws->send(response.dump(), uWS::OpCode::TEXT, true);
                  //   break;
                    
                  case 2:
                    iganet::TensorArray2 xi2 = {torch::linspace(0,1,100),
                                                torch::linspace(0,1,100)};
                    auto data2 = spline->eval(xi2);
                    response["data"] = ::iganet::to_json<double,1>(*data2[0]);
                    ws->send(response.dump(), uWS::OpCode::TEXT, true);
                    break;

                  // case 3:
                  //   iganet::TensorArray3 xi3 = {torch::linspace(0,1,100),
                  //                               torch::linspace(0,1,100),
                  //                               torch::linspace(0,1,100)};
                  //   auto data3 = spline->eval(xi3);
                  //   break;

                  // case 4:
                  //   iganet::TensorArray4 xi4 = {torch::linspace(0,1,100),
                  //                               torch::linspace(0,1,100),
                  //                               torch::linspace(0,1,100),
                  //                               torch::linspace(0,1,100)};
                  //   auto data4 = spline->eval(xi4);
                  //   break;
                  }
                }
              }
              else {
                response["status"] = 1;
                response["reason"] = "Invalid eval request. Valid requests are \"eval/<session-uuid>/<model-id>\"";
                ws->send(response.dump(), uWS::OpCode::TEXT, true);
              }              
            }
            
            else if (tokens[0] == "refine") {
              //
              // request: refine/*
              //

              if (tokens.size() == 3) {
                //
                // request: refine/<session-uuid>/<model-id>
                //

                // Get session
                auto session = ws->getUserData()->getSession(tokens[1]);
                //session->authenticate(request["token"]);

                // Get model
                auto model = session->getModel(stoi(tokens[2]));

                // Refine an existing model

                // TODO
              }

              else {
                response["status"] = 1;
                response["reason"] = "Invalid refine request. Valid requests are \"refine/<session-uuid>/<model-id>\"";
                ws->send(response.dump(), uWS::OpCode::TEXT, true);
              }              
            }

            else {
                response["status"] = 1;
                response["reason"] = "Invalid request";
                ws->send(response.dump(), uWS::OpCode::TEXT, true);
            }             
          }
          catch (std::exception& e) {
            nlohmann::json response;
            try {
              auto request = nlohmann::json::parse(message);
              response["request"] = request["id"];
              response["status"]  = 1;
              response["reason"]  = e.what();
              ws->send(response.dump(), uWS::OpCode::TEXT, true);
            } catch(...) {
              response["request"] = "unknown";
              response["status"]  = 1;
              response["reason"]  = "Malformed request";
              ws->send(response.dump(), uWS::OpCode::TEXT, true);
            }            
          }
        },
        .drain = [](auto *ws) {
          /* Check ws->getBufferedAmount() here */
        },
        .ping = [](auto *ws, std::string_view) {
          /* Not implemented yet */
        },
        .pong = [](auto *ws, std::string_view) {
          /* Not implemented yet */
        },
        .close = [](auto *ws, int code, std::string_view message) {
          /* You may access ws->getUserData() here */
          std::clog << "Connection has been closed\n";
        }
       }).listen(port_option->value(), [&port_option](auto *listen_socket) {
        if (listen_socket) {
          std::clog << "Listening on port " << port_option->value() << std::endl;
        }
    }).run();
}
