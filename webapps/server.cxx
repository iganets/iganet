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

    /// @brief InvalidAuthentication exception
    struct InvalidAuthenticationException : public std::exception {  
      const char * what() const throw() {  
        return "Invalid authentication token";  
      }
    };
    
    /// @brief InvalidObjectId exception
    struct InvalidObjectIdException : public std::exception {  
      const char * what() const throw() {  
        return "Invalid object id";  
      }
    };

    /// @brief InvalidObjectType exception
    struct InvalidObjectTypeException : public std::exception {  
      const char * what() const throw() {  
        return "Invalid object type";  
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
      /// @brief Session token
      const std::string token;
            
    public:
      /// @brief Default constructor
      Session() : token(iganet::uuid::create())
      {}

      /// @brief Returns the token
      const std::string& getToken() const {
        return token;
      }
      
      /// @brief Returns true if the given token is valid
      bool validToken(const std::string& token) const {        
        return (token == this->token);
      }

      /// @brief Throws an exception of authentication is not valid
      void authenticate(const std::string& token) const {
        if (!validToken(token))
          throw InvalidAuthenticationException();
      }

      /// Returns the requested object or throws an exception
      std::shared_ptr<iganet::core<T>> getObject(int64_t id) {
        auto it = objects.find(id);
        if (it == objects.end())
          throw InvalidObjectIdException();
        else
          return it->second;
      }

      /// Returns the object and removes it from the list of objects
      std::shared_ptr<iganet::core<T>> removeObject(int64_t id) {
        auto it = objects.find(id);
        if (it == objects.end())
          throw InvalidObjectIdException();
        else {
          auto object = it->second;
          objects.erase(it);
          return object;
        }
      }
      
      /// @brief List of objects
      std::map<int64_t, std::shared_ptr<iganet::core<T>>> objects;
    };
    
    /// @brief Sessions structure
    template<typename T>
    struct Sessions {
    public:     
      /// Returns the requested session object or throws an exception
      std::shared_ptr<Session<T>> getSession(int64_t id) {
        auto it = sessions.find(id);
        if (it == sessions.end())
          throw InvalidSessionIdException();
        else
          return it->second;
      }

      /// Returns the session and removes it from the list of sessions
      std::shared_ptr<Session<T>> removeSession(int64_t id, const std::string& token) {
        auto it = sessions.find(id);
        if (it == sessions.end())
          throw InvalidSessionIdException();
        else {
          auto session = it->second;
          session->authenticate(token);
          sessions.erase(it);
          return session;
        }
      }
      
      /// Static list of sessions shared between all sockets
      inline static std::map<int64_t, std::shared_ptr<Session<T>>> sessions;
    };
    
}} // namespace iganet::webapp


int main(int argc, char const* argv[])
{
  // iganet::JITCompiler compiler;
  // compiler << "#include <iganet.hpp>\n"
  //          << "namespace iganet {\n"
  //          << "EXPORT std::shared_ptr<iganet::core<double>> create()\n"
  //          << "{ return std::make_shared<iganet::core<double>>(iganet::uniformBSpline<double,1,1>({5})); }"
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
  //   throw std::runtime_error("An error occured while compiling the quantum expression");
  // }
  
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
                std::vector<int64_t> ids;
                for (const auto& session : ws->getUserData()->sessions)
                  ids.push_back(session.first);
                response["data"] = ids;
                ws->send(response.dump(), uWS::OpCode::TEXT, true);
              }

              else if (tokens.size() == 2) {
                //
                // request: get/<session-id>
                //

                // Get session
                auto session = ws->getUserData()->getSession(stoi(tokens[1]));
                session->authenticate(request["token"]);

                // Get list of all active objects in session
                std::vector<int64_t> ids;
                for (const auto& object : session->objects)
                  ids.push_back(object.first);
                response["data"] = ids;
                ws->send(response.dump(), uWS::OpCode::TEXT, true);
              }

              else if (tokens.size() == 3) {
                //
                // request: get/<session-id>/<object-id>
                //

                // Get session
                auto session = ws->getUserData()->getSession(stoi(tokens[1]));
                session->authenticate(request["token"]);

                // Get object
                auto object = session->getObject(stoi(tokens[2]));
                
                // Serialize object to JSON
                response["data"] = object->to_json();
                ws->send(response.dump(), uWS::OpCode::TEXT, true);
              }

              else if (tokens.size() == 4) {
                //
                // request: get/<session-id>/<object-id>/<attribute>
                //

                // Get session
                auto session = ws->getUserData()->getSession(stoi(tokens[1]));
                session->authenticate(request["token"]);

                // Get object
                auto object = session->getObject(stoi(tokens[2]));
                
                // Serialize object to JSON
                response["data"] = "Not implemented yet (get/<session-id>/<object-id>/<attribute>)";
                ws->send(response.dump(), uWS::OpCode::TEXT, true);
              }

              else {
                response["status"] = 1;
                response["reason"] = "Invalid get request. Valid requests are \"get\", \"get/<session-id>\", \"get/<session-id>/<object-id>\", and \"get/<session-id>/<object-id>/<attribute>\"";
                ws->send(response.dump(), uWS::OpCode::TEXT, true);                
              }
                
            }
            
            else if (tokens[0] == "put") {
              //
              // request: put/*
              //

              if (tokens.size() == 3) {
                //
                // request: put/<session-id>/<object-id>/<attribute>
                //

                // Get session
                auto session = ws->getUserData()->getSession(stoi(tokens[1]));
                session->authenticate(request["token"]);

                // Get object
                auto object = session->getObject(stoi(tokens[2]));

                response["data"] = "Not implemented yet (put/<session-id>/<object-id>/<attribute>)";
                ws->send(response.dump(), uWS::OpCode::TEXT, true);
              }
              
              else {
                response["status"] = 1;
                response["reason"] = "Invalid put request. Valid requests are \"put/<session-id>/<object-id>/<attribute>\"";
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
                int64_t id = (ws->getUserData()->sessions.size() > 0 ?
                              ws->getUserData()->sessions.crbegin()->first+1 : 0);
                ws->getUserData()->sessions[id] = std::make_shared<iganet::webapp::Session<double>>();
                response["data"]["id"] = std::to_string(id);
                response["data"]["token"] = ws->getUserData()->sessions[id]->getToken();
                ws->send(response.dump(), uWS::OpCode::TEXT, true);                
              }
              
              else if (tokens.size() == 3) {
                //
                // request: create/<session-id>/<object-type>
                //

                // Get session
                auto session = ws->getUserData()->getSession(stoi(tokens[1]));
                session->authenticate(request["token"]);
                
                // Create new object
                int64_t id = (session->objects.size() > 0 ?
                              session->objects.crbegin()->first+1 : 0);
                
                // Create a new object
                if (tokens[2] == "uniformBSpline") {
                  // Create a new uniform B-Spline object

                  try {
                    // nlohmann::json data = request["data"];
                    // short_t geoDim = data["geoDim"];
                    // short_t init   = data["init"];
                    // std::vector<short_t> parDim  = data["parDim"];
                    // std::vector<int64_t> ncoeffs = data["ncoeffs"];

                    session->objects[id] = std::make_shared<iganet::UniformBSpline<double,1,1,1>>(iganet::UniformBSpline<double,1,1,1>({5,6}));
                    response["data"]["id"] = std::to_string(id);
                    ws->send(response.dump(), uWS::OpCode::TEXT, true);
                    
                  } catch(...) {
                    response["status"] = 1;
                    response["reason"] = "Malformed create request";
                    ws->send(response.dump(), uWS::OpCode::TEXT, true);
                  }
                }
                else if (tokens[2] == "nonuniformBSpline") {
                  // Create a new non-uniform B-Spline object

                  try {
                    // nlohmann::json data = request["data"];
                    // short_t geoDim = data["geoDim"];
                    // short_t init   = data["init"];
                    // std::vector<short_t> parDim  = data["parDim"];
                    // std::vector<int64_t> ncoeffs = data["ncoeffs"];
                    
                    session->objects[id] = std::make_shared<iganet::UniformBSpline<double,1,1,1>>(iganet::UniformBSpline<double,1,1,1>({5,6}));
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
                  response["reason"] = "\"" + tokens[2] + "\" is not a valid object type";
                  ws->send(response.dump(), uWS::OpCode::TEXT, true);
                }
              }
              
              else {
                response["status"] = 1;
                response["reason"] = "Invalid create request. Valid requests are \"create/session\" and \"create/<session-id>/<object-type>\"";
                ws->send(response.dump(), uWS::OpCode::TEXT, true);
              }                             
            }
            
            else if (tokens[0] == "remove") {
              //
              // request: remove/*
              //

              if (tokens.size() == 2) {
                //
                // request: remove/<session-id>
                //

                // Remove session
                auto session = ws->getUserData()->removeSession(stoi(tokens[1]), request["token"]);
                ws->send(response.dump(), uWS::OpCode::TEXT, true);
              }

              else if (tokens.size() == 3) {
                //
                // request: remove/<session-id>/<object-id>
                //

                // Get session
                auto session = ws->getUserData()->getSession(stoi(tokens[1]));
                session->authenticate(request["token"]);

                // Remove object
                auto object = session->removeObject(stoi(tokens[2]));
                ws->send(response.dump(), uWS::OpCode::TEXT, true);
              }

              else {
                response["status"] = 1;
                response["reason"] = "Invalid remove request. Valid requests are \"remove/<session-id>\" and \"remove/<session-id>/<object-id>\"";
                ws->send(response.dump(), uWS::OpCode::TEXT, true);
              }
                            
            }
            
            else if (tokens[0] == "connect") {
              //
              // request: connect/*
              //

              if (tokens.size() == 2) {
                //
                // request: connect/<session-id>
                //

                // Get session
                auto session = ws->getUserData()->getSession(stoi(tokens[1]));
                session->authenticate(request["token"]);

                // Connect to an existing session

                // TODO
              }

              else {
                response["status"] = 1;
                response["reason"] = "Invalid connect request. Valid requests are \"connect/<session-id>\"";
                ws->send(response.dump(), uWS::OpCode::TEXT, true);
              }              
            }
            
            else if (tokens[0] == "disconnect") {
              //
              // request: diconnect/*
              //

              if (tokens.size() == 2) {
                //
                // request: diconnect/<session-id>
                //

                // Get session
                auto session = ws->getUserData()->getSession(stoi(tokens[1]));
                session->authenticate(request["token"]);

                // Disconnect from an existing session

                // TODO
              }

              else {
                response["status"] = 1;
                response["reason"] = "Invalid disconnect request. Valid requests are \"diconnect/<session-id>\"";
                ws->send(response.dump(), uWS::OpCode::TEXT, true);
              }              
            }
            
            else if (tokens[0] == "eval") {
              //
              // request: eval/*
              //

              if (tokens.size() == 3) {
                //
                // request: eval/<session-id>/<object-id>
                //

                // Get session
                auto session = ws->getUserData()->getSession(stoi(tokens[1]));
                session->authenticate(request["token"]);

                // Get object
                auto object = session->getObject(stoi(tokens[2]));
                
                // Evaluate an existing object
                if (auto spline = std::dynamic_pointer_cast<iganet::UniformBSpline<double,1,1,1>>(object)) {
                  switch (spline->parDim()) {
                  case 1:
                    iganet::TensorArray1 xi1 = {torch::linspace(0,1,100)};
                    auto data1 = spline->eval(xi1);
                    response["data"] = ::iganet::to_json<double,1>(*data1[0]);
                    ws->send(response.dump(), uWS::OpCode::TEXT, true);
                    break;
                    
                  // case 2:
                  //   iganet::TensorArray2 xi2 = {torch::linspace(0,1,100),
                  //                               torch::linspace(0,1,100)};
                  //   auto data2 = spline->eval(xi2);
                  //   response["data"] = ::iganet::to_json<double,1>(*data2[0]);
                  //   ws->send(response.dump(), uWS::OpCode::TEXT, true);
                  //   break;

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
                response["reason"] = "Invalid eval request. Valid requests are \"eval/<session-id>/<object-id>\"";
                ws->send(response.dump(), uWS::OpCode::TEXT, true);
              }              
            }
            
            else if (tokens[0] == "refine") {
              //
              // request: refine/*
              //

              if (tokens.size() == 3) {
                //
                // request: refine/<session-id>/<object-id>
                //

                // Get session
                auto session = ws->getUserData()->getSession(stoi(tokens[1]));
                session->authenticate(request["token"]);

                // Get object
                auto object = session->getObject(stoi(tokens[2]));

                // Refine an existing object

                // TODO
              }

              else {
                response["status"] = 1;
                response["reason"] = "Invalid refine request. Valid requests are \"refine/<session-id>/<object-id>\"";
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
