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
#include <popl.hpp>
#include <iostream>
#include <tuple>
#include <vector>

namespace iganet { namespace webapp {

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
      
      /// @brief List of objects
      std::map<int64_t, std::shared_ptr<iganet::core<T>>> objects;
    };
    
    /// @brief Sessions structure
    template<typename T>
    struct Sessions {
      /// Static list of sessions shared between all sockets
      inline static std::map<int64_t, std::shared_ptr<Session<T>>> sessions;
    };
    
}} // namespace iganet::webapp


int main(int argc, char const* argv[])
{
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

                // Authentification
                
                // Get list of all active objects of a specific session
                std::vector<int64_t> ids;
                for (const auto& object : ws->getUserData()->sessions.find(stoi(tokens[1]))->second->objects)
                  ids.push_back(object.first);
                response["data"] = ids;
                ws->send(response.dump(), uWS::OpCode::TEXT, true);
              }

              else if (tokens.size() == 3) {
                //
                // request: get/<session-id>/<object-id>
                //
                
                // Authentification

                // Serialize object to JSON
                response["data"] = ws->getUserData()->sessions.find(stoi(tokens[1]))->second->
                  objects.find(stoi(tokens[2]))->second->to_json();
                ws->send(response.dump(), uWS::OpCode::TEXT, true);
              }

              else if (tokens.size() == 4) {
                //
                // request: get/<session-id>/<object-id>/<attribute>
                //
                
                // Authentification

                // Serialize object's attribute to JSON
                if(auto object = ws->getUserData()->sessions.find(stoi(tokens[1]))->second->
                   objects.find(stoi(tokens[2]))->second; object != nullptr) {

                  // TODO
                }
                ws->send(response.dump(), uWS::OpCode::TEXT, true);
              }

              else {
                response["status"]  = 1;
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

                // Authentication

                // Update object's attribute from JSON
                // TODO
                
                ws->send(response.dump(), uWS::OpCode::TEXT, true);
              }
              
              else {
                response["status"]  = 1;
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

                // Authentication
                int64_t id = stoi(tokens[1]);
                
                // Create a new object
                if (tokens[2] == "uniformBSpline") {
                  // Create a new uniform B-Spline object
                }
                else if (tokens[2] == "nonuniformBSpline") {
                  // Create a new non-uniform B-Spline object
                }
                else {
                  response["status"]  = 1;
                  response["reason"]  = "\"" + tokens[2] + "\" is not a valid object type";
                }
              }
              else {
                response["status"]  = 1;
                response["reason"]  = "Invalid create request. Valid requests are \"create\" and \"create/<session-id>/<object-type>\"";
              }
              ws->send(response.dump(), uWS::OpCode::TEXT, true);
                
            }
            else if (tokens[0] == "remove") {
            }
            else if (tokens[0] == "connect") {
            }
            else if (tokens[0] == "disconnect") {
            }
            else if (tokens[0] == "eval") {
            }
            else if (tokens[0] == "refine") {
            }
                      
            // if (request["cmd"] == "get_geo") {
            //   nlohmann::json json;
            //   json["cmd"] = "put_geo";
            //   json["data"] = ws->getUserData()->geo.to_json();
            //   ws->send(json.dump(), uWS::OpCode::TEXT, true);
            // }
            // else if (request["cmd"] == "get_sol") {              
            //   iganet::TensorArray2 xi = {torch::linspace(0,1,100), torch::linspace(0,1,100)};
            //   auto sol = ws->getUserData()->sol.eval(xi);
              
            //   nlohmann::json json;
            //   json["cmd"] = "put_sol";
            //   json["data"] = ::iganet::to_json<double,1>(*sol[0]);
            //   ws->send(json.dump(), uWS::OpCode::TEXT, true);
            // }
            // else {
            //   nlohmann::json json;
            //   json["cmd"] = "error";
            //   json["msg"] = "Invalid request '" + request["cmd"].dump() + "'";
            //   ws->send(json.dump(), uWS::OpCode::TEXT, true);
            // }
          }
          catch (...) {
            nlohmann::json response;
            auto request = nlohmann::json::parse(message);
            response["request"] = request["id"];
            response["status"]  = 1;
            response["reason"] = "Undefined error";
            ws->send(response.dump(), uWS::OpCode::TEXT, true);
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
          std::cout << "Connection has been closed\n";
        }
       }).listen(port_option->value(), [&port_option](auto *listen_socket) {
        if (listen_socket) {
          std::clog << "Listening on port " << port_option->value() << std::endl;
        }
    }).run();
}
