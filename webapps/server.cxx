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
#include <modelmanager.hpp>
#include <popl.hpp>

namespace iganet { namespace webapp {

    /// @brief Enumerator for specifying the status
    enum class status : short_t
      {
        success                  =  0, /*!<  request was handled successfully */        
        invalidRequest           =  1, /*!<  invalid request                  */
        invalidCreateRequest     =  2, /*!<  invalid create request           */
        invalidRemoveRequest     =  3, /*!<  invalid remove request           */
        invalidConnectRequest    =  4, /*!<  invalid connect request          */
        invalidDisconnectRequest =  5, /*!<  invalid disconnect request       */
        invalidGetRequest        =  6, /*!<  invalid get request              */
        invalidPutRequest        =  7, /*!<  invalid put request              */
        invalidEvalRequest       =  8, /*!<  invalid eval request             */
        invalidRefineRequest     =  9  /*!<  invalid refine request           */
      };
    
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
      std::shared_ptr<Model> getModel(int64_t id) {
        auto it = models.find(id);
        if (it == models.end())
          throw InvalidModelIdException();
        else
          return it->second;
      }

      /// Returns the model and removes it from the list of models
      std::shared_ptr<Model> removeModel(int64_t id) {
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
      std::map<int64_t, std::shared_ptr<Model>> models;
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

      /// List of sessions shared between all sockets
      inline static std::map<std::string, std::shared_ptr<Session<T>>> sessions;

      /// List of models
      inline static ModelManager models = ModelManager("webapps/models");
    };

}} // namespace iganet::webapp


int main(int argc, char const* argv[])
{
  using PerSocketData = iganet::webapp::Sessions<float>;

  popl::OptionParser op("Allowed options");
  auto help_option = op.add<popl::Switch>("h", "help", "print help message");
  auto port_option = op.add<popl::Value<int>>("p", "port", "TCP port of the server", 9001);
  auto key_file_option = op.add<popl::Value<std::string>>("k", "keyfile", "key file for SSL encryption", "key.pem");
  auto cert_file_option = op.add<popl::Value<std::string>>("c", "certfile", "certificate file for SSL encryption", "cert.pem");
  auto passphrase_option = op.add<popl::Value<std::string>>("a", "passphrase", "passphrase for SSL encryption", "");
  op.parse(argc, argv);

  // Print auto-generated help message
  if (help_option->count() == 1)
    std::cout << op << std::endl;
  else if (help_option->count() == 2)
    std::cout << op.help(popl::Attribute::advanced) << std::endl;
  else if (help_option->count() > 2)
    std::cout << op.help(popl::Attribute::expert) << std::endl;

  // Initialize backend
  iganet::init();

  // Create WebSocket application
  try {
    uWS::SSLApp({
        .key_file_name  = key_file_option->value().c_str(),
        .cert_file_name = cert_file_option->value().c_str(),
        .passphrase     = passphrase_option->value().c_str()
        }).ws<PerSocketData>("/*", {
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
        },
        .message = [](auto *ws, std::string_view message, uWS::OpCode opCode) {
          try {
            // Tokenize request
            auto request = nlohmann::json::parse(message);
            auto tokens  = iganet::webapp::tokenize(request["request"].get<std::string>());

            // Prepare response
            nlohmann::json response;
            response["request"] = request["id"];
            response["status"]  = iganet::webapp::status::success;

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
                // request: get/<session-id>
                //

                // Get session
                auto session = ws->getUserData()->getSession(tokens[1]);

                // Get list of all active models in session
                std::vector<int64_t> ids;
                for (const auto& model : session->models)
                  ids.push_back(model.first);
                response["data"] = ids;
                ws->send(response.dump(), uWS::OpCode::TEXT, true);
              }

              else if (tokens.size() == 3) {
                //
                // request: get/<session-id>/<model-id>
                //

                // Get session
                auto session = ws->getUserData()->getSession(tokens[1]);

                // Get model
                auto model = session->getModel(stoi(tokens[2]));

                // Serialize model to JSON
                response["data"] = model->to_json();
                ws->send(response.dump(), uWS::OpCode::TEXT, true);
              }

              else if (tokens.size() == 4) {
                //
                // request: get/<session-id>/<model-id>/<attribute>
                //

                // Get session
                auto session = ws->getUserData()->getSession(tokens[1]);

                // Get model
                auto model = session->getModel(stoi(tokens[2]));

                // Serialize model attribute to JSON
                response["data"] = model->to_json(tokens[3]);
                ws->send(response.dump(), uWS::OpCode::TEXT, true);
              }

              else {
                response["status"] = iganet::webapp::status::invalidGetRequest;
                response["reason"] = "Invalid get request. Valid requests are \"get\", \"get/<session-id>\", \"get/<session-id>/<model-id>\", and \"get/<session-id>/<model-id>/<attribute>\"";
                ws->send(response.dump(), uWS::OpCode::TEXT, true);
              }

            }

            else if (tokens[0] == "put") {
              //
              // request: put/*
              //

              if (tokens.size() == 4) {
                //
                // request: put/<session-id>/<model-id>/<attribute>
                //

                // Get session
                auto session = ws->getUserData()->getSession(tokens[1]);

                // Get model
                auto model = session->getModel(stoi(tokens[2]));

                // Update model attribute
                response["data"] = model->updateAttribute(tokens[3], request);
                ws->send(response.dump(), uWS::OpCode::TEXT, true);

                // Broadcast update of model
                nlohmann::json broadcast;
                broadcast["id"] = session->getUUID();
                broadcast["request"] = "update/model";
                broadcast["data"]["id"] = stoi(tokens[2]);
                broadcast["data"]["attribute"] = tokens[3];
                ws->publish(session->getUUID(), broadcast.dump(), uWS::OpCode::TEXT);
              }

              else {
                response["status"] = iganet::webapp::status::invalidPutRequest;
                response["reason"] = "Invalid put request. Valid requests are \"put/<session-id>/<model-id>/<attribute>\"";
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
                auto session = std::make_shared<iganet::webapp::Session<float>>();
                std::string uuid = session->getUUID();
                ws->getUserData()->sessions[uuid] = session;
                response["data"]["id"] = uuid;
                response["data"]["models"] = ws->getUserData()->models.getModels();
                ws->send(response.dump(), uWS::OpCode::TEXT, true);

                // Subscribe to new session
                ws->subscribe(uuid);
              }

              else if (tokens.size() == 3) {
                //
                // request: create/<session-id>/<model-type>
                //

                // Get session
                auto session = ws->getUserData()->getSession(tokens[1]);

                // Create new model
                int64_t id = (session->models.size() > 0 ?
                              session->models.crbegin()->first+1 : 0);

                try {
                  // Create a new model
                  session->models[id] = ws->getUserData()->models.create(tokens[2], request);
                  response["data"]["id"] = std::to_string(id);
                  ws->send(response.dump(), uWS::OpCode::TEXT, true);
                  
                  // Broadcast creation of a new model
                  nlohmann::json broadcast;
                  broadcast["id"] = session->getUUID();
                  broadcast["request"] = "create/model";
                  broadcast["data"]["id"] = id;                    
                  ws->publish(session->getUUID(), broadcast.dump(), uWS::OpCode::TEXT);
                } catch(...) {
                  response["status"] = iganet::webapp::status::invalidCreateRequest;
                  response["reason"] = "Invalid create request. Valid requests are \"create/session\" and \"create/<session-id>/<model-type>\"";
                  ws->send(response.dump(), uWS::OpCode::TEXT, true);
                }
              }

              else {
                response["status"] = iganet::webapp::status::invalidCreateRequest;
                response["reason"] = "Invalid create request. Valid requests are \"create/session\" and \"create/<session-id>/<model-type>\"";
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
                auto session = ws->getUserData()->removeSession(tokens[1]);
                ws->send(response.dump(), uWS::OpCode::TEXT, true);

                // Broadcast removal of session
                nlohmann::json broadcast;
                broadcast["id"] = session->getUUID();
                broadcast["request"] = "remove/session";
                broadcast["data"]["id"] = session->getUUID();
                ws->publish(session->getUUID(), broadcast.dump(), uWS::OpCode::TEXT);
              }

              else if (tokens.size() == 3) {
                //
                // request: remove/<session-id>/<model-id>
                //

                // Get session
                auto session = ws->getUserData()->getSession(tokens[1]);

                // Remove model
                auto model = session->removeModel(stoi(tokens[2]));
                ws->send(response.dump(), uWS::OpCode::TEXT, true);

                // Broadcast removal of model
                nlohmann::json broadcast;
                broadcast["id"] = session->getUUID();
                broadcast["request"] = "remove/model";
                broadcast["data"]["id"] = stoi(tokens[2]);
                ws->publish(session->getUUID(), broadcast.dump(), uWS::OpCode::TEXT);
              }

              else {
                response["status"] = iganet::webapp::status::invalidRemoveRequest;
                response["reason"] = "Invalid remove request. Valid requests are \"remove/<session-id>\" and \"remove/<session-id>/<model-id>\"";
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
                auto session = ws->getUserData()->getSession(tokens[1]);

                // Connect to an existing session
                response["data"]["id"] = session->getUUID();
                response["data"]["models"] = ws->getUserData()->models.getModels();
                ws->send(response.dump(), uWS::OpCode::TEXT, true);
                
                // Subscribe to existing session
                ws->subscribe(session->getUUID());
              }

              else {
                response["status"] = iganet::webapp::status::invalidConnectRequest;
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
                auto session = ws->getUserData()->getSession(tokens[1]);

                // Disconnect from an existing session
                ws->unsubscribe(session->getUUID());
              }

              else {
                response["status"] = iganet::webapp::status::invalidDisconnectRequest;
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
                // request: eval/<session-id>/<model-id>
                //

                // Get session
                auto session = ws->getUserData()->getSession(tokens[1]);

                // Get model
                auto model = session->getModel(stoi(tokens[2]));

                // Evaluate an existing model
                if (auto m = std::dynamic_pointer_cast<iganet::ModelEval<1>>(model))
                  response["data"] = nlohmann::json::array()
                    .emplace_back(::iganet::to_json<float,1>(*(m->eval(request))[0]));
                else {
                  response["status"] = iganet::webapp::status::invalidEvalRequest;
                  response["reason"] = "Invalid eval request. Valid requests are \"eval/<session-id>/<model-id>\"";
                }
                ws->send(response.dump(), uWS::OpCode::TEXT, true);
              }
              else {
                response["status"] = iganet::webapp::status::invalidEvalRequest;
                response["reason"] = "Invalid eval request. Valid requests are \"eval/<session-id>/<model-id>\"";
                ws->send(response.dump(), uWS::OpCode::TEXT, true);
              }
            }

            else if (tokens[0] == "refine") {
              //
              // request: refine/*
              //

              if (tokens.size() == 3) {
                //
                // request: refine/<session-id>/<model-id>
                //

                // Get session
                auto session = ws->getUserData()->getSession(tokens[1]);

                // Get model
                auto model = session->getModel(stoi(tokens[2]));

                // Refine an existing model
                if (auto m = std::dynamic_pointer_cast<iganet::ModelRefine>(model))
                  m->refine(request);
                else {
                  response["status"] = iganet::webapp::status::invalidRefineRequest;
                  response["reason"] = "Invalid refine request. Valid requests are \"refine/<session-id>/<model-id>\"";
                }
                ws->send(response.dump(), uWS::OpCode::TEXT, true);

                // Broadcast refinement of model
                nlohmann::json broadcast;
                broadcast["id"] = session->getUUID();
                broadcast["request"] = "refine/model";
                broadcast["data"]["id"] = stoi(tokens[2]);
                ws->publish(session->getUUID(), broadcast.dump(), uWS::OpCode::TEXT);
              }

              else {
                response["status"] = iganet::webapp::status::invalidRefineRequest;
                response["reason"] = "Invalid refine request. Valid requests are \"refine/<session-id>/<model-id>\"";
                ws->send(response.dump(), uWS::OpCode::TEXT, true);
              }
            }

            else {
              response["status"] = iganet::webapp::status::invalidRequest;
                response["reason"] = "Invalid request";
                ws->send(response.dump(), uWS::OpCode::TEXT, true);
            }
          }
          catch (std::exception& e) {
            nlohmann::json response;
            try {
              auto request = nlohmann::json::parse(message);
              response["request"] = request["id"];
              response["status"]  = iganet::webapp::status::invalidRequest;
              response["reason"]  = e.what();
              ws->send(response.dump(), uWS::OpCode::TEXT, true);
            } catch(...) {
              response["request"] = "unknown";
              response["status"]  = iganet::webapp::status::invalidRequest;
              response["reason"]  = "Invalid request";
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
  } catch (std::exception& e) {
    std::cerr << e.what();
  }
  
  return 0;
}
