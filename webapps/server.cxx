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

int main(int argc, char const* argv[])
{
  struct PerSocketData {
    // Bivariate uniform B-spline of degree 3 in xi-direction and 4
    // in eta-direction with 5 x 6 control points in R^3
    iganet::UniformBSpline<double,3,3,4> geo;
    iganet::UniformBSpline<double,1,3,4> sol;
    std::string uuid;
    
    PerSocketData()
      : geo({5,6}), sol({5,6}), uuid(iganet::uuid::create())
    {
      // Map control points of geometry
      geo.transform( [](const std::array<double,2> xi){ return std::array<double,3>{(xi[0]+1)*cos(M_PI*xi[1]),
                                                                                    (xi[0]+1)*sin(M_PI*xi[1]),
                                                                                    xi[0] }; } );
      
      // Map control points of solution
      sol.transform( [](const std::array<double,2> xi){ return std::array<double,1>{ xi[0]*xi[1] }; } );
    }
  };

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
            nlohmann::json data = nlohmann::json::parse(message);
            if (data["cmd"] == "get_geo") {
              nlohmann::json json;
              json["cmd"] = "put_geo";
              json["data"] = ws->getUserData()->geo.to_json();
              ws->send(json.dump(), uWS::OpCode::TEXT, true);
            }
            else if (data["cmd"] == "get_sol") {              
              iganet::TensorArray2 xi = {torch::linspace(0,1,100), torch::linspace(0,1,100)};
              auto sol = ws->getUserData()->sol.eval(xi);
              
              nlohmann::json json;
              json["cmd"] = "put_sol";
              json["data"] = ::iganet::to_json<double,1>(*sol[0]);
              ws->send(json.dump(), uWS::OpCode::TEXT, true);
            }
            else {
              nlohmann::json json;
              json["cmd"] = "error";
              json["msg"] = "Invalid command '" + data["cmd"].dump() + "'";
              ws->send(json.dump(), uWS::OpCode::TEXT, true);
            }
          }
          catch (...) {
            nlohmann::json json;
            json["cmd"] = "error";
            json["msg"] = "Invalid message '" + std::string{message} + "'";
            ws->send(json.dump(), uWS::OpCode::TEXT, true);
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
