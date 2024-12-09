#include "env.hpp"
#include <iostream>

Env::Env(const std::string& env_name) : env_name_(env_name) {}

Env::~Env() = default;

void Env::run(zmq::socket_t& socket) {
    while (true) {
        handleMessages(socket);
    }
}
