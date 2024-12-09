#ifndef ENV_HPP
#define ENV_HPP

#include <zmq.hpp>
#include <string>

class Env {
public:
    Env(const std::string& env_name);
    virtual ~Env();

    virtual void initialize() = 0;

    virtual std::string getState() = 0;

    virtual std::string updateState(const std::string& action) = 0;

    virtual void reset() = 0;

    virtual bool registerAgent(const std::string& agent_name) = 0;

    virtual void handleMessages(zmq::socket_t& socket) = 0;

    void run(zmq::socket_t& socket);

protected:
    std::string env_name_;
};

#endif // ENV_HPP
