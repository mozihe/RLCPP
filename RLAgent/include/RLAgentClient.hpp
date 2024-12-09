#ifndef RLAGENTCLIENT_HPP
#define RLAGENTCLIENT_HPP

#include <zmq.hpp>
#include <string>
#include <Eigen/Dense>

class RLAgentClient {
public:
    RLAgentClient();
    Eigen::MatrixXi reset();
    std::pair<Eigen::MatrixXi, bool> step(int action);
    std::string get_status() const;


private:
    std::unique_ptr<zmq::context_t> context;
    std::unique_ptr<zmq::socket_t> socket;
    std::string parsed_status;
    void send_message(const std::string& message);
    std::string receive_message();
    Eigen::MatrixXi parse_state(const std::string& state);
};


#endif //RLAGENTCLIENT_HPP
