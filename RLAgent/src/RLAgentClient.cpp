#include "RLAgentClient.hpp"

#include <iterator>

RLAgentClient::RLAgentClient() {
    context = std::make_unique<zmq::context_t>(1);
    socket = std::make_unique<zmq::socket_t>(*context, zmq::socket_type::req);
    socket->connect("tcp://localhost:5555");
}

Eigen::MatrixXi RLAgentClient::reset() {
    send_message("RESET");
    receive_message();

    send_message("ACTION:None");
    std::string state = receive_message();
    return parse_state(state);
}

std::pair<Eigen::MatrixXi, bool> RLAgentClient::step(int action) {
    static const std::vector<std::string> action_map = {
        "MOVE_LEFT", "MOVE_RIGHT", "MOVE_UP", "MOVE_DOWN"
    };

    if (action < 0 || action >= static_cast<int>(action_map.size())) {
        throw std::invalid_argument("Invalid action index");
    }

    send_message("ACTION:" + action_map[action]);
    std::string state = receive_message();
    auto parsed_state = parse_state(state);
    bool done = parsed_status == "Dead" || parsed_status == "Success";
    return {parsed_state, done};
}

void RLAgentClient::send_message(const std::string& message) {
    zmq::message_t zmq_message(message.size());
    memcpy(zmq_message.data(), message.c_str(), message.size());
    socket->send(zmq_message, zmq::send_flags::none);
}

std::string RLAgentClient::receive_message() {
    zmq::message_t reply;
    socket->recv(reply, zmq::recv_flags::none);
    return std::string(static_cast<char*>(reply.data()), reply.size());
}

Eigen::MatrixXi RLAgentClient::parse_state(const std::string& state) {
    std::istringstream stream(state);
    std::string line;
    int step;
    Eigen::MatrixXi matrix;

    std::getline(stream, line);
    step = std::stoi(line.substr(line.find(": ") + 2));

    std::getline(stream, line);
    parsed_status = line.substr(line.find(": ") + 2);

    std::vector<std::vector<int>> matrix_data;
    while (std::getline(stream, line)) {
        std::istringstream row_stream(line);
        std::vector<int> row((std::istream_iterator<int>(row_stream)),
                             std::istream_iterator<int>());
        matrix_data.push_back(row);
    }

    if (!matrix_data.empty()) {
        int rows = static_cast<int>(matrix_data.size());
        int cols = static_cast<int>(matrix_data[0].size());
        matrix.resize(rows, cols);

        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                matrix(i, j) = matrix_data[i][j];
            }
        }
    }

    return matrix;
}

std::string RLAgentClient::get_status() const {
    return parsed_status;
}