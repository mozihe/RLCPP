#include "maze_env.hpp"
#include <iostream>
#include <cstdlib>
#include <ctime>

ObstacleAvoidanceEnv::ObstacleAvoidanceEnv() : Env("ObstacleAvoidanceEnv"), grid_size_(4), cell_size_(50) {
    initialize();
}

void ObstacleAvoidanceEnv::initialize() {
    std::cout << "Initializing 2D Obstacle Avoidance Environment..." << std::endl;
    agent_position_ = {0, 0};
    goal_position_ = {3, 3};
    step_count_ = 0;
    agent_state_ = AgentState::ALIVE;

    generateObstacles();

    displayEnvironment();
}

void ObstacleAvoidanceEnv::generateObstacles() {
    obstacles_.clear();
    std::srand(0);

    int num_obstacles = 2;

    for (int i = 0; i < num_obstacles; ++i) {
        std::pair<int, int> obstacle;
        do {
            obstacle = {rand() % grid_size_, rand() % grid_size_};
        } while (obstacle == agent_position_ || obstacle == goal_position_ ||
                 std::find(obstacles_.begin(), obstacles_.end(), obstacle) != obstacles_.end() ||
                 obstacle == std::make_pair(3, 2) || obstacle == std::make_pair(2, 3) ||
                 obstacle == std::make_pair(0, 1) || obstacle == std::make_pair(1, 0));

        obstacles_.push_back(obstacle);
    }

    std::cout << "Generated " << num_obstacles << " obstacles." << std::endl;
}

std::string ObstacleAvoidanceEnv::getState() {
    return generateStateMatrix();
}

std::string ObstacleAvoidanceEnv::updateState(const std::string& action) {
    if (agent_state_ != AgentState::ALIVE) {
        return "Agent is not alive. Game over or success.";
    }

    std::pair<int, int> new_position = agent_position_;

    if (action == "MOVE_UP" && agent_position_.second > 0) {
        new_position.second -= 1;
    } else if (action == "MOVE_DOWN" && agent_position_.second < grid_size_ - 1) {
        new_position.second += 1;
    } else if (action == "MOVE_LEFT" && agent_position_.first > 0) {
        new_position.first -= 1;
    } else if (action == "MOVE_RIGHT" && agent_position_.first < grid_size_ - 1) {
        new_position.first += 1;
    } else if (action == "None") {
        step_count_--;
    }

    if (std::find(obstacles_.begin(), obstacles_.end(), new_position) != obstacles_.end()) {
        agent_state_ = AgentState::DEAD;
        std::cout << "Agent hit an obstacle. Game over." << std::endl;
        return generateStateMatrix();
    }

    if (new_position == goal_position_) {
        agent_state_ = AgentState::SUCCESS;
        std::cout << "Agent reached the goal! Success!" << std::endl;
        return generateStateMatrix();
    }

    agent_position_ = new_position;
    step_count_++;

    displayEnvironment();
    return generateStateMatrix();
}

void ObstacleAvoidanceEnv::reset() {
    agent_position_ = {0, 0};
    step_count_ = 0;
    agent_state_ = AgentState::ALIVE;
    generateObstacles();
    displayEnvironment();
}

bool ObstacleAvoidanceEnv::registerAgent(const std::string& agent_name) {
    if (agents_.find(agent_name) != agents_.end()) {
        std::cout << "Agent " << agent_name << " already registered." << std::endl;
        return false;
    }

    agents_[agent_name] = {0, 0};
    std::cout << "Agent " << agent_name << " registered." << std::endl;
    return true;
}

std::string ObstacleAvoidanceEnv::generateStateMatrix() {
    std::vector<std::vector<int>> state(grid_size_, std::vector<int>(grid_size_, 0));

    state[agent_position_.second][agent_position_.first] = 1;

    state[goal_position_.second][goal_position_.first] = 2;

    for (const auto& obstacle : obstacles_) {
        state[obstacle.second][obstacle.first] = 3;
    }

    std::string state_str = "Step: " + std::to_string(step_count_) + "\n";

    std::string agent_status;
    if (agent_state_ == AgentState::ALIVE) {
        agent_status = "Alive";
    } else if (agent_state_ == AgentState::DEAD) {
        agent_status = "Dead";
    } else if (agent_state_ == AgentState::SUCCESS) {
        agent_status = "Success";
    }

    state_str += "Status: " + agent_status + "\n";

    for (const auto& row : state) {
        for (const auto& cell : row) {
            state_str += std::to_string(cell) + " ";
        }
        state_str += "\n";
    }

    return state_str;
}

void ObstacleAvoidanceEnv::displayEnvironment() {
    cv::Mat environment(grid_size_ * cell_size_, grid_size_ * cell_size_, CV_8UC3, cv::Scalar(255, 255, 255));

    for (int i = 0; i <= grid_size_ * cell_size_; i += cell_size_) {
        cv::line(environment, cv::Point(i, 0), cv::Point(i, grid_size_ * cell_size_), cv::Scalar(200, 200, 200), 1);
        cv::line(environment, cv::Point(0, i), cv::Point(grid_size_ * cell_size_, i), cv::Scalar(200, 200, 200), 1);
    }

    for (const auto& obstacle : obstacles_) {
        cv::rectangle(environment,
            cv::Point(obstacle.first * cell_size_, obstacle.second * cell_size_),
            cv::Point((obstacle.first + 1) * cell_size_, (obstacle.second + 1) * cell_size_),
            cv::Scalar(0, 0, 255), -1);
    }

    if (agent_state_ == AgentState::ALIVE) {
        cv::circle(environment,
            cv::Point(agent_position_.first * cell_size_ + cell_size_ / 2, agent_position_.second * cell_size_ + cell_size_ / 2),
            cell_size_ / 3, cv::Scalar(0, 255, 0), -1);
    }

    cv::circle(environment,
        cv::Point(goal_position_.first * cell_size_ + cell_size_ / 2, goal_position_.second * cell_size_ + cell_size_ / 2),
        cell_size_ / 3, cv::Scalar(255, 0, 0), -1);

    cv::imshow("2D Obstacle Avoidance Environment", environment);
    cv::waitKey(100);
}

void ObstacleAvoidanceEnv::handleMessages(zmq::socket_t& socket) {
    zmq::message_t request;
    socket.recv(request, zmq::recv_flags::none);
    std::string message(static_cast<char*>(request.data()), request.size());

    if (message.find("REGISTER:") == 0) {
        std::string agent_name = message.substr(9);
        bool success = registerAgent(agent_name);
        std::string reply = success ? "REGISTER_SUCCESS" : "REGISTER_FAILED";
        socket.send(zmq::buffer(reply), zmq::send_flags::none);
    } else if (message == "RESET") {
        reset();
        socket.send(zmq::buffer("RESET_DONE"), zmq::send_flags::none);
    } else if (message.find("ACTION:") == 0) {
        std::string action = message.substr(7); // 提取动作
        std::string new_state = updateState(action);
        socket.send(zmq::buffer(new_state), zmq::send_flags::none);
    } else {
        socket.send(zmq::buffer("UNKNOWN_COMMAND"), zmq::send_flags::none);
    }
}
