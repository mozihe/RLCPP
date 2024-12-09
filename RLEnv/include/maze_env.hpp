#ifndef OBSTACLE_AVOIDANCE_ENV_HPP
#define OBSTACLE_AVOIDANCE_ENV_HPP

#include "env.hpp"
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <unordered_map>

enum class AgentState {
    ALIVE,
    DEAD,
    SUCCESS
};

class ObstacleAvoidanceEnv : public Env {
public:
    ObstacleAvoidanceEnv();
    void initialize() override;
    std::string getState() override;
    std::string updateState(const std::string& action) override;
    void reset() override;
    bool registerAgent(const std::string& agent_name) override;
    void handleMessages(zmq::socket_t& socket) override;

private:
    std::pair<int, int> agent_position_;
    std::pair<int, int> goal_position_;
    std::vector<std::pair<int, int>> obstacles_;
    int grid_size_;
    int cell_size_;
    AgentState agent_state_;
    int step_count_;

    std::unordered_map<std::string, std::pair<int, int>> agents_;

    void generateObstacles();
    void displayEnvironment();
    std::string generateStateMatrix();
};

#endif // OBSTACLE_AVOIDANCE_ENV_HPP
