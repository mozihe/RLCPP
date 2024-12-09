#ifndef DQN_HPP
#define DQN_HPP

#include <iostream>

#include "RLAgentClient.hpp"
#include "DQNetwork.hpp"
#include "Memory.hpp"

class DQN {
public:
    DQN(double gamma=0.99, int input_size=16, int fc_1_dim=512, int fc_2_dim=256, int out=4, size_t max_size=50000, double lr=0.001)
    : gamma_(gamma), input_size_(input_size), lr_(lr),
      q_network_(input_size, fc_1_dim, fc_2_dim, out),
      q_network_target_(input_size, fc_1_dim, fc_2_dim, out),
      memory_(max_size)
    {
        update_target();
        goal_position_ = {3,3};
    }
    void update_target();
    int action_greedy(const Eigen::VectorXd &state);
    int action_explore(const Eigen::VectorXd &state, double epsilon);
    void pretrain(RLAgentClient &env, int n_episodes=100);
    void train(RLAgentClient &env, int n_episodes=1000, int n_pretrain=100, double epsilon_start=1.0, double epsilon_end=0.01, double decay_rate=0.001, int batch_size=32, int n_learn=4, int max_tau=100);
    void test(RLAgentClient &env, int n_episodes=10);


private:
    double gamma_;
    int input_size_;
    double lr_;
    DQNetwork q_network_;
    DQNetwork q_network_target_;
    Memory memory_;
    std::pair<int,int> goal_position_; // 目标位置

    int random_action();
    Eigen::VectorXd flatten_state(const Eigen::MatrixXi &mat);
    std::pair<int,int> get_agent_position(const Eigen::MatrixXi &mat);
    double compute_reward(const std::string &status, const Eigen::MatrixXi &mat, const std::pair<int,int> &last_pos, const std::pair<int,int> &current_pos, bool done);
    int manhattan_distance(const std::pair<int,int> &p1, const std::pair<int,int> &p2);
    void update_network(int batch_size);



};


#endif //DQN_HPP
