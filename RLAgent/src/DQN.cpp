#include "DQN.hpp"

void DQN::update_target() { q_network_target_.copy_from(q_network_); }

int DQN::action_greedy(const Eigen::VectorXd &state) {
  MatrixXd inp = state;
  MatrixXd q_values = q_network_.forward(inp);
  int max_idx;
  q_values.col(0).maxCoeff(&max_idx);
  return max_idx;
}

int DQN::action_explore(const Eigen::VectorXd &state, double epsilon) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::bernoulli_distribution bern(epsilon);
  if (bern(gen)) {
    std::uniform_int_distribution<int> dist(0, 3);
    return dist(gen);
  } else {
    return action_greedy(state);
  }
}

void DQN::pretrain(RLAgentClient &env, int n_episodes) {
  for (int ep = 0; ep < n_episodes; ++ep) {
    Eigen::MatrixXi state_mat = env.reset();
    Eigen::VectorXd state = flatten_state(state_mat);
    auto last_pos = get_agent_position(state_mat);
    bool done = false;
    while (!done) {
      int action = random_action();
      auto [next_state_mat, done_flag] = env.step(action);
      std::string status = env.get_status();
      auto current_pos = get_agent_position(next_state_mat);

      double reward = compute_reward(status, next_state_mat, last_pos,
                                     current_pos, done_flag);
      done = done_flag;

      Eigen::VectorXd next_state = flatten_state(next_state_mat);
      memory_.add(state, action, reward, next_state, done ? 0.0 : 1.0);
      state = next_state;
      last_pos = current_pos;
    }
  }
}

void DQN::train(RLAgentClient &env, int n_episodes, int n_pretrain,
                double epsilon_start, double epsilon_end, double decay_rate,
                int batch_size, int n_learn, int max_tau) {
  pretrain(env, n_pretrain);
  double epsilon = epsilon_start;
  int it = 0;
  int tau = 0;

  for (int ep = 0; ep < n_episodes; ++ep) {
    Eigen::MatrixXi state_mat = env.reset();
    Eigen::VectorXd state = flatten_state(state_mat);
    bool done = false;
    auto last_pos = get_agent_position(state_mat);

    while (!done) {
      int action = action_explore(state, epsilon);
      it++;
      tau++;
      epsilon = epsilon_end +
                (epsilon_start - epsilon_end) * std::exp(-decay_rate * it);

      auto [next_state_mat, done_flag] = env.step(action);
      std::string status = env.get_status();
      auto current_pos = get_agent_position(next_state_mat);

      double reward = compute_reward(status, next_state_mat, last_pos,
                                     current_pos, done_flag);

      std::cout << "Episode: " << ep << " Step: " << it << " Reward: " << reward
                << " Epsilon: " << epsilon << std::endl;

      done = done_flag;

      Eigen::VectorXd next_state = flatten_state(next_state_mat);
      memory_.add(state, action, reward, next_state, done ? 0.0 : 1.0);
      state = next_state;
      last_pos = current_pos;

      if (it % n_learn == 0 && memory_.size() >= (size_t)batch_size) {
        update_network(batch_size);
        if (tau >= max_tau) {
          update_target();
          tau = 0;
        }
      }
    }
  }
}

void DQN::test(RLAgentClient &env, int n_episodes) {
  for (int ep = 0; ep < n_episodes; ++ep) {
    Eigen::MatrixXi state_mat = env.reset();
    Eigen::VectorXd state = flatten_state(state_mat);
    auto last_pos = get_agent_position(state_mat);
    bool done = false;
    double total_reward = 0.0;

    while (!done) {
      int action = action_greedy(state);
      auto [next_state_mat, done_flag] = env.step(action);
      std::string status = env.get_status();
      auto current_pos = get_agent_position(next_state_mat);

      double reward = compute_reward(status, next_state_mat, last_pos,
                                     current_pos, done_flag);
      total_reward += reward;

      state = flatten_state(next_state_mat);
      last_pos = current_pos;
      done = done_flag;
    }

    std::cout << "Test Episode " << ep << " - Total Reward: " << total_reward
              << std::endl;
  }
}

int DQN::random_action() {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> dist(0, 3);
  return dist(gen);
}

Eigen::VectorXd DQN::flatten_state(const Eigen::MatrixXi &mat) {
  Eigen::VectorXd v(16);
  int idx = 0;
  for (int r = 0; r < 4; ++r) {
    for (int c = 0; c < 4; ++c) {
      v(idx++) = static_cast<double>(mat(r, c));
    }
  }
  return v;
}

std::pair<int, int> DQN::get_agent_position(const Eigen::MatrixXi &mat) {
  for (int y = 0; y < mat.rows(); ++y) {
    for (int x = 0; x < mat.cols(); ++x) {
      if (mat(y, x) == 1) {
        return {x, y};
      }
    }
  }
  return {-1, -1};
}

double DQN::compute_reward(const std::string &status,
                           const Eigen::MatrixXi &mat,
                           const std::pair<int, int> &last_pos,
                           const std::pair<int, int> &current_pos, bool done) {

  // done时:
  //   Success -> reward = 100
  //   Dead    -> reward = -100
  // 未done时:
  //   根据曼哈顿距离给予惩罚: reward = -3 * distance
  //   如果位置没变(无效动作): reward -= 10
  //   每步消耗: reward -= 1

  double reward = 0.0;
  if (done) {
    if (status == "Success") {
      reward = 100.0;
    } else {
      reward = -100.0;
    }
  } else {
    int dist = manhattan_distance(current_pos, goal_position_);
    reward = -3.0 * dist;
    if (current_pos == last_pos) {
      reward -= 10.0;
    }
    reward -= 1.0;
  }
  return reward;
}

int DQN::manhattan_distance(const std::pair<int, int> &p1,
                            const std::pair<int, int> &p2) {
  return std::abs(p1.first - p2.first) + std::abs(p1.second - p2.second);
}

void DQN::update_network(int batch_size) {
  auto transitions = memory_.sample(batch_size);

  MatrixXd states_batch(input_size_, batch_size);
  MatrixXd next_states_batch(input_size_, batch_size);
  Eigen::VectorXi actions_batch(batch_size);
  Eigen::VectorXd rewards_batch(batch_size);
  Eigen::VectorXd dones_batch(batch_size);

  for (int i = 0; i < batch_size; ++i) {
    states_batch.col(i) = transitions[i].state;
    next_states_batch.col(i) = transitions[i].next_state;
    actions_batch(i) = transitions[i].action;
    rewards_batch(i) = transitions[i].reward;
    dones_batch(i) = transitions[i].done;
  }

  MatrixXd q_next = q_network_target_.forward(next_states_batch);
  Eigen::VectorXd max_next_q(batch_size);
  for (int i = 0; i < batch_size; ++i) {
    double max_val;
    q_next.col(i).maxCoeff(&max_val);
    max_next_q(i) = max_val;
  }

  Eigen::VectorXd q_targets =
      rewards_batch +
      gamma_ * (max_next_q.array() * dones_batch.array()).matrix();

  MatrixXd q_current = q_network_.forward(states_batch);
  Eigen::VectorXd chosen_q(batch_size);
  for (int i = 0; i < batch_size; ++i) {
    chosen_q(i) = q_current(actions_batch(i), i);
  }

  Eigen::VectorXd diff = chosen_q - q_targets;
  MatrixXd grad_output = MatrixXd::Zero(q_current.rows(), q_current.cols());
  for (int i = 0; i < batch_size; ++i) {
    int a_i = actions_batch(i);
    grad_output(a_i, i) = 2.0 * diff(i) / batch_size;
  }

  q_network_.backward(grad_output);
  q_network_.update(lr_);
}
