#include <iostream>
#include "DQN.hpp"

int main() {
    try {

        RLAgentClient env;

        DQN dqn(0.99, 16, 512, 256, 4, 50000, 1e-3);

        int n_episodes = 1000;
        int n_pretrain = 64;
        double epsilon_start = 1.0;
        double epsilon_end = 0.01;
        double decay_rate = 2e-5;
        int batch_size = 64;
        int n_learn = 5;
        int max_tau = 25;

        std::cout << "Start Training..." << std::endl;
        dqn.train(env, n_episodes, n_pretrain, epsilon_start, epsilon_end, decay_rate, batch_size, n_learn, max_tau);
        std::cout << "Training Complete." << std::endl;

        int n_test_episodes = 10;
        std::cout << "Start Testing..." << std::endl;
        dqn.test(env, n_test_episodes);
        std::cout << "Testing Complete." << std::endl;

    } catch (const std::exception &e) {
        std::cerr << "Exception: " << e.what() << "\n";
    }

    return 0;
}
