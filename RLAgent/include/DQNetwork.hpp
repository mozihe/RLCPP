#ifndef DQNETWORK_HPP
#define DQNETWORK_HPP

#include "base_net.hpp"

class DQNetwork : public NeuralNetwork {
public:
    DQNetwork(int input_size = 16, int fc_1_dim = 512, int fc_2_dim = 256, int out = 4);
};


#endif //DQNETWORK_HPP
