#include "DQNetwork.hpp"

DQNetwork::DQNetwork(int input_size, int fc_1_dim, int fc_2_dim, int out) {

    add_layer(new FullyConnectedLayer(input_size, fc_1_dim));
    add_layer(new ActivationLayer(relu, relu_derivative));


    add_layer(new FullyConnectedLayer(fc_1_dim, fc_2_dim));
    add_layer(new ActivationLayer(relu, relu_derivative));

    add_layer(new FullyConnectedLayer(fc_2_dim, out));
}