#include "base_net.hpp"
#include <iostream>

FullyConnectedLayer::FullyConnectedLayer(int input_size, int output_size) {
    weights = MatrixXd::Random(output_size, input_size);
    biases = MatrixXd::Random(output_size, 1);
    grad_weights = MatrixXd::Zero(output_size, input_size);
    grad_biases = MatrixXd::Zero(output_size, 1);
}

MatrixXd FullyConnectedLayer::forward(const MatrixXd& input) {
    input_cache = input;
    return (weights * input + biases.replicate(1, input.cols()));
}

MatrixXd FullyConnectedLayer::backward(const MatrixXd& grad_output) {
    grad_weights += grad_output * input_cache.transpose() / grad_output.cols();
    grad_biases += grad_output.rowwise().sum() / grad_output.cols();

    return weights.transpose() * grad_output;
}

void FullyConnectedLayer::update(double learning_rate) {
    weights -= learning_rate * grad_weights;
    biases -= learning_rate * grad_biases;

    grad_weights.setZero();
    grad_biases.setZero();
}

void FullyConnectedLayer::print_parameters() {
    std::cout << "Weights:\n" << weights << std::endl;
    std::cout << "Biases:\n" << biases << std::endl;
}

void FullyConnectedLayer::set_parameters(const MatrixXd& new_weights, const MatrixXd& new_biases) {
    if (new_weights.rows() != weights.rows() || new_weights.cols() != weights.cols() ||
        new_biases.rows() != biases.rows() || new_biases.cols() != biases.cols()) {
        throw std::runtime_error("Parameter dimension mismatch in set_parameters.");
        }
    weights = new_weights;
    biases = new_biases;
}

void FullyConnectedLayer::get_parameters(MatrixXd& out_weights, MatrixXd& out_biases) const {
    out_weights = weights;
    out_biases = biases;
}

void NeuralNetwork::add_layer(Layer* layer) {
    layers.push_back(layer);
}

MatrixXd NeuralNetwork::forward(const MatrixXd& input) {
    MatrixXd output = input;
    for (Layer* layer : layers) {
        output = layer->forward(output);
    }
    return output;
}

void NeuralNetwork::backward(const MatrixXd& grad_output) {
    MatrixXd grad = grad_output;
    for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
        grad = (*it)->backward(grad);
    }
}

void NeuralNetwork::update(double learning_rate) {
    for (Layer* layer : layers) {
        layer->update(learning_rate);
    }
}

void NeuralNetwork::train(const std::vector<MatrixXd>& data, const std::vector<MatrixXd>& labels, int epochs, double learning_rate, int batch_size) {
    int num_batches = data.size() / batch_size;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        double total_loss = 0.0;
        for (int b = 0; b < num_batches; ++b) {
            MatrixXd batch_data = MatrixXd::Zero(data[0].rows(), batch_size);
            MatrixXd batch_labels = MatrixXd::Zero(labels[0].rows(), batch_size);
            for (int i = 0; i < batch_size; ++i) {
                batch_data.col(i) = data[b * batch_size + i];
                batch_labels.col(i) = labels[b * batch_size + i];
            }

            MatrixXd pred = forward(batch_data);
            MatrixXd loss_grad = 2 * (pred - batch_labels);
            backward(loss_grad);
            update(learning_rate);
            total_loss += (pred - batch_labels).squaredNorm() / batch_size;
        }
        if (epoch % 10 == 0) {
            std::cout << "Epoch " << epoch << ", Total Loss: " << total_loss / num_batches << std::endl;
        }
    }
}

NeuralNetwork::~NeuralNetwork() {
    for (Layer* layer : layers) {
        delete layer;
    }
}

void NeuralNetwork::print_parameters() const {
    for (Layer* layer : layers) {
        layer->print_parameters();
    }
}

void NeuralNetwork::copy_from(const NeuralNetwork& other) {
    if (this->layers.size() != other.layers.size()) {
        throw std::runtime_error("Cannot copy parameters: layer size mismatch.");
    }

    for (size_t i = 0; i < layers.size(); ++i) {
        // 尝试从other对应层获取参数并设置到this层
        FullyConnectedLayer* fc_this = dynamic_cast<FullyConnectedLayer*>(layers[i]);
        const FullyConnectedLayer* fc_other = dynamic_cast<const FullyConnectedLayer*>(other.layers[i]);

        if (fc_this && fc_other) {
            MatrixXd w, b;
            fc_other->get_parameters(w, b);
            fc_this->set_parameters(w, b);
        } else {
            // 若不是全连接层（如激活层）则跳过，因为没有参数
        }
    }
}

MatrixXd ActivationLayer::forward(const MatrixXd& input) {
    input_cache = input;
    return activation_function(input);
}

MatrixXd ActivationLayer::backward(const MatrixXd& grad_output) {
    return grad_output.array() * activation_derivative(input_cache).array();
}

MatrixXd sigmoid(const MatrixXd& x) {
    return x.unaryExpr([](double elem) { return 1 / (1 + exp(-elem)); });
}

MatrixXd sigmoid_derivative(const MatrixXd& x) {
    return sigmoid(x).array() * (1 - sigmoid(x).array());
}

MatrixXd relu(const MatrixXd& x) {
    return x.unaryExpr([](double val) { return (val > 0.0) ? val : 0.0; });
}

MatrixXd relu_derivative(const MatrixXd& x) {
    return x.unaryExpr([](double val) { return (val > 0.0) ? 1.0 : 0.0; });
}
