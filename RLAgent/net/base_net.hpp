#ifndef BASE_NET_HPP
#define BASE_NET_HPP

#include <Eigen/Dense>

using namespace Eigen;

class Layer {
public:
    virtual ~Layer() = default;

    virtual MatrixXd forward(const MatrixXd& input) = 0;
    virtual MatrixXd backward(const MatrixXd& grad_output) = 0;
    virtual void update(double learning_rate) = 0;
    virtual void print_parameters() = 0;
};

class FullyConnectedLayer : public Layer {
private:
    MatrixXd weights;
    MatrixXd biases;
    MatrixXd grad_weights;
    MatrixXd grad_biases;
    MatrixXd input_cache;
public:
    FullyConnectedLayer(int input_size, int output_size);
    MatrixXd forward(const MatrixXd& input) override;
    MatrixXd backward(const MatrixXd& grad_output) override;
    void update(double learning_rate) override;
    void print_parameters() override;
    void set_parameters(const MatrixXd& new_weights, const MatrixXd& new_biases);
    void get_parameters(MatrixXd& out_weights, MatrixXd& out_biases) const;
};

class NeuralNetwork {
protected:
    std::vector<Layer*> layers;

public:
    void add_layer(Layer* layer);
    MatrixXd forward(const MatrixXd& input);
    void backward(const MatrixXd& grad_output);
    void update(double learning_rate);
    void train(const std::vector<MatrixXd>& data, const std::vector<MatrixXd>& labels, int epochs, double learning_rate, int batch_size);
    ~NeuralNetwork();
    void print_parameters() const;
    void copy_from(const NeuralNetwork& other);
};

class ActivationLayer : public Layer {
private:
    std::function<MatrixXd(const MatrixXd&)> activation_function;
    std::function<MatrixXd(const MatrixXd&)> activation_derivative;
    MatrixXd input_cache;

public:
    ActivationLayer(std::function<MatrixXd(const MatrixXd&)> activation,
                    std::function<MatrixXd(const MatrixXd&)> activation_derivative)
                    : activation_function(activation), activation_derivative(activation_derivative) {}

    MatrixXd forward(const MatrixXd& input) override;

    MatrixXd backward(const MatrixXd& grad_output) override;

    void update(double learning_rate) override {}

    void print_parameters() override {}
};

MatrixXd sigmoid(const MatrixXd& x);

MatrixXd sigmoid_derivative(const MatrixXd& x);

MatrixXd relu(const MatrixXd& x);

MatrixXd relu_derivative(const MatrixXd& x);

#endif //BASE_NET_HPP
