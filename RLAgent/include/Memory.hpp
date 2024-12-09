#ifndef MEMORY_HPP
#define MEMORY_HPP

#include <Eigen/Dense>
#include <deque>
#include <random>
#include <vector>
#include <algorithm>

struct Transition {
    Eigen::VectorXd state;
    int action;
    double reward;
    Eigen::VectorXd next_state;
    bool done;
};

class Memory {
public:
    Memory(size_t max_size) : max_size_(max_size) {}

    void add(const Eigen::VectorXd& state, int action, double reward, const Eigen::VectorXd& next_state, bool done);

    std::vector<Transition> sample(size_t batch_size);

    size_t size() const;

private:
    size_t max_size_;
    std::deque<Transition> buffer_;
};


#endif //MEMORY_HPP
