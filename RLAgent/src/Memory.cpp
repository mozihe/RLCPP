#include "Memory.hpp"

void Memory::add(const Eigen::VectorXd& state, int action, double reward, const Eigen::VectorXd& next_state, bool done) {
    if (buffer_.size() >= max_size_) {
        buffer_.pop_front();
    }
    Transition t{state, action, reward, next_state, done};
    buffer_.push_back(t);
}

std::vector<Transition> Memory::sample(size_t batch_size) {
    std::vector<Transition> transitions;
    transitions.reserve(batch_size);

    if (batch_size > buffer_.size()) {
        transitions.assign(buffer_.begin(), buffer_.end());
        return transitions;
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<size_t> dist(0, buffer_.size() - 1);

    for (size_t i = 0; i < batch_size; ++i) {
        size_t idx = dist(gen);
        transitions.push_back(buffer_[idx]);
    }

    return transitions;
}

size_t Memory::size() const {
    return buffer_.size();
}