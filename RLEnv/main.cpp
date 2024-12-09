#include <zmq.hpp>
#include "include/maze_env.hpp"

int main() {

    zmq::context_t context(1);
    zmq::socket_t socket(context, ZMQ_REP);
    socket.bind("tcp://*:5555");

    ObstacleAvoidanceEnv env;

    env.run(socket);

    return 0;
}
