# Goals

- Build a simple neural network with only one hidden layer to expose how it works
- Use simple array to represent the internal state: double[] (make it easier to migrate to other languages)
- All in one file (about 200 lines)

# Possible improvements

- Use multiple threads
- Use the GPU for matrix multiplication
- Configure which function to use (sigmoid, tanh, ReLu, etc.)
- Possibility to add softmax layer at the end
- More layers
- Loop unrolling
- Save / Load network

# Dataset
I'm using the [Iris flower data set](https://en.wikipedia.org/wiki/Iris_flower_data_set).

# Copyright and license
Code released under the MIT license.

The neural network is a modified version of https://github.com/trentsartain/Neural-Network
(MIT license)
