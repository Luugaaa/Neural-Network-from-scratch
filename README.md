# Neural Network with Genetic Algorithm

This project implements a neural network from scratch using a genetic algorithm (GA) for training instead of traditional gradient descent. The neural network is designed to solve classification problems, where the goal is to map input data to specific target classes. The genetic algorithm is utilized to optimize the weights of the network by simulating the process of natural evolution.

## Features

- **Customizable Network Architecture**: The network allows for the specification of input size, number of hidden layers, and number of neurons per layer.
- **Genetic Algorithm Training**: Replaces backpropagation with a genetic algorithm that uses selection, crossover, and mutation to evolve the network weights.
- **Sigmoid Activation Function**: Utilizes the sigmoid function for non-linearity in the neurons.
- **Dynamic Population Management**: Includes various genetic operations such as small mutation, large mutation, and crossover, with configurable population sizes.

## Theory

### Neural Networks
A neural network is composed of layers of neurons, where each neuron processes input data using a weighted sum and an activation function. The network's learning process involves adjusting these weights to minimize the error between the predicted output and the actual target.

### Genetic Algorithms
A genetic algorithm is an optimization technique inspired by natural selection. It maintains a population of potential solutions (in this case, weight matrices) and evolves this population over generations using operators like selection, crossover, and mutation. Unlike gradient-based methods, GAs do not require differentiability and can escape local minima by exploring a broader solution space.

### Training Process
1. **Initialization**: Generate an initial population of weight matrices randomly.
2. **Evaluation**: Compute the error for each individual in the population by feeding inputs through the network and comparing the output to the target.
3. **Selection**: Rank the population by error and select the best-performing individuals.
4. **Crossover**: Create new individuals by combining parts of two parent solutions.
5. **Mutation**: Introduce small or large random changes to some individuals.
6. **Iteration**: Repeat the process for a specified number of generations or until a desired error threshold is reached.

## Results

After training for a maximum of 300,000 generations with a population size of 20, the network achieved a minimal error rate that closely matched the target output. 
