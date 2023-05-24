import numpy as np
import random

import atexit


def unpickle(filepath: str) -> dict:
    import pickle
    with open(filepath, "rb") as fh:
        dict = pickle.load(fh, encoding="bytes")
    return dict


def sigmoid(f: float) -> float:
    return 1 / (1 + np.exp(-f))


def MSE(output: list[float], target: list[float]) -> float:
    sum: float = 0.0
    for i in range(0, len(output)):
        sum += 0.5 * (target[i] - output[i])**2
    return sum


class NeuralNetwork:
    
    def __init_layout(self, layout: list[int]):
        self.input_len = layout[0]
        self.hidden_len = layout[1]
        self.output_len = layout[2]

        self.input = np.zeros(self.input_len)
        self.hidden = np.zeros(self.hidden_len)
        self.output = np.zeros(self.output_len)

        self.hid_biases = [0.1] * self.hidden_len
        self.out_biases = [0.1] * self.output_len


        # Weights
        self.ih_weights = np.zeros((self.input_len, self.hidden_len))
        self.ho_weights = np.zeros((self.hidden_len, self.output_len))
       
        for i in range(0, self.input_len):
            for h in range(0, self.hidden_len):
                self.ih_weights[i][h] = random.uniform(-1.0, +1.0)
        for h in range(0, self.hidden_len):
            for o in range(0, self.output_len):
                self.ho_weights[h][o] = random.uniform(-1.0, +1.0)

        self.ih_weight_change = np.zeros((self.input_len, self.hidden_len))
        self.ho_weight_change = np.zeros((self.hidden_len, self.output_len))
        
        self.hidden_bias_change = np.zeros(self.hidden_len)
        self.output_bias_change = np.zeros(self.output_len)



    def __init__(self, layout: list[int], learning_rate: float, epochs: int, mbs: int) -> None:
        
        self.learning_rate = learning_rate
        
        self.epoch = 0
        self.epochs = epochs
        
        self.mini_batch_size = mbs

        self.__init_layout(layout)


    def __forwardpropagate(self, input: list[float]) -> None:

        self.input = np.array([i/255.0 for i in input])
        self.hidden = sigmoid(np.dot(self.input, self.ih_weights))
        self.output = sigmoid(np.dot(self.hidden, self.ho_weights))


    def __backpropagate_collect_avg(self, target: list[float]) -> None:
        
        output_error = self.output - target
        dOut_dNet = self.output * (1.0 - self.output)
        self.ho_weight_change += np.outer(self.hidden, output_error * dOut_dNet)
        sums = np.dot(output_error * dOut_dNet, self.ho_weights.T)
        dOut_dNet = self.hidden * (1.0 - self.hidden)
        self.ih_weight_change += np.outer(self.input, sums * dOut_dNet)


    def __backpropagate_apply_avg(self) -> None:

        LRATE = self.learning_rate / self.mini_batch_size

        self.ho_weights -= LRATE * self.ho_weight_change
        self.ih_weights -= LRATE * self.ih_weight_change

        self.ho_weight_change.fill(0)
        self.ih_weight_change.fill(0)

        self.output_bias_change.fill(0)
        self.hidden_bias_change.fill(0)

