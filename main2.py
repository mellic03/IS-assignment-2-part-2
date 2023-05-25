import numpy as np
import random
import atexit
import neuralnetwork as NN
import matplotlib.pyplot as plt


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



def to_thing(category: int) -> list[int]:
    result: list[int] = [0] * 10
    result[category] = 1
    return result



def onExit(net: NN.NeuralNetwork):
    net.toFile()



def main():

    layout: list[int] = [3072, 30, 10]

    dicts = [
        unpickle("cifar-10-batches-py/data_batch_1"),
        unpickle("cifar-10-batches-py/data_batch_2"),
        unpickle("cifar-10-batches-py/data_batch_3"),
        unpickle("cifar-10-batches-py/data_batch_4"),
        unpickle("cifar-10-batches-py/data_batch_5")
    ]

    training_inputs = []
    training_outputs = []

    for d in dicts:
        for (data, label) in zip(d[b"data"], d[b"labels"]):
            training_inputs.append(np.array(data) / 255.0)
            training_outputs.append(to_thing(int(label)))

    test_data = unpickle("cifar-10-batches-py/test_batch")
    testing_inputs = []
    testing_outputs = []
    for (data, label) in zip(test_data[b"data"], test_data[b"labels"]):
        testing_inputs.append(np.array(data) / 255.0)
        testing_outputs.append(to_thing(int(label)))


    learning_rates = [10, 1.0, 0.01, 0.001]
    mini_batch_sizes = [300, 20, 5, 1]


    nn = NN.NeuralNetwork(layout, learning_rate=0.1, epochs=500, mbs=100)
    nn.train(training_inputs, training_outputs, testing_inputs, testing_outputs)
    nn.toFile()






if __name__ == "__main__":
    main()

