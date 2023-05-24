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
vecsigmoid = np.vectorize(sigmoid)


def MSE(output: list[float], target: list[float]) -> float:
    sum: float = 0.0
    for i in range(0, len(output)):
        sum += 0.5 * (target[i] - output[i])**2
    return sum


class NeuralNetwork:
    
    def __init__(self, layout: list[int], learning_rate: float, epochs: int, mbs: int) -> None:
        
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.mini_batch_size = mbs

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
                self.ih_weights[i][h] = random.uniform(-1.0/self.input_len, +1.0/self.input_len)
        for h in range(0, self.hidden_len):
            for o in range(0, self.output_len):
                self.ho_weights[h][o] = random.uniform(-1.0/self.input_len, +1.0/self.input_len)
       

        self.ih_weight_change = np.zeros((self.input_len, self.hidden_len))
        self.ho_weight_change = np.zeros((self.hidden_len, self.output_len))
        
        self.hidden_bias_change = np.zeros(self.hidden_len)
        self.output_bias_change = np.zeros(self.output_len)



    # Liat said this!!!
    # 
    # For each mini batch:
    #   Backpropagate after every sample but store weight change
    #   Apply the average weight change at the end of the mini batch
    #

    def __forwardpropagate(self, input: list[float]) -> None:

        self.input = np.array([i/1.0 for i in input])
        self.hidden = sigmoid(np.dot(self.input, self.ih_weights))
        self.output = sigmoid(np.dot(self.hidden, self.ho_weights))


    def __backpropagate_collect_avg(self, target: list[float]) -> None:
        
        sums = np.zeros(self.hidden_len)

        output_error = self.output - target
        dOut_dNet = self.output * (1.0 - self.output)

        for o in range(0, self.output_len):
            A = output_error[o]
            B = dOut_dNet[o]

            for h in range(0, self.hidden_len):
                C = self.hidden[h]
                self.ho_weight_change[h][o] += A * B * C

                D = self.ho_weights[h][o]
                sums[h] += A*B*D


        dOut_dNet = self.hidden * (1.0 - self.hidden)

        for h in range(0, self.hidden_len):
            for i in range(0, self.input_len):
                self.ih_weight_change[i][h] += sums[h] * dOut_dNet[h] * self.input[i]


    def __backpropagate_apply_avg(self) -> None:

        LRATE = self.learning_rate / self.mini_batch_size

        self.ho_weights -= LRATE * self.ho_weight_change
        self.ih_weights -= LRATE * self.ih_weight_change

        self.ho_weight_change *= 0
        self.ih_weight_change *= 0

        self.output_bias_change *= 0
        self.hidden_bias_change *= 0


    def classify(self, input: list[float]) -> int:

        self.__forwardpropagate(input)

        idx: int = -1
        max_activation = -np.inf

        for i in range(0, self.output_len):
            activation = self.output[i]
            if activation > max_activation:
                idx = i
                max_activation = activation

        v = [0] * self.output_len
        v[idx] = 1
        return v


    def train(self, training_inputs: list[float], training_outputs: list[float]) -> None:

        count: int = 0
        avg_mse: float = 0.0

        for i in range(0, self.epochs):

            temp = list(zip(training_inputs, training_outputs))
            random.shuffle(temp)
            training_inputs, training_outputs = zip(*temp)
            training_inputs, training_outputs = list(training_inputs), list(training_outputs)

            print("\nEpoch %d/%d" % (i+1, self.epochs))

            batch_num: int = 0
            for (input, output) in zip(training_inputs, training_outputs):

                if count >= self.mini_batch_size:
                    self.__backpropagate_apply_avg()
                    count = 0
                    batch_num += 1
                    print("    mb %d/%d,  mse: %.4f" % (batch_num, len(training_inputs)/self.mini_batch_size, avg_mse / self.mini_batch_size))
                    avg_mse = 0.0

                self.__forwardpropagate(input)
                self.__backpropagate_collect_avg(output)

                avg_mse += MSE([out for out in self.output], output)
                count += 1


    def toFile(self, filepath: str) -> None:
        
        fh = open(filepath, "w")

        fh.write("%d %d %d\n" % (self.input_len, self.hidden_len, self.output_len))

        for i in range(0, self.input_len):
            for h in range(0, self.hidden_len):
                fh.write("%f\n" % self.ih_weights[i][h])

        for h in range(0, self.hidden_len):
            for o in range(0, self.output_len):
                fh.write("%f\n" % self.ho_weights[h][o])


    def fromFile(self, filepath: str = "backupweights.txt") -> None:
        
        fh = open(filepath, "r")

        line = fh.readline()
        self.input_len, self.hidden_len, self.output_len = line.split(" ")
        self.input_len  = int(self.input_len)
        self.hidden_len = int(self.hidden_len)
        self.output_len = int(self.output_len)

        self.input  = [0.0] * self.input_len
        self.hidden = [0.0] * self.hidden_len
        self.output = [0.0] * self.output_len

        self.ih_weights = [[0.0] * self.hidden_len] * self.input_len
        self.ho_weights = [[0.0] * self.output_len] * self.hidden_len
        
        for ih in range(0, self.input_len * self.hidden_len):
            self.ih_weights[ih//self.hidden_len][ih%self.hidden_len] = float(fh.readline())

        for ho in range(0, self.hidden_len * self.output_len):
            self.ho_weights[ho//self.output_len][ho%self.output_len] = float(fh.readline())


    def test(self, testing_inputs, testing_outputs) -> None:

        total = len(testing_inputs)
        correct = 0

        for (input, output) in zip(testing_inputs, testing_outputs):
            result = self.classify(input)            
            
            idx = -1
            for i in range(0, len(result)):
                if result[i] == 1:
                    idx = i

            if output[idx] == result[idx]:
                correct += 1

            # print(input, output, sep=" : " ,end=" --> ")
            # print(result)

        print("Accuracy: ", correct/total)




def to_thing(category: int) -> list[int]:
    result: list[int] = [0] * 10
    result[category] = 1
    return result



def onExit(network: NeuralNetwork, inputs, outputs):
    network.toFile("backupweights.txt")
    network.test(inputs, outputs)



def main():

    layout: list[int] = [3072, 30, 10]
    nn = NeuralNetwork(layout, learning_rate=1.1, epochs=20, mbs=100)

    dict1 = unpickle("cifar-10-batches-py/data_batch_1")
    dict2 = unpickle("cifar-10-batches-py/data_batch_2")

    inputs = []
    outputs = []

    for (data, label) in zip(dict1[b"data"], dict1[b"labels"]):
        inputs.append(data)
        outputs.append(to_thing(int(label)))
  
    for (data, label) in zip(dict2[b"data"], dict2[b"labels"]):
        inputs.append(data)
        outputs.append(to_thing(int(label)))


    atexit.register(onExit, nn, inputs[1000:2000], outputs[1000:2000])


    nn.train(inputs, outputs)
    nn.test(inputs[1000:1500], outputs[1000:1500])

    # layout: list[int] = [2, 4, 2]
    # nn = NeuralNetwork(layout, learning_rate=0.5, epochs=3000, mbs=1)


    # inputs = [
    #     [0.00, 0.00],
    #     [0.00, 1.00],
    #     [1.00, 0.00],
    #     [1.00, 1.00],
    # ]
    # outputs = [
    #     [0.00, 1.00],
    #     [1.00, 0.00],
    #     [1.00, 0.00],
    #     [0.00, 1.00]
    # ]


    # nn.train(inputs, outputs)
    # nn.test(inputs, outputs)






if __name__ == "__main__":
    main()
