import time
import numpy as np
import random


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



    def __init__(self, layout: list[int] | str, learning_rate = 0.0, epochs = 0, mbs = 0) -> None:
        
        self.x_axis = [] # epoch
        self.y_axis = [] # accuracy
        self.training_time = 0.0 # time taken to train model

        if type(layout) == str:
            self.fromFile(layout)

        else:
            self.learning_rate = learning_rate
            self.epoch = 0
            self.epochs = epochs
            self.mini_batch_size = mbs
            self.__init_layout(layout)


    def __forwardpropagate(self, input: list[float]) -> None:
        self.input = np.array(input)
        self.hidden = sigmoid(np.dot(self.input, self.ih_weights) + self.hid_biases)
        self.output = sigmoid(np.dot(self.hidden, self.ho_weights) + self.out_biases)


    def __backpropagate_collect_avg(self, target: list[float]) -> None:
        
        output_error = self.output - target
        dOut_dNet = self.output * (1.0 - self.output)
        self.ho_weight_change += np.outer(self.hidden, output_error * dOut_dNet)
        sums = np.dot(output_error * dOut_dNet, self.ho_weights.T)
        dHid_dNet = self.hidden * (1.0 - self.hidden)
        self.ih_weight_change += np.outer(self.input, sums * dHid_dNet)

        self.output_bias_change += output_error * dOut_dNet
        self.hidden_bias_change += sums * dHid_dNet


    def __backpropagate_apply_avg(self) -> None:

        LRATE = self.learning_rate / self.mini_batch_size

        self.ho_weights -= LRATE * self.ho_weight_change
        self.ih_weights -= LRATE * self.ih_weight_change

        self.out_biases -= LRATE * self.output_bias_change
        self.hid_biases -= LRATE * self.hidden_bias_change

        self.ho_weight_change.fill(0)
        self.ih_weight_change.fill(0)

        self.output_bias_change.fill(0)
        self.hidden_bias_change.fill(0)


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


    def train(self, training_inputs, training_outputs, testing_inputs, testing_outputs) -> None:

        self.time = time.time()

        count: int = 0
        avg_mse: float = 0.0

        self.test(testing_inputs, testing_outputs)

        print("\nlearning rate: %.4f\tmini-batch size: %d" % (self.learning_rate, self.mini_batch_size))

        while self.epoch < self.epochs:

            if self.epoch >= 5:
                self.mini_batch_size = 100
            
            if self.epoch >= 10:
                self.learning_rate = 0.1

            temp = list(zip(training_inputs, training_outputs))
            random.shuffle(temp)
            training_inputs, training_outputs = zip(*temp)
            training_inputs, training_outputs = list(training_inputs), list(training_outputs)

            print("epoch %d/%d" % (self.epoch+1, self.epochs), end="")

            for (input, output) in zip(training_inputs, training_outputs):

                if count >= self.mini_batch_size:
                    self.__backpropagate_apply_avg()
                    count = 0

                self.__forwardpropagate(input)
                self.__backpropagate_collect_avg(output)

                avg_mse += MSE([out for out in self.output], output)
                count += 1


            self.epoch += 1
        
            avg_mse = avg_mse / len(training_inputs)
            accuracy = self.test(testing_inputs, testing_outputs)
            print("\tmse: %.4f\taccuracy: %.4f" % (avg_mse, accuracy))
            
            self.time = time.time() - self.time
            



    def toFile(self) -> None:

        s: tuple = (
            self.input_len, self.hidden_len, self.output_len, self.learning_rate,
            self.epoch, self.epochs, self.mini_batch_size
        )
        filepath = "gmodels/(%d-%d-%d)-%.4f-%dof%d-%d-weights.txt" % s

        fh = open(filepath, "w")

        tup = (self.input_len, self.hidden_len, self.output_len, self.learning_rate, self.epoch, self.epochs, self.mini_batch_size)
        fh.write("%d %d %d %.4f %d %d %d\n" % tup)

        for o in range(0, self.output_len):
            fh.write("%f\n" % self.out_biases[o])

        for h in range(0, self.hidden_len):
            fh.write("%f\n" % self.hid_biases[h])

        for i in range(0, self.input_len):
            for h in range(0, self.hidden_len):
                fh.write("%f\n" % self.ih_weights[i][h])

        for h in range(0, self.hidden_len):
            for o in range(0, self.output_len):
                fh.write("%f\n" % self.ho_weights[h][o])

        fh.close()

        filepath = "models/(%d-%d-%d)-%.4f-%dof%d-%d-weights.stats" % s
        fh = open(filepath, "w")
        for x in self.x_axis:
            fh.write("%d " % (x))
        fh.write("\n")
        for y in self.y_axis:
            fh.write("%f " % (y))

        fh.write("\n%f" % (self.time))


    def fromFile(self, filepath: str = "backupweights.txt") -> None:
        
        fh = open(filepath, "r")

        line = fh.readline()
        input_len, hidden_len, output_len, lrate, ep, eps, mbs = line.split(" ")
        input_len  = int(input_len)
        hidden_len = int(hidden_len)
        output_len = int(output_len)
        lrate = float(lrate)
        ep = int(ep)
        eps = int(eps)
        mbs = int(mbs)

        self.__init_layout([input_len, hidden_len, output_len])
        self.learning_rate = lrate
        self.epoch = ep
        self.epochs = eps
        self.mini_batch_size = mbs

        for o in range(0, self.output_len):
            self.out_biases[o] = float(fh.readline())

        for h in range(0, self.hidden_len):
            self.hid_biases[h] = float(fh.readline())
        
        for ih in range(0, self.input_len * self.hidden_len):
            self.ih_weights[ih//self.hidden_len][ih%self.hidden_len] = float(fh.readline())

        for ho in range(0, self.hidden_len * self.output_len):
            self.ho_weights[ho//self.output_len][ho%self.output_len] = float(fh.readline())


    def test(self, testing_inputs, testing_outputs) -> float:

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

        self.x_axis.append(self.epoch)
        self.y_axis.append(correct/total)

        return correct/total

