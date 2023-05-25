import numpy as np
import matplotlib.pyplot as plt
import neuralnetwork as NN

def unpickle(filepath: str) -> dict:
    import pickle
    with open(filepath, "rb") as fh:
        dict = pickle.load(fh, encoding="bytes")
    return dict

def to_thing(category: int) -> list[int]:
    result: list[int] = [0] * 10
    result[category] = 1
    return result

def onExit(net: NN.NeuralNetwork):
    net.toFile()


def load_data(filepath: str) -> tuple[list[int], list[float]]:
    fh = open(filepath, "r")

    x_axis = fh.readline().split(" ")
    y_axis = fh.readline().split(" ")

    x_axis.pop()
    y_axis.pop()

    for x in range(0, len(x_axis)):
        x_axis[x] = int(x_axis[x])
    for y in range(0, len(y_axis)):
        y_axis[y] = float(y_axis[y])

    print(x_axis)
    print(y_axis)

    return (x_axis, y_axis)


def main():

    labels = [
        "mbs = 1      lr=0.1",
        "mbs = 5      lr=0.1",
        "mbs = 20     lr=0.1",
        "mbs = 300    lr=0.1",
       
        "mbs = 100    lr = 0.001",
        "mbs = 100    lr = 0.01 ",
        "mbs = 100    lr = 0.1  ",
        "mbs = 100    lr = 1.0  ",
        "mbs = 100    lr = 10.0 "
    ]

    stats = [
        load_data("models/(3072-30-10)-0.1000-20of20-1-weights.stats"),
        load_data("models/(3072-30-10)-0.1000-20of20-5-weights.stats"),
        load_data("models/(3072-30-10)-0.1000-20of20-20-weights.stats"),
        load_data("models/(3072-30-10)-0.1000-20of20-300-weights.stats"),

        load_data("models/(3072-30-10)-0.0010-20of20-100-weights.stats"),
        load_data("models/(3072-30-10)-0.0100-20of20-100-weights.stats"),
        load_data("models/(3072-30-10)-0.1000-20of20-100-weights.stats"),
        load_data("models/(3072-30-10)-1.0000-20of20-100-weights.stats"),
        load_data("models/(3072-30-10)-10.0000-20of20-100-weights.stats")
    ]

    # for i in range(0, len(labels)):
    #     plt.legend()
    #     plt.xlabel("No. Epochs")
    #     plt.ylabel("Accuracy")
    #     plt.xticks(range(0, 21))
    #     plt.yticks(np.arange(0.0, 1.0, 0.01))
    #     plt.plot(stats[i][0], stats[i][1], label=labels[i])
    # plt.show()


    for i in range(0, len(labels)):
        plt.figure(i)
        plt.ylim(0.0, 0.5)
        plt.yticks(np.arange(0.0, 0.5, 0.02))

        plt.xlabel("No. Epochs")
        plt.ylabel("Accuracy")
        plt.xticks(range(0, 21))
        plt.plot(stats[i][0], stats[i][1], label=labels[i])
        plt.savefig("./report/figures/fig_" + str(i) + ".png")





if __name__ == "__main__":
    main()

