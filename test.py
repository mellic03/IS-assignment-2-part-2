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

    return (x_axis, y_axis)


def main():

    mbs_labels = [
        "1",
        "5",
        "20",
        "100",
        "300"
    ]

    mbs_stats = [
        load_data("models/(3072-30-10)-0.1000-50of50-1-weights.stats"),
        load_data("models/(3072-30-10)-0.1000-50of50-5-weights.stats"),
        load_data("models/(3072-30-10)-0.1000-50of50-20-weights.stats"),
        load_data("models/(3072-30-10)-0.1000-50of50-100-weights.stats"),
        load_data("models/(3072-30-10)-0.1000-50of50-300-weights.stats")
    ]


    lr_labels = [
        "0.001",
        "0.01",
        "0.1",
        "1.0",
        "10.0",
        "100.0"
    ]

    lr_stats = [
        load_data("models/(3072-30-10)-0.0010-50of50-100-weights.stats"),
        load_data("models/(3072-30-10)-0.0100-50of50-100-weights.stats"),
        load_data("models/(3072-30-10)-0.1000-50of50-100-weights.stats"),
        load_data("models/(3072-30-10)-1.0000-50of50-100-weights.stats"),
        load_data("models/(3072-30-10)-10.0000-50of50-100-weights.stats"),
        load_data("models/(3072-30-10)-100.0000-50of50-100-weights.stats")
    ]

    # # mbs
    # for i in range(0, len(mbs_labels)):
    #     plt.figure(i)
    #     plt.ylim(0.0, 0.5)
    #     plt.yticks(np.arange(0.0, 0.5, 0.02))

    #     plt.xlabel("No. Epochs")
    #     plt.ylabel("Accuracy")
    #     plt.xticks(range(0, 21))
    #     plt.plot(mbs_stats[i][0], mbs_stats[i][1], label=mbs_labels[i])
    #     plt.savefig("./report/figures/mbs_" + str(i) + ".png")
    #     plt.close()

    # # lr
    # for i in range(0, len(lr_labels)):
    #     plt.figure(i)
    #     plt.ylim(0.0, 0.5)
    #     plt.yticks(np.arange(0.0, 0.5, 0.02))

    #     plt.xlabel("No. Epochs")
    #     plt.ylabel("Accuracy")
    #     plt.xticks(range(0, 21))
    #     plt.plot(lr_stats[i][0], lr_stats[i][1], label=lr_labels[i])
    #     plt.savefig("./report/figures/lr_" + str(i) + ".png")
    #     plt.close()



    # mbs comparison
    plt.figure(1)
    for i in range(0, len(mbs_labels)):
        plt.ylim(0.0, 0.5)
        plt.yticks(np.arange(0.0, 0.5, 0.02))

        plt.xlabel("Number of Epochs")
        plt.ylabel("Model Accuracy")
        plt.xticks(range(0, 51, 5))
        plt.plot(mbs_stats[i][0], mbs_stats[i][1], label=mbs_labels[i])
    plt.legend(mbs_labels, loc="upper left").set_title("Mini-Batch Size")
    plt.grid()
    plt.savefig("./report/figures/mbs_cmp.png")
    plt.close()

    # lr comparison
    plt.figure(1)
    for i in range(0, len(lr_labels)):
        plt.ylim(0.0, 0.5)
        plt.yticks(np.arange(0.0, 0.5, 0.02))

        plt.xlabel("Number of Epochs")
        plt.ylabel("Model Accuracy")
        plt.xticks(range(0, 51, 5))
        plt.plot(lr_stats[i][0], lr_stats[i][1], label=lr_labels[i])
    plt.legend(lr_labels, loc="upper left").set_title("Learning Rate")
    plt.grid()
    plt.savefig("./report/figures/lr_cmp.png")
    plt.close()


    # all_labels = mbs_labels + lr_labels
    # all_stats = mbs_stats + lr_stats

    # mbs + lr comparison
    # plt.figure(1)
    # for i in range(0, len(all_labels)):
    #     plt.ylim(0.0, 0.5)
    #     plt.yticks(np.arange(0.0, 0.5, 0.02))

    #     plt.xlabel("No. Epochs")
    #     plt.ylabel("Accuracy")
    #     plt.xticks(range(0, 21))
    #     plt.plot(all_stats[i][0], all_stats[i][1], label=all_labels[i])
    # plt.legend(all_labels)
    # plt.savefig("./report/figures/all_cmp.png")
    # plt.close()





if __name__ == "__main__":
    main()

