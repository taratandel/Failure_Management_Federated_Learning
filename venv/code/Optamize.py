from FedAVG import runFedAvg as rFA
from modelTester import *
import pickle
import math


def optimize(epochs, batch_size, clients, rounds, name):
    current_acc = 0
    best_epochs = 0
    best_batch_size = 0
    batch_size_acc = {}
    epoch_acc = {}
    roundsi = 0
    for epoch in epochs:
        for m in batch_size + [math.inf]:
            roundsi += 1
            plot_name = name + ("round %s" % (roundsi))
            _, acc_round, clients_accuracy = rFA(epoch, m, 0.000001, clients, plot_name, rounds)
            if acc_round[-1] > current_acc:
                current_acc = acc_round[-1]
                best_epochs = epoch
                best_batch_size = m
                batch_size_acc[m] = current_acc
                epoch_acc[m] = current_acc
                print("current epoch:" + str(epoch) + "\n" +
                      "current size:" + str(m))
            current_file = open(plot_name + "bests", "w")
            current_file.write("current best epoch for trial %s automated: %s round %s \n" % (name, best_epochs, roundsi))
            current_file.write("current best batch size for trial %s automated: %s round %s\n" % (name, best_batch_size, roundsi))
            current_file.close()

            plotSimpleFigure(acc_round, 'rounds', 'accuracy', plot_name + " " + "accuracy per round")
            plotSimpleFigure(clients_accuracy[0], 'rounds', 'accuracy of client 0', plot_name + " " + "accuracy per round "
                                                                                                 "per client 0")
            plotSimpleFigure(clients_accuracy[1], 'rounds', 'accuracy of client 1', plot_name + " " + "accuracy per round "
                                                                                                 "per client 1")
            plotSimpleFigure(clients_accuracy[2], 'rounds', 'accuracy of client 2', plot_name + " " + "accuracy per round "
                                                                                                 "per client 2")

    print("best epoch:" + str(best_epochs) + "\n" +
          "best mini_batch size:" + str(best_batch_size))
    plotSimpleFigure(values=batch_size_acc.keys(), xlabel="batches", ylabel="accuracy",
                     title=plot_name + "accuracy per batch size", values2=batch_size_acc.values())
    plotSimpleFigure(values=epoch_acc.keys(), xlabel="batches", ylabel="accuracy", title=plot_name + "accuracy per epoch",
                     values2=epoch_acc.values())

    bests_file = open(plot_name + "bests", "w")
    bests_file.write("best epoch for trial %s automated: %s \n" % (plot_name, best_epochs))
    bests_file.write("best batch size for trial %s automated: %s \n" % (plot_name, best_batch_size))
    bests_file.close()
    with open('batch size accuracy array %s.pkl' % plot_name, 'wb') as output:
        # Pickle dictionary using protocol 0.
        pickle.dump(batch_size_acc, output)

    with open('epoch size accuracy array %s.pkl' % plot_name, 'wb') as output:
        # Pickle dictionary using protocol 0.
        pickle.dump(epoch_acc, output)
    return best_epochs, best_batch_size
