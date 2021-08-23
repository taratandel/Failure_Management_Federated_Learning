from coordinator import *
from clients import *
from modelTester import *
import matplotlib.pyplot as plt


# -------------------------------------------- FEDAVG Algorithm --------------------------------------------
#                                               Qiang Yang, Yang Liu, et al.
#                        Synthesis Lectures on Artificial Intelligence and Machine Learning.
#                                             Morgan & Claypool Publishers, 2020
# -----------------------------------------------------------------------------------------------------------

def runFedAvg(epoch, m, regularization, clients, name, round):
    # create an instance of a coordinator

    coordinator = Coordinator(epoch, M=m)
    # registers the clients as a participant in the coordinator
    coordinator.registerClient(clients)

    # the number of total rounds before the learning stops
    # there are other criteria to stop the training for start we say rounds.
    average_weights = None
    rounds_acc = []
    round = int(round)
    client_acc = [[None]*round]*len(clients)

    rounds = round
    for r in range(rounds):
        print("round:" + str(r))
        # coordinator pick the client this can be even a fraction of them that is parametrized by
        # rho if rho = 1 then coordinator selects all the clients. default is rho=1
        # example: coordinator.pickTheClient(rho=0.2)
        chosen_clients = coordinator.pickClients()
        # after the clients had been chosen now all of should start learning
        for client in chosen_clients:
            if average_weights is None:
                # train the client with the specified parameters set by the server
                model = client.participantUpdate(coefs=None, intercepts=None, epochs=coordinator.epochs,
                                                 M=coordinator.M, regularization=regularization)
            else:
                model = client.participantUpdate(coefs=average_weights[1], intercepts=average_weights[0],
                                                 epochs=coordinator.epochs, M=coordinator.M, regularization=
                                                 regularization)
            coordinator.receiveModels(model)
            # model = None
            # client = None

        average_weights = coordinator.aggregateTheReceivedModels()
        # chosen_clients = None
        final_model = coordinator.broadcast(average_weights)
        clc = []
        should_plot = False
        should_break = False
        if coordinator.checkForConvergence(r):
            should_plot = True
            should_break = True
        for i in range(len(chosen_clients)):
            client = chosen_clients[i]
            X_test = client.X_test
            y_test = client.y_test
            # they were used for plotting learning curve just in the ```testProcess``` function pass them as variable
            # and the function will plot you the learning curve
            X_train = client.X
            y_train = client.y
            # ----------------------
            model = final_model
            name = client.name

            if r == rounds - 1:
                should_plot = True

            acc = testProcess(X_test, y_test, None, None, model, name + "round" + str(r), should_plot)
            clc.append(acc)
            client_acc[i][r] = acc
        res_acc = list(filter(None, client_acc))
        rounds_acc.append(coordinator.averageAcc(clc))
        if should_plot:
            plotSimpleFigure(rounds_acc, "rounds", "accuracy average",
                             name + "accuracy round plot averaged for all clients" , values2=None)
            for i in range(len(chosen_clients)):
                plotSimpleFigure(res_acc[i], "rounds", "accuracy for client " + str(i),
                                 name + "accuracy round plot for client " + str(i) + " " , values2=None)
        if should_break:
            break

    return final_model, rounds_acc, client_acc


