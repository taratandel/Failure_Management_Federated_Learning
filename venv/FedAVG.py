from coordinator import *
from clients import *
from dataDivider import cleanData, loadDataFrame, prepareDataSet
from ANN_Classifier import *
from modelTester import *
import math
from numpy import linspace as lsp
import joblib
import time


# -------------------------------------------- FEDAVG Algorithm --------------------------------------------
#                                               Qiang Yang, Yang Liu, et al.
#                        Synthesis Lectures on Artificial Intelligence and Machine Learning.
#                                             Morgan & Claypool Publishers, 2020
# -----------------------------------------------------------------------------------------------------------

def runFedAvg(epoch, m, regularization):
    # create an instance of a coordinator

    coordinator = Coordinator(epoch, M=m)
    # registers the clients as a participant in the coordinator
    coordinator.registerClient(clients)

    # the number of total rounds before the learning stops
    # there are other criteria to stop the training for start we say rounds.
    average_weights = None
    rounds_acc = []
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
        joblib.dump(final_model, filename="finalmodel")
        tester_collaborative = ModelTester(X_test, y_test, final_model)
        tester_collaborative.calcStatistic()

        rounds_acc.append(tester_collaborative.acc)
    name_plt = "accuracy-round plot epoch = " + str(epoch) + "regularization_term:" + \
               str(regularization) + "mini_batch size:" + str(m)
    plotSimpleFigure(rounds_acc, 'rounds', 'accuracy', name_plt)
    return rounds_acc[-1]


#
#
# epochs = 200
# batch_size = 50
# rounds = 1000
# current_acc = 0
# best_epochs = 0
# best_batch_size = 0
# best_regularization_term = 0
# list_param = []
# for epoch in range(1, epochs, 10):
#     for m in list(range(1, batch_size, 10)) + [math.inf]:
#         for regularization in [0.1, 0.001, 0.00001, 0.0000001]:
#             list_param.append([epoch, m, regularization])
#
# for i in range(0, len(list_param)):
#     epoch = list_param[i][0]
#     m = list_param[i][1]
#     regularization = list_param[i][2]
#     acc = runFedAvg(epoch, m, regularization)
#     if acc > current_acc:
#         current_acc = acc
#         best_epochs = epoch
#         best_batch_size = m
#         best_regularization_term = regularization
#
# print("best epoch:" + str(best_epochs) + "\n" +
#       "best mini_batch size:" + str(best_batch_size) + "\n" +
#       "best regularization term:" + str(best_regularization_term))

# ------------------------------------------------ Model Comparison -------------------------------------
# here we try to compare the performance of the model in two cases: trained alone or in collaborative mode
# -------------------------------------------------------------------------------------------------------
# First TEsting model
test_separated = prepareDataSet()
test = loadDataFrame("test.csv")
X_test, y_test = cD(test)


test_not_separated = prepareDataSet(False)
for i in  range(1, len(test_separated)+1):
    test_separated[i-1].to_csv("test_separated.csv" + str(i), index=False)
    test_not_separated[i-1].to_csv("test_not_separated.csv" + str(i), index=False)
# creates the client with the given data
# clients_tns = []
# clients_ts = []
#
# for i in range(1, 4):
#     client = Client(data=test_separated[i])
#     # second scenario
#     X, y = cD(divideRandomly(test, calcFractions(test_separated)))
#     client.setTest(X,y)
#     clients_ts.append(client)
#
#     # Third scenario
#     client = Client(data=test_not_separated[i], prepare_for_testing=True)
#     clients_tns.append(client)
#
# for i in range(len(clients)):
#     # train a model for each client without collaborating with other clients
#     client = clients[i]
#     client_model = client.participantUpdate(None, None, M=math.inf, epochs=2000, regularization=0.00000001)
#     # prepare a test set to be used for the testing phase for both models
#
#     # create the tester for the client trained alone
#     tester_alone = ModelTester(X_test, y_test, client_model)
#     # create the tester for the same client but this time it trained collaboratively
#     final_model = coordinator.broadcast(average_weights, i)
#     tester_alone.calcStatistic("test_alone client number:" + str(i))
