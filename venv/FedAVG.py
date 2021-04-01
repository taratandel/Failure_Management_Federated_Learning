from coordinator import *
from clients import *
from dataDivider import cleanData, loadDataFrame
from ANN_Classifier import *
from modelTester import *
import math
from numpy import linspace as lsp
import joblib

# -------------------------------------------- FEDAVG Algorithm --------------------------------------------
#                                               Qiang Yang, Yang Liu, et al.
#                        Synthesis Lectures on Artificial Intelligence and Machine Learning.
#                                             Morgan & Claypool Publishers, 2020
# -----------------------------------------------------------------------------------------------------------
# creates the client with the given data
clients = []
for i in range(1, 4):
    name = "df" + str(i) + ".csv"
    client = Client(path=name)
    clients.append(client)
epochs = 1
# create an instance of a coordinator
coordinator = Coordinator(epochs)
# registers the clients as a participant in the coordinator
coordinator.registerClient(clients)
test = loadDataFrame("test.csv")
X_test, y_test = cD(test)
# the number of total rounds before the learning stops
# there are other criteria to stop the training for start we say rounds.
rounds = 5000
average_weights = None
rounds_acc = []
for i in range(rounds):
    print("round:" + str(i))
    # coordinator pick the client this can be even a fraction of them that is parametrized by
    # rho if rho = 1 then coordinator selects all the clients. default is rho=1
    # example: coordinator.pickTheClient(rho=0.2)
    chosen_clients = coordinator.pickClients()
    # after the clients had been chosen now all of should start learning
    for client in chosen_clients:
        if average_weights is None:
            # train the client with the specified parameters set by the server
            model = client.participantUpdate(coefs=None, intercepts=None, epochs=coordinator.epochs, M=coordinator.M, regularization=0.00000001)
        else:
            model = client.participantUpdate(coefs=average_weights[1], intercepts=average_weights[0],
                                             epochs=coordinator.epochs, M=coordinator.M,regularization=0.00000001)
        coordinator.receiveModels(model)
        model = None
        client = None

    average_weights = coordinator.aggregateTheReceivedModels()
    chosen_clients = None
    final_model = coordinator.broadcast(average_weights)
    tester_collaborative = ModelTester(X_test, y_test, final_model)
    tester_collaborative.calcStatistic("test_collaborative client number:" + str(i))

    rounds_acc.append(tester_collaborative.acc)

plotSimpleFigure(rounds_acc, 'rounds', 'accuracy', "accuracy-round plot")

# ------------------------------------------------ Model Comparison -------------------------------------
# here we try to compare the performance of the model in two cases: trained alone or in collaborative mode
# -------------------------------------------------------------------------------------------------------

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



