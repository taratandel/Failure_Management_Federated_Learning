from coordinator import *
from clients import *
from dataDivider import *
from ANN_Classifier import *
from modelTester import *
import math
from numpy import linspace as lsp
# -------------------------------------------- FEDAVG Algorithm --------------------------------------------
#                                               Qiang Yang, Yang Liu, et al.
#                        Synthesis Lectures on Artificial Intelligence and Machine Learning.
#                                             Morgan & Claypool Publishers, 2020
# -----------------------------------------------------------------------------------------------------------

# Load the dataframe
df = loadDataFrame("Labelled_Data.csv")
# On-Hot-Encode the labels
df = oneHotEncode(df)
# split test set with 20% (default is 20% if you want to change the percentage
# just call the function with desire percentage example: divideTestSet(df, test_size = 0.1)
train, test = divideTestSet(df)
# here I divided by the equipment type the rest of the code will work regardless of your division
# for your ease of use just instead of clients_train give any split you want
clients_train = divideByeqType(train)
# divides randomly the test set for the clients the reason I did that was to have all
# the labels in all the clients. this division is fractionated by number number of samples that
# each client has.
clients_test = divideRandomly(test, calcFractions(clients_train))
# initialize the weight still need work but for now don't use it
# initial_weights = coordinator.initializeWeight()

# creates the client with the given data
clients = []
for client_train in clients_train:
    client = Client(data=client_train)
    clients.append(client)

for epochs in range(1, 1500, 10):
    # create an instance of a coordinator
    coordinator = Coordinator(epochs)
    # registers the clients as a participant in the coordinator
    coordinator.registerClient(clients)

    # the number of total rounds before the learning stops
    # there are other criteria to stop the training for start we say rounds.
    for rounds in range(10, 100, 2):
        for learning_rate in lsp(1, 0.000000001, 10):

            for i in range(rounds):
                average_weights = None

# the number of total rounds before the learning stops
# there are other criteria to stop the training for start we say rounds.
rounds = 20000
average_weights = None
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
            model = client.participantUpdate(coefs=None, intercepts=None, epochs=coordinator.epochs, M=coordinator.M)
        else:
            model = client.participantUpdate(coefs=average_weights[1], intercepts=average_weights[0],
                                             epochs=coordinator.epochs, M=coordinator.M)
        coordinator.receiveModels(model)
        model = None
        client = None
    if i == rounds-1:
        average_weights = coordinator.aggregateTheReceivedModels(should_empty=False)
    else:
        average_weights = coordinator.aggregateTheReceivedModels()
    chosen_clients = None

# ------------------------------------------------ Model Comparison -------------------------------------
# here we try to compare the performance of the model in two cases: trained alone or in collaborative mode
# -------------------------------------------------------------------------------------------------------

for i in range(len(clients)):
    # train a model for each client without collaborating with other clients
    client = clients[i]
    client_model = client.participantUpdate(None, None, M=math.inf, epochs=2000)
    # prepare a test set to be used for the testing phase for both models
    test = clients_test[i]
    X_test, y_test = cleanData(test)
    # create the tester for the client trained alone
    tester_alone = ModelTester(X_test, y_test, client_model)
    # create the tester for the same client but this time it trained collaboratively
    final_model = coordinator.broadcast(average_weights, i)
    tester_collaborative = ModelTester(X_test, y_test, final_model)

    tester_alone.calcStatistic("test_alone client number:" + str(i))
    tester_collaborative.calcStatistic("test_collaborative client number:" + str(i))