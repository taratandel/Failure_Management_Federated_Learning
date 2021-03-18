from coordinator import *
from clients import *
from dataDivider import *

# -------------------------------------------- FEDAVG Algorithm --------------------------------------------
#                                               Qiang Yang, Yang Liu, et al.
#                        Synthesis Lectures on Artificial Intelligence and Machine Learning.
#                                             Morgan & Claypool Publishers, 2020
# -----------------------------------------------------------------------------------------------------------
# create an instance of a coordinator
coordinator = Coordinator(5)
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
    clients.append(Client(data=client_train))


# registers the clients as a participant in the coordinator
coordinator.registerClient(clients)

# the number of total rounds before the learning stops
# there are other criteria to stop the training for start we say rounds.
rounds = 10

for i in range(rounds):
    # coordinator pick the client this can be even a fraction of them that is parametrized by
    # rho if rho = 1 then coordinator selects all the clients. default is rho=1
    # example: coordinator.pickTheClient(rho=0.2)
    clients = coordinator.pickTheClients()
    # after the clients had been chosen now all of should start learning
    for client in clients:
        weight = client.participantUpdate()
        coordinator.receiveWeight(weight)
    coordinator.aggregateTheRecievedModels()
    coordinator.checkForConvergence()
    coordinator.broadcast()
