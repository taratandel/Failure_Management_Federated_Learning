from Coordinator import *
from Clients import *

# -------------------------------------------- FEDAVG Algorithm --------------------------------------------
#                                               Qiang Yang, Yang Liu, et al.
#                        Synthesis Lectures on Artificial Intelligence and Machine Learning.
#                                             Morgan & Claypool Publishers, 2020
# -----------------------------------------------------------------------------------------------------------
coordinator = Coordinator()
initial_weights = coordinator.initializeWeight()

# client register
rounds = 10
for i in range(rounds):
    clients = coordinator.pickTheClients()
    for client in clients:
        weight = client.participantUpdate()
        coordinator.receiveWeight(weight)
    coordinator.aggregateTheRecievedModels()
    coordinator.checkForConvergence()
    coordinator.broadcast()
