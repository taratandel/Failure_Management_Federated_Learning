from Clients import *

def initializeWeight(initial_value = 0, length = 35):
    """
    This function initializes a an array of a given value with a given length

    :parameter

    initial_value: int
        the initial value of the desired weight

    :parameter

    length: int
        the length of the desired weight

    
    """

    # don't know how to properly initialize
    weights = [initial_value] * length
    return weights


class Coordinator:
    """This class prepares a server to collaboratively train a machine learning (ML) model with the help of a clients
     (also known as parameter server or aggregation server).
     A typical assumption is that the participants are honest whereas the server is honest-but-curious"""

    def __init__(self, clients, epochs, rho=1, S=1, M=inf):
        """
        We may set M = inf and S = 1 to produce a form of SGD with a varying mini-batch size.

        Parameters
        ----------
        clients: [Client]
            the total clients that registered
        epochs: int
            the number of local of epoch for each client
        rho: float
            the fraction of clients that perform computation during each round.
             rho controls the global batch size, with rho = 1 corresponding to the full-batch gradient
             descent using all data held by all participants.
        S: int
            the number of training steps each client performs over its local dataset
            during each round (i.e., the number of local epochs)
        M: int
            the mini-batch size used for the client updates.
            We use M = inf to indicate that the full local dataset is treated as a single mini-batch.
        """

        self.clients = clients
        self.epochs = epochs
        self.rho = rho
        self.S = S
        self.M = M

    def pickTheClients(self, rho=1):
        if rho == 1 | rho < 1:
            chosen_clients = self.clients
        else:

            if rho > 1:
                chosen_clients = random.choice(self.clients)

        return chosen_clients

    def setVariables(self, ):


    def aggregateTheRecievedModels(self):
        agg_weights = weights

        return agg_weights

    def checkForConvergence(self):

        return true

    def broadcast(self):

        return weights