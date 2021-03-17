from Clients import *
import math


def initializeWeight(initial_value=0, length=35):
    """
        This function initializes a an array of a given value with a given length

    :param initial_value: int
            the initial value of the desired weight

    :param length: int
            the length of the desired weight

    :return: []
        a set of initial weights
    """

    # don't know how to properly initialize
    weights = [initial_value] * length
    return weights


class Coordinator:
    """This class prepares a server to collaboratively train a machine learning (ML) model with the help of a clients
     (also known as parameter server or aggregation server).
     A typical assumption is that the participants are honest whereas the server is honest-but-curious"""

    received_weight = []

    def __init__(self, clients, epochs, rho=1, S=1, M=math.inf):
        """
        We may set M = inf and S = 1 to produce a form of SGD with a varying mini-batch size.

        :param clients: [Client]
            the total clients that registered
        :param epochs: int
            the number of local of epoch for each client
        :param rho: float
            the fraction of clients that perform computation during each round.
             rho controls the global batch size, with rho = 1 corresponding to the full-batch gradient
             descent using all data held by all participants.
        :param S: int
            the number of training steps each client performs over its local dataset
            during each round (i.e., the number of local epochs)
        :param M: int
            the mini-batch size used for the client updates.
            We use M = inf to indicate that the full local dataset is treated as a single mini-batch.
        """

        self.clients = clients
        self.epochs = epochs
        self.rho = rho
        self.S = S
        self.M = M

    def pickTheClients(self, rho=self.rho):
        """
        The coordinator determines Ct , which is the set of randomly selected max(rho; 1) participants.

        :param rho: float
            the fraction of participants
        :return:
            an array of selected participants
        """

        if rho == 1 | rho < 1:
            chosen_clients = self.clients
        else:

            if rho > 1:
                chosen_clients = random.choice(self.clients)

        return chosen_clients

    def receiveWeight(self, weight):
        self.received_weight.append(weight)

    def aggregateTheRecievedModels(self):
        agg_weights = weights

        return agg_weights

    def checkForConvergence(self):

        return true

    def broadcast(self):

        return weights
