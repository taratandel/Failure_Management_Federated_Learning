from clients import *
import math
from modelAverage import weightedAverageModel, weightedAverageloss


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

    def __init__(self, epochs=1, rho=1, M=math.inf, epsilon=0.1):
        """
        We may set M = inf and epochs = 1 to produce a form of SGD with a varying mini-batch size.

        :param rho: float
            the fraction of clients that perform computation during each round.
             rho controls the global batch size, with rho = 1 corresponding to the full-batch gradient
             descent using all data held by all participants.
        :param epochs: int
            the number of training steps each client performs over its local dataset
            during each round (i.e., the number of local epochs)
        :param M: int
            the mini-batch size used for the client updates.
            We use M = inf to indicate that the full local dataset is treated as a single mini-batch.
        """
        self.received_intercept = []
        self.received_coefs = []
        self.received_loss = []
        self.agg_weights = []
        self.clients = []
        self.epochs = epochs
        self.rho = rho
        self.M = M
        self.total_unchanged_rounds = 0
        self.epsilon = epsilon
        self.weighted_loss = 0.00001

    def pickClients(self, rho=1):
        """
        The coordinator determines Ct , which is the set of randomly selected max(rho; 1) participants.

        :param rho: float
            the fraction of participants
        :return:
            an array of selected participants
        """

        if (rho == 1) | (rho < 1):
            chosen_clients = self.clients
        else:
            if rho > 1:
                chosen_clients = random.choice(self.clients)

        return chosen_clients

    def registerClient(self, client):
        """
        This function registers clients to participate in the learning process
        :param client: Client
            The client that wants to be registered
        """
        for cl in client:
            self.clients.append(cl)

    def receiveModels(self, model):
        """
        receive the weights from the clients and keep them all together
        :param model: model
            and array of intercepts and coefficients from the current the client
        """
        self.received_models = model
        self.received_intercept.append(model.intercepts_)
        self.received_coefs.append(model.coefs_)
        self.received_loss.append(model.loss_)

    def aggregateTheReceivedModels(self, criteria='w'):
        """
        aggregate the weights and intercepts of all the clients based in the specified criteria

        :param: criteria: string default='w'
            can be 'w', 'smw', 'normal'
            'w': weighted average by the number of samples of each client
            'smw': weighted average by the number of samples that's been seen during the training phase of the client
            'normal': normal averaging
        :return:
            and aggregated array of all the intercept and weights that are averaged
        """
        if criteria == 'w':
            agg_weights = weightedAverageModel(self.received_intercept, self.received_coefs, self.__getClientsSamples())
        self.received_coefs = []
        self.received_intercept = []

        return agg_weights

    def __getClientsSamples(self):
        """
        a private function for calculating the number of samples in each client
        :return:
            returns a list with number of samples of each client
        """
        clients_samples = []
        for client in self.clients:
            clients_samples.append(client.getNumberOfSamples())

        return clients_samples

    def checkForConvergence(self):
        return true

    def broadcast(self, avg_weights):
        model = self.received_models
        model.coefs_ = avg_weights[1]
        model.intercepts_ = avg_weights[0]
        return model

    def checkForConvergence(self, round):
        print(self.weighted_loss)
        print("+++++++++++++weighted loss before+++++++++++")
        if (self.total_unchanged_rounds > 20) and (round > 500):
            return True

        else:
            current_loss = weightedAverageloss(self.received_loss, self.__getClientsSamples())
            print(current_loss)
            print("+++++++++++++current loss+++++++++++")
            change_percentage = (current_loss - self.weighted_loss) / self.weighted_loss
            if change_percentage < self.epsilon:
                self.total_unchanged_rounds = self.total_unchanged_rounds + 1

            else:
                self.total_unchanged_rounds = 0

            self.weighted_loss = current_loss
            print(self.weighted_loss)
            print("+++++++++++++weighted loss after+++++++++++")
            self.received_loss = []
            return False

    def averageAcc(self, accs):
        return weightedAverageloss(accs, self.__getClientsSamples())
