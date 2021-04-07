from dataDivider import loadDataFrame as lDF
from dataDivider import cleanData as cD
from ANN_Classifier import *
import math
import pandas as pd
import numpy as np


class Client:
    """
    This class is a representation of the participants that are willing to collaboratively
    train a machine learning model
    """


    def __init__(self, data=None, path=None):

        """
        initialize the client class
        :param data: pandas.DataFrame
            to pass the data_frame directly
        :param path: str
            to ask to load a csv file
        :exception
            if both data and path are None. we need at least one to initialize our data
        """

        self.dataFrame: pd.DataFrame
        self.weights = []

        if data is not None:
            dataFrame = data
        elif path is not None:
            dataFrame = lDF(path)
        else:
            raise Exception("Sorry, provide a data or a path. both cannot be empty")
        self.__cleanData(dataFrame)

    def __cleanData(self, dataFrame):
        self.X, self.y = cD(dataFrame)

    def _cleanData(self):
        self.X, self.y = cD(self.dataFrame)

    def participantUpdate(self, coefs, intercepts, epochs, M, regularization):
        """
        Update it's weight according to the parameters and the weights gained by the server

        :param regularization: float
            L2 Penalty of the learner
        :param intercepts: []
            the intercepts that are coming form the server
        :param coefs: []
            weights coming from the server
        :param epochs: int
            the number of local epochs that each client should do
        :param M:
            the number of mini-batches
        :return:
            the locally updated weights
        """
        return trainANN(self.X, self.y, epochs, M, coefs, intercepts, regularization)

    def getNumberOfSamples(self):
        """
        the getter of the client
        :return:
            it returns the number of samples of the client
        """
        return len(self.X)
