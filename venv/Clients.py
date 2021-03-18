from DataDivider import loadDataFrame as lDF
import math
import pandas as pd
import numpy as np

class Client:
    """
    This class is a representation of the participants that are willing to collaboratively
    train a machine learning model
    """
    w1_1 =[]
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

        if data is not None:
            self.dataFrame = data
        elif path is not None:
            self.dataFrame = lDF(path)
        else:
            raise Exception("Sorry, provide a data or a path. both cannot be empty")

    def participantUpdate(self, weights, epochs, M):
        """
        Update it's weight according to the parameters and the weights gained by the server

        :param weights: []
            weights coming from the server
        :param epochs: int
            the number of local epochs that each client should do
        :param M:
            the number of mini-batches
        :return:
            the locally updated weights
        """
        self.w1_1 = weights
        for i in range(epochs):
            if M==math.inf:
                print("noon")
    #             run the model with the whole data set
            else:
                shuffled = self.dataFrame.sample(frac=1)
                result = np.array_split(shuffled, 5)
    #             run the model with these selected data
                return result
        return weights

    def loadTheData(self, nameOfTheFile):

