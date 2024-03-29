from dataDivider import loadDataFrame as lDF
from dataDivider import cleanData as cD
from dataDivider import divideTestSet
from ANN_Classifier import *
from dataDivider import *
import math
import pandas as pd
import numpy as np


class Client:
    """
    This class is a representation of the participants that are willing to collaboratively
    train a machine learning model
    """

    def __init__(self, data=None, path=None, prepare_for_testing=False, name=None):

        """
        initialize the client class
        :param data: pandas.DataFrame
            to pass the data_frame directly
        :param path: str
            to ask to load a csv file
        :exception
            if both data and path are None. we need at least one to initialize our data
        """

        # dataFrame: pd.DataFrame
        self.weights = []
        self.name = name
        self.dataFrame = None
        self.test = None

        if data is not None:
            dataFrame = data
        elif path is not None:
            dataFrame = lDF(path)
        else:
            raise Exception("Sorry, provide a data or a path. both cannot be empty")
        if prepare_for_testing:
            self.dataFrame, self.test = divideTestSet(dataFrame)
            self.X_test, self.y_test = self.cleanData(self.test)
        else:
            self.dataFrame = dataFrame
        self.printData()
        self._cleanData()
        # self.dataFrame = []
        # self.test = []

    def cleanData(self, dataFrame):
        return cD(dataFrame)

    def _cleanData(self):
        self.X, self.y = cD(self.dataFrame)

    def participantUpdate(self, coefs, intercepts, M, regularization, epochs=None):
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


    def setTest(self, X_test, y_test):
        self.X_test = X_test
        self.y_test = y_test

    def printData(self):
        file_name = self.name + " " + "clients data" + ".txt"
        file = open(file_name, "w")
        str1 = self.name + "" + str(len(self.dataFrame))
        print(str1)
        file.write(str1)
        gs = self.dataFrame.groupby('label').size()

        for index, value in gs.items():
            str2 = "\n " + "client" + " " + "class number" + " " + str(index) + " " + "number of labels" + " " + str(value)
            file.write(str2)
            print(str2)

        file.close()


def clientBuilder():
    test_sets, train_sets, concatenated_test, concatenated_train = prepareDataSet()
    client_set = []
    for i in range(len(train_sets)):
        train = train_sets[i]
        test = test_sets[i]

        client = Client(train)
        test_X, test_y = cD(test)
        client.setTest(test_X, test_y)
        client_set.append(client)
    return client_set, test_sets, train_sets, concatenated_test, concatenated_train


# scenario 1 is where we have three groups of randomly picked links. just to make sure that each link
# is only in one group
def clientBuilderForScenario1(name):
    df = loadDataFrame("Labelled_Data.csv")
    divided_gp = divideByLinkID(df)
    pickedGps = pickGroupsRandomly(3, divided_gp)
    gps = buildClient(pickedGps, name)
    return gps


def clientBuilderForClassesPerEach(name):
    df = loadDataFrame("Labelled_Data.csv")
    divided_gp = divideByLinkID(df)
    pickedGps = pickGPSForClassesPerEach(3, divided_gp)
    gps = buildClient(pickedGps, name)

    return gps


def clientBuilderForClassesProportional(name, proportions=[6, 4, 4]):
    df = loadDataFrame("Labelled_Data.csv")
    divided_gp = divideByLinkID(df)
    pickedGps = pickGPSForClassesSelected(divided_gp, proportions)
    gps = buildClient(pickedGps, name)

    return gps
def clientBuilderForOneClientMissing1Class(name, missingClass):
    df = loadDataFrame("Labelled_Data.csv")
    divided_gp = divideByLinkID(df)
    labels = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    labels.remove(missingClass)
    pickedGps = pickGPSForClassesSelected(divided_gp, proportions=[6,6,5], labels=[[0.0, 1.0, 2.0, 3.0, 4.0, 5.0],[0.0, 1.0, 2.0, 3.0, 4.0, 5.0],labels])
    gps = buildClient(pickedGps, name)

    return gps
def clientBuilderForClientMissing1class(name):
    df = loadDataFrame("Labelled_Data.csv")
    divided_gp = divideByLinkID(df)
    pickedGps = pickGPSForClassesSelected(divided_gp, proportions=[6,6,5], labels=[[0.0, 1.0, 2.0, 3.0, 4.0, 5.0],[0.0, 1.0, 2.0, 3.0, 4.0, 5.0],[0.0, 1.0, 3.0, 4.0, 5.0]])
    gps = buildClient(pickedGps, name)

    return gps

def buildClientWithPath(name, missingClass):
    clients = []
    for i in range(3):
        i = i + 1
        df = loadDataFrame("train"+str(i)+".csv")
        if i == 1 :
            df.drop(df[df['label'] == missingClass].index, axis=0, inplace = True)
        client = Client(data=df, path=None, prepare_for_testing=False, name=name + " " + str(i))
        testdf = loadDataFrame("test"+str(i)+".csv")
        client.test = testdf
        x_test, y_test = cD(testdf)
        client.setTest(x_test,y_test)
        clients.append(client)
    return clients

def buildWithoneTest(name, missingClass):
    clients = []
    for i in range(3):
        i = i + 1
        df = loadDataFrame("train" + str(i) + ".csv")
        if i == 1:
            df.drop(df[df['label'] == missingClass].index, axis=0, inplace=True)
        client = Client(data=df, path=None, prepare_for_testing=False, name=name + " " + str(i))
        testdf = loadDataFrame("test.csv")
        client.test = testdf
        x_test, y_test = cD(testdf)
        client.setTest(x_test, y_test)
        clients.append(client)
    return clients

def buildClient(data_sets, name):
    clients = []
    i = 0
    for data in data_sets:
        client = Client(data=data, path=None, prepare_for_testing=True, name=name + " " + str(i))
        clients.append(client)
        i = i + 1
    return clients

def buildClientFairly(datasetsName, missingClass):
    dataset = loadDataFrame(datasetsName)
    train1, test1, train2, test2, train3, test3 = divideSetEqualy(dataset)
    df = train1

    df.drop(df[df['label'] == missingClass].index, axis=0, inplace=True)

    train1 = df

    client1 = Client(data=train1, path=None, prepare_for_testing=False, name="client1"+str(missingClass))
    client1.test = test1
    x,y = cD(test1)
    client1.setTest(x,y)

    client2 = Client(data=train2, path=None, prepare_for_testing=False, name="client2"+str(missingClass))
    client2.test = test2
    x,y = cD(test2)
    client2.setTest(x,y)

    client3 = Client(data=train3, path=None, prepare_for_testing=False, name="client3"+str(missingClass))
    client3.test = test3
    x,y = cD(test3)
    client3.setTest(x,y)

    return [client1, client2, client3]