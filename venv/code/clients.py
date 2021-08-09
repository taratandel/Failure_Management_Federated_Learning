from dataDivider import loadDataFrame as lDF
from dataDivider import cleanData as cD
from dataDivider import divideTestSet
from ANN_Classifier import *
from dataDivider import *
import math
import pandas as pd
import numpy as np


def cleanData(dataFrame):
    return cD(dataFrame)


class Client:
    """
    This class is a representation of the participants that are willing to collaboratively
    train a machine learning model
    """

    def __init__(self, data=None, path=None, prepare_for_random_testing=False, name=None,
                 missing_client_train_label=None):

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

        if prepare_for_random_testing:
            self.dataFrame, self.test = divideTestSet(dataFrame)
        else:
            self.test = dataFrame.groupby('label'). \
                apply(lambda x: x.sample(int(len(x) * 0.2)))
            self.dataFrame = pd.concat([dataFrame, self.test, self.test]).drop_duplicates(keep=False)

        if missing_client_train_label:
            self.dataFrame = self.dataFrame[self.dataFrame["label"] != missing_client_train_label]

        self.dataFrame.to_csv(name + "train.csv", index=False)
        self.test.to_csv(name + "test.csv", index=False)

        self._cleanData()
        self.printData()

    def __init__(self, train_path, test_path, name):
        self.weights = []
        self.name = name
        self.test = lDF(test_path)
        self.dataFrame = lDF(train_path)
        self._cleanData()

    def _cleanData(self):

        self.X_test, self.y_test = cleanData(self.test)
        self.X, self.y = cD(self.dataFrame)
        self.test = self.test.reset_index(drop=True)
        # self.test.to_csv(self.name + " test.csv")
        # self.dataFrame.to_csv(self.name + " train.csv")

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
        str1 = self.name + " " + str(len(self.dataFrame)) + "\n "
        print(str1)
        file.write(str1)
        gs = self.dataFrame.groupby('label').size()

        for index, value in gs.items():
            str2 = str(index) + " " + "(" + str(value) + ")" + "\n "

            # "\n " + "train" + "\n " + "client" + " " + "class number" + " " + str(index) + " " + "number of labels" + " " + str(
            # value) + \
            file.write(str2)
            print(str2)
        gs = self.dataFrame.groupby('idlink').size()

        str2 = "\n " + "total train links: " + str(len(gs)) + "\n "
        file.write(str2)
        print(str2)

        gs = self.test.groupby('label').size()
        str1 = self.name + " " + str(len(self.test)) + "\n "
        file.write(str1)
        for index, value in gs.items():
            str2 = str(index) + " " + "(" + str(value) + ")" + "\n "

            # str2 = "\n " + "test" + "\n " + "client" + " " + "class number" + " " + str(
            #     index) + " " + "number of labels" + " " + str(
            #     value)
            file.write(str2)
            print(str2)
        gs = self.test.groupby('idlink').size()

        str2 = "\n " + "total test links: " + str(len(gs)) + "\n "
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


def clientBuilderForClassesPerEach(name, number_of_clients=3, missing_test_labels=[]):
    df = loadDataFrame("Labelled_Data.csv")
    divided_gp = divideByLinkID(df)
    pickedGps = pickGPSForClassesPerEach(number_of_clients, divided_gp)
    gps = buildClient(pickedGps, name)

    return gps


def clientBuilderForClassesProportional(name, proportions=[6, 4, 4], missingLabels=[-1, 5, 5]):
    df = loadDataFrame("Labelled_Data.csv")
    divided_gp = divideByLinkID(df)
    pickedGps = pickGPSForClassesSelected(divided_gp, proportions, missingLabels=missingLabels)
    gps = buildClient(pickedGps, name)

    return gps


def clientBuilderForClientMissingClasses(name, proportions, missing_train_and_test_Labels, missing_only_train_labels):
    # put -1 if you wish one client to have no missing label
    df = loadDataFrame("Labelled_Data.csv")
    divided_gp = divideByLinkID(df)
    pickedGps = pickGPSForClassesSelected(divided_gp, proportions=proportions,
                                          missing_labels=missing_train_and_test_Labels)
    gps = buildClient(pickedGps, name, missing_train_labels=missing_only_train_labels)

    return gps


def buildClient(data_sets, name, prepare_for_random_test=False, missing_train_labels=[-1, -1, -1]):
    clients = []
    i = 0
    for indx, data in enumerate(data_sets):
        if len(missing_train_labels) <= indx + 1 or missing_train_labels[indx] == -1:
            missing_client_train_label = None
        else:
            missing_client_train_label = missing_train_labels[indx]
        client = Client(data=data, path=None, prepare_for_random_testing=prepare_for_random_test,
                        name=name + " " + str(i),
                        missing_client_train_label=missing_client_train_label)
        clients.append(client)
        i = i + 1
    return clients
