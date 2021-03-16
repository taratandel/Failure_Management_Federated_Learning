from DataDivider import loadDataFrame as lDF

class Client:

    def __init__(self, data=None, path=None):
        if data is not None:
            self.dataFrame = data
        elif path is not None:
            self.dataFrame = lDF(path)
        else:
            raise Exception("Sorry, provide a data or a path. both cannot be empty")

    def updateModelWeights(self):


    def sendUpdatedModels(self):

    def loadTheData(self, nameOfTheFile):

