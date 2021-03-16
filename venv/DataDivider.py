import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import cdist, pdist
from keras.utils import to_categorical


def loadDataFrame(nameOfTheFile):
    return pd.read_csv(nameOfTheFile)


class DataDivider:
    df = pd.DataFrame()

    def __init__(self, nameOfTheFile=None, dataFrame=None):
        if dataFrame is not None:
            self.df = dataFrame
        else:
            self.df = loadDataFrame(nameOfTheFile)

    def divideByeqType(self):
        df = self.df
        # create unique list of names
        unique_eqtype = df.groupby('eqtype')
        dataFrames = [group for _, group in unique_eqtype]
        i = 0
        for dataframe in dataFrames:
            i = i + 1
            name = 'df' + str(i) + '.csv'
            dataframe.to_csv(name, index=False)
            print("In entered", i)

        return dataFrames

    def oneHotEncode(self):
        y = self.df.pop('label')
        y = y.to_numpy()
        n_label = len(set(y))
        y_cat = to_categorical(y, num_classes=n_label)
        list_of_labels = ['{:.1f}'.format(x) for x in list(set(y))]
        y_df = pd.DataFrame(y_cat, columns = list_of_labels)
        self.df = pd.concat([self.df, y_df], axis=1)

    # def divideRandomly(self):