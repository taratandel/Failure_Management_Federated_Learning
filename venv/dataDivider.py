import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import cdist, pdist
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split


def loadDataFrame(path):
    """
    reads the csv file using a path
    :param path: str
        path of the csv file
    :return:
        a data frame
    """
    return pd.read_csv(path)


def divideByeqType(df):
    # create unique list of names
    unique_eqtype = df.groupby('eqtype')
    data_frames = [group for _, group in unique_eqtype]
    i = 0
    for data_frame in data_frames:
        i = i + 1
        name = 'df' + str(i) + '.csv'
        data_frame.to_csv(name, index=False)
        print("In entered", i)

    return data_frames


def calcFractions(data_frames):
    frac = []
    total = 0
    for df in data_frames:
        total = total + df.shape[0]
    for df in data_frames:
        frac.append(df.shape[0] / total)
    return frac


def divideRandomly(df,  fractions):
    data_frames = []
    for fraction in fractions:
        data_frames.append(df.sample(frac=fraction))
    return data_frames


def oneHotEncode(df):
    y = df.pop('label')
    y = y.to_numpy()
    n_label = len(set(y))
    y_cat = to_categorical(y, num_classes=n_label)
    list_of_labels = ['{:.1f}'.format(x) for x in list(set(y))]
    y_df = pd.DataFrame(y_cat, columns=list_of_labels)
    df = pd.concat([df, y_df], axis=1)
    return df


def divideTestSet(df: pd.DataFrame ,test_size = 0.2):
    train, test = train_test_split(df, test_size=test_size)
    return train, test
