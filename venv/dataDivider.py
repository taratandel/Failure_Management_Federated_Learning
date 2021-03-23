import numpy as np
import pandas as pd
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def cleanData(df):
    windows = df
    #            ---------------------------------------- NAN PROCESSING ----------------------------------------
    # Identify the Transmitted power features
    tx = ['txMaxAN-2', 'txminAN-2', 'txMaxBN-2', 'txminBN-2', 'txMaxAN-1', 'txminAN-1',
          'txMaxBN-1', 'txminBN-1', 'txMaxAN', 'txminAN', 'txMaxBN', 'txminBN']
    # Fill the Nan value in the Tx power features with 100
    windows[tx] = windows[tx].fillna(100)
    # Fill the Nan value in the Rx power features with -150 We use two different values because in this problem the
    # Tx power and the Rx power have two different range of values, when we will normalize the data (Some steps
    # below) if the added values are too big this can eliminate the differences in the data values
    # ------------------------------------------------------------------------------------------------

    #           ---------------------------------------- LABELS PROCESSING ----------------------------------------
    # Put the labels inside a variable
    y = windows[windows.columns.values[-6:]].to_numpy()
    windows = windows.fillna(-150)

    #         ----------------------------------------- CLEAN THE FEATURES -----------------------------------------

    # Collect the columns name of the input
    columns = windows.columns.values
    # Now variable columns is a list containing all the names of the features
    # The variable is used to address and drop some features not useful in the training and cross-validation phase
    # Delete the columns that are not useful
    windows = windows.drop(columns=columns[:8])
    windows = windows.drop(columns=columns[-6:])
    # Delete the 'label' columns because we can't train a model using it as an input data
    # windows = windows.drop(columns=['label'])
    # After the elimination of the useless columns our data are described by 35 features
    # Transform the input data into an 2D-array
    X = windows.to_numpy()

    # ------------------------------------------------------------------------------------------------
    # provide to the object the data that will split
    # skf.get_n_splits(X, y)
    # Instantiate the scaler element
    scaler = StandardScaler()
    # Scale the data
    X = scaler.fit_transform(X)  # Scale the train data as (value - mean) / std
    return X, y


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


def divideRandomly(df, fractions):
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


def divideTestSet(df: pd.DataFrame, test_size=0.2):
    train, test = train_test_split(df, test_size=test_size)
    return train, test
