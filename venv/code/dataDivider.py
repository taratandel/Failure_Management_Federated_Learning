import pandas as pd
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import random


# ------------------------------------- Experiments --------------------------------------------
# divides randomly the test set for the clients the reason I did that was to have all
# the labels in all the clients. this division is fractionated by number number of samples that
# each client has.
# clients_test = divideRandomly(test, calcFractions(clients_train))
# initialize the weight still need work but for now don't use it
# initial_weights = coordinator.initializeWeight()
# -----------------------------------------------------------------------------------------------


def cleanData(windows):
    windows = windows.copy(deep=True)
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
    y = windows[windows.columns.values[-1:]].to_numpy()
    y = oneHotEncode(y)

    windows = windows.fillna(-150)

    #         ----------------------------------------- CLEAN THE FEATURES -----------------------------------------

    # Collect the columns name of the input
    columns = windows.columns.values
    # Now variable columns is a list containing all the names of the features
    # The variable is used to address and drop some features not useful in the training and cross-validation phase
    # Delete the columns that are not useful
    windows = windows.drop(columns=columns[:8])
    windows = windows.drop(columns=columns[-1:])
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


def loadDataFrame(path, should_one_hot = False, should_shuffle = False):
    """
    reads the csv file using a path
    :param should_one_hot: bool
        returns data one hot encoded
    :param should_shuffle:  bool
        shuffle data randomly
    :param path: str
        path of the csv file
    :return:
        a data frame
    """
    df = pd.read_csv(path)
    if should_one_hot:
        df = oneHotEncode(df)

    if should_shuffle:
        df = df.sample(frac=1)
    return df


def divideByeqType(df):
    # create unique list of names
    unique_eqtype = df.groupby('eqtype')
    data_frames = [group for _, group in unique_eqtype]
    print("In entered")

    return data_frames


def divideByLinkID(df):
    unique_eqtype = df.groupby('idlink')
    data_frames = [group for _, group in unique_eqtype]
    print("In entered idlink")

    return data_frames

def pickGroupsRandomly(no_of_gp, dfs):
    groups = [[] for _ in range(no_of_gp)]
    df_copy = dfs.copy()
    no_of_links = len(df_copy)
    while no_of_links > 0:
        for i in range(no_of_gp):
            upperbound = no_of_links - 1
            rand = random.randint(0, upperbound)
            groups[i].append(df_copy[rand])
            df_copy.pop(rand)
            no_of_links -= 1
    conctanated_gps = [[] for _ in range(no_of_gp)]

    for i in range(no_of_gp):
        conctanated_gps[i] = pd.concat(groups[i])

    return conctanated_gps


def pickGPSForClassesPerEach(no_of_gp, dfs):
    groups = [[] for _ in range(no_of_gp)]
    labesl = [0, 0, 0, 0, 0, 0]
    groupSeparation = [labesl for _ in range(no_of_gp)]
    df_copy = dfs.copy()
    for i, df in enumerate(df_copy):
        df_ssigned = False
        label = int(pd.unique(df['label'])[0])
        for gp in range(no_of_gp):
            if groupSeparation[gp][label] == 0 and (not df_ssigned):
                groups[gp].append(df)
                groupSeparation[gp][label] = 1
                df_ssigned = True

        if not df_ssigned:
            rand = random.randint(0, (no_of_gp - 1))
            groups[rand].append(df)
    conctanated_gps = [[] for _ in range(no_of_gp)]
    for i in range(no_of_gp):
        conctanated_gps[i] = pd.concat(groups[i])

    return conctanated_gps


def pickGPSForClassesSelected(dfs, proportions, labels=[]):
    no_of_gp = len(proportions)
    groups = [[] for _ in range(no_of_gp)]
    labels = labels
    if not labels:
        for prop in proportions:
            labels.append(random.sample([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], prop))
    df_copy = dfs.copy()
    previouse_group = 0
    for df in df_copy:
        df_ssigned = False
        unique_labels = pd.unique(df['label'])

        while not df_ssigned:
            previouse_group = (previouse_group + 1) % 4
            for gp in range(no_of_gp):
                if (all(item in labels[gp] for item in unique_labels.tolist())) and ((previouse_group - gp) == 1):
                    groups[gp].append(df)
                    df_ssigned = True
                    break

    conctanated_gps = [[] for _ in range(no_of_gp)]
    for i in range(no_of_gp):
        conctanated_gps[i] = pd.concat(groups[i])

    return conctanated_gps


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


def oneHotEncode(y):
    # df = df.copy(deep=True)
    # y = df.pop('label')
    # y = y.to_numpy()
    # n_label = len(set(y))
    y_cat = to_categorical(y, num_classes=6)
    # list_of_labels = ['{:.1f}'.format(x) for x in list([0,1,2,3,4,5])]
    # y_df = pd.DataFrame(y_cat, columns=list_of_labels)
    # new_df = pd.concat([df, y_df], axis=1)
    return y_cat


def divideTestSet(df: pd.DataFrame, test_size=0.2):
    train, test = train_test_split(df, test_size=test_size)
    return train, test


def prepareDataSet(should_divide_test=True):
    # Load the dataframe
    df = loadDataFrame("Labelled_Data.csv", True)
    # split test set with 20% (default is 20% if you want to change the percentage
    # just call the function with desire percentage example: divideTestSet(df, test_size = 0.1)
    if should_divide_test:
        return dividedTestSetPereqTyep(df)
    else:
        return divideByeqType(df)


def dividedTestSetPereqTyep(df):
    df = df.copy(deep=True)
    eqdfs = divideByeqType(df)
    test_sets = []
    train_sets = []
    for df in eqdfs:
        train, test = divideTestSet(df=df)
        test_sets.append(test)
        train_sets.append(train)
    concatenated_train = pd.concat(train_sets)
    concatenated_test = pd.concat(test_sets)

    return test_sets, train_sets, concatenated_test, concatenated_train
