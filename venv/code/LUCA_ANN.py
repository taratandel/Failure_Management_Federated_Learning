from keras.models import load_model
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np
from keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import time
import sklearn.metrics as mt

# Environmental settings
pd.options.mode.chained_assignment = None
np.set_printoptions(threshold=np.inf)

# --------------------------------------------------- OPEN THE FILES ---------------------------------------------------
# CSV file from which we take the dataframes
# Read the CSV with the hand-labelled training set
windows2 = pd.read_csv("/home/bonsai/PycharmProjects/ThesisSIAE/Analysis 50-50 again/48Features/Data/Version1/Train_set_L1_v1.csv")
# Read the CSV with the Automated-labelled training set
unlabelled = pd.read_csv(r"/home/bonsai/PycharmProjects/ThesisSIAE/Analysis 50-50 again/48Features/Extensions/Version1/KNN/U_KNN_100_v1.csv")
# Read the CSV with the hand-labelled test set
tester = pd.read_csv(r"/home/bonsai/PycharmProjects/ThesisSIAE/Analysis 50-50 again/48Features/Data/Version1/Test_set_L2_v1.csv")
# Open the Txt file for the results
result = open("ANN_time.txt", "w")
# ----------------------------------------------------------------------------------------------------------------------

# --------------------------------------------------- PRE-PROCESSING ---------------------------------------------------
# Copy of the hand-labelled training set
labelled = windows2.copy()

#            ----------------------------------- AUTOMATED LABELLING USED -----------------------------------
# Append the hand-labelled and automated-labelled dataframe together
windows = windows2.append(unlabelled, ignore_index=True)
# Update the indexes of the new Training dataframe
windows.index = range(0, len(windows))
#           --------------------------------------------------------------------------------------------------

#            ----------------------------------- ONLY HAND LABELLING USED -----------------------------------
# windows = windows2  # Change the name of the variable containing the training set, just for simplicity
#           --------------------------------------------------------------------------------------------------

#              ------------------------------------ MISSING VALUES ------------------------------------
# List of the transmitted power features
tx = ['txMaxAN-2', 'txminAN-2', 'txMaxBN-2', 'txminBN-2', 'txMaxAN-1', 'txminAN-1',
      'txMaxBN-1', 'txminBN-1', 'txMaxAN', 'txminAN', 'txMaxBN', 'txminBN']
# Fill the missing values of the transmitted power
windows[tx] = windows[tx].fillna(100)
# Fill the missing values of the received power
windows = windows.fillna(-150)
#           --------------------------------------------------------------------------------------------------

#                  ------------------------------------ LABELS ------------------------------------
# Put the labels of the training set in the variable y
y = windows['label'].to_numpy()
# Count the number of distinct labels
n_label = len(set(y))
# List of the distinct labels present in the training dataframe
labels = list(set(y))

# Transform the labels using One-Hot-Encoding for the Train Set
# e.g., Total number of label = 6 (from 0 to 5),
# Selected label = 2 ---O-H-E----> Selected label = [0, 0, 1, 0, 0, 0]
y_cat = to_categorical(y, num_classes=n_label)
#           --------------------------------------------------------------------------------------------------

#            ------------------------------------ FEATURES NORMALIZATION ------------------------------------
# Instantiate the scaler element
scal = StandardScaler()
# Fill the missing values of the hand-labelled windows used to train
labelled[tx] = labelled[tx].fillna(100)
labelled = labelled.fillna(-150)
# Take the name of the features of the dataframe
columns = windows.columns.values
# Delete the columns that are not useful
windows = windows.drop(columns=columns[:8])
# Delete the 'label' columns because we can't train a model using it as an input data
windows = windows.drop(columns=['label'])
# Delete the columns that are not useful
labelled = labelled.drop(columns=columns[:8])
# Delete the 'label' columns because we can't train a model using it as an input data
labelled = labelled.drop(columns=['label'])
# Take the mean and the standard deviation of the hand-labelled training windows
scal.fit(labelled.to_numpy())
# Transform the input data into an array
X = windows.to_numpy()
# Scale the training set
X = scal.transform(X)

# Fill the missing values of the hand-labelled windows used to test
tester[tx] = tester[tx].fillna(100)
tester = tester.fillna(-150)
# Take the name of the features of the dataframe
columns = tester.columns.values
# Keep the labels of the test dataframe
y_test = tester['label']
# Delete the columns that are not useful
tester = tester.drop(columns=columns[:8])
# Delete the 'label' columns because we can't use it to predict the label of a point
tester = tester.drop(columns=['label'])
# Transform the test set in an array and normalize it
X_tester = scal.transform(tester.to_numpy())

# Transform the labels using One-Hot-Encoding for the Test Set
# e.g., Total number of label = 6 (from 0 to 5),
# Selected label = 2 ---O-H-E----> Selected label = [0, 0, 1, 0, 0, 0]
y_test_cat = to_categorical(y_test, num_classes=n_label)
#           --------------------------------------------------------------------------------------------------


#            -------------------------------------- CODE THE FEATURES --------------------------------------
# Load the encoder
encoder = load_model(r"/home/bonsai/PycharmProjects/ThesisSIAE/Analysis 50-50 again/48Features/Encoders/Version1/Encoder_35F_100_v1.h5")
# Encode the training windows
encoded_X = encoder.predict(X)
# Encode the test windows
X_tester = encoder.predict(X_tester)
#           --------------------------------------------------------------------------------------------------
#          -------------------------------------- DON'T CODE THE FEATURES --------------------------------------
# # Change the name to the training windows
# encoded_X = X
# # Change the name to the test windows
# X_tester = X_tester
#           --------------------------------------------------------------------------------------------------

#             -------------------------------------- K-FOLD CROSS-VAL --------------------------------------
n_split_kfold = 10  # Define the number of folds
skf = StratifiedKFold(n_splits=n_split_kfold, shuffle=True)  # Instantiate the object to perform the K-fold division
skf.get_n_splits(encoded_X, y)  # Initialize the K-fold object
#           --------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------


# -------------------------------------------------- MODEL SELECTION --------------------------------------------------
i = 0  # Variable to count the iteration
Best_score = 0  # Variable to keep track of the best cross-validation accuracy
Best_layers = 0  # Best number of hidden layers
Best_neurons = 0  # Best number of neurons for hidden layer
Best_activation = ""  # Best activation function
time_avg = 0  # Variable to calculate the average time of an iteration of model selection
for activation in ['identity', 'logistic', 'tanh', 'relu']:  # Loop over activation functions
    for neurons in [10, 50, 100]:  # Loop over the number of neurons
        for layers in range(1, 11):  # Loop over the number of layers
            score = 0
            for train_index, test_index in skf.split(encoded_X, y):  # Loop over the cross validation folds
                i += 1
                # Pick the input data for the training and validation sets
                X_trainN, X_testN = encoded_X[train_index], encoded_X[test_index]
                # Pick the labels for the training and validation sets
                y_trainN, y_testN = y[train_index], y[test_index]

                # Transform the labels using One-Hot-Encoding
                # e.g., Total number of label = 6 (from 0 to 5),
                # Selected label = 2 ---O-H-E----> Selected label = [0, 0, 1, 0, 0, 0]
                y_testN = to_categorical(y_testN, num_classes=n_label)
                y_trainN = to_categorical(y_trainN, num_classes=n_label)

                # Create the object of the model that must be validated
                size = (neurons,) * layers  # Create the structure of the Artificial Neural Network
                ann = MLPClassifier(hidden_layer_sizes=size, activation=activation,
                                    solver='adam', learning_rate='invscaling', max_iter=10000)
                # Keep track of the time at the beginning of the Training
                start = time.time()
                # Train the model
                ann.fit(X_trainN, y_trainN)
                # Add the training time
                time_avg += (time.time() - start)
                # Compute the validation score
                score += ann.score(X_testN, y_testN)
            # Compute the cross validation score
            score = score/10
            # Check if the cross-validation score is higher wrt the previous best score
            if Best_score < score:
                Best_score = score  # Put the new best score in the variable
                Best_layers = layers  # New best number of layers
                Best_neurons = neurons  # New best number of neurons
                Best_activation = activation  # New best activation function
# ----------------------------------------------------------------------------------------------------------------------

# -------------------------------------------------- TEST AND RESULTS --------------------------------------------------
time_avg = time_avg/i  # Compute the average training time
result.write("AVG_time : %s\n\n" % time_avg)  # Write the average training time
size = (Best_neurons,) * Best_layers  # Create the best hidden structure of the neural network
# Instantiate the best model according to the Cross-Validation
ann = MLPClassifier(hidden_layer_sizes=size, activation=Best_activation,
                    solver='adam', learning_rate='invscaling', max_iter=10000)

# Transform the labels using One-Hot-Encoding for the Training set (Probably a repetition - I don't care)
# e.g., Total number of label = 6 (from 0 to 5),
# Selected label = 2 ---O-H-E----> Selected label = [0, 0, 1, 0, 0, 0]
y_cat = to_categorical(y, num_classes=n_label)
# Train the best model
ann.fit(encoded_X, y_cat)
# Write in the file of the results the best hyper parameters
result.write("H1 : %s\n" % Best_layers)
result.write("H2 : %s\n" % Best_activation)
result.write("H3 : %s\n\n" % Best_neurons)
# Calculate and write in the file the performances on the test set
result.write("Performance on Test set : \n")
# Predict the labels for teh test set
y_pred = ann.predict(X_tester)
# Accuracy
result.write("Accuracy : %s \n" % mt.classification.accuracy_score(y_test_cat, y_pred))
# Precision
result.write("Precision : %s \n"
             % mt.classification.precision_score(y_test_cat, y_pred, labels=labels, average='weighted'))
# Recall
result.write("Recall : %s \n" % mt.classification.recall_score(y_test_cat, y_pred, labels=labels, average='weighted'))
# F1-Score
result.write("F1 score : %s \n" % mt.classification.f1_score(y_test_cat, y_pred, labels=labels, average='weighted'))
result.write("\n\n")

# Close the Txt file of the results
result.close()
# ----------------------------------------------------------------------------------------------------------------------
