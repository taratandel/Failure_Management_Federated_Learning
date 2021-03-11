from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
import pandas as pd
import numpy as np
from keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import sklearn.metrics as mt
from sklearn.metrics import roc_auc_score, roc_curve
import time
import joblib
# Environmental settings
pd.options.mode.chained_assignment = None
np.set_printoptions(threshold=np.inf)

from data_divider import *

# ----------------------------------------------- CONFUSION MATRIX PLOT -----------------------------------------------
def plot_confusion_matrix(y_true, y_pred, classes = [],
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, [0,1,2,3,4,5])
    # Only use the labels that appear in the data
    classes = [0,1,2,3,4,5]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for w in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, w, format(cm[w, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[w, j] > thresh else "black")
    fig.tight_layout()
    return ax

# Code provided by: https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
# ----------------------------------------------------------------------------------------------------------------------
i = 3
fileName = "df" + str(i) + ".csv"
resultName = "ANN_model_selection_results" + str(i) + ".txt"
modelName = "Ann" + str(i) + ".sav"
performanceName = "Performances_result_ANN_Weighted_s_Average" + str(i) + ".txt"

nameOfModel = 'finalmodelno_SampleAverage.sav'

# --------------------------------------------------- OPEN THE FILES ---------------------------------------------------
# CSV file from which we take the dataframe containing our data
# windows = pd.read_csv("Labelled_Data.csv")
dd = DataDivider(nameOfTheFile="Labelled_Data.csv")
dd.oneHotEncode()
dfs = dd.divideByeqType()
windows = dfs[i - 1]
# Txt file in which the model selection results will be saved
result = open(resultName, "w")
# ----------------------------------------------------------------------------------------------------------------------

# ------------------------------------------------ DATA PRE-PROCESSING ------------------------------------------------

#            ---------------------------------------- NAN PROCESSING ----------------------------------------
# Identify the Transmitted power features
tx = ['txMaxAN-2', 'txminAN-2', 'txMaxBN-2', 'txminBN-2', 'txMaxAN-1', 'txminAN-1',
      'txMaxBN-1', 'txminBN-1', 'txMaxAN', 'txminAN', 'txMaxBN', 'txminBN']
# Fill the Nan value in the Tx power features with 100
windows[tx] = windows[tx].fillna(100)
# Fill the Nan value in the Rx power features with -150
# We use two different values because in this problem the Tx power and the Rx power have two different range of values,
# when we will normalize the data (Some steps below) if the added values are too big this can eliminate the differences
# in the data values
#            ------------------------------------------------------------------------------------------------

#           ---------------------------------------- LABELS PROCESSING ----------------------------------------
# Put the labels inside a variable
y = windows[windows.columns.values[-6:]].to_numpy()
windows = windows.fillna(-150)

# ----------------------------------------------------------------------------------------------------------------------
# BINARY PROBLEM SETTINGS
# index_zero = np.where(y != 5)  # Indexes of the well defined classes
# index_one = np.where(y == 5)  # Indexes of the other problems
# y[index_one] = 1  # Define the other problems as positive
# y[index_zero] = 0  # Define the well defined as negative
# ----------------------------------------------------------------------------------------------------------------------
# Calculate the number of distinct labels
n_label = 6
# Have a list of all the distinct labels
labels = [0.0,1.0,2.0,3.0,4.0,5.0]

#            ------------------------------------------------------------------------------------------------

#           ----------------------------------------- NEW FEATURES -----------------------------------------

# Create new features to improve the model results

# # Example
# windows['DeltamaxAN'] = windows['RxNominal'] - windows['rxmaxAN']


#            ------------------------------------------------------------------------------------------------

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

#            ------------------------------------------------------------------------------------------------

#         ----------------------------------------- DATAFRAME SPLITTING -----------------------------------------
test_percentage = 0.2  # Percentage of points contained in the test dataset (e.g., 20% = 0.2)

# Perform a Train-Test split maintaining the distribution of the classes inside the splits
# Parameters explanation:
# X = input data described by 35 dimensions
# y = labels assigned to the input data
# test_size = percentage of the test set
# random_state = seed of the random function, a fixed value provide to have the same splitting in different
#                execution of the script
# stratify = takes as input the list of the labels, providing the list of the labels the function provide as output
#            train and test sets where the proportion of the classes is maintained
# shuffle = if it is true, shuffle each class’s samples before splitting into batches
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_percentage,
                                                    random_state=42, stratify=y, shuffle=True)

# Select the number of splitting to perform Cross-Validation
n_split_kfold = 10

# Create the object that will split the data maintaining the proportion of the classes
# Parameters explanation:
# n_splits = number of fold in which our training dataset is divided
# random_state = seed of the random function, a fixed value provide to have the same splitting in different
#                execution of the script
# shuffle = if it is true, shuffle each class’s samples before splitting into batches
skf = StratifiedKFold(n_splits=n_split_kfold, shuffle=True, random_state=42)

# provide to the object the data that will split
skf.get_n_splits(X_train, y_train)
# Instantiate the scaler element
scaler = StandardScaler()
# ----------------------------------------------------------------------------------------------------------------------
# ---------------------------- START COMMENT HERE IF YOU ALREADY HAVE THE HYPER-PARAMETERS ----------------------------
# ------------------------------------------------MODEL SELECTION PHASE------------------------------------------------

comb = 0  # Iteration variable
Best_score = 0  # Best cross validation score of a certain model
Best_layers = 0  # Best number of layers
Best_neurons = 0  # Best number of neurons
Best_activation = ""  # Best activation function
# for activation in ['identity', 'logistic', 'tanh', 'relu']:  # Loop over activation functions
#     for neurons in [10, 50, 100]:  # Loop over the number of neurons per layer
#         for layers in range(1, 4):  # Loop over the number of hidden layers
#             comb += 1
#             print("Progress ----> ", comb, "/36")
#             score = 0
#             for train_index, test_index in skf.split(X_train, y_train):  # Loop over the cross validation folds
#
#                 # Pick the input data for the training and validation sets
#                 X_train_val, X_val = X_train[train_index], X_train[test_index]
#                 # Pick the labels for the training and validation sets
#                 y_train_val, y_val = y_train[train_index], y_train[test_index]
#
#                 # ------------------------------------- Scale the data (Automated) -------------------------------------
#                 X_train_val = scaler.fit_transform(X_train_val)  # Scale the train data as (value - mean) / std
#                 X_val = scaler.transform(X_val)  # scale the validation data as (value - mean_train) / std_train
#                 # ------------------------------------------------------------------------------------------------------
#
#                 # ------------------------------------- Scale the data (Manually) -------------------------------------
#                 # mean_train = np.mean(X_train_val, axis=0)  # Mean per feature of the training data
#                 # std_train = np.std(X_train_val, axis=0)  # Std per feature of the training data
#                 # X_train_val = (X_train_val-mean_train) / std_train  # Scaling training data
#                 # X_val = (X_val-mean_train) / std_train  # Scaling validation data
#                 # ------------------------------------------------------------------------------------------------------
#
#                 # Transform the labels using One-Hot-Encoding (O-H-E)
#                 # e.g., Total number of label = 6 (from 0 to 5),
#                 # Selected label = 2 ---O-H-E----> Selected label = [0, 0, 1, 0, 0, 0]
#                 y_train_cat = to_categorical(y_train_val, num_classes=n_label)
#                 y_val_cat = to_categorical(y_val, num_classes=n_label)
#
#                 # Create the object of the model that must be tested
#                 size = (neurons,) * layers  # Create the structure of the Artificial Neural Network
#                 ann = MLPClassifier(hidden_layer_sizes=size, activation=activation,
#                                     solver='adam', learning_rate='invscaling', max_iter=10000)
#                 # Train the model
#                 ann.fit(X_train_val, y_train_cat)
#                 # Compute the validation score
#                 score += ann.score(X_val, y_val_cat)
#             # Compute the cross validation score
#             score = score / n_split_kfold
#             # Check if the cross-validation score is higher wrt the previous best score
#             if Best_score < score:
#                 Best_score = score  # Put the new best score in the variable
#                 Best_layers = layers  # New best number of layers
#                 Best_neurons = neurons  # New best number of neurons
#                 Best_activation = activation  # New best activation function
#
# # Write the model selection results in the file
# result.write("Layers : %s\n" % Best_layers)
# result.write("Neurons : %s\n" % Best_neurons)
# result.write("Activation : %s\n" % Best_activation)
# result.write("Cross-Validation accuracy : %s \n" % Best_score)
#
# result.close()
# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------- END COMMENT HERE IF YOU ALREADY HAVE THE HYPER-PARAMETERS -----------------------------
# ----------------------------------------------------TESTING PHASE----------------------------------------------------

# Hyperparameters to set according to the model selection result
# if the hyperparameters are given the model selection part must be commented as reported in the previous comments
Best_layers = 3
Best_neurons = 100
Best_activation = "tanh"

# Scale the data (Manually)
# mean_train = np.mean(X_train, axis=0)  # Mean per feature of the training data
# std_train = np.std(X_train, axis=0)  # Std per feature of the training data
# X_train = (X_train-mean_train) / std_train  # Scaling training data
# X_test = (X_test-mean_train) / std_train  # Scaling validation data

# Scale the data
X_train = scaler.fit_transform(X_train)  # Scale the train data as (value - mean) / std
X_test = scaler.transform(X_test)  # scale the test data as (value - mean_train) / std_train

# Transform the labels using One-Hot-Encoding
# e.g., Total number of label = 6 (from 0 to 5),
# Selected label = 2 ---O-H-E----> Selected label = [0, 0, 1, 0, 0, 0]
# y_train_cat = to_categorical(y_train, num_classes=n_label)
# y_test_cat = to_categorical(y_test, num_classes=n_label)

# Create the object of the best model according to Cross-Validation
size = (Best_neurons,) * Best_layers  # Create the structure of the Artificial Neural Network
ann = MLPClassifier(hidden_layer_sizes=size, activation=Best_activation,
                    solver='adam', learning_rate='invscaling', max_iter=10000)
Tick = time.time()  # Take the time before the training
# Train the model
# ann.fit(X_train, y_train)
#
# #  here I should return the weights of the ann
# joblib.dump(ann, modelName)

ann = joblib.load(nameOfModel)
Tock = time.time() - Tick  # Calculate the training time
# Predict the label on the test set
# For each point the output is an array where the label is represented in O-H-E
y_predicted = ann.predict(X_test)
# Take the index where is present the value 1, that identify the predicted label
y_predicted = np.argmax(y_predicted, axis=1)
# For each point, predict the probability to belong to each class
y_probability = ann.predict_proba(X_test)
# ----------------------------------------------------------------------------------------------------------------------

# ------------------------------------------ SAVE THE PERFORMANCES PER CLASS ------------------------------------------

# Open the file where will be saved all the test performances
performances = open(performanceName, "w")
acc = pd.DataFrame()
y_test = np.argmax(y_test, axis=1)
# create a structure to perform the measure manually
acc['ground_truth'] = y_test  # add the ground truth
acc['predicted'] = y_predicted  # add the predicted labels
precision = [0] * n_label  # precision list
recall = [0] * n_label  # recall list
performances.write("Training Time(s): %s \n" % str(Tock))  # Print the training time needed in the Test phase
#           -------------------------------------------- ACCURACY --------------------------------------------
#                                 --------------------- AUTOMATED ---------------------

# Write the accuracy inside the performances file
performances.write("Accuracy automated: %s \n" % mt.accuracy_score(y_test, y_predicted))
#                                 --------------------- MANUALLY ---------------------

# Write the accuracy inside the performances file
performances.write("Accuracy manually: %s \n" % str(len(acc.loc[acc['ground_truth'] == acc['predicted']]) / len(acc)))
#           --------------------------------------------------------------------------------------------------

#           -------------------------------------------- PRECISION --------------------------------------------
#                                 --------------------- AUTOMATED ---------------------

# Write the precision inside the performances file
performances.write("Precision per class automated: %s \n"
                   % mt.precision_score(y_test, y_predicted, labels=labels, average=None))
#                                 --------------------- MANUALLY ---------------------

# Write the precision inside the performances file
# performances.write("\nPrecision per class manually: \n")
# for i in range(n_label):
#     performances.write("Class number %s \n" % str(i))
#     human = acc.loc[acc['predicted'] == i]['ground_truth']  # we pick all the ground truth in a list
#     predicted = acc.loc[acc['predicted'] == i]['predicted']  # we pick the label assigned to the clustering
#     measure = pd.DataFrame()
#     measure['ground_truth'] = human
#     measure['predicted'] = predicted
#     precision[i] = len(measure.loc[measure['ground_truth'] == measure['predicted']]) / len(measure)
#     performances.write("Precision: %s \n\n" % precision[i])
#           --------------------------------------------------------------------------------------------------

#           -------------------------------------------- RECALL --------------------------------------------
#                                 --------------------- AUTOMATED ---------------------

# Write the recall inside the performances file
performances.write("Recall per class automated: %s \n"
                   % mt.recall_score(y_test, y_predicted, labels=labels, average=None))
#                                 --------------------- MANUALLY ---------------------
#
# # Write the recall inside the performances file
# performances.write("\nRecall per class manually: \n")
# for i in range(n_label):
#     performances.write("Class number %s \n" % str(i))
#     human = acc.loc[acc['ground_truth'] == i]['ground_truth']  # we pick all the ground truth in a list
#     predicted = acc.loc[acc['ground_truth'] == i]['predicted']  # we pick the label assigned to the clustering
#     measure = pd.DataFrame()
#     measure['ground_truth'] = human
#     measure['predicted'] = predicted
#     recall[i] = len(measure.loc[measure['ground_truth'] == measure['predicted']]) / len(measure)
#     performances.write("Recall: %s \n\n" % recall[i])
#           --------------------------------------------------------------------------------------------------

#           -------------------------------------------- F1-SCORE --------------------------------------------
#                                 --------------------- AUTOMATED ---------------------

# Write the f1-score inside the performances file
performances.write("F1-score automated: %s \n"
                   % mt.f1_score(y_test, y_predicted, labels=labels, average=None))
#                                 --------------------- MANUALLY ---------------------

# Write the f1-score inside the performances file
# performances.write("\nF1-score per class manually: \n")
# for i in range(n_label):
#     performances.write("Class number %s \n" % str(i))
#     f1score = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])
#     performances.write("Recall: %s \n\n" % f1score)
#           --------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
performances.close()
# ----------------------------------------------------- ROC CURVE -----------------------------------------------------
if n_label == 2:
    # Calculate the False positive rate and the True positive rate
    fpr, tpr, thresholds = roc_curve(y_test, y_probability[:, 1], pos_label=1)
    # Calculate the Area under the Roc Curve
    auc_score = roc_auc_score(y_test, y_probability[:, 1])
    # Plot the Roc Curve
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % auc_score)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
    # https: // scikit - learn.org / stable / auto_examples / model_selection / plot_roc.html
# ----------------------------------------------------------------------------------------------------------------------

# --------------------------------------------- PLOT THE CONFUSION MATRIX ---------------------------------------------
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plot_confusion_matrix(y_test, y_predicted,
                      title='Confusion matrix, without normalization')

# # Plot normalized confusion matrix
# plot_confusion_matrix(y_test, y_predicted, classes=labels, normalize=True,
#                       title='Normalized confusion matrix')

plt.show()
# ----------------------------------------------------------------------------------------------------------------------
