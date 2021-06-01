import pandas as pd
import numpy as np
from keras.models import Model
from keras.layers import Input, Dense
from keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn import metrics
import time
from sklearn.model_selection import train_test_split
from keras.layers import Activation
import math
from numpy.random import randint
from keras.models import load_model
from sklearn.neighbors import KNeighborsClassifier
import sklearn.metrics as mt
from sklearn.gaussian_process import GaussianProcessClassifier


Tick = time.time()
# SOME SETTINGS OF THE LIBRARIES
pd.options.mode.chained_assignment = None  # Allows to perform changes directly in the Dataframe
np.set_printoptions(threshold=np.inf)  # The representation of the arrays is not limited in size

# Load the dataset (csv file) that we want to split
labelled = pd.read_csv(r"Labelled_35F.csv")

# tx select all the transmitted power features
tx = ['txMaxAN-2', 'txminAN-2', 'txMaxBN-2', 'txminBN-2', 'txMaxAN-1', 'txminAN-1',
      'txMaxBN-1', 'txminBN-1', 'txMaxAN', 'txminAN', 'txMaxBN', 'txminBN']
# Fill the transmitted power Nan with 100
labelled[tx] = labelled[tx].fillna(100)
# Fill the received power Nan with -150
labelled = labelled.fillna(-150)

# Take the labels of the points
labelstot = labelled['label']

# Be sure that the RxNominal is a negative value:
# Take the RxNominal value
Nom = labelled['RxNominal']
# Assign again the value of RxNominal to be sure that is a negative value
labelled['RxNominal'] = abs(Nom) * (-1)

# Define how much labelled points keep to train our models
labelled_percentage = 0.5

for split in range(2):

    # -------------------------------------------------SPLITTING PART--------------------------------------------------

    # Split the labelled set in two parts keeping to the train part the percentage set before
    # the splits keep the same label distribution
    Train, Test = train_test_split(labelled, test_size=1-labelled_percentage, shuffle=True, stratify=labelstot)

    # Reassign the indexes to the new dataset to make the program work well
    Train.index = range(len(Train))
    Test.index = range(len(Test))

    # Save the test dataset in a csv file

    Test.to_csv(str(split)+"_Iter_Test_set.csv", index=False)

    # Save te train dataset in a csv file
    Train.to_csv(str(split)+"_Iter_Train_set.csv", index=False)
    print("Split done: test and train sets created")

    # --------------------------------------------------ENCODER SEARCH--------------------------------------------------

    # Dataframe structure used later on for the measurements
    label = pd.DataFrame()

    # OPEN THE FILE
    # CSV file from which we take the dataframe to train the Encoder
    windows = Train

    # Create an array with the labels to ue it in the Training
    y = windows['label']
    # Count the number of classes
    n_label = len(set(y))
    # Add the ground truth to the structure 'label' used to perform performance measures
    label['label'] = windows['label']
    # Create One-Hot-Coding structure for the labels
    y_cat = to_categorical(y, num_classes=n_label)
    # Print the number of points for each label
    # print(label['label'].value_counts(sort=False, dropna=False))

    # Keep all the columns name of the database
    columns = windows.columns.values
    # Drop the not needed information about the window
    windows = windows.drop(columns=columns[:8])
    # Drop the label columns
    windows = windows.drop(columns=['label'])
    # Print the name of the remaining columns to check that we drop the right columns
    # print(windows.columns.values)

    # Transform the Dataframe in an Array
    X = windows.to_numpy()
    # Scaling the inputs (Value-Mean)/std var
    X = StandardScaler().fit_transform(X)
    # Keep the number of columns in the Database - Number of features
    input_dim = X.shape[1]

    # CREATING THE FRAMEWORK FOR THE AUTOENCODER
    best_ari = 0
    best_fms = 0
    activation = 'tanh'  # define the activation function
    for hidden_layers in range(5, 10, 2):  # Define the possible structure as number of hidden layers
        if (best_ari > 0.98) & (best_fms > 0.98):
            break
        encoder_dim = math.ceil(hidden_layers / 2)  # Number of layers in the encoder without counting the input one
        for trial in range(20):  # Define the trial for each structure
            print("Iteration number ", trial)
            if (best_ari > 0.98) & (best_fms > 0.98):
                break
            n_neurons = [0] * hidden_layers  # For each hidden layer define the number of neurons

            #  what is 35? why we used this
            n_neurons[encoder_dim - 1] = 35 #randint(1, input_dim - 1)  # Decide the number of coded features

            for i in range(encoder_dim - 1):
                if i == 0:  # Define the dimension of the first and last hidden layers
                    # how he found the numbers ?
                    n_neurons[i] = randint(input_dim+165, 400)
                    n_neurons[hidden_layers - 1 - i] = n_neurons[i]
                else:  # define the dimension of all the others hidden layers
                    n_neurons[i] = randint(math.ceil(n_neurons[i - 1]/2), n_neurons[i - 1]) #n_neurons[encoder_dim - 1]
                    n_neurons[hidden_layers - 1 - i] = n_neurons[i]

            # Creation of the Autoencoder
            in_data = Input(shape=(input_dim,))  # Input layer
            fe = Dense(n_neurons[0], activation=activation)(in_data)  # First hidden layer
            for i in range(1, hidden_layers):  # All the other hidden layers
                fe = Dense(n_neurons[i], activation=activation)(fe)
            fe = Dense(n_label)(fe)  # Output layer
            sup_out_layer = Activation(activation)(fe)  # Activation function of the output layer
            autoencoder = Model(in_data, sup_out_layer)  # Create the complete Autoencoder model
            # define and compile supervised discriminator model

            # print(autoencoder.summary())

            # CREATE THE FRAMEWORK OF THE ENCODER
            # Create automatically the Encoder
            input_enc = Input(shape=(input_dim,))  # Input layer
            encoder = autoencoder.layers[1](input_enc)  # First Hidden Layer always present
            for i in range(2, encoder_dim + 1):  # Create all the remaining hidden layers
                encoder = autoencoder.layers[i](encoder)

            encoder_model = Model(input_enc, encoder)  # Create the complete Encoder model

            # print(encoder_model.summary())

            # Define the optimizer, the loss function and a performance metric calculated during the training phase
            autoencoder.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
            # Train the AutoEncoder
            autoencoder.fit(X, y_cat,
                            epochs=300,  # How many times the entire dataset is fed to the autoencoder
                            batch_size=100,  # At each step how many points are fed to train the autoencoder
                            shuffle=True,  # The input data are shuffled
                            validation_split=0.05,  # Percentage off data used as Validation test
                            verbose=0  # Visual option that shows only a line of code at screen for each epoch
                            )

            # Use the Encoder to predict the entries and in this way perform dimensionality reduction
            encoded_entries = encoder_model.predict(X)

            # K-MEANS CLUSTERING WITH K FIXED
            # Set the number of clusters
            n_cluster = n_label
            # Instantiate the algorithm instance
            kmean = KMeans(n_clusters=n_cluster, init='k-means++', n_init=100, max_iter=100000, tol=1e-6)

            # Fit the algorithm and directly produce the clustering result
            clulabels = kmean.fit_predict(encoded_entries)
            # Extract the cluster centroids from the algorithm
            centroids = kmean.cluster_centers_

            # we put the label assigned by the clustering together with the human label in the 'label' structure
            label['clulabel'] = clulabels

            # WRITE THE RESULTS IN THE FILE FOR ALL THE CLASSES
            # measure the ARI of the clustering
            ari = metrics.adjusted_rand_score(label['label'],
                                              label['clulabel'])
            # measure the Fowlkes-Mallows scores of the clustering
            fms = metrics.fowlkes_mallows_score(label['label'],
                                                label['clulabel'])
            # measure the Silhouette score over the clustering result
            sil = metrics.silhouette_score(X, label['clulabel'],
                                           metric='euclidean')
            if (ari > best_ari) & (fms > best_fms):
                best_ari = ari
                best_fms = fms
                # File that contains the results - it is a txt file (ex. Result_Encoder_1.txt)
                result = open(str(split) + "_Iter_Best_encoder_results.txt", "w")
                # Save the Encoder structure and weights to use them after - the saved file is a h5 file
                encoder_model.save(str(split)+"_Iter_Best_Encoder.h5")
                # Lines to write the results inside the file
                result.write("\n AVERAGE RESULTS OVER ALL  THE CLASSES: \n")
                result.write("ARI : %s \n" % ari)
                result.write("FMS : %s \n" % fms)
                result.write("Silhouette : %s \n" % sil)

                # WRITE THE RESULTS IN THE FILE FOR THE FIVE CLASSES WELL DEFINED
                # measure the ARI of the clustering
                ari = metrics.adjusted_rand_score(label.loc[label['label'] != 5]['label'],
                                                  label.loc[label['label'] != 5]['clulabel'])
                # measure the Fowlkes-Mallows scores
                fms = metrics.fowlkes_mallows_score(label.loc[label['label'] != 5]['label'],
                                                    label.loc[label['label'] != 5]['clulabel'])

                # Lines to write the results inside the file
                result.write("\n AVERAGE RESULTS OVER THE 5 WELL DEFINED CLASSES: \n")
                result.write("ARI : %s \n" % ari)
                result.write("FMS : %s \n" % fms)

                # WRITE THE FMI RESULTS IN THE FILE FOR EACH CLASS
                result.write("\n FMS PER CLASS: \n")
                for i in range(0, n_cluster):
                    result.write("Class : %s\n" % i)
                    human = label.loc[label['label'] == i]['label']  # we pick all the ground truth in a list
                    cluster = label.loc[label['label'] == i]['clulabel']  # we pick the label assigned to the clustering
                    # write the FMS per class
                    result.write("FMS : %s \n" % metrics.fowlkes_mallows_score(human, cluster))

                # INFORMATION PER CLUSTER
                farthest = []
                for i in range(0, n_cluster):
                    larger = 0
                    result.write("\n\nCluster : %s" % i)
                    # Write the centroid
                    result.write("\nThe centroid is : %s " % str(centroids[i]))
                    # Take the indexes of the points contained in the cluster i
                    index_cluster = np.where(clulabels == i)[0]
                    # For each point calculate the distance from the centroid of the cluster i
                    for index in index_cluster:
                        dist = np.linalg.norm(encoded_entries[index] - centroids[i])
                        # Keep only the farthest point, the distance is used as radius of the cluster
                        if larger < dist:
                            larger = dist
                            farthest = encoded_entries[index]
                    # Write the radius in the file
                    result.write("\nRadius of the cluster: %s" % larger)
                    # Write the farthest point in the cluster
                    result.write("\nThe Farthest point is: %s " % str(farthest))
                    # Calculate the distance between all the centroids
                    result.write("\nDistances between this centroid and the others:")
                    for centr in range(0, n_cluster):
                        if centr != i:
                            result.write("\nCluster : %s" % centr)
                            result.write("\nDistance : %s" % np.linalg.norm(centroids[centr] - centroids[i]))
                    # Report in the file the number of points per each class contained in the cluster
                    result.write("\nClasses inside the cluster:")
                    result.write("\n%s" %
                                 label.loc[label['clulabel'] == i]['label'].value_counts(sort=False,
                                                                                         dropna=False).to_string())
                    result.write("\n")

                result.close()

    # ---------------------------------------------EXTENSION PERFORMANCES----------------------------------------------

    # -------------------------------------------PERFORMANCE PRE-PROCESSING--------------------------------------------

    # Dataframe structure used later on for the measurements
    label = pd.DataFrame()

    # OPEN THE FILES
    # CSV file from which we take the dataframe to train the KNN model
    windows = Train
    # CSV file from which we take the dataframe to test the KNN extension
    nolab = Test

    # Create an array with the labels to ue it in the Training
    y = windows['label']
    # Count the number of classes
    n_label = len(set(y))
    # Add the ground truth to the structure 'label' used to perform performance measures
    label['label'] = nolab['label']
    # Labels inside the label set
    labels = list(set(label['label']))

    # TRAINING DATASET COLUMNS DROP
    # Keep all the columns name of the database
    columns = windows.columns.values
    # Drop the not needed information about the window
    windows = windows.drop(columns=columns[:8])
    # Drop the label columns
    windows = windows.drop(columns=['label'])
    # TEST DATASET COLUMNS DROP
    # Keep all the columns name of the database
    columns = nolab.columns.values
    # Drop the not needed information about the window
    nolab = nolab.drop(columns=columns[:8])
    # Drop the label columns
    nolab = nolab.drop(columns=['label'])

    # SCALING THE INPUT
    # Transform the Dataframe in an Array
    X = windows.to_numpy()
    Nol = nolab.to_numpy()
    # Instantiate the scaling algorithm
    scal = StandardScaler()
    # Scaling the data (Value-Mean)/Variance
    X = scal.fit_transform(X)
    Nol = scal.transform(Nol)

    # Load the best encoder found for the specific training set - the file that must be loaded is the h5 file
    encoder = load_model(str(split)+"_Iter_Best_Encoder.h5")

    # Codify the training data - perform dimensionality reduction
    encoded_entries = encoder.predict(X)
    # Codify the test data - perform dimensionality reduction
    Nol_encoded = encoder.predict(Nol)

    # ------------------------------------------------KNN PERFORMANCES-------------------------------------------------

    # TXT File that contains the results
    result = open(str(split) + "_Iter_KNN_performances.txt", "w")
    # Recap of the class division in L1
    result.write("Class division of L1:\n")
    result.write(y.value_counts(sort=False, dropna=False).to_string())

    # Recap of the class division in L2
    result.write("\nClass division of L2:\n")
    result.write(label['label'].value_counts(sort=False, dropna=False).to_string())
    result.write("\n")
    # KNN CLASSIFIER
    # The number of neighbors is selected as the double of the number in the less representative class minus 1
    neighbors = min(y.value_counts(sort=False, dropna=False)) * 2 - 1
    # Instantiate the KNN classifier
    knn = KNeighborsClassifier(n_neighbors=neighbors, weights='distance', algorithm='auto')
    # Fit the KNN classifier
    knn.fit(encoded_entries, y)
    # Predict the labels of the test set using the KNN
    label['predicted'] = knn.predict(Nol_encoded)

    # Recap of the label extension over the points
    result.write("\nClass division of Predicted L2:\n")
    result.write(label['predicted'].value_counts(sort=False, dropna=False).to_string())
    result.write("\n")

    # Write the performance measures associated to the extension
    result.write("Average Performances:\n")
    # Accuracy
    result.write("Accuracy : %s \n" % mt.classification.accuracy_score(label['label'], label['predicted']))
    # Precision
    result.write("Precision : %s \n"
                 % mt.classification.precision_score(label['label'], label['predicted'], labels=labels,
                                                     average='weighted'))
    # Recall
    result.write("Recall : %s \n"
                 % mt.classification.recall_score(label['label'], label['predicted'], labels=labels,
                                                  average='weighted'))
    # F1-Score
    result.write("F1 score : %s \n"
                 % mt.classification.f1_score(label['label'], label['predicted'], labels=labels, average='weighted'))

    # Write the accuracy per class
    result.write("\nAccuracy per class: \n")
    for i in range(6):
        result.write("Class number %s \n" % str(i))
        human = label.loc[label['label'] == i]['label']  # we pick all the ground truth in a list
        predicted = label.loc[label['label'] == i]['predicted']  # we pick the label assigned to the clustering
        result.write("Accuracy: %s \n\n" % mt.classification.accuracy_score(human, predicted))

    result.close()

    # -------------------------------------------------GP PERFORMANCES--------------------------------------------------

    # Kernel used in the Gaussina Process
    ker = None
    # Instantiate the Gaussian Process classifier
    gp = GaussianProcessClassifier(kernel=ker, optimizer='fmin_l_bfgs_b', n_restarts_optimizer=50, max_iter_predict=150,
                                   copy_X_train=True, multi_class='one_vs_rest')
    # Fit the Gaussian Process classifier
    gp.fit(encoded_entries, y)
    # Predict the probability of the classes for each point
    k = gp.predict_proba(Nol_encoded)

    # TXT File that contains the results
    result = open(str(split) + "_Iter_GP_performances.txt", "w")

    #                        ---------------------------RHO = 90---------------------------

    # Set the confidence bound
    rho = 0.90
    result.write("RHO = %s \n" % rho)
    # Create an array of labels initialized to "-1"
    lab = [-1] * len(nolab)

    # Assign the fictitious initialized label to the predicted labels
    label['predicted'] = lab

    # For each point in the test set we check that the maximum predicted probability is bigger than rho
    # and at that point we assign the predicted label
    for i in range(0, len(k)):
        # Check if the predicted probability is bigger than rho
        if np.amax(k[i]) >= rho:
            # Assign as label the class in which we have the higher predicted probability
            label['predicted'][i] = int(np.where(k[i] == np.amax(k[i]))[0])

    # Take the index of the point that have an associated new label
    index_nolab = label.index[label['predicted'] != -1].tolist()

    # Write the number of points at which we add the label
    result.write("Labeled points : %s \n" % len(index_nolab))

    # Recap of the label extension over the points
    result.write("\nClass division of Predicted L2:\n")
    result.write(label.loc[index_nolab]['predicted'].value_counts(sort=False, dropna=False).to_string())
    result.write("\n")

    # Write the performance measures associated to the extension
    # Accuracy
    result.write("Accuracy : %s \n" % mt.classification.accuracy_score(label.loc[index_nolab]['label'],
                                                                       label.loc[index_nolab]['predicted']))
    # Precision
    result.write("Precision : %s \n"
                 % mt.classification.precision_score(label.loc[index_nolab]['label'],
                                                     label.loc[index_nolab]['predicted'],
                                                     labels=labels, average='weighted'))
    # Recall
    result.write("Recall : %s \n"
                 % mt.classification.recall_score(label.loc[index_nolab]['label'], label.loc[index_nolab]['predicted'],
                                                  labels=labels, average='weighted'))
    # F1-Score
    result.write("F1 score : %s \n"
                 % mt.classification.f1_score(label.loc[index_nolab]['label'], label.loc[index_nolab]['predicted'],
                                              labels=labels, average='weighted'))

    predicted_classes = list(set(label.loc[index_nolab]['predicted']))
    # Write the accuracy per class
    result.write("\nAccuracy per class: \n")
    for i in predicted_classes:
        result.write("Class number %s \n" % str(i))
        # we pick all the ground truth in a list
        human = label.loc[(label['label'] == i) & (label['predicted'] != -1)]['label']
        # we pick the label assigned to the clustering
        predicted = label.loc[(label['label'] == i) & (label['predicted'] != -1)]['predicted']
        result.write("Accuracy: %s \n\n" % mt.classification.accuracy_score(human, predicted))

    #                        ---------------------------RHO = 60---------------------------

    # Set the confidence bound
    rho = 0.60
    result.write("RHO = %s \n" % rho)
    # Create an array of labels initialized to "-1"
    lab = [-1] * len(nolab)
    # Assign the fictitious initialized label to the predicted labels
    label['predicted'] = lab

    # For each point in the test set we check that the maximum predicted probability is bigger than rho
    # and at that point we assign the predicted label
    for i in range(0, len(k)):
        # Check if the predicted probability is bigger than rho
        if np.amax(k[i]) >= rho:
            # Assign as label the class in which we have the higher predicted probability
            label['predicted'][i] = int(np.where(k[i] == np.amax(k[i]))[0])

    # Take the index of the point that have an associated new label
    index_nolab = label.index[label['predicted'] != -1].tolist()

    # Write the number of points at which we add the label
    result.write("Labeled points : %s \n" % len(index_nolab))

    # Recap of the label extension over the points
    result.write("\nClass division of Predicted L2:\n")
    result.write(label.loc[index_nolab]['predicted'].value_counts(sort=False, dropna=False).to_string())
    result.write("\n")

    # Write the performance measures associated to the extension
    # Accuracy
    result.write("Accuracy : %s \n" % mt.classification.accuracy_score(label.loc[index_nolab]['label'],
                                                                       label.loc[index_nolab]['predicted']))
    # Precision
    result.write("Precision : %s \n"
                 % mt.classification.precision_score(label.loc[index_nolab]['label'],
                                                     label.loc[index_nolab]['predicted'],
                                                     labels=labels, average='weighted'))
    # Recall
    result.write("Recall : %s \n"
                 % mt.classification.recall_score(label.loc[index_nolab]['label'], label.loc[index_nolab]['predicted'],
                                                  labels=labels, average='weighted'))
    # F1-Score
    result.write("F1 score : %s \n"
                 % mt.classification.f1_score(label.loc[index_nolab]['label'], label.loc[index_nolab]['predicted'],
                                              labels=labels, average='weighted'))

    predicted_classes = list(set(label.loc[index_nolab]['predicted']))
    # Write the accuracy per class
    result.write("\nAccuracy per class: \n")
    for i in predicted_classes:
        result.write("Class number %s \n" % str(i))
        # we pick all the ground truth in a list
        human = label.loc[(label['label'] == i) & (label['predicted'] != -1)]['label']
        # we pick the label assigned to the clustering
        predicted = label.loc[(label['label'] == i) & (label['predicted'] != -1)]['predicted']
        result.write("Accuracy: %s \n\n" % mt.classification.accuracy_score(human, predicted))

    result.close()

Tock = time.time() - Tick
print("Time to execute the code: ", Tock)
