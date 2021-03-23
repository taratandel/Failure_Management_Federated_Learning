class ModelTester:
    """
    Predict the labels and gives test statistics
    """

    def __init__(self, X, y, model):
        self.X = X
        self.y = y
        self.model = model
        self._predict()

    def _predict(self):
        self.y_predicted = ann.predict(self.X)
        # Take the index where is present the value 1, that identify the predicted label
        self.y_predicted = np.argmax(y_predicted, axis=1)
        # For each point, predict the probability to belong to each class
        self.y_probability = ann.predict_proba(X_test)

    def calcStatistic(self):
        # Open the file where will be saved all the test performances
        performances = open(performanceName, "w")
        acc = pd.DataFrame()
        y_test = np.argmax(y_test, axis=1)
        # create a structure to perform the measure manually
        acc['ground_truth'] = y_test  # add the ground truth
        acc['predicted'] = y_predicted  # add the predicted labels
        precision = [0] * n_label  # precision list
        recall = [0] * n_label  # recall list
    # def plotConfusionMatrix(self):

    # def plotStatistic(self):



    performances.write("Training Time(s): %s \n" % str(Tock))  # Print the training time needed in the Test phase
    #           -------------------------------------------- ACCURACY --------------------------------------------
    #                                 --------------------- AUTOMATED ---------------------

    # Write the accuracy inside the performances file
    performances.write("Accuracy automated: %s \n" % mt.accuracy_score(y_test, y_predicted))
    #                                 --------------------- MANUALLY ---------------------

    # Write the accuracy inside the performances file
    performances.write(
        "Accuracy manually: %s \n" % str(len(acc.loc[acc['ground_truth'] == acc['predicted']]) / len(acc)))
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
