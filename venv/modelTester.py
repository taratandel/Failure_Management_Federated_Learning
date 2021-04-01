from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as mt
import matplotlib.pyplot as plt


def plot_confusion_matrix(y_true, y_pred, classes=[],
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
    cm = confusion_matrix(y_true, y_pred, classes)
    # Only use the labels that appear in the data
    classes = classes
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
def plotSimpleFigure(values, xlabel, ylabel, title):
    plt.plot(range(1, len(values)+1), values, label='accuracy')  # Plot some data on the (implicit) axes.
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.show()

class ModelTester:
    """
    Predict the labels and gives test statistics
    """

    def __init__(self, X, y, model):
        self.f1 = []
        self.recall = []
        self.acc = []
        self.prec = []
        self.X = X
        self.y = y
        self.model = model
        self._predict()

    def _predict(self):
        self.y_predicted = self.model.predict(self.X)
        # Take the index where is present the value 1, that identify the predicted label
        self.y_predicted = np.argmax(self.y_predicted, axis=1)
        # For each point, predict the probability to belong to each class
        self.y_probability = self.model.predict_proba(self.X)

    def calcStatistic(self, performance_name):
        # Open the file where will be saved all the test performances
        y_test = np.argmax(self.y, axis=1)
        # create a structure to perform the measure manually
        labels = list(set(y_test))
        self.acc = mt.accuracy_score(y_test, self.y_predicted)

        self.prec = mt.precision_score(y_test, self.y_predicted, labels=labels, average=None)

        self.recall = mt.recall_score(y_test, self.y_predicted, labels=labels, average=None)

        self.f1 = mt.f1_score(y_test, self.y_predicted, labels=labels, average=None)

    def outputStatistics(self):
        performances = open(performance_name, "w")

        performances.write("Accuracy automated: %s \n" % self.acc[-1])
        performances.write("Precision per class automated: %s \n"
                           % self.prec[-1])
        performances.write("Recall per class automated: %s \n"
                           % self.recall[-1])
        performances.write("F1-score automated: %s \n"
                           % self.f1[-1])
        performances.close()

    def plotROCCurve(self):
        if n_label == 2:
            # Calculate the False positive rate and the True positive rate
            fpr, tpr, thresholds = roc_curve(self.y, self.y_probability[:, 1], pos_label=1)
            # Calculate the Area under the Roc Curve
            auc_score = roc_auc_score(self.y, self.y_probability[:, 1])
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

    def plotConfusionMatrix(self):
        np.set_printoptions(precision=2)

        # Plot non-normalized confusion matrix
        plot_confusion_matrix(self.y, self.y_predicted,
                              title='Confusion matrix, without normalization')

        # # Plot normalized confusion matrix
        # plot_confusion_matrix(y_test, y_predicted, classes=labels, normalize=True,
        #                       title='Normalized confusion matrix')

        plt.show()
