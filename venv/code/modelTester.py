from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as mt
import matplotlib.pyplot as plt

print(__doc__)

from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit


def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    estimator : estimator instance
        An estimator instance implementing `fit` and `predict` methods which
        will be cloned for each validation.

    title : str
        Title for the chart.

    X : array-like of shape (n_samples, n_features)
        Training vector, where ``n_samples`` is the number of samples and
        ``n_features`` is the number of features.

    y : array-like of shape (n_samples) or (n_samples, n_features)
        Target relative to ``X`` for classification or regression;
        None for unsupervised learning.

    axes : array-like of shape (3,), default=None
        Axes to use for plotting the curves.

    ylim : tuple of shape (2,), default=None
        Defines minimum and maximum y-values plotted, e.g. (ymin, ymax).

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like of shape (n_ticks,)
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the ``dtype`` is float, it is regarded
        as a fraction of the maximum size of the training set (that is
        determined by the selected validation method), i.e. it has to be within
        (0, 1]. Otherwise it is interpreted as absolute sizes of the training
        sets. Note that for classification the number of samples usually have
        to be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")
    plt.savefig(title + "learning_curve" + ".png")
    plt.clf()

    return plt


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
    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    acc_per_class = cmn.diagonal()

    # Only use the labels that appear in the data
    classes = classes
    if normalize:
        cm = cmn
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
    # plt.show()
    plt.savefig('cfm %s.png' % title)
    plt.clf()

    return acc_per_class


# Code provided by: https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
def plotSimpleFigure(values, xlabel, ylabel, title, values2=None):
    if values2 is None:
        plt.plot(range(1, len(values) + 1), values, label='accuracy')  # Plot some data on the (implicit) axes.
    else:
        plt.plot(values2, values)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title('simple curve %s.png' % title)
    plt.legend()
    # plt.show()
    plt.savefig('simple curve %s.png' % title)
    plt.clf()


def testProcess(X_test, y_test, X_train, y_train, model, name, should_plt=True):
    np.set_printoptions(precision=2)
    tester_alone = ModelTester(X_test, y_test, model)
    # create the tester for the same client but this time it trained collaboratively
    # final_model = coordinator.broadcast(average_weights, i)
    tester_alone.calcStatistic()
    if not should_plt:
        return tester_alone.acc
    tester_alone.outputStatistics(name + ":")
    if (X_train is not None) and (y_train is not None):
        title = name + " learning curve"

        plot_learning_curve(model, title,
                            X_train, y_train, axes=None, ylim=None, cv=10,
                            n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5))
    # title = name + " confusion matrix normalized"
    # plot_confusion_matrix(y_test.argmax(axis=1), model.predict(X_test).argmax(axis=1),
    #                       classes=[0, 1, 2, 3, 4, 5],
    #                       normalize=False,
    #                       title=title,
    #                       cmap=plt.cm.Blues)
    title = name + " confusion matrix not normalized"
    acc_per_class = plot_confusion_matrix(y_test.argmax(axis=1), model.predict(X_test).argmax(axis=1),
                                          classes=[0, 1, 2, 3, 4, 5],
                                          normalize=True,
                                          title=title,
                                          cmap=plt.cm.Blues)
    tester_alone.accuracy_per_class = acc_per_class
    # plt.show()
    # plt.savefig("%s.png" %title)
    return tester_alone.acc


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
        self.accuracy_per_class = None
        self._predict()

    def _predict(self):
        self.y_predicted = self.model.predict(self.X)
        # Take the index where is present the value 1, that identify the predicted label
        self.y_predicted = np.argmax(self.y_predicted, axis=1)
        # For each point, predict the probability to belong to each class
        self.y_probability = self.model.predict_proba(self.X)

    def calcStatistic(self):
        # Open the file where will be saved all the test performances
        y_test = np.argmax(self.y, axis=1)
        # create a structure to perform the measure manually
        labels = list(set(y_test))
        self.acc = mt.accuracy_score(y_test, self.y_predicted)

        self.prec = mt.precision_score(y_test, self.y_predicted, labels=labels, average=None)

        self.recall = mt.recall_score(y_test, self.y_predicted, labels=labels, average=None)

        self.f1 = mt.f1_score(y_test, self.y_predicted, labels=labels, average=None)

    def outputStatistics(self, performance_name):
        performances = open(performance_name + " analytics", "w")

        performances.write("Accuracy automated: %s \n" % self.acc)
        performances.write("Precision per class automated: %s \n"
                           % self.prec)
        performances.write("Recall per class automated: %s \n"
                           % self.recall)
        performances.write("F1-score automated: %s \n"
                           % self.f1)
        performances.write("Accuracy per class: %s \n"
                           % self.accuracy_per_class)
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
            # plt.savefig('%s.png' % title)
            # https: // scikit - learn.org / stable / auto_examples / model_selection / plot_roc.html

    # ----------------------------------------------------------------------------------------------------------------------
