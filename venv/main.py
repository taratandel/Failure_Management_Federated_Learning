from dataDivider import cleanData, loadDataFrame, divideRandomly, calcFractions, divideByeqType
from clientbuilder import clientBuilder
from ANN_Classifier import *
from modelTester import *
import math
from numpy import linspace as lsp
import joblib
import time
from modelTester import plot_learning_curve as plc
from FedAVG import *
import matplotlib.pyplot as plt
df = loadDataFrame('Labelled_Data.csv')
dfs = divideByeqType(df)
for dfss in dfs:
    print(len(dfss))
    print(dfss['label'].value_counts())
clients_ts, client_tns = clientBuilder()
#
final_model_ts = runFedAvg(5, math.inf, 0.0001, clients_ts, "test_separated")
final_model_tns = runFedAvg(5, math.inf, 0.0001, client_tns, "test_not_separated")

joblib.dump(final_model_tns, filename="test_not_separetad")
joblib.dump(final_model_ts, filename="test_separated")
#
#
# epochs = 200
# batch_size = 50
# rounds = 1000
# current_acc = 0
# best_epochs = 0
# best_batch_size = 0
# best_regularization_term = 0
# list_param = []
# for epoch in range(1, epochs, 10):
#     for m in list(range(1, batch_size, 10)) + [math.inf]:
#         for regularization in [0.1, 0.001, 0.00001, 0.0000001]:
#             list_param.append([epoch, m, regularization])
#
# for i in range(0, len(list_param)):
#     epoch = list_param[i][0]
#     m = list_param[i][1]
#     regularization = list_param[i][2]
#     acc = runFedAvg(epoch, m, regularization)
#     if acc > current_acc:
#         current_acc = acc
#         best_epochs = epoch
#         best_batch_size = m
#         best_regularization_term = regularization
#
# print("best epoch:" + str(best_epochs) + "\n" +
#       "best mini_batch size:" + str(best_batch_size) + "\n" +
#       "best regularization term:" + str(best_regularization_term))

# ------------------------------------------------ Model Comparison -------------------------------------
# here we try to compare the performance of the model in two cases: trained alone or in collaborative mode
# -------------------------------------------------------------------------------------------------------
# First TEsting model


# acc_ts = []
# acc_tns = []
# for m in list(range(1, 51, 10)) + [math.inf]:
#     acc_ts.append(runFedAvg(5, math.inf, 0.000001, clients_ts, "final_model_test_separated"))
#     acc_tns.append(runFedAvg(4, math.inf, 0.000001, clients_tns, "final_model_test_not_separated"))
#
# name_plt = "accuracy-batch-sieze ts plot" + "\n" + " epoch = " + str(m) + "\n"
# print(name_plt)
# plotSimpleFigure(acc_ts, 'epochs', 'accuracy', name_plt, list(range(1, 51, 10)) + [60 + 1])
# name_plt = "accuracy-batch-size tns plot" + "\n" + " epoch = " + str(m) + "\n"
# print(name_plt)
# plotSimpleFigure(acc_tns, 'epochs', 'accuracy', name_plt, list(range(1, 51, 10)) + [60 + 1])
#
#
# final_model_ts = joblib.load("final_model_test_separated")
# final_model_tns = joblib.load("final_model_test_not_separated")
#
for i in range(len(clients_ts)):
    # train a model for each client without collaborating with other clients
    client = clients_ts[i]

    client_model = client.participantUpdate(None, None, epochs=10000, M=math.inf, regularization=0.0001)
    name = "final_isolated_model_ts" + str(i)
    joblib.dump(client_model, filename=name)
    # prepare a test set to be used for the testing phase for both models
    # client_model = joblib.load("final_isolated_model_ts" + str(i))
    # create the tester for the client trained alone
    tester_alone = ModelTester(client.X_test, client.y_test, client_model)
    # create the tester for the same client but this time it trained collaboratively
    # final_model = coordinator.broadcast(average_weights, i)
    tester_alone.calcStatistic()
    tester_alone.outputStatistics("test_alone_test_separated client number:" + str(i))
    plc(client_model, "learning curve test not separated model alone" + str(i),
        client.X, client.y, axes=None, ylim=None, cv=10,
        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5))
    plot_confusion_matrix(client.y_test.argmax(axis=1), client_model.predict(client.X_test).argmax(axis=1),
                          classes=[0, 1, 2, 3, 4, 5],
                          normalize=False,
                          title="test not separated model alone not normalized" + str(i),
                          cmap=plt.cm.Blues)

    plot_confusion_matrix(client.y_test.argmax(axis=1), client_model.predict(client.X_test).argmax(axis=1),
                          classes=[0, 1, 2, 3, 4, 5],
                          normalize=True,
                          title="test not separated model alone normalized"+ str(i),
                          cmap=plt.cm.Blues)

    # final_model_ts = joblib.load("test_separated")

    tester_collaborative = ModelTester(client.X_test, client.y_test, final_model_ts)
    tester_collaborative.calcStatistic()
    tester_collaborative.outputStatistics("test_collaborative test set separated client number:" + str(i))
    plc(final_model_ts, "learning curve test separated model collaborative" + str(i),
        client.X, client.y, axes=None, ylim=None, cv=10,
        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5))
    plot_confusion_matrix(client.y_test.argmax(axis=1), final_model_ts.predict(client.X_test).argmax(axis=1),
                          classes=[0, 1, 2, 3, 4, 5],
                          normalize=False,
                          title="test separated collaborative not normalized"+ str(i),
                          cmap=plt.cm.Blues)
    plot_confusion_matrix(client.y_test.argmax(axis=1), final_model_ts.predict(client.X_test).argmax(axis=1),
                          classes=[0, 1, 2, 3, 4, 5],
                          normalize=True,
                          title="test separated collaborative normalized"+ str(i),
                          cmap=plt.cm.Blues)
    # scenario 3
    client = client_tns[i]

    # client_model = joblib.load("final_isolated_model_tns" + str(i))
    client_model = client.participantUpdate(None, None, epochs=10000, M=math.inf, regularization=0.0001)
    name = "final_isolated_model_ts" + str(i)
    joblib.dump(client_model, filename=name)
    # create the tester for the client trained alone
    tester_alone = ModelTester(client.X_test, client.y_test, client_model)
    # create the tester for the same client but this time it trained collaboratively
    # final_model = coordinator.broadcast(average_weights, i)
    tester_alone.calcStatistic()
    tester_alone.outputStatistics("test_alone_test_not_separated client number:" + str(i))
    plc(client_model, "learning curve test not separated model alone" + str(i),
        client.X, client.y, axes=None, ylim=None, cv=10,
        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5))
    plot_confusion_matrix(client.y_test.argmax(axis=1), client_model.predict(client.X_test).argmax(axis=1),
                          classes=[0, 1, 2, 3, 4, 5],
                          normalize=False,
                          title="test not separated model alone not normalized"+ str(i),
                          cmap=plt.cm.Blues)
    plot_confusion_matrix(client.y_test.argmax(axis=1), client_model.predict(client.X_test).argmax(axis=1),
                          classes=[0, 1, 2, 3, 4, 5],
                          normalize=True,
                          title="test not separated model alone normalized"+ str(i),
                          cmap=plt.cm.Blues)


    # final_model_tns = joblib.load("test_not_separated")

    tester_collaborative = ModelTester(client.X_test, client.y_test, final_model_tns)
    tester_collaborative.calcStatistic()
    tester_collaborative.outputStatistics("test_collaborative test set not separated client number:" + str(i))
    plc(final_model_tns, "learning curve test not separated model collaborative" + str(i),
        client.X, client.y, axes=None, ylim=None, cv=10,
        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5))
    plot_confusion_matrix(client.y_test.argmax(axis=1), final_model_tns.predict(client.X_test).argmax(axis=1),
                          classes=[0, 1, 2, 3, 4, 5],
                          normalize=False,
                          title="test not separated collaborative not normalized"+ str(i),
                          cmap=plt.cm.Blues)

    plot_confusion_matrix(client.y_test.argmax(axis=1), final_model_tns.predict(client.X_test).argmax(axis=1),
                          classes=[0, 1, 2, 3, 4, 5],
                          normalize=True,
                          title="test not separated collaborative normalized"+ str(i),
                          cmap=plt.cm.Blues)
    plt.show()
