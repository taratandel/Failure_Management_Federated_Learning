from clients import clientBuilder as cB
from dataDivider import *
from modelTester import testProcess as tP
from clients import *
from modelAverage import weightedAverageloss
from FedAVG import runFedAvg as rFA
import matplotlib.pyplot as plt
import math
import joblib
from numpy import save, load

round = 10000
epochs = 200
batch_size = 50
rounds = 1000
number_of_trail = 10
total_acc = [[] for _ in range(number_of_trail)]
total_fed_client_acc = [[] for _ in range(number_of_trail)]
total_fed_round_acc = [[] for _ in range(number_of_trail)]
for i in range(number_of_trail):
    clients_data = clientBuilderForScenario1()
    clients = []
    for client_data in clients_data:
        client = Client(data=client_data, prepare_for_testing=True)
        clients.append(client)
    for indx in range(len(clients)):
        client = clients[indx]
        X_train = client.X
        y_train = client.y
        X_test = client.X_test
        y_test = client.y_test
        name = "training alone for client" + str(indx) + "round" + str(i)
        ann = trainANN(X_train, y_train, epochs=round, M=math.inf, coef=None, intercept=None)
        joblib.dump(ann, filename=name)
        total_acc[i].append(tP(X_test, y_test, X_train, y_train, ann, name))
    final_model, round_acc, clients_acc = rFA(4, math.inf, 0.0001, clients, "federated average", round)
    total_fed_round_acc.append(round_acc)
    total_fed_client_acc.append(clients_acc)
    namefl = "federated training random mode round:" + str(i)
    joblib.dump(final_model, filename=namefl)
save('test_alone_accuracy_data', total_acc)
save('fed_round_acc', total_fed_client_acc)
save('fed_client_acc', total_fed_round_acc)

# round = 10000
# # scenario 1
# # we have 100% of data we divided 80 20 then we train on the whole and test on the equipment type normally
# X_train, y_train = cD(concatenated_train)
# X_test, y_test = cD(concatenated_test)
# # ann = trainANN(X_train, y_train, epochs=round, M=math.inf, coef=None, intercept=None)
# name = "training with whole data"
# # joblib.dump(ann, filename=name)
# ann = joblib.load(name)
# total_acc_sce_1 = tP(X_test, y_test, X_train, y_train, ann, name)
#
# i = 0
# total_accs_sce_1 = []
# ts = dividedTestSetPereqTyep(concatenated_test)
# for test in ts:
#     X_test, y_test = cD(test)
#     name = "test per eqType on the whole data"
#     total_accs_sce_1.append(tP(X_test, y_test, None, None, ann, name + str(i)))
#     i = i + 1
#

#
# total_accs_sce_2 = []
# total_accs_sce_2_fl = []
# for i in range(len(client_sets)):
#     client = client_sets[i]
#
#     # client_model = client.participantUpdate(None, None, epochs=round, M=math.inf, regularization=0.0001)
#     name = "training and testing per eqType" + str(i)
#     # joblib.dump(client_model, filename=name)
#     client_model = joblib.load(name)
#     total_accs_sce_2.append(tP(client.X_test, client.y_test, client.X, client.y, client_model, name))
#
#     name = namefl + str(i)
#     total_accs_sce_2_fl.append(tP(client.X_test, client.y_test, client.X, client.y, final_model, name))
#
#     plt.plot(list(range(round)), [total_accs_sce_1[i]]*round, 'r', label="scenario 1 full knowledge eqTpe" + str(i))
#     plt.xlabel("round")
#     plt.ylabel("accuracy")
#     plt.title("full knowledge per eq type " + str(i))
#     plt.legend()
#
#     plt.plot(list(range(round)), [total_accs_sce_2[i]]*round, 'b',
#              label="scenario 2 isolated knowledge eqTpe" + str(i))
#     plt.xlabel("round")
#     plt.ylabel("accuracy")
#     plt.title("isolated knowledge per eq type" + str(i))
#     plt.legend()
#
#     plt.plot(list(range(len(clients_acc[i]))), clients_acc[i], 'g',
#              label="scenario 2 federated knowledge eqTpe" + str(i))
#     plt.xlabel("round")
#     plt.ylabel("accuracy")
#     plt.title("federated knowledge per eq type" + str(i))
#     plt.legend()
#     plt.show()
#
# plt.plot(list(range(len(total_acc_sce_1))), [total_acc_sce_1]*round, 'r', label= "full knowledge")
# plt.plot(list(range(round)), round_acc, 'b', label="federated knowledge average")
# avg = weightedAverageloss(total_accs_sce_2, [1588, 43, 555])
# plt.plot(list(range(round)), [avg]*round, 'g', label="partial learning average")
# plt.plot(list(range(round)), [total_acc_sce_1]*round, 'r', label= "full knowledge")
#
