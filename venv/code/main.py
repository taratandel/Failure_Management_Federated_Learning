from numpy import save, load
import os
os.chdir(os.path.dirname(__file__))
from modelTester import testProcess as tP
from dataDivider import cleanData as cD
from clients import *
import joblib
import numpy as np
from FedAVG import runFedAvg as rFA
from Optamize import *
from ANN_Classifier import *
from matplotlib import pyplot as plt

def tolerant_mean(arrs):
    lens = [len(i) for i in arrs]
    arr = np.ma.empty((np.max(lens),len(arrs)))
    arr.mask = True
    for idx, l in enumerate(arrs):
        arr[:len(l),idx] = l
    return arr.mean(axis = -1), arr.std(axis=-1)


name = "Completely Random choice trial 4"
per_trial_total_acc = load(name + "per_trial_accuracy_per_client_alone.npy", allow_pickle=True)

per_trial_per_round_per_client_acc_fed = load(name + "per_trial_per_round_per_client_acc_fed.npy", allow_pickle=True)
per_trial_per_round_total_acc_fed = load(name + "per_trial_per_round_total_averaged_acc_fed.npy", allow_pickle=True)
per_trial_total_fed_cd = load(name + "per_trial_total_fed_cd.npy", allow_pickle=True)

average_per_trial_total_acc_alone = load(name + "per_trial_accuracy_total_alone.npy", allow_pickle=True)
average_per_trial_total_acc = load("average_per_trial_accuracy_per_client_alone.npy", allow_pickle=True)

average_per_trial_per_round_per_client_acc_fed = load(name + "average_per_trial_per_round_per_client_acc_fed.npy", allow_pickle=True)
average_per_trial_per_round_total_acc_fed = load(name + "average_per_trial_per_round_total_averaged_acc_fed.npy", allow_pickle=True)
average_per_trial_total_fed_cd = load(name + "average_per_trial_total_fed_cd.npy")

average_per_trial_total_acc_alone = load(name + "average_per_trial_accuracy_total_alone.npy", allow_pickle=True)

cr0 = per_trial_per_round_per_client_acc_fed[0][0][0][per_trial_per_round_per_client_acc_fed[0][0][0] != 0]
cr1 = per_trial_per_round_per_client_acc_fed[1][0][0][per_trial_per_round_per_client_acc_fed[1][0][0] != 0]
cr2 = per_trial_per_round_per_client_acc_fed[2][0][0][per_trial_per_round_per_client_acc_fed[2][0][0] != 0]
cr3 = per_trial_per_round_per_client_acc_fed[3][0][0][per_trial_per_round_per_client_acc_fed[3][0][0] != 0]
cr4 = per_trial_per_round_per_client_acc_fed[4][0][0][per_trial_per_round_per_client_acc_fed[4][0][0] != 0]
cr5 = per_trial_per_round_per_client_acc_fed[0][0][1][per_trial_per_round_per_client_acc_fed[0][0][1] != 0]
cr6 = per_trial_per_round_per_client_acc_fed[1][0][1][per_trial_per_round_per_client_acc_fed[1][0][1] != 0]
cr7 = per_trial_per_round_per_client_acc_fed[2][0][1][per_trial_per_round_per_client_acc_fed[2][0][1] != 0]
cr8 = per_trial_per_round_per_client_acc_fed[3][0][1][per_trial_per_round_per_client_acc_fed[3][0][1] != 0]
cr9 = per_trial_per_round_per_client_acc_fed[4][0][1][per_trial_per_round_per_client_acc_fed[4][0][1] != 0]
cr10 = per_trial_per_round_per_client_acc_fed[0][0][2][per_trial_per_round_per_client_acc_fed[0][0][2] != 0]
cr11 = per_trial_per_round_per_client_acc_fed[1][0][2][per_trial_per_round_per_client_acc_fed[1][0][2] != 0]
cr12 = per_trial_per_round_per_client_acc_fed[2][0][2][per_trial_per_round_per_client_acc_fed[2][0][2] != 0]
cr13 = per_trial_per_round_per_client_acc_fed[3][0][2][per_trial_per_round_per_client_acc_fed[3][0][2] != 0]
cr14 = per_trial_per_round_per_client_acc_fed[4][0][2][per_trial_per_round_per_client_acc_fed[4][0][2] != 0]

y1, error = tolerant_mean([cr0,cr1,cr2,cr3,cr4])
y2, error = tolerant_mean([cr5,cr6,cr7,cr8,cr9])
y3, error = tolerant_mean([cr10,cr11,cr12,cr13,cr14])
y4 = np.mean([y1,y2,y3], axis = 0)
# plt.plot(np.arange(len(y4))+1, (y4 * 100).tolist(), color='b', linestyle='-', label="Total Accuracy for fedAVG", linewidth=0.2)
# plt.plot(np.arange(len(y3))+4, [88.2] * 10000, color='r', label="Total Accuracy when trained with all data", linewidth=2)
# plt.plot(np.arange(len(y3))+4, [90] * 10000, color='g', label="Total Accuracy Averaged between three clients", linewidth=2)


plt.plot(np.arange(len(y1))+1, y1, color='b', linestyle='-', label="client 1", linewidth=0.1)
plt.plot(np.arange(len(y3))+1, [0.874] * 10000, color='b', label="client 1_trained_alone", linewidth=5)
plt.plot(np.arange(len(y2))+1, y2, color='g', linestyle='-', label="client 2", linewidth=0.1)
plt.plot(np.arange(len(y3))+1, [0.90] * 10000, color='g', label="client 2_trained_alone", linewidth=5)
plt.plot(np.arange(len(y3))+1, y3, color='r', linestyle='-', label="client 3", linewidth=0.1)

plt.plot(np.arange(len(y3))+1, [0.87] * 10000, color='r', label = "client 3_trained_alone", linewidth=5)
plt.xlabel("round")
plt.ylabel("accuracy")
plt.legend()
ymin, ymax = plt.ylim()

plt.ylim(.8, ymax)
plt.show()
# plt.xlabel("round")
# plt.ylabel("accuracy")
# plt.legend()
# plt.show()
#
#
# plt.xlabel("round")
# plt.ylabel("accuracy")
# plt.legend()
# plt.show()
#

# plt.xlabel("round")
# plt.ylabel("accuracy")
# plt.legend()
#
# plt.show()
# -------------- Trial variables
# we need also confusion matrix for it
# epochs = [10,100,200,500]
# batch_size = [32,64,128]
# rounds = 100
# total_trails = 5
# train_alone_epochs = None
# average_accuracy_per_client_alone = 0
# average_accuracy_per_client_fed = 0
# result_of_optimization = []
# average_accuracy_per_class_alone = []
# average_accuracy_per_class_def = []
# # ------------------------------------
#
# switcher = {
#     0: "Completely Random choice",
#     1: "All classes in all groups",
#     2: "proportions of [6,4,4]"
# }
# per_trial_total_acc = []
#
# per_trial_per_round_total_acc_fed = []
# per_trial_per_round_per_client_acc_fed = []
#
# per_trial_total_acc_alone = []
#
# per_trial_total_fed_cd = []
# for trial in range(total_trails):
#     name = switcher.get(0, "nothing") + " " + "trial" + " " + str(trial)
#     total_scenarios_data = [
#          clientBuilderForScenario1(name),
#                            # clientBuilderForClassesPerEach(
#                             #    switcher.get(1, "nothing") + " " + "trial" + " " + str(trial)),
#                             # clientBuilderForClassesProportional(switcher.get(2, "nothing") + "trial" + str(trial))
#     ]
#     per_scenario_total_acc = []
#
#     per_scenario_total_acc_fed = []
#     per_scenario_client_acc_fed = []
#
#     per_scenario_total_acc_alone = []
#
#     per_scenario_total_acc_fed_cd = []
#
#     i = 0
#     for scenarios in total_scenarios_data:
#         total_df = []
#         # total_y = []
#         total_test = []
#         # total_X_test = []
#         client_acc = []
#
#         for client in scenarios:
#             X_train = client.X
#             y_train = client.y
#             X_test = client.X_test
#             y_test = client.y_test
#             total_df.append(client.dataFrame)
#             total_test.append(client.test)
#
#
#             name_client = client.name + (" trial number %s " %(trial))
#             # ------------ Train alone
#             train_alone_name = name_client + "" + "train_alone"
#             ann = client.participantUpdate(coefs=None, intercepts=None, M='auto', regularization=0.000001, epochs=train_alone_epochs)
#             joblib.dump(ann, train_alone_name)
#             acc = tP(X_test, y_test, X_train, y_train, ann, train_alone_name)
#             client_acc.append(acc)
#
#             i = i % 3
#             # --------------------------------
#         # -------------------- Train alone with all the data
#         train_alone_name = name + "" + "train_alone_with all the data"
#         df = pd.concat(total_df)
#         test = pd.concat(total_test)
#         X, y = cD(df)
#         X_test, y_test = cD(test)
#         ann_total = trainANN(X, y, epochs=train_alone_epochs, M='auto', coef=None, intercept=None)
#         joblib.dump(ann_total, filename=train_alone_name)
#         acc = tP(X_test, y_test, X, y, ann_total, "total_test_for_train_alone_with_concatdata")
#         per_scenario_total_acc_alone.append(acc)
#
#         # ------------- OPTIMIZE
#         optimize_name = name + " optimization"
#         best_epoch, best_batch = optimize(epochs, batch_size, scenarios, rounds, name)
#         # ------------- FedAvg
#         fedavg_name = name + " fedavg"
#         rounds_fed = rounds * 100
#         model, round_accuracy, per_client_accuracy = rFA(epoch=best_epoch, m=best_batch, regularization=0.000001, clients=scenarios, name=fedavg_name, round=rounds_fed)
#
#         per_scenario_client_acc_fed.append(per_client_accuracy)
#         per_scenario_total_acc_fed.append(round_accuracy)
#         joblib.dump(model, filename=fedavg_name + " model " + "train with FedAVG")
#         # --------- Test FedAvg with all the data
#         acc = tP(X_test, y_test, X, y, model, fedavg_name+ " total_test_Fed_avg_with_concatdata")
#         per_scenario_total_acc_fed_cd.append(acc)
#
#         per_scenario_total_acc_alone.append(acc)
#         per_scenario_total_acc.append(client_acc)
#
#     per_trial_total_acc.append(per_scenario_total_acc)
#
#     per_trial_per_round_total_acc_fed.append(per_scenario_total_acc_fed)
#     per_trial_per_round_per_client_acc_fed.append(per_scenario_client_acc_fed)
#
#     per_trial_total_acc_alone.append(per_scenario_total_acc_alone)
#
#     per_trial_total_fed_cd.append(per_scenario_total_acc_fed_cd)
#
#
# average_per_trial_total_acc = np.array(per_trial_total_acc).mean(axis=0).astype(int).tolist()
#
#
# average_per_trial_per_round_total_acc_fed = np.array(per_trial_total_acc).mean(axis=0).astype(int).tolist()
# average_per_trial_per_round_per_client_acc_fed = np.array(per_trial_per_round_per_client_acc_fed).mean(axis=0).astype(int).tolist()
#
# average_per_trial_total_acc_alone = np.array(per_trial_total_acc_alone).mean(axis=0).astype(int).tolist()
#
# average_per_trial_total_fed_cd = np.array(per_trial_total_fed_cd).mean(axis=0).astype(int).tolist()
#
# save(name + "per_trial_accuracy_per_client_alone", per_trial_total_acc)
#
# save(name + "per_trial_per_round_per_client_acc_fed", per_trial_per_round_per_client_acc_fed)
# save(name + "per_trial_per_round_total_averaged_acc_fed", per_trial_per_round_total_acc_fed)
# save(name + "per_trial_total_fed_cd", per_trial_total_fed_cd)
#
# save(name + "per_trial_accuracy_total_alone", average_per_trial_total_acc_alone)
# save("average_per_trial_accuracy_per_client_alone", average_per_trial_total_acc)
#
# save(name + "average_per_trial_per_round_per_client_acc_fed", average_per_trial_per_round_per_client_acc_fed)
# save(name + "average_per_trial_per_round_total_averaged_acc_fed", average_per_trial_per_round_total_acc_fed)
# save(name + "average_per_trial_total_fed_cd", average_per_trial_total_fed_cd)
#
# save(name + "average_per_trial_accuracy_total_alone", average_per_trial_total_acc_alone)
#
#
