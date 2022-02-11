from numpy import save, load
import os
from modelTester import testProcess as tP
from dataDivider import cleanData as cD
from clients import *
import joblib
from numpy import save, load
import numpy as np
from FedAVG import runFedAvg as rFA
from Optamize import *
from ANN_Classifier import *

os.chdir(os.path.dirname(__file__))
def tolerant_mean(arrs):
    lens = [len(i) for i in arrs]
    arr = np.ma.empty((np.max(lens),len(arrs)))
    arr.mask = True
    for idx, l in enumerate(arrs):
        arr[:len(l),idx] = l
    return arr.mean(axis = -1), arr.std(axis=-1)

# -------------- Trial variables
# we need also confusion matrix for it
epochs = [10, 100, 200, 500]
batch_size = [32, 64, 128]
rounds = 100
total_trails = 1
train_alone_epochs = None
# ------------------------------------
save_path = '/Users/loopsie/Downloads/Tara_Tandel_2021-2/Source Code/venv/code'

switcher = {
    0: "Completely Random choice",
    1: "All classes in all groups",
    2: "proportions of [6,4,4]",
    3: "proportions to eq type",
    4: "1 client missing 1 class (hardwareFailure)",
    5: "1 client missing 1 class (label 4)",
    6: "1 client missing 1 class (label 2)",
    7: "1 client missing 1 class (all)"
}

per_trial_total_acc = []
per_trial_per_round_total_acc_fed = []
per_trial_per_round_per_client_acc_fed = []
per_trial_total_acc_alone = []
per_trial_total_fed_cd = []
per_trial_per_client_accuracy_on_cd_data = []

for trial in range(total_trails):



    total_scenarios_data = [
        # clientBuilderForScenario1(name),
        # clientBuilderForClassesPerEach(switcher.get(1, "nothing") + " " + "trial" + " " + str(trial)),
        #  clientBuilderForClassesProportional(switcher.get(2, "nothing") + "trial" + str(trial)),
        # clientBuilderForClientMissing1class(switcher.get(6, "nothing") + "trial" + str(trial))
    ]

    for j in range(6):
        total_scenarios_data.append(buildClientWithPath('splitTest' + str(j), j))

    for t in range(6):
        total_scenarios_data.append(buildClientWithPath('nonSplitTest' + str(t), t))

    per_scenario_total_acc = []

    per_scenario_total_acc_fed = []
    per_scenario_client_acc_fed = []

    per_scenario_total_acc_alone = []

    per_scenario_total_acc_fed_cd = []

    per_scenario_per_client_accuracy_on_cd_data = []
    name_counter = 0
    i = 0
    for scenarios in total_scenarios_data:
        name_counter = name_counter + 1
        name = switcher.get(7, "nothing") + "trial" + str(trial) + " " + str(name_counter)
        total_df = []
        total_test = []

        client_acc = []
        client_model = []
        client_acc_on_cd_data = []

        for client in scenarios:
            X_train = client.X
            y_train = client.y
            X_test_alone = client.X_test
            y_test_alone = client.y_test
            total_df.append(client.dataFrame)
            total_test.append(client.test)

            name_client = client.name + (" trial number %s " % (trial))
            # ------------ Train alone
            train_alone_name = name_client + "" + "train_alone"
            ann = client.participantUpdate(coefs=None, intercepts=None, M='auto', regularization=0.000001,
                                           epochs=train_alone_epochs)
            completeName = save_path+'/clients_in_scenario'
            if not os.path.exists(completeName):
                os.makedirs(completeName)

            completeName = os.path.join(completeName, train_alone_name)

            joblib.dump(ann, completeName)
            acc = tP(X_test_alone, y_test_alone, None, None, ann, completeName)
            client_model.append(ann)
            client_acc.append(acc)
            i = i % 3


            # --------------------------------

        # -------------------- Train alone with all the data
        train_alone_name = name + (" trial number %s " % (trial)) + " " + "train_alone_with all the data"
        completeName = save_path + '/Train_alone_all_data'
        if not os.path.exists(completeName):
            os.makedirs(completeName)

        completeName = os.path.join(completeName, train_alone_name)
        df = pd.concat(total_df)
        test = pd.concat(total_test).drop_duplicates().reset_index(drop=True)
        X, y = cD(df)
        X_test, y_test = cD(test)
        ann_total = trainANN(X, y, epochs=train_alone_epochs, M='auto', coef=None, intercept=None)
        joblib.dump(ann_total, filename=completeName)
        acc = tP(X_test, y_test, None, None, ann_total, completeName + "total_test_for_train_alone_with_concatdata")
        per_scenario_total_acc_alone.append(acc)

        # --------------------- Test on client model with concatenated data
        completeName = save_path + '/Test_on_client_model_with_concatenated_data'
        if not os.path.exists(completeName):
            os.makedirs(completeName)

        completeName = os.path.join(completeName, train_alone_name)
        for model in client_model:
            acc = tP(X_test, y_test, None, None, model, completeName + "test on cd data")
            client_acc_on_cd_data.append(acc)
        per_scenario_per_client_accuracy_on_cd_data.append(client_acc_on_cd_data)
        # ------------- OPTIMIZE
        # optimize_name = name + " optimization"
        # best_epoch, best_batch = optimize(epochs, batch_size, scenarios, rounds, name)
        # ------------- FedAvg
        best_epoch = 500
        best_batch = 64
        fedavg_name = name + " fedavg"

        completeName = save_path + '/fedAvg'
        if not os.path.exists(completeName):
            os.makedirs(completeName)

        completeName = os.path.join(completeName, fedavg_name)
        rounds_fed = rounds * 100
        model, round_accuracy, per_client_accuracy = rFA(epoch=best_epoch, m=best_batch, regularization=0.000001,
                                                         clients=scenarios, name=completeName, round=rounds_fed)

        per_scenario_client_acc_fed.append(per_client_accuracy)
        per_scenario_total_acc_fed.append(round_accuracy)
        joblib.dump(model, filename=completeName + " model " + "train with FedAVG")
        # --------- Test FedAvg with all the data
        acc = tP(X_test, y_test, None, None, model, completeName + " total_test_Fed_avg_with_concatdata")
        per_scenario_total_acc_fed_cd.append(acc)

        per_scenario_total_acc_alone.append(acc)
        per_scenario_total_acc.append(client_acc)

    per_trial_total_acc.append(per_scenario_total_acc)

    per_trial_per_round_total_acc_fed.append(per_scenario_total_acc_fed)
    per_trial_per_round_per_client_acc_fed.append(per_scenario_client_acc_fed)

    per_trial_total_acc_alone.append(per_scenario_total_acc_alone)

    per_trial_total_fed_cd.append(per_scenario_total_acc_fed_cd)
    per_trial_per_client_accuracy_on_cd_data.append(per_scenario_per_client_accuracy_on_cd_data)

save(name + "per_trial_accuracy_per_client_alone", per_trial_total_acc)
save(name + "per_trial_per_round_per_client_acc_fed", per_trial_per_round_per_client_acc_fed)
save(name + "per_trial_per_round_total_averaged_acc_fed", per_trial_per_round_total_acc_fed)
save(name + "per_trial_total_fed_cd", per_trial_total_fed_cd)
save(name + "per_trial_per_client_accuracy_on_cd_data", per_trial_per_client_accuracy_on_cd_data)
name = "1 client missing 1 class (label 2) trial 4"
per_trial_total_acc = load(name + "per_trial_accuracy_per_client_alone.npy", allow_pickle=True)

per_trial_per_round_per_client_acc_fed = load(name + "per_trial_per_round_per_client_acc_fed.npy", allow_pickle=True)
per_trial_per_round_total_acc_fed = load(name + "per_trial_per_round_total_averaged_acc_fed.npy", allow_pickle=True)
per_trial_total_fed_cd = load(name + "per_trial_total_fed_cd.npy", allow_pickle=True)

average_per_trial_total_acc_alone = load(name + "per_trial_per_client_accuracy_on_cd_data.npy", allow_pickle=True)


y = []
for j in range(3):
    cr = []
    for i in range(5):
        cr.append(per_trial_per_round_per_client_acc_fed[i][0][j][per_trial_per_round_per_client_acc_fed[i][0][j] != 0])
    y1, error = tolerant_mean(cr)
    y.append(y1)

y4 = np.mean(y, axis = 0)
plt.plot(np.arange(len(y4))+1, y4, color='b', linestyle='-', label="Average All clients when train with FedAvg", linewidth=0.1)
plt.plot(np.arange(len(y4))+1, [0.90] * len(y4), color='r', label="Train on complete Data set", linewidth=5)
plt.plot(np.arange(len(y4))+1, [0.88] * len(y4), color='g', label="Average All clients when train alone", linewidth=5)
plt.plot(np.arange(len(y4))+1, [0.66] * len(y4), color='y', label="Average All Clients when tested on whole test set", linewidth=5)

plt.xlabel("round")
plt.ylabel("accuracy")
plt.legend()
ymin, ymax = plt.ylim()

plt.ylim(.5, ymax)
plt.show()

plt.plot(np.arange(len(y[0]))+1, y[0], color='b', linestyle='-', label="client 1", linewidth=0.1)
plt.plot(np.arange(len(y[0]))+1, [0.85] * len(y[0]), color='r', label="client 1_trained_alone", linewidth=5)
plt.plot(np.arange(len(y[0]))+1, [0.68] * len(y[0]), color='g', label="client 1_trained_alone_tested_on_whole_set", linewidth=5)
plt.xlabel("round")
plt.ylabel("accuracy")
plt.legend()
ymin, ymax = plt.ylim()

plt.ylim(.65, ymax)
plt.show()
plt.plot(np.arange(len(y[1]))+1, y[1], color='b', linestyle='-', label="client 2", linewidth=0.1)
plt.plot(np.arange(len(y[1]))+1, [0.87] * len(y[0]), color='r', label="client 2_trained_alone", linewidth=5)
plt.plot(np.arange(len(y[0]))+1, [0.68] * len(y[0]), color='g', label="client 2_trained_alone_tested_on_whole_set", linewidth=5)
plt.xlabel("round")
plt.ylabel("accuracy")
plt.legend()
ymin, ymax = plt.ylim()

plt.ylim(.65, ymax)
plt.show()
plt.plot(np.arange(len(y[0]))+1, y[2], color='b', linestyle='-', label="client 3", linewidth=0.1)
plt.plot(np.arange(len(y[0]))+1, [0.91] * len(y[0]), color='r', label="client 3_trained_alone", linewidth=5)
plt.plot(np.arange(len(y[0]))+1, [0.62] * len(y[0]), color='g', label="client 3_trained_alone_tested_on_whole_set", linewidth=5)

plt.xlabel("round")
plt.ylabel("accuracy")
plt.legend()
ymin, ymax = plt.ylim()

plt.ylim(.6, ymax)

plt.show()