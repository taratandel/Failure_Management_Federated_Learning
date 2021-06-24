from modelTester import testProcess as tP
from dataDivider import cleanData as cD
from clients import *
import joblib
import os
from numpy import save, load
import numpy as np
from FedAVG import runFedAvg as rFA
from Optamize import *
from ANN_Classifier import *

os.chdir(os.path.dirname(__file__))
# -------------- Trial variables
# we need also confusion matrix for it
epochs = 50
batch_size = 30
rounds = 500
total_trails = 5
average_accuracy_per_client_alone = 0
average_accuracy_per_client_fed = 0
result_of_optimization = []
average_accuracy_per_class_alone = []
average_accuracy_per_class_def = []
# ------------------------------------

switcher = {
    0: "Completely Random choice",
    1: "All classes in all groups",
    2: "proportions of [6,4,4]"
}
per_trial_total_acc = []

per_trial_per_round_total_acc_fed = []
per_trial_per_round_per_client_acc_fed = []

per_trial_total_acc_alone = []

per_trial_total_fed_cd = []
for trial in range(total_trails):
    total_scenarios_data = [
        # clientBuilderForScenario1(switcher.get(0, "nothing") + " " + "trial" + " " + str(trial)),
                            clientBuilderForClassesPerEach(
                                switcher.get(1, "nothing") + " " + "trial" + " " + str(trial)),
                            # clientBuilderForClassesProportional(switcher.get(2, "nothing") + "trial" + str(trial))
    ]
    per_scenario_total_acc = []

    per_scenario_total_acc_fed = []
    per_scenario_client_acc_fed = []

    per_scenario_total_acc_alone = []

    per_scenario_total_acc_fed_cd = []

    i = 0
    name = ""
    for scenarios in total_scenarios_data:
        total_df = []
        # total_y = []
        total_test = []
        # total_X_test = []
        client_acc = []

        for client in scenarios:
            X_train = client.X
            y_train = client.y
            X_test = client.X_test
            y_test = client.y_test
            total_df.append(client.dataFrame)
            total_test.append(client.test)


            name = client.name
            # ------------ Train alone
            ann = client.participantUpdate(coefs=None, intercepts=None, M='auto', regularization=0.000001, epochs=None)
            joblib.dump(ann, filename=name + "" + "train alone")
            acc = tP(X_test, y_test, X_train, y_train, ann, name)
            client_acc.append(acc)

            i = i % 3
            # --------------------------------
        # -------------------- Train alone with all the data
        df = pd.concat(total_df)
        test = pd.concat(total_test)
        X, y = cD(df)
        X_test, y_test = cD(test)
        ann_total = trainANN(X, y, epochs=None, M='auto', coef=None, intercept=None)
        joblib.dump(ann_total, filename=name + "" + "train alone")
        acc = tP(X_test, y_test, X, y, ann_total, "total_test_for_train_alone_with_cd")
        per_scenario_total_acc_alone.append(acc)

        # ------------- OPTIMIZE
        best_epoch, best_batch = optimize(epochs, batch_size, scenarios, rounds, name)
        # ------------- FedAvg
        rounds_fed = rounds * 10
        model, round_accuracy, per_client_accuracy = rFA(epoch=best_epoch, m=best_batch, regularization=0.000001, clients=scenarios, name=name, round=rounds_fed)

        per_scenario_client_acc_fed.append(per_client_accuracy)
        per_scenario_total_acc_fed.append(round_accuracy)
        joblib.dump(model, filename=name + "" + "train with FedAVG")
        # --------- Test FedAvg with all the data
        acc = tP(X_test, y_test, X, y, model, "total_test_Fed_avg_with_cd")
        per_scenario_total_acc_fed_cd.append(acc)

        per_scenario_total_acc_alone.append(acc)
        per_scenario_total_acc.append(client_acc)

    per_trial_total_acc.append(per_scenario_total_acc)

    per_trial_per_round_total_acc_fed.append(per_scenario_total_acc_fed)
    per_trial_per_round_per_client_acc_fed.append(per_scenario_client_acc_fed)

    per_trial_total_acc_alone.append(per_scenario_total_acc_alone)

    per_trial_total_fed_cd.append(per_scenario_total_acc_fed_cd)


average_per_trial_total_acc = np.array(per_trial_total_acc).mean(axis=0).astype(int).tolist()


average_per_trial_per_round_total_acc_fed = np.array(per_trial_total_acc).mean(axis=0).astype(int).tolist()
average_per_trial_per_round_per_client_acc_fed = np.array(per_trial_per_round_per_client_acc_fed).mean(axis=0).astype(int).tolist()

average_per_trial_total_acc_alone = np.array(per_trial_total_acc_alone).mean(axis=0).astype(int).tolist()

average_per_trial_total_fed_cd = np.array(per_trial_total_fed_cd).mean(axis=0).astype(int).tolist()

save("per_trial_accuracy_per_client_alone", per_trial_total_acc)

save("per_trial_per_round_per_client_acc_fed", per_trial_per_round_per_client_acc_fed)
save("per_trial_per_round_total_averaged_acc_fed", per_trial_per_round_total_acc_fed)
save("per_trial_total_fed_cd", per_trial_total_fed_cd)

save("per_trial_accuracy_total_alone", average_per_trial_total_acc_alone)
save("average_per_trial_accuracy_per_client_alone", average_per_trial_total_acc)

save("average_per_trial_per_round_per_client_acc_fed", average_per_trial_per_round_per_client_acc_fed)
save("average_per_trial_per_round_total_averaged_acc_fed", average_per_trial_per_round_total_acc_fed)
save("average_per_trial_total_fed_cd", average_per_trial_total_fed_cd)

save("average_per_trial_accuracy_total_alone", average_per_trial_total_acc_alone)


