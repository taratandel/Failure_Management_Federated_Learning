# from settings import PROJECT_ROOT
from modelTester import testProcess as tP
from clients import *
import joblib
import os
from numpy import save, load
os.chdir(os.path.dirname(__file__))
# -------------- Trial variables
# we need also confusion matrix for it
total_trails = 11
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
for trial in range(total_trails):
    total_scenarios_data = [clientBuilderForScenario1(switcher.get(0, "nothing") + " " + "trial" + " " + str(trial)),
                            clientBuilderForClassesPerEach(
                                switcher.get(1, "nothing") + " " + "trial" + " " + str(trial)),
                            clientBuilderForClassesProportional(switcher.get(2, "nothing") + "trial" + str(trial))]
    per_scenario_total_acc = []
    for scenarios in total_scenarios_data:
        client_acc = []
        for client in scenarios:
            X_train = client.X
            y_train = client.y
            X_test = client.X_test
            y_test = client.y_test
            name = client.name
            # ------------ Train alone
            ann = client.participantUpdate(coefs=None, intercepts=None, M='auto', regularization=0.000001, epochs=None)
            joblib.dump(ann, filename=name)
            acc = tP(X_test, y_test, X_train, y_train, ann, name)
            client_acc.append(acc)
            # --------------------------------

            # ------------- OPTIMIZE


        per_scenario_total_acc.append(client_acc)
    per_trial_total_acc.append(per_scenario_total_acc)

save("per_trial_accuracy_per_client_alone", per_trial_total_acc)
