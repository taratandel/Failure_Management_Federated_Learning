from dataDivider import cleanData, loadDataFrame, divideRandomly, calcFractions
from clients import *

def clientBuilder():

    test_separated = []
    test = loadDataFrame("test.csv")
    X_test, y_test = cleanData(test)


    test_not_separated = []
    for i in range(1, 4):
        name = "test_separated.csv" + str(i)
        test_separated.append(loadDataFrame(name))
        name = "test_not_separated.csv" + str(i)
        test_not_separated.append(loadDataFrame(name))

    # creates the client with the given data
    clients_tns = []
    clients_ts = []
    # tests_df = divideRandomly(test, calcFractions(test_separated))
    for i in range(0, 3):
        client = Client(data=test_separated[i])
        client.setTest(X_test, y_test)
        clients_ts.append(client)

        client = Client(data=test_not_separated[i], prepare_for_testing=True)

        clients_tns.append(client)

    return clients_ts, clients_tns