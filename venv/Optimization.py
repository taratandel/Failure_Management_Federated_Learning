from FedAVG import runFedAvg as rFA


def optimize(epochs, batch_size, clients, rounds):
    current_acc = 0
    best_epochs = 0
    best_batch_size = 0
    best_regularization_term = 0
    list_param = []
    for epoch in range(1, epochs, 10):
        for m in list(range(1, batch_size, 10)) + [math.inf]:
            for regularization in [0.1, 0.001, 0.00001, 0.0000001]:
                list_param.append([epoch, m, regularization])

    for i in range(0, len(list_param)):
        epoch = list_param[i][0]
        m = list_param[i][1]
        acc = runFedAvg(epoch, m, clients, rounds)
        if acc > current_acc:
            current_acc = acc
            best_epochs = epoch
            best_batch_size = m

    print("best epoch:" + str(best_epochs) + "\n" +
          "best mini_batch size:" + str(best_batch_size) + "\n" +
          "best regularization term:" + str(best_regularization_term))
