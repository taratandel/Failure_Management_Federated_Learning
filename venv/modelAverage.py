from joblib import load, dump
from copy import deepcopy as dc


def weightedAverageModel(received_intercept, received_coef, no_samples):
    total = sum(no_samples)
    for i in range(len(no_samples)):
        ns = no_samples[i]
        intc = received_intercept[i]
        coef = received_coef[i]
        if i == 0:
            intercept = intc
            intercept[:] = [x * ns for x in intercept]

            weight = coef
            weight[:] = [x * ns for x in weight]

        else:
            model2_interc = [x * ns for x in intc]
            model2_coef = [x * ns for x in coef]
            intercept = [sum(x) for x in zip(model2_interc, intercept)]
            weight = [sum(x) for x in zip(model2_coef, weight)]

    intercept[:] = [x / total for x in intercept]
    weight[:] = [x / total for x in weight]

    return intercept, weight


def iterAverageModel():
    models_params = models_param
    total = 0
    intercept = []
    weight = []
    for param in models_params:
        no_samples = param.t_
        total = total + no_samples

        model_intercept = param.intercepts_
        model_intercept[:] = [x * no_samples for x in model_intercept]

        model_weight = param.coefs_
        model_weight[:] = [x * no_samples for x in model_weight]
        if intercept:
            intercept = [sum(x) for x in zip(model_intercept, intercept)]
            weight = [sum(x) for x in zip(model_weight, weight)]
        else:
            intercept = model_intercept
            weight = model_weight

    intercept[:] = [x / total for x in intercept]
    weight[:] = [x / total for x in weight]

    final_model = dc(models_params[2])
    final_model.coefs_ = weight
    final_model.intercepts_ = intercept
    dump(final_model, 'finalmodelno_SampleAverage.sav')
    print(final_model.coefs_, final_model.intercepts_)


def onlyAverage():
    models_params = models_param
    intercept = []
    weight = []
    for param in models_params:

        model_intercept = param.intercepts_
        model_weight = param.coefs_

        if intercept:
            intercept = [sum(x) for x in zip(model_intercept, intercept)]
            weight = [sum(x) for x in zip(model_weight, weight)]
        else:
            intercept = model_intercept
            weight = model_weight

    intercept[:] = [x / 3 for x in intercept]
    weight[:] = [x / 3 for x in weight]

    final_model = dc(models_params[2])
    final_model.coefs_ = weight
    final_model.intercepts_ = intercept
    dump(final_model, 'finalmodelOnlyAverage.sav')
    print(final_model.coefs_, final_model.intercepts_)
