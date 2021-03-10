from joblib import load, dump
from copy import deepcopy as dc

models_params = []
for i in range(1, 4):
    name = "Ann" + str(i) + ".sav"
    model = load(name)
    # params = model.get_params(deep=True)
    models_params.append(model)
intercept = []
weight = []
total = 593 + 1877 + 43
for i in range(len(models_params)):
    param = models_params[i]
    if i == 0:
        intercept = param.intercepts_
        intercept[:] = [x * 1877 for x in intercept]

        weight = param.coefs_
        weight[:] = [x * 1877 for x in weight]

    elif i == 1:
        model2_interc = [x * 43 for x in param.intercepts_]
        model2_coef = [x * 43 for x in param.coefs_]
        intercept = [sum(x) for x in zip(model2_interc, intercept)]
        weight = [sum(x) for x in zip(model2_coef, weight)]
    else:
        model2_interc = [x * 593 for x in param.intercepts_]
        model2_coef = [x * 593 for x in param.coefs_]
        intercept = [sum(x) for x in zip(model2_interc, intercept)]
        weight = [sum(x) for x in zip(model2_coef, weight)]

intercept[:] = [x / total for x in intercept]
weight[:] = [x / total for x in weight]

final_model = dc(models_params[2])
final_model.coefs_ = weight
final_model.intercepts_ = intercept
dump(final_model, 'finalmodelWeightedAverage.sav')
print(final_model)