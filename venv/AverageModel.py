from joblib import load


def load_model(name):
    return load(name)


models_params = []
for i in range(1, 4):
    name = "Ann" + str(i) + ".sav"
    model = load_model(name)
    params = model.get_params(deep=True)
    models_params.append(params)

models_params