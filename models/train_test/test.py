from keras import Model

def test_model(model:Model, Xs, prediction_threshold)->list:
    print('Num of samples :', len(Xs))
    predictions = model.predict(Xs)
    predictions[predictions >= prediction_threshold] = 1
    predictions[predictions < prediction_threshold] = 0
    return predictions
