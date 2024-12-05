import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as skl_lm
import sklearn.discriminant_analysis as skl_da
import sklearn.preprocessing as skl_pre
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV


np.random.seed(1) # same seed always
data = pd.read_csv("./data/siren_data_train.csv") # read in training and test data


def process_data(data, train_size):

    data["distance"] = np.sqrt( np.square( data["near_x"] - data["xcoor"] ) + np.square( data["near_y"] - data["ycoor"] ) )
    data = data.drop(columns=["near_x", "near_y", "xcoor", "ycoor"])

    trainI = np.random.choice(data.shape[0], size=train_size, replace=False)
    trainIndex = data.index.isin(trainI)

    train = data.iloc[trainIndex]
    test = data.iloc[~trainIndex]

    x_train = train.drop(columns=["heard"])
    y_train = np.ravel(train[["heard"]])

    x_test = test.drop(columns=["heard"])
    y_test = np.ravel(test[["heard"]])

    # normalize the input data
    scaler = skl_pre.StandardScaler()
    scaler.fit(x_train)
    x_train_scaled = scaler.transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    return x_train_scaled, y_train, x_test_scaled, y_test


def model_tune_shrinkage(model, x, y):
    # cross validation grid search
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    grid = dict()
    grid['shrinkage'] = np.arange(0, 1, 0.01)
    search = GridSearchCV(model, grid, scoring='accuracy', cv=cv, n_jobs=-1)
    res = search.fit(x, y)
    print('Mean accuracy of best shrinkage: %.3f' % res.best_score_)
    print('Shrinkage config: %s' % round(res.best_params_['shrinkage'], 2))
    return round(res.best_params_['shrinkage'], 2)


# process data
train_size = 200
x_train_scaled, y_train, x_test_scaled, y_test = process_data(data=data, train_size=train_size)

# tune shrinkage
model = skl_da.LinearDiscriminantAnalysis(solver="lsqr")
shrinkage = model_tune_shrinkage(model=model, x=x_train_scaled, y=y_train)

# choose model:
model = skl_da.LinearDiscriminantAnalysis(solver="lsqr", shrinkage=shrinkage)

model.fit(x_train_scaled, y_train)
print(f'Model: {model}')

predict_prob = model.predict_proba(x_test_scaled)
print(f'The model classes are: {model.classes_}, where 1 represents heard.\n')

prediction = np.empty(len(x_test_scaled), dtype=object)
naive_prediction = np.ones(len(x_test_scaled))

prediction = np.where(predict_prob[:, 0] >= 0.5, 0, 1)
print(f"First 20 predictions: {prediction[0:20]}")
print(f"First 20 y-values:    {y_test[0:20]}\n")

cross = pd.crosstab(prediction, y_test)
print("Confusion matrix, top is predicted outcomes:")
print(cross,'\n')

x = cross[1][0] / ( cross[1][0] + cross[0][0] )
print(f"Prc of people that were predicted to hear but didn't: {x:.3f}")

print(f"Accuracy: {np.mean(prediction == y_test):.3f}") # Accuracy
print(f"Naive accuracy (i.e. everyone heard): {np.mean(naive_prediction == y_test):.3f}")
