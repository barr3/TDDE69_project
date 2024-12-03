import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as skl_lm
import sklearn.discriminant_analysis as skl_da

np.random.seed(1) # same seed always

data = pd.read_csv("./data/siren_data_train.csv") # read in training and test data
#test = pd.read_csv("data/siren_data_test_without_labels.csv")

train_size = 2400
trainI = np.random.choice(data.shape[0], size=train_size, replace=False)
trainIndex = data.index.isin(trainI)

train = data.iloc[trainIndex]  # training set
test = data.iloc[~trainIndex] # test set

# take out inputs and outputs
x_train = train.drop(columns="heard")
y_train = np.ravel(train[["heard"]])

x_test = test.drop(columns="heard")
y_test = np.ravel(test[["heard"]])

# choose qda or lda model:
# model = skl_da.LinearDiscriminantAnalysis()
model = skl_da.QuadraticDiscriminantAnalysis()

model.fit(x_train, y_train)
print(f'Model: {model}')

predict_prob = model.predict_proba(x_test)
print(f'The model classes are: {model.classes_}, where 1 represents heard.\n')

prediction = np.empty(len(x_test), dtype=object)
naive_prediction = np.ones(len(x_test))

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

"""
I guess below is the whole actual code, everything else is just to see that it works somewhat:
One thing to note is that QDA becomes better with larger test size, while LDA is fine with small size.

# read in data
data = pd.read_csv("./data/siren_data_train.csv")
x_train = data.drop(columns="heard")
y_train = data[["heard"]]
y_train = np.ravel(y_train)

# train it
model = skl_da.LinearDiscriminantAnalysis()
model.fit(x_train, y_train)
"""