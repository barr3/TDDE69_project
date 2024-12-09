import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.neighbors as skl_nb
import sklearn.model_selection as skl_ms


def distance(x, y, x_horn, y_horn):
    return ((x-x_horn)**2 + (y-y_horn)**2)**(1/2)

D = pd.read_csv("./data/siren_data_train.csv")
y = D['heard']
X = D.drop(columns='heard')


X = X.drop(columns='near_fid')
#X = X.drop(columns='near_x')
#X = X.drop(columns='near_y')
X = X.drop(columns='near_angle')
X = X.drop(columns='building')
#X = X.drop(columns='xcoor')
#X = X.drop(columns='ycoor')
X = X.drop(columns='noise')
X = X.drop(columns='in_vehicle')
X = X.drop(columns='asleep')
#X = X.drop(columns='no_windows')
#X = X.drop(columns='age')

# Adds distance to data set

def add_distance(X):
    X['distance'] = distance(X.xcoor, X.ycoor, X.near_x, X.near_y)
    X = X.drop(columns='near_x')
    X = X.drop(columns='near_y') 
    X = X.drop(columns='xcoor')
    X = X.drop(columns='ycoor')
    return X
    
X = add_distance(X)

print(X.info())


#np.random.seed(1)

#N = len(X)
#M = np.ceil(0.7*N).astype(int) # Number of training data

# Using Scikit learn KFold
n_fold = 10

cv = skl_ms.KFold(n_splits=n_fold, random_state=2, shuffle=True)
K = np.arange(1,50)
misclassification = np.zeros(len(K))
for train_index, val_index in cv.split(X):
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]
    
    for j, k in enumerate(K):
        model = skl_nb.KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train, y_train)
        prediction = model.predict(X_val)
        misclassification[j] += np.mean(prediction != y_val)

misclassification /= n_fold
print(misclassification)
print(min(misclassification))
print(np.argmin(misclassification))
plt.plot(K, misclassification)
plt.title('Cross validation error for kNN')
plt.xlabel('k')
plt.ylabel('Validation error')
plt.show()
## Around 3-5 neighbours seem to be the best k

