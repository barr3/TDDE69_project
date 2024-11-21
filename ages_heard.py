import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv("./data/siren_data_train.csv")

heard_prc = []
label = []

for i in range(0, 100, 20):
    start = i
    end = i + 20

    d_sub = data.loc[((start <= data["age"]) & (data["age"] < end))]
    label.append(f"{start} to {end}")
    heard_prc.append(np.average(d_sub["heard"]))

plt.bar(label, heard_prc)
plt.title('% of people that heard within an age group')
plt.xlabel('Age span')
plt.ylabel('% that heard')
plt.show()

