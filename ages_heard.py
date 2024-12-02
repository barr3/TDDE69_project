import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv("./data/siren_data_train.csv")
ages_data = data["age"]
data_heard = data.loc[data["heard"] == 1]
data_not_heard = data.loc[data["heard"] == 0]

std_heard = np.std(data_heard["age"])
std_not_heard = np.std(data_not_heard["age"])

avg_age_heard = np.round(np.average(data_heard["age"]))
avg_age_not_heard = np.round(np.average(data_not_heard["age"]))

print(data.head(100))
print(f"Youngest person is: { min(ages_data) } y/o")
print(f"Oldest person is: { max(ages_data) } y/o")
print(f"The average age of people that heard is: { avg_age_heard }")
print(f"The average age of people that did not hear is: { avg_age_not_heard }")
print(f"The std of people that did hear is: { std_heard }")
print(f"The std of people that did not hear is: { std_not_heard }")
print("Finished")

heard_prc = []
label = []

for i in range(18, 88, 10):
    start = i
    end = i + 10

    d_sub = data.loc[((start <= data["age"]) & (data["age"] < end))]
    label.append(f"{start} to {end}")
    heard_prc.append(np.average(d_sub["heard"]))

heard_prc = [round(num, 2) for num in heard_prc]

plt.bar(label, heard_prc)
plt.title('% of people that heard within an age group')
plt.xlabel('Age span')
plt.ylabel('% that heard')
plt.show()
