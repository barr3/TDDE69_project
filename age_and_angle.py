import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

D = pd.read_csv("data/siren_data_train.csv")

# Age analysis
D_heard = D.loc[D['heard'] == 1]
D_not_heard = D.loc[D['heard'] == 0]

heard_mean = np.mean(D_heard['age'])
heard_std = np.std(D_heard['age'])

not_heard_mean = np.mean(D_not_heard['age'])
not_heard_std = np.std(D_not_heard['age'])
print(f"Age of those who heard\n  Mean: {heard_mean:.3f}, Standard deviation: {heard_std:.3f}")
print(f"Age of those who didn't hear\n  Mean: {not_heard_mean:.3f}, Standard deviation: {not_heard_std:.3f}")

# Angle analysis
num_subdivs  = 4
subsdiv_size = 360 / num_subdivs
subset_labels = []
heard_prc = []
for i in range(num_subdivs):
    start = -180 + i*subsdiv_size
    end = start + subsdiv_size
    subset_labels.append(f"{start}° to {end}°")
    D_subdiv = D.loc[((D['near_angle'] >= start) & (D['near_angle'] < end))]
    heard_prc.append(np.average(D_subdiv['heard']) * 100)

print(subset_labels)
print(heard_prc)

right = (heard_prc[0] + heard_prc[3]) / 2
left = (heard_prc[1] + heard_prc[2]) / 2
print(right)
print(left)

plt.bar(["left", "right"], [left, right])
plt.xlabel('Direction to the nearest horn')
plt.ylabel('Percentage of subjects who heard the horn')
plt.show()
