import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

D = pd.read_csv("./data/siren_data_train.csv")

def distance(x, y, x_horn, y_horn):
    return ((x-x_horn)**2 + (y-y_horn)**2)**(1/2)

num_subdivids = 75
subdiv_size = 1000
subset_labels = []
heard_prc = []
numbers = []
number_label = []
for i in range(num_subdivids):
    start = 0 + i * subdiv_size
    end = start+subdiv_size
    subset_labels.append(f"{start/1000} - {end/1000}")
    D_subdiv = D.loc[lambda x: (distance(x.xcoor, x.ycoor, x.near_x, x.near_y) > start) & (distance(x.xcoor, x.ycoor, x.near_x, x.near_y) < end) ]
    numbers.append(len(D_subdiv))
    number_label.append(f"{i} - {i+1}")
    heard_prc.append(np.mean(D_subdiv['heard']) * 100)


mpl.rcParams['font.size'] = 18

# Prints distance plot
print(subset_labels)
print(heard_prc)
plt.bar(subset_labels, heard_prc)
plt.xlabel('Distance to nearest horn (kilometers)')
plt.ylabel('Percentage who heard the horn')
plt.xticks(ticks=range(0, len(subset_labels), 2), labels=subset_labels[::2], rotation=60)
plt.show()


# Removes first 3 datapoints since they are large compared to the others
number_label = number_label[3:]
numbers = numbers[3:]
print(number_label)

# Plots number of people in every interval
plt.plot(number_label, numbers, 'o-')
plt.xlabel("Distance to nearest horn (kilometers)")
plt.ylabel("Number in each distance interval")
plt.xticks(ticks=range(0, len(number_label), 2), labels=number_label[::2], rotation=60)
plt.show()