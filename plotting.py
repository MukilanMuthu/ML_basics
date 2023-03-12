import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

x = np.linspace(0, 5, 11)
y = x ** 2
data = pd.read_csv('histogram.csv')
a = np.array(data["DN"])
b = np.array(data["Frequency"])

data2 = pd.read_csv("histogram2.csv", header=None)
histo = np.array(data2).reshape(1, -1)
print(histo[0])

fig1, ax1 = plt.subplots()
ax1.hist(histo[0], bins=(max(histo[0])-min(histo[0])))

# Functional way of creating plot
plt.plot(x, y)
plt.xlabel("X axis")
plt.ylabel("Y axis")
plt.title("Basic plot")


# Object-oriented method
fig = plt.figure()  # create a figure, essentially a canvas
axes1 = fig.add_axes([0.1, 0.1, 0.8, 0.8])  # add axes
axes1.plot(x, y)  # plot in the created axis in the figure
axes1.set_xlabel('X axis')
axes1.set_ylabel('Y axis')
axes2 = fig.add_axes([0.15, 0.4, 0.4, 0.4])
axes2.plot(y, x)

fig, axes = plt.subplots()  # when calling multiple subplots, axes becomes an array with multiple axes
# index the axes and plot them individually
axes.plot(x, y, label="Normal", marker="o")
axes.plot(y, x, label="Inverted")
axes.set_xlabel('X')
axes.legend()

plt.tight_layout()
plt.show()
