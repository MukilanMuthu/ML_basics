import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

x = np.linspace(0, 5, 11)
y = x ** 2
data = pd.read_csv('histogram.csv')
a = np.array(data["DN"])
b = np.array(data["Frequency"])
plt.plot(a, b)


# Functional way of creating plot
plt.plot(x, y)
plt.xlabel("X axis")
plt.ylabel("Y axis")
plt.title("Basic plot")

plt.plot(a, b)
plt.xlim(8, 244)
plt.bar(x, y, width=1, linewidth=0.7)

# Object-oriented method
fig = plt.figure()  # create a figure, essentially a canvas
axes1 = fig.add_axes([0.1, 0.1, 0.8, 0.8])  # add axes
axes1.plot(x, y)  # plot in the created axis in the figure
axes1.set_xlabel('X axis')
axes1.set_ylabel('Y axis')
axes2 = fig.add_axes([0.15, 0.4, 0.4, 0.4])
axes2.plot(y, x)

plt.show()
