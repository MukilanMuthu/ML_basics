import numpy as np

a = np.arange(0.01, 1.01, 0.01)
x = np.arange(1, 10, 1)
y = x.reshape(3, 3)
b = a.reshape(2, -1, 5)  # -1 to indicate python to calculate the no. or elements required based on given row or column
print(b)
print(y)

c = b.reshape(2, 5, -1)
print(c)
print(c.sum())
print(c.prod())
print(c.mean())
print(c.max(), c.min())
print(c.ptp())  # max - min peak to peak

# flatten
print(c.reshape(c.size))  # or
print(c.flatten())  # creates a copy
print(c.ravel())  # creates a view

# repeat and unique
z = np.repeat(y, 3, axis=0)
print(z)
print(np.unique(z, axis=0))

# transpose: swapping rows and columns
print(np.swapaxes(y, 1, 0))
print(y.transpose(1, 0))
print(y.T)  # means transpose

# simple ops this includes element multiplication
addition = (y.T + y - y.T*y)
print(addition)

# matrix multiplication
print(np.matmul(y, y.T))  # can also be done by y.dot(y.T) or y @ y.T
print(y * y.T)
