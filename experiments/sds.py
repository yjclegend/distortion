import numpy as np
import matplotlib.pyplot as plt
x = np.linspace(1, 10, 1000)
k = 1/300
y = x/(1+ k*x**2)
z = x*(1- k*x**2)
plt.plot(x, y)
plt.plot(x, z)
plt.show()