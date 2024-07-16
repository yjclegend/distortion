import numpy as np
import matplotlib.pyplot as plt

from simulation.gen_sample import gen_grid

params = [-3.22074475e-01, 9.96646441e-01, 1.92880304e-04, 1.89391227e-03, -7.49772786e-02]


# image_u = gen_grid(1.5, 1000)

xs = np.arange(5496) - 5496/2
ys = np.arange(3672) - 3672/2
x, y = np.meshgrid(xs, ys)
image_i = np.column_stack([x.flatten(), y.flatten()])

image_u = image_i / 5710

r2 = image_u[:, 0]**2 + image_u[:, 1]**2

radial = params[0] * r2 + params[1] * r2**2 + params[2] * r2**3


p1, p2 = params[2], params[3]
x, y = image_u[:, 0], image_u[:, 1]
r2 = x**2 + y**2
tan_x = 2 * p1 * x * y + p2 * (r2 + 2 * x**2)
tan_y = p1 * (r2 + 2 * y**2) + 2 * p1 * x * y

tan = np.sqrt(tan_x**2 + tan_y**2)

tan *= 5710

tan = np.reshape(tan, (3672, 5496, 1))
print(np.mean(tan))
plt.imshow(tan)
plt.colorbar()
plt.show()
