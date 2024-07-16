import math
import numpy as np

def gen_grid(r, num=100, extend=None):
    if extend is not None:
        r *= extend
    xs_i = np.linspace(-r/2, r/2, num)
    x, y = np.meshgrid(xs_i, xs_i)
    image_i = np.column_stack([x.flatten(), y.flatten(), np.ones_like(x.flatten())])
    return image_i

def gen_rect(r, num=100):
    rs = np.linspace(-r/2, r/2, num)
    x = np.concatenate((rs, rs, np.zeros_like(rs) - r/2, np.zeros_like(rs) + r/2)) 
    y = np.concatenate((np.zeros_like(rs) + r/2, np.zeros_like(rs) - r/2, rs, rs))
    points = np.column_stack((x, y, np.ones_like(x)))
    return points

def gen_ring(r, num = 100):
    rad = np.linspace(0, np.pi, num)
    xs = r*np.cos(rad)
    ys = r*np.sin(rad)
    x = np.concatenate((xs, xs))
    y = np.concatenate((ys, -ys))
    points = np.column_stack((x, y, np.ones_like(x)))
    return points

def gen_norm_line(norm_rad, r, c, num=1000, crop=True):
    a = math.cos(norm_rad)
    b = math.sin(norm_rad)
    x0 = -a*c*r
    y0 = -b*c*r
    dx = r/num
    line = [(x0, y0)]
    for i in range(1, num):
        pa = (x0 - b * i * dx, y0 + a * i * dx)
        pb = (x0 + b * i * dx, y0 - a * i * dx)
        line.append(pa)
        line.append(pb)
    line.sort()
    line = np.array(line)
    line = np.column_stack((line, np.ones_like(line[:, 0])))
    if crop:
        import random
        line = line[:-random.randint(200, 800)]
    # line[:, :2] *= w
    return line

if __name__ == "__main__":
    image = gen_ring(r=10, num=10)
    import matplotlib.pyplot as plt
    plt.scatter(image[:, 0], image[:, 1])
    plt.show()