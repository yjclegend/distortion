import matplotlib.pyplot as plt
import numpy as np

def violin(data, pos, facecolor, edgecolor, width):
    vio = plt.violinplot(data, pos, showmeans=False, showmedians=False, showextrema=False, widths=width)
    for pc in vio["bodies"]:
        pc.set_facecolor(facecolor)
        pc.set_edgecolor(edgecolor)
        pc.set_alpha(0.5)
        

def box(data, pos, facecolor, edgecolor, width):
    bo = plt.boxplot(data, positions=pos, showfliers=False,widths=width,
                    boxprops=dict(color=edgecolor),
                    capprops=dict(color=edgecolor),
                    whiskerprops=dict(color=edgecolor),
                    medianprops=dict(color='red'))
    

class MatPlotBase:
    def __init__(self, xlabel, ylabel):
        self.fig = plt.figure()
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

class LinePlot(MatPlotBase):
    def __init__(self, xlabel, ylabel, discrete=False):
        super().__init__(xlabel, ylabel)
    
    def plot_line(self, x, y, label):
        plt.plot(x, y, label=label)

    def plot_lines(self, x, ys, labels):
        for y in ys:
            self.plot_line(x, y)
