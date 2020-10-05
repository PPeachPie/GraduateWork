import numpy as np
import scipy.special as sc

class MyNeuroNet:
    def __init__(self, nodes_in, nodes_out, nodes_hidden, learningrate):
        self.inode = nodes_in
        self.onodes = nodes_out
        self.hnodes = nodes_hidden
        self.lrate = learningrate
        self.weight_in = np.random.rand(nodes_hidden, nodes_in)
        self.weight_out = np.random.rand(nodes_out, nodes_hidden)
        self.activation_func = lambda x: sc.expit(x)


