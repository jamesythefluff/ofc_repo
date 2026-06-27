import matplotlib.pyplot as plt

class Visualiser:
    def __init__(self, im):
        self.im = im

    def update(self, F):
        self.im.set_data(F)
        plt.pause(0.0001)