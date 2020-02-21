import os
import json
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D

from Doc2Vec.lib.train import train


class visualizeDatasets():
    def __init__(self):
        self.names, self.vectorized_encodings = train()

    def viz2D(self):
        tsne_2d = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, random_state=32)
        embeddings_2d = tsne_2d.fit_transform(self.vectorized_encodings)
        self.tsne_plot_2d('Dataset Correspondance', embeddings_2d, words=self.names, a=0.1)

    def viz3D(self):
        tsne_3d = TSNE(perplexity=30, n_components=3, init='pca', n_iter=5000, random_state=12)
        embeddings_3d = tsne_3d.fit_transform(self.vectorized_encodings)
        self.tsne_plot_3d('Dataset Embeddings using t-SNE', 'Datasets', embeddings_3d, a=0.1)

    def tsne_plot_2d(self, label, embeddings, words=[], a=1):
        plt.figure(figsize=(16, 9))
        colors = cm.rainbow(np.linspace(0, 1, 1))
        x = embeddings[:,0] / 100
        y = embeddings[:,1] / 100
        plt.scatter(x, y, c=colors, alpha=a, label=label)
        for i, word in enumerate(words):
            plt.annotate(word, alpha=0.3, xy=(x[i], y[i]), xytext=(5, 2),
                         textcoords='offset points', ha='right', va='bottom', size=10)
        plt.legend(loc=4)
        plt.grid(True)
        plt.savefig("hhh.png", format='png', dpi=150, bbox_inches='tight')
        plt.show()

    def tsne_plot_3d(self, title, label, embeddings, a=1):
        fig = plt.figure()
        ax = Axes3D(fig)
        colors = cm.rainbow(np.linspace(0, 1, 1))
        x = embeddings[:,0] / 100
        y = embeddings[:,1] / 100
        z = embeddings[:,2] / 100
        plt.scatter(x, y, z, c=colors, alpha=a, label=label)
        plt.legend(loc=4)
        plt.title(title)
        plt.show()

if __name__ == '__main__':
    tester = visualizeDatasets()
    tester.viz2D()
    tester.viz3D()
