from PIL import Image
import os
from sklearn.feature_extraction.image import extract_patches
import numpy as np
import pdb

patchSize = 128
graph = Image.open('./archer.jpg')
#graph.show()
img = np.asarray(graph, dtype=np.float32)
img = img.transpose(2, 0, 1)
img1 = np.zeros((1, 3, img.shape[1], img.shape[2]))
img1[0, :, :, :] = img
patches = extract_patches(img, (3, patchSize, patchSize), patchSize)
X = patches.reshape((-1, 3, patchSize, patchSize))
pdb.set_trace()