#!/usr/bin/env python
from patched_cnn import *

shadows = open_images("./segments/shadows")
non_shadows = open_images("./segments/non_shadows", len(shadows))

x = [] # input features.
y = [] # labels

x.extend(shadows)
x.extend(non_shadows)

y.extend([ 1 for i in range(len(shadows)) ])
y.extend([ 0 for i in range(len(non_shadows)) ])

cnn = Patched_CNN()
cnn.build_model()

batch_size = 50
epochs = 50
patience = 5
cnn.train(x, y, batch_size, epochs, patience)
cnn.save_model()
