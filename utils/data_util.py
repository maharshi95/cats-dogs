import os, concurrent
import numpy as np
from datetime import datetime as dt
from scipy import misc
from concurrent import futures

train_data_path = 'data/train'
test_data_path = 'data/test'

X_train = []

all_files = [os.path.join(test_data_path, filename) for filename in os.listdir(test_data_path)][:4000]

dog_files = [filename for filename in all_files if 'dog' in filename]

cat_files = [filename for filename in all_files if 'cat' in filename]

def get_shape(filepath):
    img = misc.imread(filepath)
    return img.shape

tic = dt.now()

with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
    future_to_url = [executor.submit(get_shape, filename) for filename in all_files]
    shapes = [shape.result() for shape in concurrent.futures.as_completed(future_to_url)]
    mu, sigma, min, max = np.mean(shapes, axis=0), np.std(shapes, axis=0), np.min(shapes, axis=0), np.max(shapes, axis=0)
    print('mu:', mu)
    print('sigma:', sigma)
    print('min:', min)
    print('max:', max)

delta = dt.now() - tic
print(delta)

tic = dt.now()

shapes = list(map(get_shape, all_files))
mu, sigma, min, max = np.mean(shapes, axis=0), np.std(shapes, axis=0), np.min(shapes, axis=0), np.max(shapes, axis=0)
print('mu:', mu)
print('sigma:', sigma)
print('min:', min)
print('max:', max)

delta = dt.now() - tic
print(delta)

