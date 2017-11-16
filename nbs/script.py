import os, multiprocessing, time, random
import numpy as np

from scipy import misc
from concurrent import futures

from datetime import datetime as dt
from matplotlib import pyplot as pp

pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())

train_data_path = '../data/train'
test_data_path = '../data/test'

input_img_size = (256, 256)
sl = 1.2


def img_resize_and_crop(img, output_size=input_img_size):
    img_x, img_y, img_z = img.shape

    crop_x, crop_y = output_size

    img_f, crop_f = img_y / img_x, crop_y / crop_x

    trgt_f = max(min(crop_f, img_f * sl), img_f / sl)

    trgt_x, trgt_y = int(max(crop_x, crop_y / trgt_f)), int(max(crop_y, crop_x * trgt_f))

    img = misc.imresize(img, (trgt_x, trgt_y))

    start_x, end_x = (trgt_x - crop_x) // 2, (trgt_x + crop_x) // 2
    start_y, end_y = (trgt_y - crop_y) // 2, (trgt_y + crop_y) // 2

    return img[start_x:end_x, start_y:end_y, :]


def test_img_resize_and_crop():
    img = images[1343]
    pp.imshow(img)
    pp.show()
    img = img_resize_and_crop(img, input_img_size)
    pp.imshow(img)
    pp.show()

    # test_img_resize_and_crop()


def get_full_path(image_name):
    return os.path.join(train_data_path, image_name)


def get_scaled_image_from_file(filepath):
    img = misc.imread(filepath)
    return img_resize_and_crop(img)


def get_images_from_files(filepaths):
    with futures.ProcessPoolExecutor(max_workers=4) as executor:
        futures_to_images = [executor.submit(get_scaled_image_from_file, filepath) for filepath in filepaths]
        completed_images = futures.as_completed(futures_to_images)
        images = list(map(lambda img: img.result(), completed_images))

    return images


def get_X_Y(dir_path):
    filenames = os.listdir(dir_path)
    filepaths = list(map(get_full_path, filenames))

    n_examples = len(filepaths)

    dog_files = list(filter(lambda f: 'dog' in f, filepaths))
    cat_files = list(filter(lambda f: 'cat' in f, filepaths))

    tic = dt.now()

    dog_images = get_images_from_files(dog_files)
    cat_images = get_images_from_files(cat_files)

    tac = dt.now()

    print("Time to read dogs and cats:", tac - tic)

    X = dog_images + cat_images
    Y = [1] * len(dog_images) + [0] * len(cat_images)

    #     rand_perm = np.arange(n_examples, dtype=np.uint16)
    #     np.random.shuffle(rand_perm)
    #     print(rand_perm, rand_perm.dtype)
    #     X = X[rand_perm]
    #     Y = Y[rand_perm]

    return X, Y


def main():
    train_files = os.listdir(train_data_path)

    test_files = os.listdir(test_data_path)

    all_train_files = list(map(get_full_path, train_files))

    all_test_files = list(map(get_full_path, test_files))

    X, Y = get_X_Y(train_data_path)

if __name__ == '__main__':
    main()