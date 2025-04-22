import cv2
import os

import numpy as np
import matplotlib.pyplot as plt


def imread(filepath):
    return cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)


def imwrite(filepath, img):
    cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def make_mask(images_path, labels_path, image_filename):
    img = imread(os.path.join(images_path, image_filename))
    h, w, _ = img.shape

    label_filename = image_filename[:-4] + '.txt'
    with open(os.path.join(labels_path, label_filename), 'r') as f:
        lines = f.readlines()

        lines = [list(map(float, l.split()[1:])) for l in lines]

    polygons = []
    for line in lines:
        polygon = []
        for i in range(0, len(line), 2):
            polygon.append((w * line[i], h * line[i + 1]))
        polygon = np.array(polygon).round().astype(np.int32)
        polygons.append(polygon)

    mask = np.zeros_like(img)
    for polygon in polygons:
        cv2.fillPoly(mask, pts=[polygon], color=(255, 255, 255))

    return img, mask


def make_new_dataset(dataset_path, new_dataset_path):
    os.makedirs(new_dataset_path, exist_ok=True)
    assert len(os.listdir(new_dataset_path)) == 0
    os.makedirs(os.path.join(new_dataset_path, 'images'), exist_ok=True)
    os.makedirs(os.path.join(new_dataset_path, 'labels'), exist_ok=True)

    images_path = os.path.join(dataset_path, 'images')
    labels_path = os.path.join(dataset_path, 'labels')

    images_filenames = list(filter(lambda f: f[-4:] == '.jpg', os.listdir(images_path)))
    print(len(images_filenames))

    # labels_filenames = [
    #     f[:-4] + '.txt'
    #     for f in images_filenames
    # ]

    for image_filename in images_filenames:
        img, mask = make_mask(images_path, labels_path, image_filename)
        imwrite(os.path.join(new_dataset_path, 'images', image_filename), img)
        imwrite(os.path.join(new_dataset_path, 'labels', image_filename), mask)


def main():
    working_directory = os.path.abspath(os.path.dirname(__file__))
    dataset_path = os.path.join(
        working_directory,
        '../data/datasets/junior-cell-segmentation-poly'
    )
    new_dataset_path = os.path.join(
        working_directory,
        '../data/datasets/junior-cell-segmentation'
    )

    # dataset_path = '../data/datasets/junior-cell-segmentation-poly'
    # new_dataset_path = dataset_path + '../data/datasets/junior-cell-segmentation'

    make_new_dataset(dataset_path, new_dataset_path)


if __name__ == '__main__':
    main()
