import os
import shutil
import random


def split_dataset(datasets_path, dataset_name,
                  train_ratio=0.8, exist_ok=False
                  ):
    dst_name = f'{dataset_name}-tv'
    srcpath = os.path.join(datasets_path, dataset_name)
    dstpath = os.path.join(datasets_path, dst_name)
    assert os.path.isdir(srcpath)
    assert not os.path.exists(dstpath)

    random.seed(42)
    classes = os.listdir(srcpath)

    for cls in classes:
        os.makedirs(os.path.join(dstpath, 'train', cls))
        os.makedirs(os.path.join(dstpath, 'val', cls))

        filenames = os.listdir(os.path.join(srcpath, cls))
        random.shuffle(filenames)

        split_index = round(train_ratio * len(filenames))
        train_filenames = filenames[:split_index]
        val_filenames = filenames[split_index:]

        for filename in train_filenames:
            shutil.copy(
                os.path.join(srcpath, cls, filename),
                os.path.join(dstpath, 'train', cls, filename)
            )
        for filename in val_filenames:
            shutil.copy(
                os.path.join(srcpath, cls, filename),
                os.path.join(dstpath, 'val', cls, filename)
            )


if __name__ == "__main__":
    working_directory = os.path.abspath(os.path.dirname(__file__))
    datasets_path = os.path.join(working_directory, '../data/datasets')

    dataset_name = 'field-classification'

    split_dataset(datasets_path, dataset_name)
