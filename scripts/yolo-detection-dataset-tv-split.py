import os
import yaml     # type: ignore
import shutil
import random

# maybe todos maybe not
#
# todo: files classes.txt and notes.json must be also copied
# todo: think how to parametrize image_is_val(...)
# todo: parametrize dataset path automatically


def image_is_val(filename: str):
    first_val = '2025-01-18-03-25-53-549325.jpg'

    filename = filename[filename.find('-')+1:]
    return filename >= first_val


def split_dataset(datasets_path, dataset_name,
                  train_ratio=0.8, exist_ok=False
                  ):
    dataset_yaml_path = os.path.join(datasets_path, f'{dataset_name}.yaml')
    with open(dataset_yaml_path, 'r') as f:
        config = yaml.safe_load(f)

    dataset_root = os.path.join(os.path.dirname(dataset_yaml_path),
                                config['path'])
    images_dir = os.path.join(dataset_root, 'images')
    labels_dir = os.path.join(dataset_root, 'labels')

    new_dataset_root = dataset_root + '-tv'
    train_images_dir = os.path.join(new_dataset_root, 'train', 'images')
    val_images_dir = os.path.join(new_dataset_root, 'val', 'images')
    train_labels_dir = os.path.join(new_dataset_root, 'train', 'labels')
    val_labels_dir = os.path.join(new_dataset_root, 'val', 'labels')

    os.makedirs(train_images_dir, exist_ok=exist_ok)
    os.makedirs(val_images_dir, exist_ok=exist_ok)
    os.makedirs(train_labels_dir, exist_ok=exist_ok)
    os.makedirs(val_labels_dir, exist_ok=exist_ok)

    images = sorted([
        f for f in os.listdir(images_dir)
        if f.endswith(('.jpg', '.png', '.jpeg'))
    ])
    random.shuffle(images)

    split_idx = int(train_ratio * len(images))
    train_images = images[:split_idx]
    val_images = images[split_idx:]

    # train_images = list(filter(lambda x: not image_is_val(x), images))
    # val_images = list(filter(lambda x: image_is_val(x), images))

    for img in train_images:
        shutil.copy(os.path.join(images_dir, img), train_images_dir)
        label_file = os.path.splitext(img)[0] + '.txt'
        if os.path.exists(os.path.join(labels_dir, label_file)):
            shutil.copy(os.path.join(labels_dir, label_file), train_labels_dir)

    for img in val_images:
        shutil.copy(os.path.join(images_dir, img), val_images_dir)
        label_file = os.path.splitext(img)[0] + '.txt'
        if os.path.exists(os.path.join(labels_dir, label_file)):
            shutil.copy(os.path.join(labels_dir, label_file), val_labels_dir)

    new_config = {
        'path': os.path.basename(new_dataset_root),
        'train': 'train/images',
        'val': 'val/images',
        'names': config.get('names', {})
    }

    new_dataset_yaml_filename = f'{dataset_name}-tv.yaml'
    new_dataset_yaml_filepath = \
        os.path.join(datasets_path, new_dataset_yaml_filename)
    with open(new_dataset_yaml_filepath, 'w') as f:
        yaml.dump(new_config, f, default_flow_style=False, allow_unicode=True)

    print(f'Done. Train: {len(train_images)}, Val: {len(val_images)}')


if __name__ == "__main__":
    working_directory = os.path.abspath(os.path.dirname(__file__))
    datasets_path = os.path.join(working_directory, '..\\data\\datasets')
    # dataset_name = 'field-detection'
    dataset_name = 'junior-cell-segmentation-poly'

    split_dataset(datasets_path, dataset_name)
