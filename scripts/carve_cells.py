import os
from ultralytics import YOLO    # type: ignore
import cv2


working_directory = os.path.abspath(os.path.dirname(__file__))
model_path = os.path.join(
    working_directory,
    '../ml/field-detection/runs/detect/train4/weights/last.pt'
)
input_images_dirpath = os.path.join(
    working_directory,
    '../data/raw_images'
)
output_images_dirpath = os.path.join(
    working_directory,
    '../data/carved_cells'
)


def imread(filepath):
    return cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)


def imwrite(filepath, img):
    cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def detect_field(model, img):
    res = model.predict(img)[0].cpu()
    if res.boxes:
        index = int(res.boxes.conf.argmax())
        x1, y1, x2, y2 = res.boxes.xyxy[index].round().int()
        return x1, y1, x2, y2
    return None


def carve_cells(model_path, input_images_dirpath, output_images_dirpath):
    os.makedirs(output_images_dirpath, exist_ok=True)

    model = YOLO(model_path)

    images_filenames = sorted([
        f for f in os.listdir(input_images_dirpath)
        if f.endswith(('.jpg', '.png'))
    ])

    for img_filename in images_filenames:
        img_filepath = os.path.join(input_images_dirpath, img_filename)
        img = imread(img_filepath)

        field = detect_field(model, img)
        if not field:
            continue
        x1, y1, x2, y2 = field
        img_field = img[y1:y2, x1:x2]

        h, w, _ = img_field.shape
        nrows, ncols = 4, 4
        hstep, wstep = h / nrows, w / ncols

        for i in range(4):
            for j in range(4):
                x1 = round(j * wstep)
                x2 = round((j + 1) * wstep)
                y1 = round(i * hstep)
                y2 = round((i + 1) * hstep)

                img_cell = img_field[y1:y2, x1:x2]
                number = i * ncols + j
                img_cell_filename = f'{img_filename[:-4]}-cell-{number:02d}{img_filename[-4:]}'
                img_cell_filepath = \
                    os.path.join(output_images_dirpath, img_cell_filename)
                imwrite(img_cell_filepath, img_cell)


def main():
    carve_cells(model_path, input_images_dirpath, output_images_dirpath)


if __name__ == '__main__':
    main()
