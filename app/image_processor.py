# from pathlib import Path

import cv2
import numpy as np

from ultralytics import YOLO

from settings import PROJECT_ROOT

from ml.starter_cell_classiffication.starter_cell_classifier \
    import MobileNetV3Classifier as StarterMobileNetV3Classifier

from ml.field_classification.field_classifier \
    import MobileNetV3Classifier as FieldMobileNetV3Classifier


from solution_finder.solution_finder import SolutionFinder


class StarterCellClassifier:
    def __init__(self):
        pt_path = (PROJECT_ROOT /
                   'ml/starter_cell_classiffication' /
                   'runs_players_take/07/best.pt')
        self.classes = ['down', 'left', 'right', 'sky', 'up']
        self.model = StarterMobileNetV3Classifier(pt_path, self.classes)

    def predict(self, np_images: list):
        return self.model.predict(np_images)


class StarterFieldProcessor:
    def __init__(self):
        self.cell_classifier = StarterCellClassifier()

    @staticmethod
    def _carve_cells(img_field):
        h, w, _ = img_field.shape
        nrows, ncols = 4, 4
        hstep, wstep = h / nrows, w / ncols

        imgs_cells = []
        for i in range(4):
            for j in range(4):
                x1 = round(j * wstep)
                x2 = round((j + 1) * wstep)
                y1 = round(i * hstep)
                y2 = round((i + 1) * hstep)

                img_cell = img_field[y1:y2, x1:x2]
                imgs_cells.append(img_cell)
        return imgs_cells

    def process(self, img_field):
        imgs_cells = self._carve_cells(img_field)

        pred_labels = self.cell_classifier.predict(imgs_cells)

        label_index = 0
        matrix = []
        mapping = {k: v for k, v in zip(
            self.cell_classifier.classes, ['d', 'l', 'r', '.', 'u'])
        }
        for i in range(4):
            row = []
            for j in range(4):
                elem = pred_labels[label_index]
                label_index += 1
                row.append(mapping[elem])
            matrix.append(row)

        field_data = {
            'level': 'starter',
            'matrix': matrix,
            'roads': None,

            'img2': None,
        }
        return field_data


class JuniorCellSegmentation:
    def __init__(self):
        pt_path = (PROJECT_ROOT /
                   'ml/junior_cell_segmentation' /
                   'runs/segment/train2/weights/best.pt')
        self.model = YOLO(pt_path)

    @staticmethod
    def _make_result_mask(yolo_seg_result):
        mask_sum = np.zeros((640, 640, 1))

        for o in range(len(yolo_seg_result)):
            for i in range(len(yolo_seg_result[o].masks)):
                mask = yolo_seg_result[o].masks[i].data
                mask = mask.detach().cpu().numpy().transpose((1, 2, 0))

                mask_sum = np.maximum(mask_sum, mask)   # \
                # if mask_sum is not None else mask

        return mask_sum

    def predict(
            self,
            np_images: list[np.ndarray],
            ignore_empty=False
            ) -> list[np.ndarray]:

        results = self.model.predict(np_images, verbose=False)

        masks = []
        for result, orig_img in zip(results, np_images):
            if ignore_empty and len(result) == 0:
                # there are roads in cell found
                mask = None
            else:
                mask = self._make_result_mask(result)

            # assert mask.shape[2] == 1
            # mask = np.concat((mask, mask, mask), axis=-1)
            # h, w, _ = orig_img.shape
            # mask = cv2.resize(mask, (w, h))
            # assert mask.shape == orig_img.shape

            masks.append(mask)

        return masks


class JuniorFieldProcessor:
    def __init__(self):
        self.segmentation = JuniorCellSegmentation()

    @staticmethod
    def _compute_imgs_cells(img_field):
        h, w, _ = img_field.shape
        hs, ws = h / 4, w / 4

        imgs_cells = []
        for i in range(16):
            r, c = i // 4, i % 4
            img_cell = img_field[round(r * hs):round((r + 1) * hs),
                                 round(c * ws):round((c + 1) * ws)]
            imgs_cells.append(img_cell)
        return imgs_cells

    @staticmethod
    def _mask_to_directions(mask: np.ndarray):
        mask = mask.reshape(*mask.shape[:2], 1)
        h, w, _ = mask.shape

        centers = {
            'r': (w, h / 2),
            'd': (w / 2, h),
            'l': (0, h / 2),
            'u': (w / 2, 0),
        }
        scores = {d: 0 for d in centers.keys()}

        def dist2(x1, y1, x2, y2):
            return (x1 - x2) ** 2 + (y1 - y2) ** 2

        for i in range(h):
            for j in range(w):
                pix = mask[i, j, 0] / 255
                dists = [
                    (dist2(v[0] / w, v[1] / h, j / w, i / h), k)
                    for k, v in centers.items()
                ]
                min_dist, min_center = min(dists)
                scores[min_center] += pix / (1 + min_dist)
        scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        scores = scores[:2]
        return ''.join(sorted([x[0] for x in scores], key='udlr'.find))

    @staticmethod
    def _mask_to_cellclass(mask, directions=None) -> tuple:
        if np.max(mask) < 0.01:
            return 'sky', None

        if directions is None:
            directions = JuniorFieldProcessor._mask_to_directions(mask)

        if set(directions) not in [set('ud'), set('lr')]:
            return 'road', directions

        mid_mask = np.zeros((32, 32))
        m, s = 16, 2
        mid_mask[m-s:m+s+1, m-s:m+s+1] = 1

        score = float((mid_mask * mask).sum() / mid_mask.sum())
        if score < 0.05:
            return 'plane', directions
        return 'road', directions

    @staticmethod
    def _compute_matrices(masks):
        assert len(masks) == 16
        matrix_cls = [[None] * 4 for _ in range(4)]
        matrix_dir = [[None] * 4 for _ in range(4)]

        for i in range(4):
            for j in range(4):
                index = 4 * i + j
                mask = masks[index]
                cellclass, direction = \
                    JuniorFieldProcessor._mask_to_cellclass(mask)

                if direction is None:
                    direction = '..'

                matrix_cls[i][j] = cellclass
                matrix_dir[i][j] = direction

        return matrix_cls, matrix_dir

    @staticmethod
    def _matrix_dir_to_matrix_planes_dir(matrix_dir):
        """
        example input:

        [['ud', '..', 'dr', 'lr'],
        ['ud', '..', 'ur', 'lr'],
        ['ur', 'lr', 'dl', 'dr'],
        ['lr', 'dl', 'ud', 'ud']]

        example output [0]:
            (contains a possible combination of planes directions
            in every straight road segment)

        [['u', ' ', 'd', 'l'],
        ['u', ' ', 'r', 'r'],
        ['u', 'l', 'l', 'd'],
        ['l', 'l', 'u', 'd']]

        example output [1]:
            (contains lists of same-road road segments, unordered)

        [[(0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (3, 2)],
            [(0, 2), (1, 2), (0, 3), (1, 3)],
            [(2, 3), (3, 3)],
            [(3, 0), (3, 1)]]
        """
        moves = [('u', 'd', -1, 0),
                 ('d', 'u', 1, 0),
                 ('l', 'r', 0, -1),
                 ('r', 'l', 0, 1)]
        visited = [[False] * 4 for _ in range(4)]
        matrix_dir_planes = [[' '] * 4 for _ in range(4)]
        roads = []

        for start_i in range(4):
            for start_j in range(4):
                if set(matrix_dir[start_i][start_j]) & set('udlr') and \
                        not visited[start_i][start_j]:

                    visited[start_i][start_j] = True
                    matrix_dir_planes[start_i][start_j] = \
                        matrix_dir[start_i][start_j][0]

                    queue = [(start_i, start_j)]
                    road = []

                    while queue:
                        cur_i, cur_j = queue.pop(0)
                        road.append((cur_i, cur_j))

                        for cur_d, nxt_d, di, dj in moves:
                            nxt_i, nxt_j = cur_i + di, cur_j + dj
                            if 0 <= nxt_i < 4 and 0 <= nxt_j < 4 and \
                                    cur_d in matrix_dir[cur_i][cur_j] and \
                                    nxt_d in matrix_dir[nxt_i][nxt_j] and \
                                    not visited[nxt_i][nxt_j]:
                                visited[nxt_i][nxt_j] = True

                                if matrix_dir_planes[cur_i][cur_j] == cur_d:
                                    matrix_dir_planes[nxt_i][nxt_j] = (
                                        set(matrix_dir[nxt_i][nxt_j])
                                        .difference([nxt_d]).pop()
                                    )
                                else:
                                    matrix_dir_planes[nxt_i][nxt_j] = nxt_d

                                queue.append((nxt_i, nxt_j))

                    roads.append(road)
        return matrix_dir_planes, roads

    def process(self, img_field):
        imgs_cells = self._compute_imgs_cells(img_field)

        masks = self.segmentation.predict(imgs_cells, ignore_empty=False)
        masks = [cv2.resize(mask, (32, 32)) for mask in masks]

        masks0 = np.hstack(masks[0:4])
        masks1 = np.hstack(masks[4:8])
        masks2 = np.hstack(masks[8:12])
        masks3 = np.hstack(masks[12:16])
        masks16 = np.vstack((masks0, masks1, masks2, masks3))
        roads_mask = np.stack((masks16, masks16, masks16), axis=-1)
        roads_mask = (roads_mask * 255).round().astype(np.uint8)
        assert roads_mask.shape == (128, 128, 3)

        matrix_cls, matrix_dir = self._compute_matrices(masks)

        matrix_planes_dir, roads_all = self._matrix_dir_to_matrix_planes_dir(matrix_dir)

        # a possible direction for a plane (example: 'r')
        # a '.' for not a plane
        matrix = [['.'] * 4 for _ in range(4)]
        for i in range(4):
            for j in range(4):
                if matrix_cls[i][j] == 'plane':
                    matrix[i][j] = matrix_planes_dir[i][j]

        # leave only (i, j) where there is a plane
        roads = [
            [(i, j) for i, j in road if matrix_cls[i][j] == 'plane']
            for road in roads_all
        ]

        field_data = {
            'level': 'junior',
            'matrix': matrix,
            'roads': roads,
            'masks': masks,
            'img2': roads_mask,
            'matrix_cls': matrix_cls,
            'matrix_dir': matrix_dir,
        }
        return field_data


class FieldDetector:
    def __init__(self):
        self.model = YOLO(
            PROJECT_ROOT /
            'ml/field-detection/runs/detect/train4/weights/last.pt'
        )

    def predict(self, img):
        result = {'xyxy': None}

        # one [0] for the first image from batch
        results = self.model.predict(img, verbose=False)[0]

        # boxes of the image
        boxes_xyxy = results.boxes.xyxy.detach().cpu().numpy()

        result['boxes_xyxy'] = boxes_xyxy
        return result


class FieldClassifier:
    def __init__(self):
        pt_path = (PROJECT_ROOT /
                   'ml/field_classification' /
                   'runs/05/best.pt')
        self.classes = ['expert', 'junior', 'master', 'starter']
        self.model = FieldMobileNetV3Classifier(pt_path, self.classes)

    def predict(self, np_image):
        np_images = [np_image]
        preds = self.model.predict(np_images)
        return preds[0]


class ImageProcessor:
    def __init__(self):
        self.field_detector = FieldDetector()
        self.field_classifier = FieldClassifier()
        self.field_processors = {
            'starter': StarterFieldProcessor(),
            'junior': JuniorFieldProcessor(),
        }

        self.solution_finder = SolutionFinder()

    @staticmethod
    def _draw_answer_field_img(img_field, solution):
        if solution is None:
            return img_field.copy()

        color_mapping = {
            'owo': (255, 200, 0),
            'gwg': (0, 255, 0),
            'bwb': (0, 50, 200),
            'rwr': (150, 0, 0),
            'wrw': (255, 100, 100),
            'wbw': (0, 255, 255),
        }
        h, w, _ = img_field.shape
        hstep, wstep = h / 4, w / 4

        img_field_rects = img_field.copy()
        for i in range(4):
            for j in range(4):
                x1, y1 = j * wstep, i * hstep
                x2, y2 = x1 + hstep, y1 + wstep
                x1, y1, x2, y2 = map(round, (x1, y1, x2, y2))

                piece_color = solution[i][j]
                color = color_mapping[piece_color]
                color = (*color, 0.3)

                img_field_rects = cv2.rectangle(
                    img_field_rects, (x1, y1), (x2, y2),
                    color, thickness=-1
                )

        img_field_gray = img_field.mean(2).round().astype(np.uint8)
        img_field_gray = np.stack(
            (img_field_gray, img_field_gray, img_field_gray), -1
        )
        assert img_field_gray.shape == img_field.shape, \
            f'{img_field_gray.shape}'
        assert img_field_gray.dtype == img_field.dtype, \
            f'{img_field.dtype}'

        img_field_rects = cv2.addWeighted(
            img_field_gray, 0.7, img_field_rects, 0.3, 0
        )
        return img_field_rects

    @staticmethod
    def _draw_field_bbox(img, field_xyxy):
        x1, y1, x2, y2 = field_xyxy
        img = cv2.rectangle(img.copy(), (x1, y1), (x2, y2), (0, 255, 0), 2)
        return img

    def process(self, img_field):
        # detect field
        result = self.field_detector.predict(img_field)
        boxes_xyxy = result['boxes_xyxy']

        if boxes_xyxy.shape[0] > 0:
            xyxy = boxes_xyxy[0]
            x1, y1, x2, y2 = xyxy
            x1, y1, x2, y2 = map(round, (x1, y1, x2, y2))
            field_xyxy = (x1, y1, x2, y2)

            img_with_bbox = self._draw_field_bbox(img_field, field_xyxy)

            # carve field
            img_field = img_field[y1:y2, x1:x2]

            # predict field_class with FieldClassifier
            field_class = self.field_classifier.predict(img_field)

            if field_class not in self.field_processors:
                return {
                    'img': img_with_bbox,
                    'field': {
                        'xyxy': field_xyxy,
                        'img': img_field,
                        'img2': img_field,
                        'ans_img': img_field,
                        'class': field_class,
                        'results': None,
                        'solution': None,
                    },
                }

            field_processor = self.field_processors[field_class]
            field_results = field_processor.process(img_field)
            img_field2 = field_results['img2']
            solution = self.solution_finder.find_solution(
                field_results
            )
            img_field_ans = self._draw_answer_field_img(
                img_field.copy(), solution
            )

            return {
                'img': img_with_bbox,
                'field': {
                    'xyxy': field_xyxy,
                    'img': img_field,
                    'img2': img_field2,
                    'ans_img': img_field_ans,
                    'class': field_class,
                    'results': field_results,
                    'solution': solution,
                },
            }

        return {
            'img': img_field,
            'field': None,
        }
