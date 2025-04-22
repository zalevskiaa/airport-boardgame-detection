# import sys
import time

import cv2
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QApplication, QLabel, QHBoxLayout, QWidget
from PyQt5.QtGui import QImage, QPixmap

import settings

from image_processor import ImageProcessor


class VideoCapture(QThread):
    frame_captured = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self.cap = cv2.VideoCapture(settings.CAMERA_SOURCE)

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.frame_captured.emit(frame)
            time.sleep(0.03)


class VideoProcessor(QThread):
    frame_processed = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self.video_capture = VideoCapture()
        self.video_capture.frame_captured.connect(self.set_frame)
        self.video_capture.start()

        self.frame = None

        self.image_processor = ImageProcessor()

    def run(self):
        while True:
            frame = self.frame
            if frame is not None:
                results = self.image_processor.process(frame.copy())
                self.frame_processed.emit(results)
                time.sleep(0.05)
            time.sleep(0.01)

    def set_frame(self, frame):
        self.frame = frame


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.start_video_processing()

    def init_ui(self):
        self.frame_label = QLabel(self)
        self.frame_field_label = QLabel(self)
        self.frame_field2_label = QLabel(self)
        self.frame_field_ans_label = QLabel(self)

        self.text_label = QLabel("")
        layout = QHBoxLayout()
        layout.addWidget(self.frame_label)
        layout.addWidget(self.frame_field_label)
        layout.addWidget(self.frame_field2_label)
        layout.addWidget(self.frame_field_ans_label)

        layout.addWidget(self.text_label)

        self.setLayout(layout)

    def start_video_processing(self):
        self.thread = VideoProcessor()
        self.thread.frame_processed.connect(self.update_frame)
        self.thread.start()

    @staticmethod
    def _img_resize_by_larger(np_img, h_max=400, w_max=400):
        h, w, c = np_img.shape
        h_scale, w_scale = h_max / h, w_max / w
        scale = min(h_scale, w_scale)
        h, w = round(scale * h), round(scale * w)
        np_img = cv2.resize(np_img, (w, h))
        return np_img

    @staticmethod
    def _img_resize_to_square(np_img, size):
        return cv2.resize(np_img, (size, size))

    @staticmethod
    def _img_to_pixmap(np_img):
        if len(np_img.shape) == 2:
            np_img = np_img.reshape(*np_img.shape, 1)

        h, w, c = np_img.shape

        bytes_per_line = c * w
        qimage = QImage(np_img.data, w, h, bytes_per_line,
                        QImage.Format_RGB888)

        qpixmap = QPixmap.fromImage(qimage)
        return qpixmap

    @staticmethod
    def str_mat(mat):
        if mat is None:
            return 'None'
        return '\n'.join([' '.join(map(str, row)) for row in mat])

    def update_frame(self, results):
        img = results['img']
        self.frame_label.setPixmap(self._img_to_pixmap(
            self._img_resize_by_larger(img, h_max=300)
        ))

        if results['field'] is not None:
            results_field = results['field']
            field_img = results_field['img']
            field_img2 = results_field['img2']
            if field_img2 is None:
                field_img2 = field_img.copy()

            self.frame_field_label.setPixmap(self._img_to_pixmap(
                self._img_resize_to_square(field_img, 300)
            ))
            self.frame_field2_label.setPixmap(self._img_to_pixmap(
                self._img_resize_to_square(field_img2, 300)
            ))

            field_ans_img = results_field['ans_img']
            self.frame_field_ans_label.setPixmap(self._img_to_pixmap(
                self._img_resize_to_square(field_ans_img, 300)
            ))

            if results_field['results'] is not None:
                self.text_label.setText(
                    self.str_mat(results_field['results']['matrix']) + '\n' +
                    self.str_mat(results_field['solution'])
                )
            else:
                self.text_label.setText(results_field['class'])


app = QApplication([])
window = MainWindow()
window.show()
app.exec_()
