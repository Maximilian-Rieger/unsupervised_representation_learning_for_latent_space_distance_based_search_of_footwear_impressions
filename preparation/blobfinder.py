from PyQt5 import QtGui
from PyQt5.QtWidgets import QLabel, QMainWindow, QApplication, QWidget, QVBoxLayout
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import sys
from preparation.siftCorrespondance import mainBlob

from preparation.qrangeslider import QRangeSlider
boot = '/home/rhinigtassalvex/Desktop/Windows Documents/Uni/Bachelorarbeit/datasets/impress schuhe+spezial/001/001_01_2.jpg'


class App(QMainWindow):
    def __init__(self):
        super().__init__()

        self.params = dict()
        self.continuous = False

        self.setWindowTitle("BlobFinder")

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        lay = QVBoxLayout(self.central_widget)

        self.teshhold_slider = QRangeSlider()
        self.teshhold_slider.setMin(0)
        self.teshhold_slider.setMax(255)
        self.teshhold_slider.setRange(10, 100)
        self.teshhold_slider.setObjectName('treshhold')
        self.teshhold_slider.startValueChanged.connect(self.slider_changed)
        self.teshhold_slider.endValueChanged.connect(self.slider_changed)

        self.area_slider = QRangeSlider()
        self.area_slider.setMin(0)
        self.area_slider.setMax(500 * 500)
        self.area_slider.setRange(10, 25000)
        self.area_slider.setObjectName('area')
        self.area_slider.startValueChanged.connect(self.slider_changed)
        self.area_slider.endValueChanged.connect(self.slider_changed)

        self.inertia_slider = QRangeSlider()
        self.inertia_slider.setMin(0)
        self.inertia_slider.setMax(100)
        self.inertia_slider.setRange(10, 100)
        self.inertia_slider.setObjectName('inertia')
        self.inertia_slider.startValueChanged.connect(self.slider_changed)
        self.inertia_slider.endValueChanged.connect(self.slider_changed)

        lay.addWidget(self.teshhold_slider)
        lay.addWidget(self.area_slider)
        lay.addWidget(self.inertia_slider)

        self.label = QLabel(self)
        self.updateImage(path=boot)

        self.resize(600, 600)

        lay.addWidget(self.label)
        self.show()

    def updateImage(self, data=None, path=None):
        pixmap = None
        if path is not None:
            pixmap = QPixmap(path)
        if data is not None:
            height, width, channel = data.shape
            bytesPerLine = 3 * width
            qImg = QImage(data.data, width, height, bytesPerLine, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qImg)

        pixmap = pixmap.scaled(500, 500, Qt.KeepAspectRatio)
        self.label.setPixmap(pixmap)

    def slider_changed(self):
        sender = self.sender()
        self.params[sender.objectName()] = {
            'start': sender.start(),
            'end': sender.end(),
        }
        # self.statusBar().showMessage('{} start: {} | end: {}'.format(sender.objectName(), sender.start(), sender.end()))
        self.statusBar().showMessage(str(self.params))

        if self.continuous:
            img = mainBlob(boot, self.params)
            self.updateImage(data=img)

    def keyPressEvent(self, e: QtGui.QKeyEvent) -> None:
        super().keyPressEvent(e)

        self.statusBar().showMessage('MainWindow pressed {}'.format(str(e.key())))

        if e.key() == Qt.Key_Return:
            self.statusBar().showMessage('Enter pressed {}'.format(str(self.params)))
            img = mainBlob(boot, self.params)
            self.updateImage(data=img)

        if e.key() == Qt.Key_Escape:
            self.params = dict()

        if e.key() == Qt.Key_Alt:
            self.continuous = not self.continuous


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())