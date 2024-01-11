from PyQt6 import QtWidgets, uic
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QFileDialog, QGridLayout, QLabel, QScrollArea, QWidget
from similarity_search.similarity_search_gui import load_images_features, find_similar_images


class SearchWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(SearchWindow, self).__init__()
        uic.loadUi('gui/similarity.ui', self)
        self.button_select = self.findChild(QtWidgets.QPushButton, 'buttonSelect')
        self.button_select.clicked.connect(self.select_clicked);
        self.slider = self.findChild(QtWidgets.QSlider, "baseWeight")
        self.slider.sliderReleased.connect(self.slider_released);
        self.slider.setMinimum(1)
        self.slider.setMaximum(9)
        self.slider.setValue(5)
        self.slider.setTickInterval(1)
        self.grid = self.findChild(QtWidgets.QScrollArea, 'gridImages');
        self.query_img = self.findChild(QtWidgets.QLabel, 'queryImg');

        nn, handcrafted_features = load_images_features()
        self.nn_features = nn;
        self.handcrafted = handcrafted_features
        self.filename = None;

    def slider_released(self):
        if self.filename is not None:
            self.load_images_in_grid(self.filename)

    def select_clicked(self):
        dialog = QFileDialog()
        dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        dialog.setNameFilter("Immagini (*.jpg *.png *.jpeg *.bmp);; Tutti i file (*.*)")
        if dialog.exec():
            filename = dialog.selectedFiles()[0]
            self.load_images_in_grid(filename)
            

    def load_images_in_grid(self, filename):
        print(filename)
        pixmap = QPixmap(filename)
        self.query_img.setPixmap(pixmap.scaledToWidth(200))
        self.query_img.setAlignment(Qt.AlignmentFlag.AlignCenter)

        use_intersection = False,
        use_nn = True

        most_similar_filenames = find_similar_images(filename, self.handcrafted, self.nn_features, use_intersection, use_nn, self.slider.value()/10)
        row, col = 0, 0
        number_limit = 100

        scrollAreaWidget = QWidget()
        grid_layout = QGridLayout(scrollAreaWidget)
        for path in most_similar_filenames:
            if number_limit <= 0:
                continue;
            number_limit -= 1;
            pixmap = QPixmap(path)
            label = QLabel(self)
            label.setPixmap(pixmap.scaledToWidth(200))
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)

            grid_layout.addWidget(label, row, col)
            col += 1;

            if col == 2:
                col = 0;
                row += 1

        self.filename = filename
        self.grid.setWidget(scrollAreaWidget);
            
