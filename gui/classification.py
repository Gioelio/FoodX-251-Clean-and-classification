from PyQt6 import QtWidgets, uic
from PyQt6.QtWidgets import QFileDialog


class ClassificationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(ClassificationWindow, self).__init__()
        uic.loadUi('gui/classification.ui', self)
        self.button_select = self.findChild(QtWidgets.QPushButton, 'buttonSelect')
        self.button_select.clicked.connect(self.select_clicked)

    def select_clicked(self):
        dialog = QFileDialog()
        dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        dialog.setNameFilter("Immagini (*.jpg *.png *.jpeg *.bmp);; Tutti i file (*.*)")
        if dialog.exec():
            filename = dialog.selectedFiles()[0]
            print(filename)
