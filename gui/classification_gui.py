from PyQt6 import QtWidgets, uic
from PyQt6.QtWidgets import QFileDialog
from classification.classify import classify_image

class ClassificationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(ClassificationWindow, self).__init__()
        uic.loadUi('gui/classification.ui', self)
        self.button_select = self.findChild(QtWidgets.QPushButton, 'buttonSelect')
        self.results_label = self.findChild(QtWidgets.QLabel, 'labelResults')
        self.button_select.clicked.connect(self.select_clicked)

    def select_clicked(self):
        dialog = QFileDialog()
        dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        dialog.setNameFilter("Immagini (*.jpg *.png *.jpeg *.bmp);; Tutti i file (*.*)")
        if dialog.exec():
            filename = dialog.selectedFiles()[0]
            try:
                ordered_classes, prediction, classnames = classify_image(filename)
                result_string = ""
                for i in range(10):
                    result_string += classnames[ordered_classes[i]] + ": " + "{0:.4f}".format(prediction[ordered_classes[i]] * 100) + "%\n"
                self.results_label.setText(result_string)
            except Exception as e:
                self.results_label.setText(str(e))
