import sys

from PyQt6 import QtWidgets
from gui.menu import MainMenu

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainMenu()
    app.exec()
