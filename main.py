import sys
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from PyQt6 import QtWidgets
from gui.menu import MainMenu

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainMenu()
    app.exec()
