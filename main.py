import sys
from PySide6.QtWidgets import QApplication, QMainWindow

class Main(QMainWindow):
  def __init__(self):
    super().__init__()
    self.setWindowTitle("RSPile Student Edition")
    self.resize(800, 600)

if __name__ == "__main__":
  app = QApplication(sys.argv)
  w = Main()
  w.show()
  sys.exit(app.exec())
