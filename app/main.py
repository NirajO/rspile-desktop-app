"Entry point for the GUI."
"Spins up a Qt application and shows the main window"
                     
import sys
from PySide6.QtWidgets import QApplication

def main():
  app = QApplication(sys.argv)
  from app.ui.main_window import MainWindow
  w = MainWindow()
  w.show()
  sys.exit(app.exec())

if __name__ == "__main__":
  main()