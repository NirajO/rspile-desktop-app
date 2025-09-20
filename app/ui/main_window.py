"""
Main Qt window

This wires together:
- project lifecycle (new/open/save)
- data entry dialogs (pile, loads, soil layers)
- curve generation (t-z, p-y, q-z)
- a simple axial analysis (load-settlement) with quick exports
"""

import pathlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog, QMessageBox, QStatusBar, QTextEdit, QTabWidget)
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from ..models.curves import get_tz_curve, get_qz_curve, get_py_curve
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from ..models.axial import axial_analysis

# relative imports within the package
from .dialogs import PileDialog, LoadDialog, SoilLayerDialog
from ..io.serializer import load_project, save_project

class MainWindow(QMainWindow):
  def __init__(self):
    super().__init__()
    self.setWindowTitle("RSPile Student Edition by Niraj, Hemraj and Manish")
    self.resize(1000, 700)

    self.project: dict | None = None

    #-------central area------
    central = QWidget()
    self.setCentralWidget(central)
    self.layout = QVBoxLayout(central)

    self.info = QTextEdit(readOnly=True)
    self.info.setMinimumHeight(140)

    self.plot_area = QTabWidget()
    self.layout.addWidget(self.info)
    self.layout.addWidget(self.plot_area)

    self.btn_gen = QPushButton("Generate Curves")
    self.btn_gen.clicked.connect(self.generate_curves)
    self.layout.addWidget(self.btn_gen, alignment=Qt.AlignLeft)

    #-----status bar-----
    self.setStatusBar(QStatusBar(self))

    #----menu----
    self._build_menu()

    #initial UI state
    self.refresh_ui()

    #---------Menus---------
  def _build_menu(self):
      # File
      m_file = self.menuBar().addMenu("&File")

      act_new = QAction("New Project", self)
      act_new.triggered.connect(self.new_project)
      m_file.addAction(act_new)

      act_open = QAction("Open...", self)
      act_open.triggered.connect(self.open_project)
      m_file.addAction(act_open)

      act_save = QAction("Save As...", self)
      act_save.triggered.connect(self.save_project)
      m_file.addAction(act_save)

      m_file.addSeparator()
      act_exit = QAction("Exit", self)
      act_exit.triggered.connect(self.close)
      m_file.addAction(act_exit)

      #Edit
      m_edit = self.menuBar().addMenu("&Edit")

      act_axial = QAction("Run Axial Analysis", self)
      act_axial.triggered.connect(self.run_axial_analysis)
      m_edit.addAction(act_axial)

      act_pile = QAction("Edit Pile...", self)
      act_pile.triggered.connect(self.edit_pile)
      m_edit.addAction(act_pile)

      act_loads = QAction("Edit Loads...", self)
      act_loads.triggered.connect(self.edit_loads)
      m_edit.addAction(act_loads)

      act_soil_add = QAction("Add Soil Layer...", self)
      act_soil_add.triggered.connect(self.add_soil_layer)
      m_edit.addAction(act_soil_add)

      #Help
      m_help = self.menuBar().addMenu("&Help")
      act_about = QAction("About", self)
      act_about.triggered.connect(self.about)
      m_help.addAction(act_about)

  #-------Actions--------
  def about(self):
        QMessageBox.information(
          self,
          "About",
          "Single pile analysis under axial (up/down) and lateral (side-ways) " \
          " loads using soil-pile interaction models "
          " (t-z, q-z for axial; p-y for lateral)" \
          " Developed by Niraj Ojha, Hemraj Khatri and Manish Lohani at McNeese State University"
        )

  def new_project(self):
        #Minimal schema we will expand later
        self.project = {
          "meta" : {"version": 1, "units": "SI"},
          "pile": {},            #e.g., length_m, diameter_m, E, unit weight
          "soil_profile": [],    #list of layers with properties
          "loads": {},           #axial, lateral, moment
          "analysis": {"segments": 40}
        }
        self.plot_area.clear()
        self.statusBar().showMessage("New Project created", 3000)
        self.refresh_ui()

  def open_project(self):
        fn, _ = QFileDialog.getOpenFileName(
          self, "Open Project", ".", "RSPile (*.rspile.json)"
        )
        if not fn:
          return
        try:
          self.project = load_project(fn)
          self.statusBar().showMessage(f"Loaded{pathlib.Path(fn).name}", 3000)
          self.refresh_ui()
        except Exception as e:
          QMessageBox.critical(self, "Open Failed", f"could not open file.\n\n{e}")

  def save_project(self):
          if self.project is None:
            QMessageBox.warning(self, "No Project", "Create or open a project first.")
            return
          fn, _ = QFileDialog.getSaveFileName(
            self, "Save Project As", ".", "RSPile (*.rspile.json)"
          )
          if not fn:
            return
          try:
            save_project(self.project, fn)
            self.statusBar().showMessage("Project Saved", 3000)
          except Exception as e:
            QMessageBox.critical(self, "Save Failed", f"Could not save file.\n\n{e}")

  def generate_curves(self):
            if self.project is None:
              QMessageBox.warning(self, "No Project", "Create or open a project first.")
              return
            pile = self.project.get("pile", {})

            if not pile:
                QMessageBox.warning(self, "No Pile", "Edit the pile properties first.")
                return
            D = pile.get("diameter_m", 1.0)
            L = pile.get("length_m", 10.0)
            layers = self.project.get("soil_profile", [])

            if not layers:
                QMessageBox.warning(self, "No Soil", "Add soil layers first.")
                return
            self.plot_area.clear()
            
            # Assume tip in last layer for q-z
            tip_layer = layers[-1] if layers else None
            for i, layer in enumerate(layers):
                mid_depth = (layer.get("from_m", 0) + layer.get("to_m", 0)) / 2
                if "gamma_kNpm3" not in layer:
                    QMessageBox.warning(self, "Data Error", f"Layer {i+1} missing gamma_kNpm3. Please re-enter soil data.")
                    continue
                
                # t-z
                fig_tz, ax_tz = plt.subplots(figsize=(8, 6))
                z_tz, t = get_tz_curve(layer, D, mid_depth)
                ax_tz.plot(z_tz, t)
                ax_tz.set_title(f't-z Curve for Layer {i+1} ({layer["type"]}) at {mid_depth: .2f} m')
                ax_tz.set_xlabel('Displacement z (m)')
                ax_tz.set_ylabel('Shaft Friction t (kPa)')
                ax_tz.grid(True)
                canvas_tz = FigureCanvas(fig_tz)
                self.plot_area.addTab(canvas_tz, f"t-z layer {i+1}")

                # p-y
                fig_py, ax_py = plt.subplots(figsize=(8, 6))
                y_py, p = get_py_curve(layer, D, mid_depth)
                ax_py.plot(y_py, p)
                ax_py.set_title(f'p-y Curve for Layer {i+1} ({layer["type"]}) at {mid_depth:.2f} m')
                ax_py.set_xlabel('Deflection y (m)')
                ax_py.set_ylabel('Lateral Resistance p (kN/m)')
                ax_py.grid(True)
                canvas_py = FigureCanvas(fig_py)
                self.plot_area.addTab(canvas_py, f"p-y Layer {i+1}")

            if tip_layer:
                if "gamma_kNpm3" not in tip_layer:
                    QMessageBox.warning(self, "Data Error", "Tip layer missing gamma_kNpm3. Please re-enter soil data")
                    return
                
                # q-z at tip
                fig_qz, ax_qz = plt.subplots(figsize=(8, 6))
                z_qz, q = get_qz_curve(tip_layer, D, L)
                ax_qz.plot(z_qz, q)
                ax_qz.set_title(f'q-z Curve at Pile Tip ({tip_layer["type"]}) at {L:.2f} m')
                ax_qz.set_xlabel('Displacement z (m)')
                ax_qz.set_ylabel('Tip Resistance q (kPa)')
                ax_qz.grid(True)
                canvas_qz = FigureCanvas(fig_qz)
                self.plot_area.addTab(canvas_qz, "q-z Tip")
            self.statusBar().showMessage("Curves generated and plotted", 3000)
            QMessageBox.information(self, "Curves", "Preview plots generated")

  def run_axial_analysis(self):
    if self.project is None or not self.project.get("pile") or not self.project.get("loads") or not self.project.get("soil_profile"):
        QMessageBox.warning(self, "Missing Data", "Edit Pile, loads, and add soil layers first.")
        return
    results = axial_analysis(self.project["pile"], self.project["loads"], self.project["soil_profile"])

    # Plot load-settlement
    fig_ls, ax_ls = plt.subplots(figsize=(8, 6))
    ax_ls.plot(results['settlements_m'], results['loads_kN'])
    ax_ls.set_title('Load-Settlement Curve')
    ax_ls.set_xlabel('Head Settlement (m)')
    ax_ls.set_ylabel('Axial Load (kN)')
    ax_ls.grid(True)
    canvas_ls = FigureCanvas(fig_ls)
    self.plot_area.addTab(canvas_ls, "Load-settlement")

    # Plot shear vs depth
    fig_sd, ax_sd = plt.subplots(figsize=(8, 6))
    ax_sd.plot(results['plots']['shear_N'], results['plots']['z_m'])
    ax_sd.set_title('Cumulative Shaft Shear vs Depth')
    ax_sd.set_xlabel('Cumulative Shear (N)')
    ax_sd.set_ylabel('Depth (m)')
    ax_sd.grid(True)
    canvas_sd = FigureCanvas(fig_sd)
    self.plot_area.addTab(canvas_sd, "Shear vs Depth")

    # CSV
    df = pd.DataFrame({'Load_kN': results['loads_kN'], 'Settlement_m': results['settlements_m']})
    csv_path = QFileDialog.getSaveFileName(self, "Save CSV", "", "CSV (*.csv)")
    if csv_path:
        df.to_csv(csv_path, index=False)

    # PDF
    pdf_path = QFileDialog.getSaveFileName(self, "Save PDF", "", "PDF (*.pdf)")
    if pdf_path:
        c = canvas.Canvas(pdf_path, pagesize=letter)
        c.drawString(100, 750, "Axial Analysis Report")
        c.drawString(100, 700, f"Max Settlement: {max(results['settlements_m']):.4f} m at {max(results['loads_kN'])} kN")
        c.drawString(100, 650, f"Toe Resistance: {results['plots']['toe_res_N']:.2f} N")
        c.save()
            

  #--------edit menu handlers----------
  def _ensure_project(self):
      if self.project is None:
          self.new_project()

  def edit_pile(self):
      self._ensure_project()
      dlg = PileDialog(self.project.get("pile", {}), self)
      if dlg.exec():
          try:
            self.project["pile"] = dlg.result_data()
            self.refresh_ui()
          except ValueError as e:
              QMessageBox.critical(self, "Input Error", str(e))

  def edit_loads(self):
      self._ensure_project()
      dlg = LoadDialog(self.project.get("loads", {}), self)
      if dlg.exec():
          self.project["loads"] = dlg.result_data()
          self.refresh_ui()

  def add_soil_layer(self):
      self._ensure_project()
      dlg = SoilLayerDialog(self)
      if dlg.exec():
          layer = dlg.result_data()
          self.project.setdefault("soil_profile", []).append(layer)
          # keeping layers ordered by start depth
          self.project["soil_profile"].sort(key=lambda L: L.get("from_m", 0.0))
          self.refresh_ui()

  #-----UI helper-------
  def refresh_ui(self):
            if self.project is None:
              self.info.setPlainText(
                "No project laoded.\n\n"
                "Use File -> New to start a new project, or File -> Open.. to load one."
              )
              self.btn_gen.setEnabled(False)
              return
            
            # Show a simple summary of the current project
            meta = self.project.get("meta", {})
            units = meta.get("units", "SI")
            pile = self.project.get("pile", {})
            loads = self.project.get("loads", {})
            layers = self.project.get("soil_profile", [])

            lines = [
              "Project Summary",
              "----------------",
              f"Units: {units}",
              f"Pile: {pile if pile else '(empty)'}",
              f"Loads: {loads if loads else '(empty)'}",
              f"Soil Layers: {len(layers)} layer(s)",
              "",
            ]

            # pretty-print layers
            if layers:
                lines.append("Layers:")
                for i, L in enumerate(layers, 1):
                    if L.get("type") == "clay":
                        extra = f"su={L.get('undrained_shear_strength_kPa', 0)} kPa"
                    else:
                        extra = f"phi={L.get('phi_deg', 0)}"
                    lines.append(
                        f" {i}. {L.get('type', '?')} {L.get('from_m', 0)}-{L.get('to_m',0)} m, "
                        f"U+0079={L.get('gamma_kNpm3',0)} kN/m³, {extra}"
                    )
            lines += ["", "Tip: Use Edit menu to enter data."]
            self.info.setPlainText("\n".join(lines))
            self.btn_gen.setEnabled(True)
