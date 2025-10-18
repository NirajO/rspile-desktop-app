"""
Main Qt window

This wires together:
- project lifecycle (new/open/save)
- data entry dialogs (pile, loads, soil layers)
- curve generation (t-z, p-y, q-z)
- a simple axial analysis (load-settlement) with quick exports as csv and pdf
"""

import pathlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar
from PySide6.QtWidgets import (QApplication, QMainWindow, QMenu, QHBoxLayout, QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog, QMessageBox, QStatusBar, QTextEdit, QTabWidget)
from PySide6.QtCore import Qt, QPoint, QSettings
from PySide6.QtGui import QAction, QKeySequence
from ..models.curves import get_tz_curve, get_qz_curve, get_py_curve, make_py_spring
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from ..models.axial import axial_analysis
from ..models.lateral import PileProps as LatPileProps, LateralLoadCase, BCType, LateralConfig, lateral_analysis

# relative imports within the package
from .dialogs import PileDialog, LoadDialog, SoilLayerDialog
from ..io.serializer import load_project, save_project

class MainWindow(QMainWindow):
  def __init__(self):
    super().__init__()
    self.setWindowTitle("RSPile Student Edition by Niraj, Hemraj and Manish")
    self.resize(1000, 700)

    self.project: dict | None = None
    self.last_axial_results = None

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

    #-------Initialize themes-------
    self._init_theme()

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

      actRunlat = QAction("Run Lateral Analysis", self)
      actRunlat.triggered.connect(self.run_lateral_analysis)
      m_edit.addAction(actRunlat)

      act_pile = QAction("Edit Pile...", self)
      act_pile.triggered.connect(self.edit_pile)
      m_edit.addAction(act_pile)

      act_loads = QAction("Edit Loads...", self)
      act_loads.triggered.connect(self.edit_loads)
      m_edit.addAction(act_loads)

      act_soil_add = QAction("Add Soil Layer...", self)
      act_soil_add.triggered.connect(self.add_soil_layer)
      m_edit.addAction(act_soil_add)

      # Settings
      m_settings = self.menuBar().addMenu("&Settings")

      self.actLight = QAction("Light Mode", self, checkable=True)
      self.actDark = QAction("Dark Mode", self, checkable=True)

      self.actLight.triggered.connect(lambda: self.set_theme("light"))
      self.actDark.triggered.connect(lambda: self.set_theme("dark"))

      m_settings.addAction(self.actLight)
      m_settings.addAction(self.actDark)
      m_settings.addSeparator()

      self._sync_theme_checks()

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
        self.last_axial_results = None
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
          self.statusBar().showMessage(f"Loaded {pathlib.Path(fn).name}", 3000)
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
    self.last_axial_results = results

    def make_export_bar():
      bar = QHBoxLayout()
      btn_csv = QPushButton("Save Axial CSV")
      btn_pdf = QPushButton("Save Axial PDF")
      btn_csv.clicked.connect(self.export_axial_csv)
      btn_pdf.clicked.connect(self.export_axial_pdf)
      bar.addWidget(btn_csv)
      bar.addWidget(btn_pdf)
      bar.addStretch(1)
      return bar

    # Plot load-settlement
    loads_kN = list(results['loads_kN'])
    sett_mm = [1000.0 * abs(s) for s in results['settlements_m']]

    from matplotlib.ticker import MaxNLocator, FormatStrFormatter

    fig_ls, ax_ls = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ax_ls.plot(sett_mm, loads_kN, marker='o', linewidth=2)
    ax_ls.set_title('Load-Settlement Curve')
    ax_ls.set_xlabel('Head Settlement (mm)')
    ax_ls.set_ylabel('Axial Load (kN)')
    ax_ls.set_xlim(0, max(1.0, (max(sett_mm) if sett_mm else 0) * 1.10))
    ax_ls.set_ylim(0, (max(loads_kN) if loads_kN else 0) * 1.10)
    ax_ls.grid(True, alpha=0.35)
    ax_ls.xaxis.set_major_locator(MaxNLocator(6))
    ax_ls.xaxis.set_minor_locator(MaxNLocator(12))
    ax_ls.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    canvas_ls = FigureCanvas(fig_ls)
    self._attach_export_context_menu(canvas_ls)
    
    ls_tab = QWidget()
    ls_vbox = QVBoxLayout(ls_tab)
    ls_vbox.setContentsMargins(6, 6, 6, 6)
    ls_vbox.addWidget(canvas_ls)
    ls_vbox.addLayout(make_export_bar())
    self.plot_area.addTab(ls_tab, "Load-Settlement")


    # Plot shear vs depth
    shear_kN = [s / 1000.0 for s in results['plots']['shear_N']]
    depth_m = list(results['plots']['z_m'])
    shear_max = max(shear_kN) if shear_kN else 0.0

    fig_sd, ax_sd = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ax_sd.plot(shear_kN, depth_m, marker='o', linewidth=2)
    ax_sd.set_title('Cumulative Shaft Shear vs Depth')
    ax_sd.set_xlabel('Cumulative Shear (kN)')
    ax_sd.set_ylabel('Depth (m)')
    ax_sd.invert_yaxis()
    ax_sd.set_xlim(0, max(1e-3, shear_max * 1.10))
    ax_sd.grid(True, alpha=0.35)

    canvas_sd = FigureCanvas(fig_sd)
    self._attach_export_context_menu(canvas_sd)
    
    sd_tab = QWidget()
    sd_vbox = QVBoxLayout(sd_tab)
    sd_vbox.setContentsMargins(6, 6, 6, 6)
    sd_vbox.addWidget(canvas_sd)
    sd_vbox.addLayout(make_export_bar())
    self.plot_area.addTab(sd_tab, "Shear vs Depth")

    self.plot_area.setCurrentWidget(ls_tab)
    self.statusBar().showMessage(
        "Axial Analysis complete. Use File -> Export or right-click a plot to save.", 5000
    )
  
  def run_lateral_analysis(self):
    if self.project is None:
         QMessageBox.warning(self, "No Project", "Create or open a project first.")
         return
     
     #------Gathers pile data------
    pile_data = self.project.get("pile", {})
    if not pile_data:
         QMessageBox.warning(self, "Missing Data", "Please define pile properties first.")
         return
     
    try: 
         L = float(pile_data.get("length_m", 0))
         D = float(pile_data.get("diameter_m", 0))
         E = float(pile_data.get("elastic_modulus_pa", 0))

    except (ValueError, TypeError):
         QMessageBox.critical(self, "Invalid Data", "Pile parameters must be numeric.")
         return

    if L <= 0 or D <= 0 or E <= 0:
        QMessageBox.critical(self, "Inavlid Input", "Pile length, diameter, or E must be positive.")
        return

    #----Compute stiffness (EI)------
    I = 3.14159265358979 * (D ** 4) / 64.0 # moment of inertia for circular crosss-section
    EI = E * I 

    pile = LatPileProps(length_m=L, EI_Nm2=EI, d_m=D, n_nodes=81)

    #------Get soil profile layers-------
    soil_layers = self.project.get("soil_profile", [])
    if not soil_layers:
        QMessageBox.warning(self, "Missing Soil Data", "Please define soil layers first.")
        return
    
    def layer_at_depth(depth_m: float):
        for layer in soil_layers:
            if layer.get("from_m", 0.0) <= depth_m < layer.get("to_m", 0.0):
                return layer
        return soil_layers[-1]
    
    def py_backbone(y_val: float, z_val: float) -> float:
        lyr = layer_at_depth(z_val)
        y_vals, p_vals_kNm = get_py_curve(lyr, D, z_val)
        
        if callable(y_vals) or callable(p_vals_kNm):
            return 0.0
        try:
            y_np = np.asarray(y_vals, dtype=float).ravel()
            p_np = np.asarray(p_vals_kNm, dtype=float).ravel()
        except Exception:
            return 0.0
        if y_np.size == 0 or p_np.size == 0 or y_np.size != p_np.size:
            return 0.0
        
        order = np.argsort(y_np)
        y_sorted = y_np[order]
        p_sorted = p_np[order]

        yq = float(np.clip(y_val, y_sorted[0], y_sorted[-1]))
        p_kNm = float(np.interp(yq, y_sorted, p_sorted))
        return p_kNm * 1000.0

    py_spring = make_py_spring(py_backbone)

    # Define Load steps (kN -> N)
    steps = [LateralLoadCase(H_N=h * 1e3) for h in [0, 200, 400, 600]]

    # Solver configuration
    cfg = LateralConfig(bc=BCType.FREE_HEAD, max_iters=60, tol=1e-7, relax=0.8)

    # Run analysis
    out = lateral_analysis(pile, steps, py_spring, cfg)

    pairs = out.get("head_curve", [])
    if not pairs:
        QMessageBox.warning(self, "No Data", "Lateral solver returned no head curve points.")
        return

    #------Plot Head Load-Deflection Curve-------
    H_kN = np.array([h for (h, y0) in pairs], dtype=float) / 1e3
    y_head_mm = np.array([y0 for (h, y0) in pairs], dtype=float) * 1e3

    fig1, ax1 = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ax1.plot(y_head_mm, H_kN, marker="o", linewidth=2)
    ax1.set_title("Lateral Load-Deflection")
    ax1.set_xlabel("Head Deflection (mm)")
    ax1.set_ylabel("Applied Lateral Load H (kN)")
    ax1.grid(True, alpha=0.35)

    canvas1 = FigureCanvas(fig1)
    tab1 = QWidget()
    vbox1 = QVBoxLayout(tab1)
    vbox1.setContentsMargins(6, 6, 6, 6)
    vbox1.addWidget(NavigationToolbar(canvas1, tab1))
    vbox1.addWidget(canvas1)
    self.plot_area.addTab(tab1, "Lateral H-y")
    

    #-------Plot Deflection vs Depth for Last Step---------
    if not out.get("steps"):
        QMessageBox.warning(self, "No Data", "No step results to plot.")
        return
    last = out["steps"][-1]
    fig2, ax2 = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ax2.plot(last.y_m * 1e3, last.z_m, linewidth=2)
    ax2.invert_yaxis()
    ax2.set_xlabel("Deflection (mm)")
    ax2.set_ylabel("Depth (m)")
    ax2.grid(True, alpha=0.35)

    canvas2 = FigureCanvas(fig2)
    tab2 = QWidget()
    vbox2 = QVBoxLayout(tab2)
    vbox2.setContentsMargins(6, 6, 6, 6)
    vbox2.addWidget(NavigationToolbar(canvas2, tab2))
    vbox2.addWidget(canvas2)
    self.plot_area.addTab(tab2, "Deflection vs Depth")

    self.plot_area.setCurrentWidget(tab1)

  #---------Right click export menu------------------
  def _attach_export_context_menu(self, widget: QWidget):
      widget.setContextMenuPolicy(Qt.CustomContextMenu)
      widget.customContextMenuRequested.connect(
          lambda pos, w=widget: self._show_export_menu(w, pos)
      )  

  def _show_export_menu(self, widget: QWidget, pos: QPoint):
      menu = QMenu(widget)
      a_csv = menu.addAction("Export Axial CSV")
      a_pdf = menu.addAction("Export Axial PDF")
      chosen = menu.exec(widget.mapToGlobal(pos))
      if chosen == a_csv:
          self.export_axial_csv()
      elif chosen == a_pdf:
          self.export_axial_pdf()

  #---------Export Handlers------------
  def export_axial_csv(self):
      if not self.last_axial_results:
          QMessageBox.information(self, "No results", "Run an axial analysis first.")
          return
      
      results = self.last_axial_results
      df = pd.DataFrame({
          'Loads_kN': results['loads_kN'],
          'Settlement_m': results['settlements_m']
      })

      csv_path, _ = QFileDialog.getSaveFileName(self, "Save Axial Curve CSV", "axial_curve.csv", "CSV (*.csv)")
      if not csv_path:
          return
      if not csv_path.lower().endswith(".csv"):
          csv_path += ".csv"

      try:
          df.to_csv(csv_path, index=False)
          self.statusBar().showMessage(f"Saved: {csv_path}", 5000)
      except Exception as e:
          QMessageBox.critical(self, "Save Failed", f"could not save CSV:\n{e}")

  def export_axial_pdf(self):
      if not self.last_axial_results:
          QMessageBox.information(self, "No results", "Run an axial analysis first.")
          return
      
      results = self.last_axial_results
      pdf_path, _ = QFileDialog.getSaveFileName(self, "Save Axial Report (PDF)", "axial_report.pdf", "PDF (*.pdf)")
      if not pdf_path:
          return
      if not pdf_path.lower().endswith(".pdf"):
          pdf_path += ".pdf"

      try:
          c = canvas.Canvas(pdf_path, pagesize=letter)
          width, height = letter
          y = height - 72  # 1 inch margin

          c.setFont("Helvetica-Bold", 14)
          c.drawString(72, y, "Axial Analysis Report")
          y -= 24

          c.setFont("Helvetica", 11)
          max_set = float(max(results['settlements_m']))
          max_load = float(max(results['loads_kN']))
          toe_res = float(results['plots']['toe_res_N'])

          c.drawString(72, y, f"Max Settlement: {max_set:.4f} m at {max_load:.0f} kN")
          y -= 16
          c.drawString(72, y, f"Toe Resistance (final step): {toe_res:.1f} N")

          c.showPage()
          c.save()
          self.statusBar().showMessage(f"Saved: {pdf_path}", 5000)

      except Exception as e:
          QMessageBox.critical(self, "Save failed", f"Could not save PDF:\n{e}")
  
        
          

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
                "No project loaded.\n\n"
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
                        f"gamma={L.get('gamma_kNpm3',0)} kN/m³, {extra}"
                    )
            lines += ["", "Tip: Use Edit menu to enter data."]
            self.info.setPlainText("\n".join(lines))
            self.btn_gen.setEnabled(True)

  def _init_theme(self):
    s = QSettings("RSPile", "StudentEdition")
    theme = s.value("theme", "light")
    self._current_theme = theme
    self._apply_theme(theme)
    self._sync_theme_checks()

  def set_theme(self, theme: str):
    if theme not in ("light", "dark"):
      theme = "light"
    self._current_theme = theme
    self._apply_theme(theme)
    QSettings("RSPile", "StudentEdition").setValue("theme", theme)
    self._sync_theme_checks()
    
  def toggle_theme(self):
    self.set_theme("dark" if getattr(self, "_current_theme", "light") != "dark" else "light")

  def _sync_theme_checks(self):
    if hasattr(self, "actLight"):
      self.actLight.setChecked(getattr(self, "_current_theme", "light") == "light")
    if hasattr(self, "actDark"):
      self.actDark.setChecked(getattr(self, "_current_theme", "light") == "dark")
      
  def _apply_theme(self, theme: str):
    app = QApplication.instance()
    if not app:
      return
    
    if theme == "dark":
      app.setStyleSheet("""
        QWidget {background: #232323; color:#eeeeee;}
        QMenuBar, QMenu { background: #2c2c2c; color: #eeeeee;}
        QToolTip { color: #eeeeee; background:#444;}
        QPushButton { background:#3a3a3a; border: 1px solid #555; padding: 6px;}
        QPushButton:hover {background:#444;}
        QTextEdit, QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
          background:#2b2b2b; border: 1px solid #555; color:#eee;
        }
        QTabBar::tab { background:#2c2c2c; padding:6px 10px;}
        QTabBar::tab:selected {background:#3a3a3a;}
        QStatusBar {background:#2c2c2c;}

      """)
    else:
      app.setStyleSheet("")

    #-----------Matplotlib: light mode for curves-----------
    mpl.rcParams.update(mpl.rcParamsDefault)
    mpl.style.use("default")
    mpl.rcParams.update({
      "figure.facecolor": "white",
      "axes.facecolor": "white",
      "savefig.facecolor": "white",
      "axes.edgecolor": "black",
      "axes.labelcolor": "black",
      "xtick.color": "black",
      "ytick.color": "black",
      "grid.color": "#CCCCCC",
    })
      
      


  
