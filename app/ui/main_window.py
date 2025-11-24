"""
Main Qt window

This wires together:
- project lifecycle (new/open/save)
- data entry dialogs (pile, loads, soil layers)
- curve generation (t-z, p-y, q-z)
- a simple axial analysis (load-settlement) with quick exports as csv and pdf
"""

import pathlib, io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
from PySide6.QtWidgets import (QApplication, QMainWindow, QMenu, QHBoxLayout, QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog, QMessageBox, QStatusBar, QTextEdit, QTabWidget, QToolBar, QDockWidget, QListWidget, QListWidgetItem, QFrame, QDoubleSpinBox, QFormLayout, QToolButton, QStyle, QTextBrowser)
from PySide6.QtCore import Qt, QPoint, QSize, QSettings, QPropertyAnimation, QEasingCurve, QUrl
from PySide6.QtGui import QAction, QKeySequence, QIcon
from ..models.curves import get_tz_curve, get_qz_curve, get_py_curve, make_py_spring
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas as rl_canvas
from ..models.axial import axial_analysis
from ..models.lateral import PileProps as LatPileProps, LateralLoadCase, BCType, LateralConfig, lateral_analysis

# relative imports within the package
from .dialogs import PileDialog, LoadDialog, SoilLayerDialog
from ..io.serializer import load_project, save_project

class MainWindow(QMainWindow):
  def __init__(self):
    super().__init__()
    self.setWindowTitle("Pile Analysis Tool")
    self.setWindowIcon(QIcon("icons/app_icon.png"))
    self.resize(1200, 800)
    self.setMinimumSize(900, 600)

    self.project: dict | None = None
    self.last_axial_results = None

    #-------central area------
    central = QWidget()
    self.setCentralWidget(central)
    self.layout = QVBoxLayout(central)

    self.welcome = self._make_welcome()
    self.layout.addWidget(self.welcome)

    self.info = QTextBrowser()
    self.info.setOpenExternalLinks(False)
    self.info.setOpenLinks(False)
    self.info.anchorClicked.connect(self._on_summary_link_clicked)
    self.info.setMinimumHeight(140)
    self.info.setObjectName("ProjectSummary")
    self.info.setFrameShape(QFrame.NoFrame)
    self.info.setStyleSheet("""
            QTextEdit#ProjectSummary{
            font-family: 'Segoe UI', 'Helvetica Neue', Arial;
            font-size: 18px;
            background: transparent;
            padding: 25px 30px;
    }
    """)

    self.plot_area = QTabWidget()
    self.plot_area.setTabsClosable(True)
    self.plot_area.tabCloseRequested.connect(self._close_plot_tab)
    self.layout.addWidget(self.info)
    self.layout.addWidget(self.plot_area)

    self.btn_gen = QPushButton("Generate Curves")
    self.btn_gen.clicked.connect(self.generate_curves)
    self.layout.addWidget(self.btn_gen, alignment=Qt.AlignLeft)

    #-----status bar-----
    self.setStatusBar(QStatusBar(self))

    #---------status bar widgets + dirty flag ---------
    self._dirty = False
    self.status_project = QLabel("No Project Loaded.")
    self.status_dirty = QLabel("")
    self.status_dirty.setStyleSheet("color:#d9534f; font-weight:600;")

    self.statusBar().addWidget(self.status_project)
    self.statusBar().addPermanentWidget(self.status_dirty)

    #----menu----
    self._build_menu()

    #-------Initialize themes-------
    self._init_theme()

    #initial UI state
    self.refresh_ui()

    self._update_status_bar()

    self._refresh_recent_list()

    self._axial_figures = []

    self._lateral_figures = []
    self.last_lateral_out = None

    # Restore window geometry and dock/toolbar layout
    s = QSettings("Pile Analysis", "StudentEdition")

    g = s.value("win/geo")
    st = s.value("win/state")
    try:
        if g:
            self.restoreGeometry(g)
        if st:
            self.restoreState(st, 1)
    except Exception as e:
        print("restoreState/restoreGeometry failed, using default layout:", e)
    
    if hasattr(self, "_dock"):
        self._dock.show()
        self._dock.raise_()
    

    # 3D view state
    self._dock3d = None
    self._3d_canvas = None
    self._3d_fig = None
    self._3d_ax = None
    self._3d_scale_spin = None

  #---------Menus---------
  def _build_menu(self):
      #----------- Toolbar ---------------
      tb = QToolBar("Main")
      tb.setObjectName("MainToolbar")
      tb.setMovable(False)
      self.addToolBar(Qt.TopToolBarArea, tb)

      # File
      m_file = self.menuBar().addMenu("&File")

      act_new = QAction(QIcon.fromTheme("document-new"), "New Project", self)
      act_new.setShortcut(QKeySequence.New)
      act_new.triggered.connect(self.new_project)
      tb.addAction(act_new); m_file.addAction(act_new)

      act_open = QAction(QIcon.fromTheme("document-open"), "Open", self)
      act_open.setShortcut(QKeySequence.Open)
      act_open.triggered.connect(self.open_project)
      tb.addAction(act_open); m_file.addAction(act_open)

      act_save = QAction(QIcon.fromTheme("document-save"), "Save As", self)
      act_save.setShortcut(QKeySequence.SaveAs)
      act_save.triggered.connect(self.save_project)
      tb.addAction(act_save); m_file.addAction(act_save)

      # Recent submenu
      self._m_recent = m_file.addMenu("Open &Recent")
      self._rebuild_recent_menu = lambda: (
          self._m_recent.clear(),
          [self._m_recent.addAction(pathlib.Path(p).name, lambda p=p: self._open_path(p)) for p in self._recent_files()]
      )
      self._rebuild_recent_menu()
      m_file.addSeparator()

      act_exit = QAction("Exit", self)
      act_exit.setShortcut(QKeySequence.Quit)
      act_exit.triggered.connect(self.close)
      m_file.addAction(act_exit)

      #Edit
      m_edit = self.menuBar().addMenu("&Edit")

      act_axial = QAction("Run Axial Analysis", self)
      act_axial.setShortcut("Ctrl+Shift+A")
      act_axial.triggered.connect(self.run_axial_analysis)
      tb.addAction(act_axial); m_edit.addAction(act_axial)

      act_lat = QAction("Run Lateral Analysis", self)
      act_lat.setShortcut("Ctrl+Shift+L")
      act_lat.triggered.connect(self.run_lateral_analysis)
      tb.addAction(act_lat); m_edit.addAction(act_lat)

      # 3D view
      act_3d = QAction("Open 3D View", self)
      act_3d.setShortcut("Ctrl+3")
      act_3d.triggered.connect(self.open_3d_view)
      tb.addAction(act_3d)

      m_edit.addSeparator()

      act_pile = QAction("Edit Pile...", self)
      act_pile.setShortcut("Ctrl+P")
      act_pile.triggered.connect(self.edit_pile)
      m_edit.addAction(act_pile)

      act_loads = QAction("Edit Loads...", self)
      act_loads.setShortcut("Ctrl+L")
      act_loads.triggered.connect(self.edit_loads)
      m_edit.addAction(act_loads)

      act_soil_add = QAction("Add Soil Layer...", self)
      act_soil_add.setShortcut("Ctrl+S")
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

      # Lateral boundary condition submenu
      m_settings.addSeparator()
      bc_menu = m_settings.addMenu("Lateral Boundary Condition")

      self.actBCFree = QAction("Free Head", self, checkable=True)
      self.actBCFixed = QAction("Fixed Head", self, checkable=True)

      self.actBCFree.triggered.connect(lambda: self._set_lateral_bc("free_head"))
      self.actBCFixed.triggered.connect(lambda: self._set_lateral_bc("fixed_head"))

      bc_menu.addAction(self.actBCFree)
      bc_menu.addAction(self.actBCFixed)

      self._sync_lateral_bc_checks()

      #Help
      m_help = self.menuBar().addMenu("&Help")
      act_about = QAction("About", self)
      act_about.triggered.connect(self.about)
      m_help.addAction(act_about)

      #----------Left Dock: Project Inspector----------
      self._dock = QDockWidget("Project Inspector", self)
      self._dock.setObjectName("ProjectInspectorDock")
      self._dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
      self._dock_expanded_width = 400
      self._dock_collapsed_width = 40
      self._dock_last_width = self._dock_expanded_width

      self._dock.setMinimumWidth(self._dock_collapsed_width)
      self._dock.setMaximumWidth(self._dock_expanded_width)
      self._dock.resize(self._dock_expanded_width, self._dock.height())

      self._dock.setFeatures(
          QDockWidget.DockWidgetMovable |
          QDockWidget.DockWidgetFloatable |
          QDockWidget.DockWidgetClosable
      )
      
      self._dock_is_collapsed = False
      self._dock_anim = None

      # dock toggle bar
      title_bar = QWidget()
      title_layout = QHBoxLayout(title_bar)
      title_layout.setContentsMargins(6, 0, 6, 0)

      title_label = QLabel("Project Inspector")
      title_label.setStyleSheet("font-weight:600")
      self._dock_title_label = title_label

      btn_toggle = QToolButton()
      btn_toggle.setToolTip("Hide/Show Project Inspector")
      btn_toggle.setAutoRaise(True)

      btn_toggle.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_ArrowLeft))

      btn_toggle.setFixedSize(30, 30)
      btn_toggle.setIconSize(QSize(14, 14))
      btn_toggle.setStyleSheet("QToolButton {padding: 0px; margin: 0px;}")

      btn_toggle.clicked.connect(self.toggle_project_inspector)

      title_layout.addWidget(title_label)
      title_layout.addStretch(1)
      title_layout.addWidget(btn_toggle)

      title_bar.setMinimumHeight(32)

      self._dock_toggle_btn = btn_toggle
      self._dock.setTitleBarWidget(title_bar)

      dockw = QWidget()

      dlay = QVBoxLayout(dockw)
      self._lbl_meta = QLabel("")
      self._lbl_meta.setWordWrap(True)

      # Quick actions
      line = QFrame(); line.setFrameShape(QFrame.HLine); line.setFrameShadow(QFrame.Sunken)
      b_row = QVBoxLayout()
      b_pile = QPushButton("Edit Pile..."); b_pile.clicked.connect(self.edit_pile)
      b_loads = QPushButton("Edit Loads..."); b_loads.clicked.connect(self.edit_loads)
      b_soil = QPushButton("Add Soil Layer..."); b_soil.clicked.connect(self.add_soil_layer)
      b_curves = QPushButton("Generate Curves"); b_curves.clicked.connect(self.generate_curves)
      b_axial = QPushButton("Run Axial Analysis"); b_axial.clicked.connect(self.run_axial_analysis)
      b_lat = QPushButton("Run Lateral Analysis"); b_lat.clicked.connect(self.run_lateral_analysis)
      for b in (b_pile, b_loads, b_soil, b_curves, b_axial, b_lat):
          b.setMinimumHeight(28); dlay.addWidget(b)

      dlay.insertWidget(0, self._lbl_meta)
      dlay.insertWidget(1, line)

      self.recent_list = QListWidget()
      self.recent_list.setMaximumHeight(160)
      self.recent_list.itemDoubleClicked.connect(lambda it: self._open_path(it.text()))

      row_recent = QHBoxLayout()
      lbl_recent = QLabel("Recent Projects")
      btn_clear_recent = QPushButton("Clear")
      btn_clear_recent.setFixedHeight(22)
      btn_clear_recent.setStyleSheet("font-size: 10px; padding: 2px 6px;")
      btn_clear_recent.clicked.connect(self._clear_recent_files)
      row_recent.addWidget(lbl_recent)
      row_recent.addStretch(1)
      row_recent.addWidget(btn_clear_recent)

      dlay.addLayout(row_recent)
      dlay.addWidget(self.recent_list)

      lbl_layers = QLabel("Soil Layers")
      lbl_layers.setStyleSheet("margin-top: 8px; font-weight: 600;")
      dlay.addWidget(lbl_layers)

      self.layers_list = QListWidget()
      self.layers_list.setMaximumHeight(220)
      self.layers_list.setSpacing(2)
      dlay.addWidget(self.layers_list)

      dlay.addStretch(1)

      dockw.setLayout(dlay)
      self._dock.setWidget(dockw)
      self.addDockWidget(Qt.LeftDockWidgetArea, self._dock)

      self._dock.show()


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

  def _make_welcome(self) -> QWidget:
      w = QWidget()
      lay = QVBoxLayout(w)
      title = QLabel("Pile Analysis - Student Edition")
      title.setStyleSheet("font-size: 22px; font-weight: 600;")
      subtitle = QLabel("Analyze single piles under axial and lateral loads")
      subtitle.setStyleSheet("color: #666; margin-bottom: 8px;")

      btn_row = QHBoxLayout()
      bnew = QPushButton("New Project")
      bopen = QPushButton("Open Project")
      bnew.setFixedWidth(160)
      bopen.setFixedWidth(160)
      bnew.clicked.connect(self.new_project)
      bopen.clicked.connect(self.open_project)
      btn_row.addWidget(bnew)
      btn_row.addWidget(bopen)
      btn_row.addStretch(1)

      # recent
      rec = QListWidget()
      rec.setMaximumHeight(160)
      self._welcome_recent = rec
      for p in self._recent_files():
          QListWidgetItem(p, rec)
      rec.itemDoubleClicked.connect(lambda it: self._open_path(it.text()))

      lay.addStretch(1)
      lay.addWidget(title, 0, Qt.AlignLeft)
      lay.addWidget(subtitle, 0, Qt.AlignLeft)
      lay.addLayout(btn_row)
      lay.addWidget(QLabel("Recent Projects"))
      lay.addWidget(rec)
      lay.addStretch(3)
      return w
  
  def _refresh_recent_list(self):
      files = self._recent_files()

      # Left dock list
      if hasattr(self, "recent_list"):
        self.recent_list.clear()
        for p in files:
            QListWidgetItem(p, self.recent_list)
    
      # Welcome screen list
      if hasattr(self, "_welcome_recent"):
        self._welcome_recent.clear()
        for p in files:
          QListWidgetItem(p, self._welcome_recent)

  def new_project(self):
        #Minimal schema we will expand later
        self.project = {
          "meta" : {"version": 1, "units": "SI"},
          "pile": {},            #e.g., length_m, diameter_m, E, unit weight
          "soil_profile": [],    #list of layers with properties
          "loads": {},           #axial, lateral, moment
          "analysis": {"segments": 40, "lateral_bc": "free_head"}
        }
        self.last_axial_results = None
        self.plot_area.clear()
        self.statusBar().showMessage("New Project created", 3000)
        self.refresh_ui()
        self._set_dirty(True)

  def open_project(self):
        fn, _ = QFileDialog.getOpenFileName(
          self, "Open Project", ".", "RSPile (*.pile.json);;All files (*)"
        )
        if not fn:
          return
        try:
          self.project = load_project(fn)
          self.statusBar().showMessage(f"Loaded {pathlib.Path(fn).name}", 3000)
          self.refresh_ui()
          self._sync_lateral_bc_checks()
          self._set_dirty(False)
          self._push_recent_file(fn)
        except Exception as e:
          QMessageBox.critical(self, "Open Failed", f"could not open file.\n\n{e}")
        
  def save_project(self):
          if self.project is None:
            QMessageBox.warning(self, "No Project", "Create or open a project first.")
            return
          fn, _ = QFileDialog.getSaveFileName(
            self, "Save Project As", ".", "RSPile (*.pile.json)"
          )
          if not fn.lower().endswith(".pile.json"):
            fn += ".pile.json"
          try:
            save_project(self.project, fn)
            self.statusBar().showMessage("Project Saved", 3000)
            self._set_dirty(False)
            self._push_recent_file(fn)
          except Exception as e:
            QMessageBox.critical(self, "Save Failed", f"Could not save file.\n\n{e}")

  def _recent_files(self) -> list[str]:
      s = QSettings("Pile Analysis", "StudentEdition")
      return s.value("recent_files", [], list)
  
  def _push_recent_file(self, path: str) -> None:
      s = QSettings("Pile Analysis", "StudentEdition")
      items = [p for p in self._recent_files() if p != path]
      items.insert(0, path)
      s.setValue("recent_files", items[:10])
      if hasattr(self, "_rebuild_recent_menu"):
          self._rebuild_recent_menu()
      self._refresh_recent_list()

  def _clear_recent_files(self):
      # Clear stored recent projects and refresh UI lists/menyus.
      s = QSettings("Pile Analysis", "StudentEdition")
      s.setValue("recent_files", [])
      if hasattr(self, "_rebuild_recent_menu"):
          self._rebuild_recent_menu()
      self._refresh_recent_list()

  def _open_path(self, path: str):
      try:
          self.project = load_project(path)
          self._push_recent_file(path)
          self.statusBar().showMessage(f"Loaded {pathlib.Path(path).name}", 3000)
          self.plot_area.clear()
          self.refresh_ui()
          self._sync_lateral_bc_checks()
      except Exception as e:
          QMessageBox.critical(self, "Open Failed", f"Could not open file.\n\n{e}")

  def _update_status_bar(self):
      """Refresh the bottom status labels based on current project + dirty flag"""
      if self.project:
          meta = self.project.get("meta", {})
          name = meta.get("name", "Untitled")
          units = meta.get("units", "SI")
          self.status_project.setText(f"Project: {name} - Units: {units}")
      else:
          self.status_project.setText("No Project Loaded")
      self.status_dirty.setText("● Unsaved" if self._dirty else "")

  def toggle_project_inspector(self):
    """Toggle: collapse to a thin strip (arrow visible) or expand back."""
    if not hasattr(self, "_dock") or self._dock is None:
        return

    collapsed_w = getattr(self, "_dock_collapsed_width", 28)
    expanded_w  = getattr(self, "_dock_last_width", getattr(self, "_dock_expanded_width", 300))

    # ---------- Collapse ----------
    if self._dock.isVisible() and not self._dock_is_collapsed:
        start_w = self._dock.width()
        self._dock_last_width = max(180, start_w)
        end_w = collapsed_w   # <-- NOT 0 anymore

        if self._dock_anim and self._dock_anim.state() == QPropertyAnimation.Running:
            self._dock_anim.stop()

        # keep dock from fighting the animation
        self._dock.setMinimumWidth(collapsed_w)

        self._dock_anim = QPropertyAnimation(self._dock, b"maximumWidth")
        self._dock_anim.setDuration(180)
        self._dock_anim.setEasingCurve(QEasingCurve.InOutCubic)
        self._dock_anim.setStartValue(start_w)
        self._dock_anim.setEndValue(end_w)

        def on_value_changed(val):
            self._dock.setFixedWidth(int(val))

        def on_finished():
            self._dock_is_collapsed = True

            # hide content, keep titlebar
            if self._dock.widget():
                self._dock.widget().setVisible(False)

            # hide label so arrow has space
            if hasattr(self, "_dock_title_label"):
                self._dock_title_label.setVisible(False)

            self._dock.setFixedWidth(end_w)
            self._dock.setMaximumWidth(end_w)

            if hasattr(self, "_dock_toggle_btn"):
                self._dock_toggle_btn.setIcon(
                    self.style().standardIcon(QStyle.StandardPixmap.SP_ArrowRight)
                )

        self._dock_anim.valueChanged.connect(on_value_changed)
        self._dock_anim.finished.connect(on_finished)
        self._dock_anim.start()
        return

    # ---------- Expand ----------
    self._dock.show()
    if self._dock.widget():
        self._dock.widget().setVisible(True)

    if hasattr(self, "_dock_title_label"):
        self._dock_title_label.setVisible(True)

    start_w = collapsed_w
    end_w = expanded_w

    if self._dock_anim and self._dock_anim.state() == QPropertyAnimation.Running:
        self._dock_anim.stop()

    self._dock.setMaximumWidth(start_w)
    self._dock.setMinimumWidth(collapsed_w)

    self._dock_anim = QPropertyAnimation(self._dock, b"maximumWidth")
    self._dock_anim.setDuration(180)
    self._dock_anim.setEasingCurve(QEasingCurve.InOutCubic)
    self._dock_anim.setStartValue(start_w)
    self._dock_anim.setEndValue(end_w)

    def on_value_changed(val):
        self._dock.setFixedWidth(int(val))

    def on_finished():
        self._dock_is_collapsed = False
        self._dock.setFixedWidth(end_w)
        self._dock.setMaximumWidth(end_w)
        self._dock.setMinimumWidth(collapsed_w)
        if hasattr(self, "_dock_toggle_btn"):
            self._dock_toggle_btn.setIcon(
                self.style().standardIcon(QStyle.StandardPixmap.SP_ArrowLeft)
            )

    self._dock_anim.valueChanged.connect(on_value_changed)
    self._dock_anim.finished.connect(on_finished)
    self._dock_anim.start()


  def _set_dirty(self, value: bool = True):
      self._dirty = bool(value)
      self._update_status_bar()

  def _set_lateral_bc(self, bc: str):
    """Save lateral BC to project and refresh checkmarks."""
    if self.project is None:
        return
    self.project.setdefault("analysis", {})

    if bc not in ("free_head", "fixed_head"):
        bc = "free_head"
    
    self.project["analysis"]["lateral_bc"] = bc
    self._sync_lateral_bc_checks()
    self._set_dirty(True)

  def _sync_lateral_bc_checks(self):
    """Update menu checkmarks to match project settings."""
    bc = "free_head"
    if self.project is not None:
        bc = self.project.get("analysis", {}).get("lateral_bc", "free_head")
    
    if hasattr(self, "actBCFree"):
        self.actBCFree.setChecked(bc == "free_head")
    if hasattr(self, "actBCFixed"):
        self.actBCFixed.setChecked(bc == "fixed_head")

  def dragEnterEvent(self, e):
      if e.mimeData().hasUrls():
          for u in e.mimeData().urls():
              if u.toLocalFile().lower().endswith(".pile.json"):
                  e.acceptProposedAction()
                  self.setStyleSheet("QMainWindow {border: 2px dashed #4a90e2;}")
                  return
      e.ignore()

  def dragLeaveEvent(self, e):
      self.setStyleSheet("")

  def dropEvent(self, e):
      self.setStyleSheet("")
      for u in e.mimeData().urls():
          p = u.toLocalFile()
          if p.lower().endswith(".pile.json"):
              self._open_path(p)
              break

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
                wrap = QWidget(); v = QVBoxLayout(wrap); v.setContentsMargins(6,6,6,6)
                v.addWidget(NavigationToolbar(canvas_tz, wrap))
                v.addWidget(canvas_tz)
                self.plot_area.addTab(wrap, f"t-z Layer {i+1}")

                # p-y
                fig_py, ax_py = plt.subplots(figsize=(8, 6))
                y_py, p = get_py_curve(layer, D, mid_depth)
                ax_py.plot(y_py, p)
                ax_py.set_title(f'p-y Curve for Layer {i+1} ({layer["type"]}) at {mid_depth:.2f} m')
                ax_py.set_xlabel('Deflection y (m)')
                ax_py.set_ylabel('Lateral Resistance p (kN/m)')
                ax_py.grid(True)
                canvas_py = FigureCanvas(fig_py)
                wrap = QWidget(); v = QVBoxLayout(wrap); v.setContentsMargins(6,6,6,6)
                v.addWidget(NavigationToolbar(canvas_py, wrap))
                v.addWidget(canvas_py)
                self.plot_area.addTab(wrap, f"p-y Layer {i+1}")

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
                wrap = QWidget(); v = QVBoxLayout(wrap); v.setContentsMargins(6,6,6,6)
                v.addWidget(NavigationToolbar(canvas_qz, wrap))
                v.addWidget(canvas_qz)
                self.plot_area.addTab(wrap, f"q-z Layer {i+1}")
            self.statusBar().showMessage("Curves generated and plotted", 3000)
            QMessageBox.information(self, "Curves", "Preview plots generated")

  def run_axial_analysis(self):
    if self.project is None or not self.project.get("pile") or not self.project.get("loads") or not self.project.get("soil_profile"):
        QMessageBox.warning(self, "Missing Data", "Edit Pile, loads, and add soil layers first.")
        return
    results = axial_analysis(self.project["pile"], self.project["loads"], self.project["soil_profile"])
    self.last_axial_results = results

    self._axial_figures = []

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
    ls_vbox.addWidget(NavigationToolbar(canvas_ls, ls_tab))
    ls_vbox.addWidget(canvas_ls)
    ls_vbox.addLayout(make_export_bar())
    self.plot_area.addTab(ls_tab, "Load-Settlement")

    self._axial_figures.append(fig_ls)

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
    sd_vbox.addWidget(NavigationToolbar(canvas_sd, sd_tab))
    sd_vbox.addWidget(canvas_sd)
    sd_vbox.addLayout(make_export_bar())
    self.plot_area.addTab(sd_tab, "Shear Vs Depth")

    self._axial_figures.append(fig_sd)

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
        QMessageBox.critical(self, "Invalid Input", "Pile length, diameter, or E must be positive.")
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
    
    def make_export_bar():
        bar = QHBoxLayout()
        btn_csv = QPushButton("Save Lateral CSV")
        btn_pdf = QPushButton("Save Lateral PDF")
        btn_csv.clicked.connect(self.export_lateral_csv)
        btn_pdf.clicked.connect(self.export_lateral_pdf)
        bar.addWidget(btn_csv)
        bar.addWidget(btn_pdf)
        bar.addStretch(1)
        return bar
    
    def layer_at_depth(depth_m: float):
        for layer in soil_layers:
            if layer.get("from_m", 0.0) <= depth_m < layer.get("to_m", 0.0):
                return layer
        return soil_layers[-1]
    
    def py_backbone(y_val: float, z_val: float) -> float:
        lyr = layer_at_depth(z_val)
        y_vals, p_vals_kNm = get_py_curve(lyr, D, z_val)
        
        if callable(y_vals) or callable(p_vals_kNm):
            print(f"[p-y] invalid curve (callable) at z={z_val:.2f} m, layer={lyr.get('type')}")
            return 0.0
        try:
            y_np = np.asarray(y_vals, dtype=float).ravel()
            p_np = np.asarray(p_vals_kNm, dtype=float).ravel()
        except Exception as e:
            print(f"[p-y] exception building curve at z={z_val:.2f} m, layer={lyr.get('type')}: {e}")
            return 0.0
        if y_np.size == 0 or p_np.size == 0 or y_np.size != p_np.size:
            print(f"[p-y] bad sizes at z={z_val:.2f} m (|y|={y_np.size}, |p|={p_np.size}), layer={lyr.get('type')}")
            return 0.0
        
        order = np.argsort(y_np)
        y_sorted = y_np[order]
        p_sorted = p_np[order]

        yq = float(np.clip(y_val, y_sorted[0], y_sorted[-1]))
        p_kNm = float(np.interp(yq, y_sorted, p_sorted))
        return p_kNm * 1000.0

    py_spring = make_py_spring(py_backbone)

    for z_test in [0.0, 0.5*L, 0.9*L]:
        p_test, k_test = py_spring(1e-3, z_test)
        print(f"[probe] z={z_test:.2f} m -> p={p_test:.1f} N/m, k={k_test:.1f} N/m^2")

    # Define Load steps (kN -> N)
    loads = (self.project.get("loads") or {})
    H_user_kN = float(loads.get("lateral_kN", 400.0))
    M_user_kNm = float(loads.get("moment_kNm", 0.0))
    M_user_Nm = M_user_kNm * 1e3

    grid = np.linspace(0.0, H_user_kN, 4)
    steps = [LateralLoadCase(H_N=float(h) * 1e3, M_Nm=M_user_Nm) for h in grid]

    # Solver configuration
    bc_str = self.project.get("analysis", {}).get("lateral_bc", "free_head")
    bc_enum = BCType.FREE_HEAD if bc_str == "free_head" else BCType.FIXED_HEAD

    cfg = LateralConfig(bc=bc_enum, max_iters=80, tol=1e-6, relax=0.8)

    # Run analysis
    out = lateral_analysis(pile, steps, py_spring, cfg)
    self.last_lateral_out = out
    self._lateral_figures = []

    pairs = out.get("head_curve", [])
    if not pairs:
        QMessageBox.warning(self, "No Data", "Lateral solver returned no head curve points.")
        return

    #------Plot Head Load-Deflection Curve-------
    H_kN = np.array([h for (h, y0) in pairs], dtype=float) / 1e3
    y_head_mm = np.array([y0 for (h, y0) in pairs], dtype=float) * 1e3

    fig1, ax1 = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ax1.plot(y_head_mm, H_kN, marker="o", linewidth=2)
    ax1.set_title(f"Lateral Load-Deflection (H up to {grid[-1]:.0f} kN)")
    ax1.set_xlabel("Head Deflection (mm)")
    ax1.set_ylabel("Applied Lateral Load H (kN)")
    ax1.grid(True, alpha=0.35)

    canvas1 = FigureCanvas(fig1)
    try:
        self._attach_export_context_menu(canvas1, kind="lateral")
    except TypeError:
        self._attach_export_context_menu(canvas1)
    tab1 = QWidget()
    vbox1 = QVBoxLayout(tab1)
    vbox1.setContentsMargins(6, 6, 6, 6)
    vbox1.addWidget(NavigationToolbar(canvas1, tab1))
    vbox1.addWidget(canvas1)
    vbox1.addLayout(make_export_bar())
    self.plot_area.addTab(tab1, "Lateral H-y")
    self._lateral_figures.append(fig1)
    

    #-------Plot Deflection vs Depth for Last Step---------
    if not out.get("steps"):
        QMessageBox.warning(self, "No Data", "No step results to plot.")
        return
    last = out["steps"][-1]
    print("Lateral post max |y|: ", float(np.max(np.abs(last.y_m))))
    print("Lateral post max |M| :(", float(np.max(np.abs(last.M_Nm))))
    print("Lateral post max |V|: ", float(np.max(np.abs(last.V_N))))

    fig2, ax2 = plt.subplots(figsize=(9, 5), constrained_layout=True)
    ax2.plot(last.y_m * 1e3, last.z_m, linewidth=2)
    ax2.invert_yaxis()
    ax2.set_xlabel("Deflection (mm)")
    ax2.set_ylabel("Depth (m)")
    ax2.grid(True, alpha=0.35)

    canvas2 = FigureCanvas(fig2)
    try:
        self._attach_export_context_menu(canvas2, kind="lateral")
    except TypeError:
        self._attach_export_context_menu(canvas2)
    tab2 = QWidget()
    vbox2 = QVBoxLayout(tab2)
    vbox2.setContentsMargins(6, 6, 6, 6)
    vbox2.addWidget(NavigationToolbar(canvas2, tab2))
    vbox2.addWidget(canvas2)
    vbox2.addLayout(make_export_bar())
    self.plot_area.addTab(tab2, "Deflection vs Depth")
    self._lateral_figures.append(fig2)

    self.plot_area.setCurrentWidget(tab1)

    #------Moment & Shear vs Depth (derived from deflection)-------
    z = last.z_m
    y = last.y_m
    M = last.M_Nm
    V = last.V_N
    
    # Moment vs Depth
    figM, axM = plt.subplots(figsize=(9, 5), constrained_layout=True)
    axM.plot(M / 1e3, z, linewidth=2)
    axM.invert_yaxis()
    axM.set_xlabel("Moment (kN.m)")
    axM.set_ylabel("Depth (m)")
    axM.set_title("Moment vs Depth")
    axM.grid(True, alpha=0.35)

    canvasM = FigureCanvas(figM)
    self._attach_export_context_menu(canvasM, kind="lateral")
    tabM = QWidget()
    vM = QVBoxLayout(tabM)
    vM.setContentsMargins(6,6,6,6)
    vM.addWidget(NavigationToolbar(canvasM, tabM))
    vM.addWidget(canvasM)
    vM.addLayout(make_export_bar())
    self.plot_area.addTab(tabM, "Moment vs Depth")
    self._lateral_figures.append(figM)

    # Shear vs Depth
    figV, axV = plt.subplots(figsize=(9, 5), constrained_layout=True)
    axV.plot(V / 1e3, z, linewidth=2)
    axV.invert_yaxis()
    axV.set_xlabel("Shear (kN)")
    axV.set_ylabel("Depth (m)")
    axV.set_title("Shear vs Depth")
    axV.grid(True, alpha=0.35)

    canvasV = FigureCanvas(figV)
    self._attach_export_context_menu(canvasV, kind="lateral")
    tabV = QWidget()
    vV = QVBoxLayout(tabV)
    vV.setContentsMargins(6,6,6,6)
    vV.addWidget(NavigationToolbar(canvasV, tabV))
    vV.addWidget(canvasV)
    vV.addLayout(make_export_bar())
    self.plot_area.addTab(tabV, "Shear vs Depth (Lateral)")
    self._lateral_figures.append(figV)

  #---------Right click export menu------------------
  def _attach_export_context_menu(self, widget: QWidget, kind: str = "axial"):
      widget.setContextMenuPolicy(Qt.CustomContextMenu)
      widget.customContextMenuRequested.connect(
          lambda pos, w=widget, k=kind: self._show_export_menu(w, pos, k)
      )  

  def open_3d_view(self):
    """create/show the 3D pile viewer dock and render the current projects."""
    if self._dock3d is None:
        self._dock3d = QDockWidget("3D View", self)
        self._dock3d.setObjectName("ThreeDViewDock")
        self._dock3d.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)

        host = QWidget()
        v = QVBoxLayout(host)
        v.setContentsMargins(6,6,6,6)

        # controls row
        ctrl = QWidget()
        fl = QFormLayout(ctrl)
        fl.setContentsMargins(0,0,0,0)
        self._3d_scale_spin = QDoubleSpinBox()
        self._3d_scale_spin.setRange(1.0, 1000.0)
        self._3d_scale_spin.setDecimals(1)
        self._3d_scale_spin.setValue(50.0)
        self._3d_scale_spin.setSuffix(" x")
        self._3d_scale_spin.valueChanged.connect(self._refresh_3d_view)
        fl.addRow("Deflection scale:", self._3d_scale_spin)
        ctrl.setLayout(fl)
        v.addWidget(ctrl)

        # matplotlib 3D figure
        self._3d_fig = plt.figure(figsize=(6.5, 6), constrained_layout=True)
        self._3d_ax = self._3d_fig.add_subplot(111, projection='3d')
        self._3d_canvas = FigureCanvas(self._3d_fig)
        v.addWidget(NavigationToolbar(self._3d_canvas, host))
        v.addWidget(self._3d_canvas)

        host.setLayout(v)
        self._dock3d.setWidget(host)
        self.addDockWidget(Qt.RightDockWidgetArea, self._dock3d)

    self._dock3d.show()
    self._dock3d.raise_()
    self._refresh_3d_view()

  def _refresh_3d_view(self):
      """Render pile  + soil + optional lateral deflection in the 3D axes."""
      if self._3d_ax is None:
          return
      self._3d_ax.clear()

      # read inputs
      proj = self.project or {}
      pile = proj.get("pile", {})
      layers = proj.get("soil_profile", [])

      try:
          L = float(pile.get("length_m", 0.0))
          D = float(pile.get("diameter_m", 0.0))
      except (TypeError, ValueError):
          L, D = 0.0, 0.0

      if L <= 0.0 or D <= 0.0:
          self._3d_ax.text2D(0.05, 0.95, "Define pile (L, D) to view 3D", transform=self._3d_ax.transAxes)
          self._3d_canvas.draw()
          return
      
      R = 0.5 * D
      n_theta = 120
      theta = np.linspace(0, 2*np.pi, n_theta)

      # soil colors
      def soil_color(layer):
          typ = (layer.get("type") or "").lower()
          if typ == "clay":
              return (0.55, 0.37, 0.3, 0.55)  #brownish, semi-transparent
          if typ == "sand":
              return (0.95, 0.82, 0.52, 0.45)  #sandy yellow
          return (0.7, 0.7, 0.7, 0.5)  #default gray
      
      seen_soil = set() # track which soil types appear for the legend
      
      # draw soil layers as translucent rectangular bands
      half = 2.5 * D
      for Lyr in layers:
          try:
              z0 = float(Lyr.get("from_m", 0.0))
              z1 = float(Lyr.get("to_m", 0.0))
          except (TypeError, ValueError):
              continue
          if z1 <= z0:
              continue
          c = soil_color(Lyr)
          typ = (Lyr.get("type") or "").lower()
          seen_soil.add(typ)
          Xs = np.array([[-half, -half], [half, half]])
          Ys = np.array([[-half, half], [-half, half]])
          Z0 = -z0 * np.ones_like(Xs)
          Z1 = -z1 * np.ones_like(Xs)
          # four sides
          self._3d_ax.plot_surface(-half*np.ones_like(Xs), Ys, np.vstack([Z0[0], Z1[0]]), alpha=c[3], color=c[:3], linewidth=0, shade=True)
          self._3d_ax.plot_surface(half*np.ones_like(Xs), Ys, np.vstack([Z0[0], Z1[0]]), alpha=c[3], color=c[:3], linewidth=0, shade=True)
          self._3d_ax.plot_surface(Xs, -half*np.ones_like(Xs), np.vstack([Z0[0], Z1[0]]), alpha=c[3], color=c[:3], linewidth=0, shade=True)
          self._3d_ax.plot_surface(Xs, half*np.ones_like(Xs), np.vstack([Z0[0], Z1[0]]), alpha=c[3], color=c[:3], linewidth=0, shade=True)
          #top cap (horizontal plane at z0)
          Zcap_top = -z0 * np.ones_like(Xs)
          self._3d_ax.plot_surface(
              Xs, Ys, Zcap_top,
              alpha=max(0.15, c[3]-0.2),
              color=c[:3],
              linewidth=0,
              shade=False
          )

      # draw pile cylinder (undeformed)
      z_line = np.linspace(0.0, L, 260)
      Z = -z_line[None, :]
      Xc = (R * np.cos(theta))[:, None]
      Yc = (R * np.sin(theta))[:, None]
      X = np.repeat(Xc, Z.shape[1], axis=1)
      Y = np.repeat(Yc, Z.shape[1], axis=1)
      Z = np.repeat(Z, X.shape[0], axis=0)
      self._3d_ax.plot_surface(X, Y, Z, color=(0.75, 0.78, 0.84), alpha=0.98, rstride=1, cstride=1, linewidth=0.0, antialiased=True, shade=True)

      # demo: overlay lateral deflection of last step, scaled
      scale = float(self._3d_scale_spin.value() if self._3d_scale_spin else 50.0)
      y_defl = None
      z_defl = None
      if getattr(self, "last_lateral_out", None):
          steps = self.last_lateral_out.get("steps", [])
          if steps:
              last = steps[-1]
              z_defl = np.asarray(getattr(last, "z_m", []), dtype=float).reshape(-1)
              y_defl = np.asarray(getattr(last, "y_m", []), dtype=float).reshape(-1)

      if y_defl is not None and z_defl is not None and z_defl.size > 2 and y_defl.size == z_defl.size:
          xd = scale * y_defl
          # centerline
          self._3d_ax.plot(xd, np.zeros_like(xd), -z_defl, lw=2.2, color="C3", label="Deflected centerline")
          # deflected cylinder (translate x by deflection)
          y_interp = np.interp(-Z[0, :], -z_defl, xd)
          Xdef = X + y_interp[None, :]
          self._3d_ax.plot_surface(Xdef, Y, Z, color=(0.92, 0.55, 0.55), alpha=0.35, rstride=1, cstride=6, linewidth=0, antialiased=True, shade=True)

      # ground plane
      gx = np.linspace(-2*D, 2*D, 10)
      gy = np.linspace(-2*D, 2*D, 10)
      GX, GY = np.meshgrid(gx, gy)
      GZ = np.zeros_like(GX)
      self._3d_ax.plot_surface(GX, GY, GZ, alpha=0.15, color=(0.5, 0.7, 0.95))

      # axes styling
      self._3d_ax.set_xlabel("X (m)")
      self._3d_ax.set_ylabel("Y (m)")
      self._3d_ax.set_zlabel("Z (m)")
      self._3d_ax.set_title("Pile 3D view (soil layers + pile; deflection scaled)")

      span = max(2*D, L/12)
      self._3d_ax.set_xlim(-span, span)
      self._3d_ax.set_ylim(-span, span)
      self._3d_ax.set_zlim(-L, 0)
      self._3d_ax.set_box_aspect((1, 1, max(1.0, L/(3*D))))

      # Reduce clutter on lateral axes; keep z readable
      self._3d_ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
      self._3d_ax.yaxis.set_major_locator(MaxNLocator(nbins=3))

      # z every ~2 m (adjust automatically with nbins)
      self._3d_ax.zaxis.set_major_locator(MaxNLocator(nbins=9))
      self._3d_ax.zaxis.set_major_formatter(FormatStrFormatter('%g'))

      # Give Z label a little air so it does not collide with tick labels
      self._3d_ax.set_zlabel("Z (m)", labelpad=12)

      # Slightly smaller tick text; pull numbers off the panes a touch
      self._3d_ax.tick_params(axis='both', which='major', labelsize=8, pad=4)
      self._3d_ax.tick_params(axis='z', which='major', labelsize=9, pad=6)

      # Softer panes so labels pop
      self._3d_ax.xaxis.pane.set_alpha(0.04)
      self._3d_ax.yaxis.pane.set_alpha(0.04)
      self._3d_ax.zaxis.pane.set_alpha(0.06)

      # Legend
      handles = []

      # Soil patches only for soil present
      if "clay" in seen_soil:
          handles.append(mpatches.Patch(facecolor=(0.55,0.37,0.3), alpha=0.55, edgecolor='none', label='Clay'))
      if "sand" in seen_soil:
          handles.append(mpatches.Patch(facecolor=(0.95,0.82,0.52), alpha=0.45, edgecolor='none', label='Sand'))

      # Pile (undeformed) proxy
      handles.append(mpatches.Patch(facecolor=(0.75,0.78,0.84), alpha=0.95, edgecolor='none', label='Pile (undeformed)'))

      # Deflected pile + centerline proxies (only if we drew them)
      if y_defl is not None and z_defl is not None and z_defl.size > 2 and y_defl.size == z_defl.size:
          handles.append(mpatches.Patch(facecolor=(0.92,0.55,0.55), alpha=0.35, edgecolor='none', label='Pile (deflected x{})'.format(int(scale))))
          handles.append(Line2D([0], [0], color="C3", lw=2.2, label='Deflected centerline'))
      
      # Place legend in the upper-left corner inside the axes pane
      self._3d_ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(0.2,0.98), fontsize=8, frameon=True)
      
      self._3d_canvas.draw()
          

  #------include axial analysis graphs in pdf report-------
  def _figure_to_imagereader(self, fig, dpi=200):
      buf = io.BytesIO()
      fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
      buf.seek(0)
      return ImageReader(buf)
      
  def _show_export_menu(self, widget: QWidget, pos: QPoint, kind: str):
      menu = QMenu(widget)
      if kind == "lateral":
          a_csv = menu.addAction("Export Lateral CSV")
          a_pdf = menu.addAction("Export Lateral PDF")
      else:
          a_csv = menu.addAction("Export Axial CSV")
          a_pdf = menu.addAction("Export Axial PDF")
      chosen = menu.exec(widget.mapToGlobal(pos))
      if chosen == a_csv:
        (self.export_lateral_csv if kind == "lateral" else self.export_axial_csv)()
      elif chosen == a_pdf:
        (self.export_lateral_pdf if kind == "lateral" else self.export_axial_pdf)()
      

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

  def export_lateral_csv(self):
    if not getattr(self, "last_lateral_out", None):
        QMessageBox.information(self, "Export Lateral CSV", "Run Lateral Analysis First.")
        return

    out = self.last_lateral_out
    steps = out.get("steps", [])
    if not steps:
        QMessageBox.information(self, "Export Lateral CSV", "No step results to export.")
        return
    
    def _get(step, key, default=None):
        if isinstance(step, dict):
            return step.get(key, default)
        return getattr(step, key, default)

    # last step arrays
    last = steps[-1]
    z = np.asarray(getattr(last, "z_m", []), dtype=float)
    y = np.asarray(getattr(last, "y_m", []), dtype=float)
    slope = np.asarray(getattr(last, "theta_rad", []), dtype=float)
    M = np.asarray(getattr(last, "M_Nm", []), dtype=float)
    V = np.asarray(getattr(last, "V_N", []), dtype=float)
    p = np.asarray(getattr(last, "p_N_per_m", []), dtype=float)

    if z.size == 0:
        QMessageBox.critical(self, "Export Lateral CSV", "Solver returned no depth array. Re-run analysis")
        return

    def _clean(a, n_ref):
        a = np.asarray(a, dtype=float).reshape(-1)
        if a.size != n_ref:
            out = np.zeros(n_ref, dtype=float)
            out[:min(a.size, n_ref)] = a[:min(a.size, n_ref)]
            a = out
        return np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    

    n = z.size
    y = _clean(y, n)
    slope = _clean(slope, n)
    M = _clean(M, n)
    V = _clean(V, n)
    p = _clean(p, n)

    # EI for curvature calculation
    EI = float(out.get("meta", {}).get("EI_Nm2", 0.0))
    if not np.isfinite(EI) or EI <= 0.0:
        # fallback from current pile if meta is missing
        pile = (self.project or {}).get("pile", {})
        D = float(pile.get("diameter_m", 0.0))
        E = float(pile.get("elastic_modulus_pa", 0.0))
        I = (np.pi * (D ** 4)) / 64.0 if D > 0 else 0.0
        EI = E * I if (E > 0 and I > 0) else 1.0

    curvature = np.nan_to_num(M / EI, nan=0.0, posinf=0.0, neginf=0.0)

    df = pd.DataFrame({
        "Depth_m": z,
        "Deflection_m": y,
        "Slope_rad": slope,
        "Curvature_1_per_m": curvature,
        "Moment_Nm": M,
        "Shear_N": V,
        "SoilReaction_N_per_m": p,
    })

    path, _ = QFileDialog.getSaveFileName(self, "Save Lateral Results CSV", "lateral_results_csv", "CSV (*.csv)")
    if not path:
        return
    if not path.lower().endswith(".csv"):
        path += ".csv"

    try:
        df.to_csv(path, index=False)
        self.statusBar().showMessage(f"Saved: {path}", 5000)
    except Exception as e:
        QMessageBox.critical(self, "Save Failed", f"Could not save CSV:\n{e}")

  def export_axial_pdf(self):
      if not getattr(self, "last_axial_results", None):
          QMessageBox.warning(self, "Export Axial PDF", "Run Axial Analysis first.")
          return
      
      # Where to save
      default_name = "axial_analysis.pdf"
      path, _ = QFileDialog.getSaveFileName(self, "Save Axial PDF", default_name, "PDF Files (*.pdf)")
      if not path:
          return
      
      c = rl_canvas.Canvas(path, pagesize=letter)
      width, height = letter
      margin = 0.75 * inch
      cursor_y = height - margin

      # Header
      c.setFont("Helvetica-Bold", 14)
      c.drawString(margin, cursor_y, "Axial Load-Settlement Analysis")
      cursor_y -= 18

      # Text helper
      c.setFont("Helvetica", 10)
      def line(txt):
          nonlocal cursor_y
          c.drawString(margin, cursor_y, txt)
          cursor_y -= 12

      # Gather inputs
      meta = self.project.get("meta", {})
      pile = self.project.get("pile", {})
      loads = self.project.get("loads", {})
      layers = self.project.get("soil_profile", [])

      units = meta.get("units", "SI")

      E = pile.get("elastic_modulus_pa")
      E_txt = f"{E/1e9:.2f} GPa" if E else "-"
      gamma = pile.get("unit_weight_kNpm3", "-")

      moment_kNm = loads.get("moment_kNm", loads.get("moment_kN", "-"))

      # Inputs summary
      line(f"Units: {units}")
      line(
          f"Pile: L = {pile.get('length_m', '-')} m, "
          f"D = {pile.get('diameter_m', '-')} m, "
          f"E = {E_txt}, "
          f"γ = {gamma} kN/m³"
      )
      line (
          f"Loads: Axial = {loads.get('axial_kN', '-')} kN, "
          f"Lateral = {loads.get('lateral_kN', '-')} kN, "
          f"Moment = {moment_kNm} kN.m"
      )

      # Soil layers
      if layers:
          line("Soil Layers:")
          for i, L in enumerate(layers, 1):
              soil_type = (L.get("type") or "?").title()
              gL = L.get("gamma_kNpm3", "-")
              if L.get("type") == "clay":
                  strength = f"su = {L.get('undrained_shear_strength_kPa', '-')} kPa"
              else:
                  strength = f"φ = {L.get('phi_deg', '-')}°"
              line(
                  f" {i}) {soil_type}, "
                  f"{L.get('from_m', '?')}-{L.get('to_m', '?')} m, "
                  f"γ = {gL} kN/m³, {strength}"
              )
      else:
          line("Soil Layers: (none defined)")

      cursor_y -= 4
      line("Notes: Results generated from the project inputs listed above.")
      cursor_y -= 10

      # Draw axial figures captured during run
      figs = list(getattr(self, "_axial_figures", []))
      if not figs:
          try:
              fig, ax = plt.subplots(figsize=(6,4))
              s = self.last_axial_results.get("settlements_m", [])
              q = self.last_axial_results.get("loads_kN", [])
              # Convert settlement to mm for a nice axis
              s_mm = [val*1000.0 for val in s]
              ax.plot(s_mm, q)
              ax.set_xlabel("Settlement s (mm)")
              ax.set_ylabel("Load Q (kN)")
              ax.set_title("Q-s Curve")
              ax.grid(True)
              figs = [fig]
          except Exception:
              pass
          
      if not figs:
          cursor_y -= 14
          c.setFont("Helvetica-Oblique", 10)
          c.drawString(margin, cursor_y, "No plots were generated; run Axial Analysis to generate curves")
          c.showPage()
          c.save()
          QMessageBox.information(self, "Export Axial PDF", f"Saved: {path}")
          return
      
      # Layout: one figure per page (scaled to fit within margins)
      max_plot_w = width - 2 * margin
      max_plot_h = height - 2 * margin - 80
      for idx, fig in enumerate(figs, start=1):
          # Start a new page for each figure except the first
          if idx > 1:
              c.showPage()
              cursor_y = height - margin
              c.setFont("Helvetica-Bold", 14)
              c.drawString(margin, cursor_y, "Axial Load-Settlement Analsysis")
              cursor_y -= 24

          img = self._figure_to_imagereader(fig, dpi=220)

          # Compute image aspect and scale to fit
          iw, ih = fig.get_size_inches()
          aspect = (ih / iw)
          draw_w = max_plot_w
          draw_h = draw_w * aspect
          if draw_h > max_plot_h:
              draw_h = max_plot_h
              draw_w = draw_h / aspect

          x = margin + (max_plot_w - draw_w) / 2.0
          y = margin + (max_plot_h - draw_h) / 2.0

          c.drawImage(img, x, y, width=draw_w, height=draw_h, preserveAspectRatio=True, anchor='c')
    
      c.showPage()
      c.save()
      
      QMessageBox.information(self, "Export Axial PDF", f"Saved: {path}")


  def export_lateral_pdf(self):
      if not getattr(self, "last_lateral_out", None):
          QMessageBox.warning(self, "Export Lateral PDF", "Run Lateral Analysis first.")
          return
      
      path, _ = QFileDialog.getSaveFileName(self, "Save Lateral PDF", "lateral_analysis.pdf", "PDF (*.pdf)")
      if not path:
          return
      
      c = rl_canvas.Canvas(path, pagesize=letter)
      width, height = letter
      margin = 0.75 * inch
      cursor_y = height - margin

      # Header
      c.setFont("Helvetica-Bold", 14)
      c.drawString(margin, cursor_y, "Lateral Analysis Report")
      cursor_y -= 18

      # Text Helper
      c.setFont("Helvetica", 10)
      def line(txt):
          nonlocal cursor_y
          c.drawString(margin, cursor_y, txt)
          cursor_y -= 12

      # Gather inputs
      meta = self.project.get("meta", {})
      pile = self.project.get("pile", {})
      loads = self.project.get("loads", {})
      layers = self.project.get("soil_profile", [])

      units = meta.get("units", "SI")

      E = pile.get("elastic_modulus_pa")
      E_txt = f"{E/1e9:.2f} GPa" if E else "-"
      gamma = pile.get("unit_weight_kNpm3", "-")

      moment_kNm = loads.get("moment_kNm", loads.get("moment_kN", "-"))

      # Inputs summary
      line(f"Units: {units}")
      line(
          f"Pile: L = {pile.get('length_m', '-')} m, "
          f"D = {pile.get('diameter_m', '-')} m, "
          f"E = {E_txt}, "
          f"γ = {gamma} kN/m³"
      )
      line (
          f"Loads: Axial = {loads.get('axial_kN', '-')} kN, "
          f"Lateral = {loads.get('lateral_kN', '-')} kN, "
          f"Moment = {moment_kNm} kN.m"
      )

      # Soil layers
      if layers:
          line("Soil Layers:")
          for i, L in enumerate(layers, 1):
              soil_type = (L.get("type") or "?").title()
              gL = L.get("gamma_kNpm3", "-") 
              if L.get("type") == "clay":
                  strength = f"su = {L.get('undrained_shear_strength_kPa', '-')} kPa"
              else:
                  strength = f"φ = {L.get('phi_deg', '-')}°"
              line(
                  f" {i}) {soil_type}, "
                  f"{L.get('from_m', '?')}-{L.get('to_m', '?')} m, "
                  f"γ = {gL} kN/m³, {strength}"
              )
      else:
          line("Soil Layers: (none defined)")

      cursor_y -= 4
      line("Notes: Results generated from the project inputs listed above.")
      cursor_y -= 10


      # Figures captured during lateral analysis
      figs = list(getattr(self, "_lateral_figures", []))

      # If none captured, synthesize key plots from last step
      if not figs and self.last_lateral_out.get("steps"):
          last = self.last_lateral_out["steps"][-1]
          z = np.asarray(last.z_m, dtype=float)
          y = np.asarray(last.y_m, dtype=float)

          # H-y (head) curve if present
          pairs = self.last_lateral_out.get("head_curve", [])
          if pairs:
              figHy, axHy = plt.subplots(figsize=(6, 4))
              H_kN = np.array([h for (h, y0) in pairs]) / 1e3
              y_mm = np.array([y0 for (h, y0) in pairs]) * 1e3
              axHy.plot(y_mm, H_kN, marker="o")
              axHy.set_xlabel("Head Deflection (mm)")
              axHy.set_ylabel("Head Load H (kN)")
              axHy.set_title("Lateral H-y")
              axHy.grid(True)
              figs.append(figHy)

          # Deflection, Moment, Shear
          if z.size >= 3:
              dz = float(np.mean(np.diff(z)))
              dy_dz = np.gradient(y, dz)
              d2y_dz2 = np.gradient(dy_dz, dz)
              D = float(pile.get("diameter_m", 0.0))
              E = float(pile.get("elastic_modulus_pa", 0.0))
              I = 3.14159265358979 * (D ** 4) / 64.0
              EI = E * I
              M = -EI * d2y_dz2
              V = np.gradient(M, dz)

              figDefl, axDefl = plt.subplots(figsize=(6,4))
              axDefl.plot(y * 1e3, z)
              axDefl.invert_yaxis()
              axDefl.set_xlabel("Deflection (mm)")
              axDefl.set_ylabel("Depth (m)")
              axDefl.set_title("Deflection vs Depth")
              axDefl.grid(True)
              figs.append(figDefl)

              figM, axM = plt.subplots(figsize=(6,4))
              axM.plot(M / 1e3, z)
              axM.invert_yaxis()
              axM.set_xlabel("Moment (kN.m)")
              axM.set_ylabel("Depth (m)")
              axM.set_title("Moment vs Depth")
              axM.grid(True)
              figs.append(figM)

              figV, axV = plt.subplots(figsize=(6, 4))
              axV.plot(V / 1e3, z)
              axV.invert_yaxis()
              axV.set_xlabel("Shear (kN)")
              axV.set_ylabel("Depth (m)")
              axV.set_title("Shear vs Depth")
              axV.grid(True)
              figs.append(figV)

      if not figs:
        cursor_y -= 14
        c.setFont("Helvetica-Oblique", 10)
        c.drawString(margin, cursor_y, "No plots were generated. Run Lateral Analysis to generate curves.")
        c.showPage()
        c.save()
        QMessageBox.information(self, "Export Lateral PDF", f"Saved: {path}")
        return
      
      # draw figures (one per page)
      max_plot_w = width - 2 * margin
      max_plot_h = height - 2 * margin - 80

      for idx, fig in enumerate(figs, start=1):
          if idx > 1:
              c.showPage()
              cursor_y = height - margin
              c.setFont("Helvetica-Bold", 14)
              c.drawString(margin, cursor_y, "Lateral Analysis Report")
              cursor_y -= 24

          img = self._figure_to_imagereader(fig, dpi=220)
          iw, ih = fig.get_size_inches()
          aspect = ih / iw
          draw_w = max_plot_w
          draw_h = draw_w * aspect
          if draw_h > max_plot_h:
              draw_h = max_plot_h
              draw_w = draw_h / aspect

          x = margin + (max_plot_w - draw_w) / 2.0
          y = margin + (max_plot_h - draw_h) / 2.0
          c.drawImage(img, x, y, width=draw_w, height=draw_h, preserveAspectRatio=True)

      c.showPage()
      c.save()
      QMessageBox.information(self, "Export Lateral PDF", f"Saved: {path}")


      
  
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
            self._set_dirty(True)
          except ValueError as e:
              QMessageBox.critical(self, "Input Error", str(e))

  def edit_loads(self):
      self._ensure_project()
      dlg = LoadDialog(self.project.get("loads", {}), self)
      if dlg.exec():
          self.project["loads"] = dlg.result_data()
          self.refresh_ui()
          self._set_dirty(True)

  def add_soil_layer(self):
      self._ensure_project()
      dlg = SoilLayerDialog(parent=self)
      if dlg.exec():
          layer = dlg.result_data()
          self.project.setdefault("soil_profile", []).append(layer)
          # keeping layers ordered by start depth
          self.project["soil_profile"].sort(key=lambda L: L.get("from_m", 0.0))
          self.refresh_ui()
          self._set_dirty(True)

  def edit_soil_layer(self, index: int):
      """Simplest edit: open dialog blank, then replace layer."""
      self._ensure_project()
      layers = self.project.setdefault("soil_profile", [])
      if index < 0 or index >= len(layers):
          return
      
      current_layer = layers[index]
      
      dlg = SoilLayerDialog(current_layer, self)
      if dlg.exec():
          new_layer = dlg.result_data()
          layers[index] = new_layer
          layers.sort(key=lambda L: L.get("from_m", 0.0))
          self.refresh_ui()
          self._set_dirty(True)

  def delete_soil_layer(self, index: int):
      self._ensure_project()
      layers = self.project.setdefault("soil_profile", [])
      if index < 0 or index >= len(layers):
          return
      
      L = layers[index]
      typ = (L.get("type") or "soil").title()
      z0 = L.get("from_m", 0)
      z1 = L.get("to_m", 0)

      ok = QMessageBox.question(
          self, "Delete Soil Layer",
          f"Delete layer {index+1}: {typ} ({z0:g}-{z1:g} m)?",
          QMessageBox.Yes | QMessageBox.No
      )
      if ok != QMessageBox.Yes:
          return
      
      layers.pop(index)
      self.refresh_ui()
      self._set_dirty(True)
    
  def closeEvent(self, e):
      s = QSettings("Pile Analysis", "StudentEdition")
      s.setValue("win/geo", self.saveGeometry())
      s.setValue("win/state", self.saveState(1))
      super().closeEvent(e)

  def _close_plot_tab(self, index: int):
      w = self.plot_area.widget(index)
      self.plot_area.removeTab(index)
      if w is not None:
          w.deleteLater()

  def _add_plot_tab(self, fig, title: str):
      """Warap a Matplotlib fig with a canvas + toolbar and add as a tab"""
      try:
          fig.tight_layout()
      except Exception:
          pass
      
      canvas = FigureCanvas(fig)
      wrap = QWidget()
      v = QVBoxLayout(wrap)
      v.setContentsMargins(6,6,6,6)

      toolbar = NavigationToolbar(canvas, wrap)
      v.addWidget(toolbar)
      v.addWidget(canvas)

      idx = self.plot_area.addTab(wrap, title)
      self.plot_area.setCurrentIndex(idx)
      return canvas, toolbar
  
  def _rebuild_layers_list(self):
      """Fill left dock soil list with Edit/Delete buttons."""
      if not hasattr(self, "layers_list"):
          return
      
      self.layers_list.clear()
      layers = (self.project or {}).get("soil_profile", [])

      if not layers:
          item = QListWidgetItem("(no soil layers)")
          item.setFlags(Qt.NoItemFlags)
          self.layers_list.addItem(item)
          return
      
      for idx, layer in enumerate(layers):
          row = QWidget()
          h = QHBoxLayout(row)
          h.setContentsMargins(6, 2, 6, 2)
          h.setSpacing(6)

          typ = (layer.get("type") or "soil").title()
          z0 = layer.get("from_m", 0)
          z1 = layer.get("to_m", 0)

          lbl = QLabel(f"{idx+1}. {typ} ({z0:g}-{z1:g} m)")
          lbl.setStyleSheet("font-size:12px;")

          btn_edit = QToolButton()
          btn_edit.setText("Edit")
          btn_edit.setAutoRaise(True)
          btn_edit.clicked.connect(lambda _=False, i=idx: self.edit_soil_layer(i))

          btn_del = QToolButton()
          btn_del.setText("Delete")
          btn_del.setAutoRaise(True)
          btn_del.clicked.connect(lambda _=False, i=idx: self.delete_soil_layer(i))

          h.addWidget(lbl)
          h.addStretch(1)
          h.addWidget(btn_edit)
          h.addWidget(btn_del)

          item = QListWidgetItem()
          item.setSizeHint(row.sizeHint())
          self.layers_list.addItem(item)
          self.layers_list.setItemWidget(item, row)

  def _on_summary_link_clicked(self, url: QUrl):
      """Handles clicks from Project Summary soil-layer actions."""
      if self.project is None:
          return
      
      s = url.toString()

      try:
          if s.startswith("edit-layer:"):
              idx = int(s.split("edit-layer:")[1])
              self.edit_soil_layer(idx)
          
          elif s.startswith("del-layer:"):
              idx = int(s.split("del-layer:")[1])
              self.delete_soil_layer(idx)

      except Exception as e:
          print("summary link error:", e)

      self.refresh_ui()

  #-----UI helper-------
  def refresh_ui(self):
            if self.project is None:
              self.info.hide()
              self.plot_area.hide()
              self.btn_gen.hide()
              self.welcome.show()
              self._lbl_meta.setText("No Projects Loaded.")
              if hasattr(self, "layers_list"):
                  self.layers_list.clear()
              return
            
            # existing summary
            self.welcome.hide()
            self.info.show()
            self.plot_area.show()
            self.btn_gen.show()
            
            # Show a simple summary of the current project
            meta = self.project.get("meta", {})
            units = meta.get("units", "SI")
            pile = self.project.get("pile", {})
            loads = self.project.get("loads", {})
            layers = self.project.get("soil_profile", [])
            
            is_dark = getattr(self, "_current_theme", "light") == "dark"
            fg = "#e9edf5" if is_dark else "#111827"
            muted = "#a6aec2" if is_dark else "#555"
            tip = "#9aa4c7" if is_dark else "#888"
            line_col = "#596073" if is_dark else "#d0d7e2"
            header_col = "#8ab4ff" if is_dark else "#2b6cb0"

            self._lbl_meta.setText(
                f"<b>Units</b>: {meta.get('units', 'SI')}<br>"
                f"<b>Pile</b>: {('✓' if pile else '—')} &nbsp; "
                f"<b>Loads</b>: {('✓' if loads else '—')} &nbsp; "
                f"<b>Layers</b>: {len(layers)}"
            )

            # Pile summary
            if pile:
                E = pile.get("elastic_modulus_pa")
                E_txt = f"{E/1e9:.1f} GPa" if E else "-"
                pile_text = (
                    f"L = {pile.get('length_m', '-')} m &nbsp;|&nbsp; "
                    f"D = {pile.get('diameter_m', '-')} m &nbsp;|&nbsp; "
                    f"E = {E_txt} &nbsp;|&nbsp; "
                    f"γ = {pile.get('unit_weight_kNpm3', '-')} kN/m³ &nbsp;|&nbsp;"
                )
            else:
                pile_text = "<span style='color:#888;'>(not defined)</span>"

            # Loads summary
            if loads:
                loads_text = (
                    f"Axial = {loads.get('axial_kN', 0)} kN &nbsp;|&nbsp; "
                    f"Lateral = {loads.get('lateral_kN', 0)} kN &nbsp;|&nbsp; "
                    f"Moment = {loads.get('moment_kNm', 0)} kN.m"
                )
            else:
                loads_text = "<span style='color:#888;'>(not defined)</span>"

            # Soil layers table
            layer_rows = ""
            for i, L in enumerate(layers, 1):
                if L.get("type") == "clay":
                    soil_type = "Clay"
                    strength = f"su = {L.get('undrained_shear_strength_kPa', 0)} kPa"
                else:
                    soil_type = "Sand"
                    strength = f"φ = {L.get('phi_deg', 0)}°"  

                actions_html = (
                    f"<a href='edit-layer:{i-1}' style='"
                    "display: inline-block; padding: 2px 8px; border: 1px solid #777; border-radius: 6px;"
                    "text-decoration: none; font-size: 13px; margin-right: 10px;'>Edit</a>"
                    "&nbsp;&nbsp;&nbsp;"
                    f"<a href='del-layer:{i-1}' style='"
                    "display: inline-block; padding: 2px 8px; border: 1px solid #b55; border-radius: 6px;"
                    "text-decoration: none; font-size: 13px; color: #b55;'>Delete</a>"
                )

                layer_rows += (
                    "<tr>"
                    f"<td style='padding: 2px 18px 2px 0;'>{i}</td>"
                    f"<td style='padding: 2px 18px 2px 0;'>{soil_type}</td>"
                    f"<td style='padding: 2px 18px 2px 0;'>{L.get('from_m', 0):g}-{L.get('to_m', 0):g} m</td>"
                    f"<td style='padding: 2px 18px 2px 0;'>{L.get('gamma_kNpm3', 0):g} kN/m³</td>"
                    f"<td style='padding: 2px 18px 2px 0;'>{strength}</td>"
                    f"<td style='padding: 2px 0 2px 0;'>{actions_html}</td>"
                    "</tr>"
                )    

            layers_section = ""
            if layers:
                layers_section = f"""
                    <table style="border-collapse:collapse; margin-top: 8px; font-size: 16px;">
                        <tr style="font-weight:600; color:{muted};">
                            <th align="left" style="padding: 0 18px 4px 0;">#</th>
                            <th align="left" style="padding: 0 18px 4px 0;">Type</th>
                            <th align="left" style="padding: 0 18px 4px 0;">Depth</th>
                            <th align="left" style="padding: 0 18px 4px 0;">γ</th>    
                            <th align="left" style="padding: 0 18px 4px 0;">Strength</th>
                            <th align="left">Actions</th>
                        </tr>
                        {layer_rows}
                    </table>"""
            
            box_bg = "#2f3340" if is_dark else "#ffffff"
            box_border = "#596073" if is_dark else "#d0d7e2"

            html = f"""
                <div style="color:{fg}; background: {box_bg}; border: 1px solid {box_border}; border-radius: 1opx; padding: 10px;">
                    <div style="font-size: 13px; font-weight: 600; letter-spacing: 0.5px; color: {header_col}; margin-bottom: 2px;">
                        Project Summary
                    </div>
                    <div style="height: 1px; background: {line_col}; margin-bottom: 6px;"></div>

                    <table style="border-collapse: collapse; margin: 4px 0 8px 0; font-size: 16px; color: {fg};">
                        <tr>
                            <th align="left" style="padding: 4px 18px 4px 0; color: {muted};">Units</th>
                            <td style="padding: 4px 0;">{units}</td>
                        </tr>
                        <tr>
                            <th align="left" style="padding: 4px 18px 4px 0; color: {muted};">Pile</th>
                            <td style="padding: 4px 0;">{pile_text}</td>
                        </tr>
                        <tr>
                            <th align="left" style="padding: 4px 18px 4px 0; color: {muted};">Loads</th>
                            <td style="padding: 4px 0;">{loads_text}</td>
                        </tr>
                        <tr>
                            <th align="left" style="padding: 4px 18px 4px 0; color: {muted};">Soil Layers</th>
                            <td style="padding: 4px 0;">{len(layers)} defined</td>
                        </tr>
                    </table>

                    {layers_section}

                    <p style="margin-top: 6px; color: {tip}; font-size: 16px;">
                        Tip: Use the Edit Controls to update inputs.
                        Drag &amp; drop a <code>.pile.json</code> file anywhere to open an existing project.
                    </p>
                </div>"""

            self.info.setHtml(html)
            self.btn_gen.setEnabled(True)
            self._update_status_bar()
            self._rebuild_layers_list()

            if getattr(self, "_dock3d", None) and self._dock3d.isVisible():
                try:
                    self._refresh_3d_view()
                except Exception as e:
                    print("[3D] refresh_ui -> _refresh_3d_view_error:", e)

  def _init_theme(self):
    s = QSettings("Pile Analysis", "StudentEdition")
    theme = s.value("theme", "light")
    self._current_theme = theme
    self._apply_theme(theme)
    self._sync_theme_checks()

  def set_theme(self, theme: str):
    if theme not in ("light", "dark"):
      theme = "light"
    self._current_theme = theme
    self._apply_theme(theme)
    QSettings("Pile Analysis", "StudentEdition").setValue("theme", theme)
    self._sync_theme_checks()
    self.refresh_ui()
    
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
        /* Base */
        QWidget {background: #2f3340; color:#e9edf5;}
        QToolTip { color: #e9edf5; background:#3a3f4e; border: 1px solid #596073;}
        
        /* Top chrome */
        QMenuBar, QMenu, QToolBar, QStatusBar { background: #3a3f4e; color: #e9edf5;}
        QMenu::item:selected {background: #4a5161;}
        QStatusBar::item {border: none;}
        
        /* Inputs */
        QTextEdit, QPlainTextEdit, QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
            background: #3a3f4e; color: #e9edf5; border: 1px solid #6a7288; border-radius: 6px; selection-background-color: #58627a; selection-color: #ffffff;}
        QComboBox QAbstractItemView {background: #3a3f4e; selection-background-color: #58627a;}
        
        /* Buttons */
        QPushButton { background:#41485a; border: 1px solid #6a7288;  border-radius: 8px; padding: 6px 10px;}
        QPushButton:hover {background:#4a5161;}
        QPushButton:pressed {background: #424959;}
        QPushButton:disabled {color: #a6aec2; border-color: #545b6d;}
                        
        /* Tabs */
        QTabWidget::pane {border: 1px solid #596073; border-radius: 8px; top:-1px; background: #2f3340;}
        QTabBar::tab { background:#3a3f4e; padding:6px 12px; border: 1px solid #596073; border-bottom: none; border-top-left-radius: 8px; border-top-right-radius: 8px; margin-right: 2px;}
        QTabBar::tab:selected {background:#41485a;}
                        
        /* Splitters & frames */
        QFrame[frameShape="4"], QFrame[frameShape="5"] {background: #596073; max-height: 1px; max-width: 1px;}
                        
        /* Scrollbars */
        QScrollBar:vertical, QScrollBar:horizontal {background: #2f3340; border: none;}
        QScrollBar::handle:vertical, QScrollBar::handle:horizontal {background: #50586b; border-radius: 6px; min-height: 24px; min-width: 24px;}
        QScrollBar::handle:hover {background: #5a6277;}
        QScrollBar::add-page, QScrollBar::sub-page {background: transparent;}
                        
        /* Links & selection */
        a, QLabel[foregroundRole="link"] {color: #8f4bff;}
        *::selection {background: #58627a; color: #ffffff;}
        """)
    else:
      app.setStyleSheet("")

    # Make Matplotlib also dark if it is suggested in the future
    """
    if theme == "dark":
        mpl.rcParams.update({
            "figure.facecolor": "#2f3340",
            "axes.facecolor": "2f3340",
            "savefig.facecolor": "#2f3340",
            "axes.edgecolor": "#cfd6e6",
            "axes.labelcolor": "#e9edf5",
            "xtick.color": "#cfd6e6",
            "ytick.color": "#cfd6e6",
            "grid.color": "#5d6678",
            "text.color": "#e9edf5",
            "lines.linewidth": 1.6,
            "axes.grid": True,
    
    })"""
    
    # Matplotlib always light
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
        "axes.grid": True,
    })