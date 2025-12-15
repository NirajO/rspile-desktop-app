"""
Small Qt dialog windows for editing model inputs:
- Pile properties
- Loads
- Soil layer

Each dialog returns a simple dict on accept() so the main window can update state.
"""

from PySide6.QtWidgets import (
  QDialog, QFormLayout, QDialogButtonBox, QDoubleSpinBox, QComboBox
)

class PileDialog(QDialog):
    #Collect basic pile properties in SI units.
    def __init__(self, pile: dict | None, parent=None):
        super().__init__(parent)
        print("Initializing PileDialog")  # Debug
        self.setWindowTitle("Edit Pile")
        lay = QFormLayout(self)
        
        self.len_m = QDoubleSpinBox()
        self.len_m.setRange(0.0, 1e6)
        self.len_m.setDecimals(3)
        self.len_m.setSuffix(" m")

        self.d_m = QDoubleSpinBox()
        self.d_m.setRange(0.0, 1e6)
        self.d_m.setDecimals(3)
        self.d_m.setSuffix(" m")

        self.E_pa = QDoubleSpinBox()
        self.E_pa.setRange(1e6, 1e13)  # Reasonable elastic modulus range
        self.E_pa.setDecimals(0)
        self.E_pa.setSuffix(" Pa")

        self.gamma = QDoubleSpinBox()
        self.gamma.setRange(1, 100)  # kN/m³, typical soil unit weights
        self.gamma.setDecimals(2)
        self.gamma.setSuffix(" kN/m^3")

        if pile:
            self.len_m.setValue(pile.get("length_m", 0.0))
            self.d_m.setValue(pile.get("diameter_m", 0.0))
            self.E_pa.setValue(pile.get("elastic_modulus_pa", 0.0))
            self.gamma.setValue(pile.get("unit_weight_kNpm3", 0.0))

        lay.addRow("Length", self.len_m)
        lay.addRow("Diameter", self.d_m)
        lay.addRow("Elastic modulus", self.E_pa)
        lay.addRow("Unit weight", self.gamma)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        lay.addWidget(btns)

    def result_data(self):
        #Returns a validated dict of pile properties.
        data = {
            "length_m": self.len_m.value(),
            "diameter_m": self.d_m.value(),
            "elastic_modulus_pa": self.E_pa.value(),
            "unit_weight_kNpm3": self.gamma.value(),
        }
        if any(v <= 0 for v in data.values()):
            raise ValueError("All pile properties must be greater than 0.")
        if data["length_m"] < data["diameter_m"]:
            raise ValueError("Pile length must exceed diameter.")
        return data
    
class LoadDialog(QDialog):
  # Collect axial, lateral, and moment loads (kN / kN.m)
  def __init__(self, loads: dict | None, parent=None):
    super().__init__(parent)
    self.setWindowTitle("Edit Loads")
    lay = QFormLayout(self)

    self.axial = QDoubleSpinBox()
    self.axial.setRange(-1e6, 1e6)
    self.axial.setDecimals(2)
    self.axial.setSuffix(" kN")

    self.lat = QDoubleSpinBox()
    self.lat.setRange(-1e6, 1e6)
    self.lat.setDecimals(2)
    self.lat.setSuffix(" kN")

    self.moment = QDoubleSpinBox()
    self.moment.setRange(-1e9, 1e9)
    self.moment.setDecimals(2)
    self.moment.setSuffix(" kN.m")

    if loads:
      self.axial.setValue(loads.get("axial_kN", 0.0))
      self.lat.setValue(loads.get("lateral_kN", 0.0))
      self.moment.setValue(loads.get("moment_kNm", 0.0))

    lay.addRow("Axial", self.axial)
    lay.addRow("Lateral", self.lat)
    lay.addRow("Moment", self.moment)

    btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
    btns.accepted.connect(self.accept)
    btns.rejected.connect(self.reject)
    lay.addWidget(btns)

  def result_data(self):
    return{
      "axial_kN": self.axial.value(),
      "lateral_kN": self.lat.value(),
      "moment_kNm": self.moment.value(),
    }
  
class SoilLayerDialog(QDialog):
  # Adds a single soil layer either clay or sand.
  def __init__(self, layer: dict | None = None, parent=None):
    super().__init__(parent)
    self.setWindowTitle("Add Soil Layer" if not layer else "Edit Soil Layer")
    self._layer = layer or {}
    lay = QFormLayout(self)

    self.from_m = QDoubleSpinBox()
    self.from_m.setRange(0, 1e6)
    self.from_m.setDecimals(2)
    self.from_m.setSuffix(" m")

    self.to_m = QDoubleSpinBox()
    self.to_m.setRange(0, 1e6)
    self.to_m.setDecimals(2)
    self.to_m.setSuffix(" m")

    self.type = QComboBox()
    self.type.addItems(["clay", "sand"])

    self.gamma = QDoubleSpinBox()
    self.gamma.setRange(0, 100)
    self.gamma.setDecimals(2)
    self.gamma.setValue(18.0)
    self.gamma.setSuffix(" kN/m³")

    # clay-only property
    self.su = QDoubleSpinBox()
    self.su.setRange(0, 1000)
    self.su.setDecimals(1)
    self.su.setSuffix(" kPa")

    # sand-only property
    self.phi = QDoubleSpinBox()
    self.phi.setRange(20, 45)
    self.phi.setDecimals(1)
    self.phi.setSuffix(" °")

    lay.addRow("From depth", self.from_m)
    lay.addRow("To depth", self.to_m)
    lay.addRow("Type", self.type)
    lay.addRow("Unit Weight", self.gamma)
    lay.addRow("Undrained shear (su, clay)", self.su)
    lay.addRow("Friction angle (phi, sand)", self.phi)

    if self._layer:
       self.from_m.setValue(float(self._layer.get("from_m", 0.0)))
       self.to_m.setValue(float(self._layer.get("to_m", 0.0)))

       t = (self._layer.get("type") or "clay").lower()
       if t not in ("clay", "sand"):
          t = "clay"
       self.type.setCurrentText(t)

       self.gamma.setValue(float(self._layer.get("gamma_kNpm3", 18.0)))

       self.su.setValue(float(self._layer.get("undrained_shear_strength_kPa", 0.0)))
       self.phi.setValue(float(self._layer.get("phi_deg", 30.0)))

    btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
    btns.accepted.connect(self.accept)
    btns.rejected.connect(self.reject)
    lay.addWidget(btns)

  def result_data(self):
    # Returns a layer dict; includes su for clay or phi for sand.
    d = {
      "from_m": self.from_m.value(),
      "to_m": self.to_m.value(),
      "type": self.type.currentText(),
      "gamma_kNpm3": self.gamma.value(),
    }
    if d["type"] == "clay":
      d["undrained_shear_strength_kPa"] = self.su.value()
    else:
      d["phi_deg"] = self.phi.value()
    return d