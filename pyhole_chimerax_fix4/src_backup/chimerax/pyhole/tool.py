
from chimerax.core.tools import ToolInstance
from chimerax.ui import MainToolWindow
from chimerax.core.commands import run
from chimerax.atomic import selected_residues
from Qt.QtWidgets import (QGridLayout, QLabel, QComboBox, QLineEdit, QPushButton,
                          QDoubleSpinBox, QSpinBox, QCheckBox, QHBoxLayout, QVBoxLayout,
                          QFileDialog, QRadioButton)
import os, subprocess, sys, tempfile, json
from pathlib import Path

ONE_LETTER={'ALA':'A','CYS':'C','ASP':'D','GLU':'E','PHE':'F','GLY':'G','HIS':'H','ILE':'I','LYS':'K','LEU':'L','MET':'M','ASN':'N','PRO':'P','GLN':'Q','ARG':'R','SER':'S','THR':'T','VAL':'V','TRP':'W','TYR':'Y','SEC':'U','PYL':'O'}

def _format_reslist(residues):
    toks=[]
    for r in residues:
        try:
            one=ONE_LETTER.get(r.name.upper(),''); num=getattr(r,'number',None); ch=getattr(r,'chain_id','')
            if num is None or ch is None: continue
            toks.append(f"{one}{num}/{ch}" if one else f"{num}/{ch}")
        except: pass
    return ", ".join(toks)

class PyHoleTool(ToolInstance):
    def __init__(self, session, tool_name):
        super().__init__(session, tool_name)
        self.display_name = "pyHole"
        self.tw = MainToolWindow(self)
        parent = self.tw.ui_area
        L = QVBoxLayout(parent)

        # Header: model chooser and Cite button
        H = QHBoxLayout()
        self.model_combo = QComboBox(); H.addWidget(self.model_combo, 1)
        cite = QPushButton("Cite pyHole"); cite.clicked.connect(self._cite); H.addWidget(cite, 0)
        L.addLayout(H)
        self._refresh_models()
        session.triggers.add_handler('models changed', self._models_changed)

        # Inputs
        G = QGridLayout(); r=0
        G.addWidget(QLabel("Top plane"),r,0); self.top = QLineEdit(); G.addWidget(self.top,r,1)
        b=QPushButton("Get atoms"); b.clicked.connect(lambda:self._fill(self.top)); G.addWidget(b,r,2); r+=1
        G.addWidget(QLabel("Bottom plane"),r,0); self.bot = QLineEdit(); G.addWidget(self.bot,r,1)
        b=QPushButton("Get atoms"); b.clicked.connect(lambda:self._fill(self.bot)); G.addWidget(b,r,2); r+=1
        G.addWidget(QLabel("Ion radii file"),r,0); self.vdw = QLineEdit(); G.addWidget(self.vdw,r,1)
        b=QPushButton("Select file"); b.clicked.connect(self._pick_vdw); G.addWidget(b,r,2); r+=1
        G.addWidget(QLabel("Output prefix"),r,0); self.out = QLineEdit(); G.addWidget(self.out,r,1)
        b=QPushButton("Select"); b.clicked.connect(self._pick_out); G.addWidget(b,r,2); r+=1
        L.addLayout(G)

        # Numeric options
        O = QGridLayout(); r=0
        self.probe=QDoubleSpinBox(); self.probe.setDecimals(2); self.probe.setRange(0,20); self.probe.setValue(1.4)
        self.cond=QDoubleSpinBox(); self.cond.setDecimals(2); self.cond.setRange(0,10); self.cond.setValue(1.5)
        self.noH=QCheckBox("No hydrogens"); self.noH.setChecked(True)
        O.addWidget(QLabel("Probe (Å)"),r,0); O.addWidget(self.probe,r,1)
        O.addWidget(QLabel("Conductivity (S/m)"),r,2); O.addWidget(self.cond,r,3)
        O.addWidget(self.noH,r,4); r+=1
        self.step=QDoubleSpinBox(); self.step.setDecimals(2); self.step.setRange(0.05,20); self.step.setValue(1.0)
        self.eps=QDoubleSpinBox(); self.eps.setDecimals(2); self.eps.setRange(0,5); self.eps.setValue(0.25)
        O.addWidget(QLabel("Step (Å)"),r,0); O.addWidget(self.step,r,1)
        O.addWidget(QLabel("eps (Å)"),r,2); O.addWidget(self.eps,r,3); r+=1
        self.rings=QSpinBox(); self.rings.setRange(6,180); self.rings.setValue(24)
        O.addWidget(QLabel("Rings"),r,0); O.addWidget(self.rings,r,1)
        L.addLayout(O)

        # Occupancy / scale
        R = QHBoxLayout()
        R.addWidget(QLabel("Occupancy:"))
        self.occH=QRadioButton("hydrophobicity"); self.occE=QRadioButton("electrostatics")
        self.occH.setChecked(True); R.addWidget(self.occH); R.addWidget(self.occE)
        R.addStretch(1); R.addWidget(QLabel("scale"))
        self.scale=QComboBox(); self.scale.addItems(["raw","01"]); R.addWidget(self.scale)
        L.addLayout(R)

        # Adaptive straight vs curved
        A = QGridLayout(); rr=0
        self.adapt=QCheckBox("Adaptive sampling (straight)"); self.adapt.setChecked(True); A.addWidget(self.adapt,rr,0)
        self.slope=QDoubleSpinBox(); self.slope.setDecimals(2); self.slope.setRange(0,10); self.slope.setValue(0.5)
        A.addWidget(QLabel("slope"),rr,1); A.addWidget(self.slope,rr,2)
        self.iters=QSpinBox(); self.iters.setRange(0,10); self.iters.setValue(3)
        A.addWidget(QLabel("iterations"),rr,3); A.addWidget(self.iters,rr,4); rr+=1
        self.curved=QCheckBox("Curved centerline"); self.curved.setChecked(False); A.addWidget(self.curved,rr,0)
        self.crad=QDoubleSpinBox(); self.crad.setDecimals(2); self.crad.setRange(0,10); self.crad.setValue(2.0)
        A.addWidget(QLabel("radius (Å)"),rr,1); A.addWidget(self.crad,rr,2)
        self.cit=QSpinBox(); self.cit.setRange(0,10); self.cit.setValue(3)
        A.addWidget(QLabel("iterations"),rr,3); A.addWidget(self.cit,rr,4)
        L.addLayout(A)

        # Run button
        H2 = QHBoxLayout(); runBtn=QPushButton("Run pyHole"); runBtn.clicked.connect(self._run)
        H2.addStretch(1); H2.addWidget(runBtn); L.addLayout(H2)

        self.tw.manage(None)

    # Helpers
    def _cite(self):
        self.session.logger.info("pyHole: cite this tool (and UCSF ChimeraX). 2025.")

    def _models_changed(self, *a, **k): self._refresh_models()
    def _refresh_models(self):
        self.model_combo.clear()
        for m in self.session.models.list():
            try: self.model_combo.addItem(f"#{m.id_string}  {m.name}", m.id_string)
            except: pass

    def _fill(self, line):
        res = selected_residues(self.session)
        if not res: self.session.logger.warning("Select residues first."); return
        line.setText(_format_reslist(res))

    def _pick_vdw(self):
        fn,_=QFileDialog.getOpenFileName(self.tw.ui_area,"Select species radii JSON","","JSON (*.json);;All files (*)")
        if fn: self.vdw.setText(fn)

    def _pick_out(self):
        fn,_=QFileDialog.getSaveFileName(self.tw.ui_area,"Select output prefix","","All files (*)")
        if fn: self.out.setText(fn)

    def _run(self):
        idx=self.model_combo.currentIndex()
        if idx<0: self.session.logger.error("No model selected."); return
        idstr=self.model_combo.itemData(idx)
        top=self.top.text().strip(); bot=self.bot.text().strip()
        if not top or not bot: self.session.logger.error("Provide TOP/BOTTOM residue lists."); return
        outprefix=self.out.text().strip() or str(Path(tempfile.gettempdir())/"pyhole_out")
        tmp_pdb=str(Path(tempfile.gettempdir())/"pyhole_tmp_model.pdb")
        try: run(self.session, f"save {tmp_pdb} #{idstr}")
        except Exception as e: self.session.logger.error(f"Failed to save model: {e}"); return

        holepy_path = Path(__file__).with_name("holepy.py")
        if not holepy_path.exists(): self.session.logger.error("holepy.py missing from bundle."); return

        args=[sys.executable, str(holepy_path), tmp_pdb,
              "--top", top, "--bottom", bot,
              "--step", str(self.step.value()), "--eps", str(self.eps.value()),
              "--probe", str(self.probe.value()), "--conductivity", str(self.cond.value()),
              "--rings", str(self.rings.value()), "--outprefix", outprefix]
        if self.noH.isChecked(): args+=["--noH"]
        if self.occE.isChecked(): args+=["--occupancy","electro","--electroscale",self.scale.currentText()]
        else: args+=["--occupancy","hydro","--hydroscale",self.scale.currentText()]
        if self.curved.isChecked():
            args+=["--centerline","curved","--curve_radius",str(self.crad.value()),"--curve_iters",str(self.cit.value())]
        elif self.adapt.isChecked():
            args+=["--adaptive","--slope_thresh",str(self.slope.value()),"--max_refine",str(self.iters.value())]
        vdw=self.vdw.text().strip()
        if vdw: args+=["--vdwjson", vdw]

        self.session.logger.status("Running pyHole…")
        try:
            out = subprocess.run(args, capture_output=True, text=True)
        except Exception as e:
            self.session.logger.error(f"Failed to run holepy: {e}"); return
        if out.returncode!=0:
            self.session.logger.error("pyHole failed:\n"+out.stderr); return
        if out.stdout: self.session.logger.info(out.stdout)

        # auto-open PDB outputs
        for f in (outprefix+"_mesh.pdb", outprefix+"_centerline.pdb"):
            if os.path.exists(f): run(self.session, f"open {f}")
