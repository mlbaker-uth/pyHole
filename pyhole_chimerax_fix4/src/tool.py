import threading
import traceback
import tempfile
from chimerax.core.commands import run as cxrun
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
            one=ONE_LETTER.get(r.name.upper(),''); num=getattr(r,'number',None)
            ch=getattr(r,'chain_id','')
            if ch is None or (isinstance(ch,str) and not str(ch).strip()):
                ch='*'
            if num is None: continue
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

        # Inputs
        G = QGridLayout(); r=0
        G.addWidget(QLabel("Top plane"),r,0); self.top = QLineEdit(); G.addWidget(self.top,r,1)
        b=QPushButton("Get atoms"); b.clicked.connect(lambda:self._fill(self.top)); G.addWidget(b,r,2); r+=1
        G.addWidget(QLabel("Bottom plane"),r,0); self.bot = QLineEdit(); G.addWidget(self.bot,r,1)
        b=QPushButton("Get atoms"); b.clicked.connect(lambda:self._fill(self.bot)); G.addWidget(b,r,2); r+=1
        self.use_mid = QCheckBox("Mid plane (optional waypoint for bent pores)")
        self.use_mid.toggled.connect(lambda on: (self.mid.setEnabled(on), self.mid_btn.setEnabled(on)))
        G.addWidget(self.use_mid,r,0)
        self.mid = QLineEdit(); self.mid.setEnabled(False); G.addWidget(self.mid,r,1)
        self.mid_btn=QPushButton("Get atoms"); self.mid_btn.setEnabled(False)
        self.mid_btn.clicked.connect(lambda:self._fill(self.mid)); G.addWidget(self.mid_btn,r,2); r+=1
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
        self.noHet=QCheckBox("Ignore HETATM (ligands/waters/ions)"); self.noHet.setChecked(False)
        O.addWidget(QLabel("Probe (Å)"),r,0); O.addWidget(self.probe,r,1)
        O.addWidget(QLabel("Conductivity (S/m)"),r,2); O.addWidget(self.cond,r,3)
        O.addWidget(self.noH,r,4)
        O.addWidget(self.noHet,r,5); r+=1
        self.step=QDoubleSpinBox(); self.step.setDecimals(2); self.step.setRange(0.05,20); self.step.setValue(1.0)
        self.eps=QDoubleSpinBox(); self.eps.setDecimals(2); self.eps.setRange(0,5); self.eps.setValue(0.25)
        O.addWidget(QLabel("Step (Å)"),r,0); O.addWidget(self.step,r,1)
        O.addWidget(QLabel("eps (Å)"),r,2); O.addWidget(self.eps,r,3); r+=1
        self.rings=QSpinBox(); self.rings.setRange(6,180); self.rings.setValue(24)
        O.addWidget(QLabel("Rings"),r,0); O.addWidget(self.rings,r,1)
        self.surface=QCheckBox("Render true pore surface (isosurface)"); self.surface.setChecked(False)
        O.addWidget(self.surface,r,2,1,2); r+=1
        L.addLayout(O)

        # Occupancy / scale
        R = QHBoxLayout()
        R.addWidget(QLabel("Occupancy:"))
        self.occH=QRadioButton("hydrophobicity"); self.occE=QRadioButton("electrostatics"); self.occR=QRadioButton("radii")
        self.occH.setChecked(True); R.addWidget(self.occH); R.addWidget(self.occE); R.addWidget(self.occR)
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
        H2 = QHBoxLayout(); self.run_btn = QPushButton("Run pyHole"); self.run_btn.clicked.connect(self._on_run_clicked)
        H2.addStretch(1); H2.addWidget(self.run_btn); L.addLayout(H2)        
        
        
        self._refresh_models()        # fill menu once
        self._install_model_triggers()# hook triggers defensively

        self.tw.manage(None)
    def _on_run_clicked(self):
        # Resolve selected model (unchanged from your version)
        model = None
        if hasattr(self, "_selected_model"):
            model = self._selected_model()
        if model is None:
            idx = self.model_combo.currentIndex()
            if idx < 0:
                self.session.logger.warning("pyHole: no model selected.")
                return
            idstr = self.model_combo.itemData(idx)
            model = next((m for m in self.session.models.list()
                          if getattr(m, "id_string", "") == idstr), None)
            if model is None:
                self.session.logger.warning("pyHole: could not resolve selected model.")
                return

        # ---- Export the model to a temporary PDB ON THE UI THREAD ----
        id_str = str(getattr(model, 'id_string', getattr(model, 'id', 'model'))).lstrip('#')
        tmp_root = Path(getattr(self.session, 'user_data_dir', '') or tempfile.gettempdir())
        tmp_root.mkdir(parents=True, exist_ok=True)
        tmp_pdb = tmp_root / f"pyhole_tmp_{id_str}.pdb"
        cxrun(self.session, f'save "{tmp_pdb}" models #{id_str} format pdb')

        # ---- Collect GUI options ----
        params = dict(
            # IMPORTANT: pass pdb path; do NOT pass session/model to engine
            pdb=str(tmp_pdb),

            top=self.top.text().strip(),
            bottom=self.bot.text().strip(),
            mid=(self.mid.text().strip() if self.use_mid.isChecked() else ""),
            vdwjson=self.vdw.text().strip(),
            outprefix=(self.out.text().strip() or "pyhole"),

            step=float(self.step.value()),
            eps=float(self.eps.value()),
            rings=int(self.rings.value()),
            noH=self.noH.isChecked(),
            noHet=self.noHet.isChecked(),

            probe=float(self.probe.value()),
            conductivity=float(self.cond.value()),

            centerline=("curved" if self.curved.isChecked() else "straight"),
            adaptive=self.adapt.isChecked(),
            slope_thresh=float(self.slope.value()),
            max_refine=int(self.iters.value()),
            curve_radius=float(self.crad.value()),
            curve_iters=int(self.cit.value()),

            occupancy=("radii" if self.occR.isChecked() else ("electro" if self.occE.isChecked() else "hydro")),
            hydroscale=self.scale.currentText().lower(),
            electroscale=self.scale.currentText().lower(),

            surface=self.surface.isChecked(),
        )

        # UI feedback + start worker
        self.run_btn.setEnabled(False)
        self.run_btn.setText("Running…")
        self.session.logger.status("pyHole: running…")

        t = threading.Thread(target=self._run_pyhole_worker, args=(params,), daemon=True)
        t.start()
    def _run_pyhole_worker(self, p):
        """Background thread: call engine once; schedule UI updates on main thread."""
        try:
            from . import holepy as engine
            outputs = engine.run_pyhole(**p) or {}

        except Exception:
            msg = traceback.format_exc()  # capture before leaving 'except'

            def _err(m=msg):
                self.run_btn.setEnabled(True)
                self.run_btn.setText("Run pyHole")
                self.session.logger.error("pyHole error:\n" + m)
                self.session.logger.status("pyHole: failed.")
            self.session.ui.thread_safe(_err)
            return  # make sure we don't fall through

        # define success callback AFTER the try/except block
        def _done():
            self.run_btn.setEnabled(True)
            self.run_btn.setText("Run pyHole")

            mp = outputs.get("mesh_pdb")
            cp = outputs.get("centerline_pdb")
            metric = (p.get("occupancy") or "hydro").lower()
            if metric.startswith("elect"):
                metric = "electro"
            elif metric.startswith("rad"):
                metric = "radii"
            else:
                metric = "hydro"
            electro_scale = str(p.get("electroscale") or "raw").lower()

            if mp:
                self._style_pore_model(mp, metric=metric, size_centerline=False, electro_scale=electro_scale)
            if cp:
                # Keep centerline radius sizing from B-factor, but color by selected metric.
                self._style_pore_model(cp, metric=metric, size_centerline=True, electro_scale=electro_scale)

            if outputs.get("csv"):
                self.session.logger.info(f'pyHole CSV: {outputs["csv"]}')

            dm = outputs.get("density_map")
            if dm:
                self._render_true_surface(dm)

            self.session.logger.status("pyHole: finished.")

        # run success callback on the UI thread
        self.session.ui.thread_safe(_done)
   
    # Helpers
    def _pdb_attr_minmax(self, path, attr):
        """Return (min, max) for occupancy or bfactor column in a PDB."""
        if attr == "occupancy":
            c0, c1 = 54, 60
        else:
            c0, c1 = 60, 66
        minv, maxv = None, None
        try:
            with open(path, 'r') as f:
                for line in f:
                    if line.startswith(('ATOM  ', 'HETATM')):
                        vs = line[c0:c1].strip()
                        if not vs:
                            continue
                        v = float(vs)
                        minv = v if minv is None or v < minv else minv
                        maxv = v if maxv is None or v > maxv else maxv
        except Exception:
            pass
        return minv, maxv

    def _style_pore_model(self, pdb_path, metric="hydro", size_centerline=False, electro_scale="raw"):
        """Open mesh/centerline and color by selected metric with tool-specific palette."""
        from pathlib import Path
        pre = {getattr(m, 'id_string', getattr(m, 'id', '')) for m in self.session.models.list()}

        # Open model
        try:
            cxrun(self.session, f'open "{pdb_path}"')
        except Exception as e:
            self.session.logger.warning(f"pyHole: could not open PDB: {e}")
            return

        # Find the newly opened model (prefer name match)
        new_models = [m for m in self.session.models.list()
                      if getattr(m, 'id_string', getattr(m, 'id', '')) not in pre]
        model = None
        if new_models:
            stem = Path(pdb_path).stem
            for m in new_models:
                if stem in getattr(m, 'name', ''):
                    model = m
                    break
            if model is None:
                model = new_models[-1]

        if model is None:
            self.session.logger.warning("pyHole: opened model not found after open.")
            return

        mid_str = getattr(model, 'id_string', getattr(model, 'id', ''))

        # Optional centerline sizing uses B-factor radius values.
        if size_centerline:
            minr, maxr = self._pdb_attr_minmax(pdb_path, "bfactor")
            if minr is None or maxr is None or maxr <= minr:
                minr, maxr = 0.5, 8.0
            cxrun(
                self.session,
                f'size byattribute a:bfactor #{mid_str} {minr:.2f}:{minr:.2f} {maxr:.2f}:{maxr:.2f}'
            )

        # Color by selected metric:
        # hydro: cyan -> white -> yellow (range from file)
        # electro: mean formal charge — fixed anchors like legacy pyHole / HOLE-style utilities:
        #   raw in [-1,1]; scale "01" maps that onto [0,1] with neutral -> 0.5 in writer
        # radii: red (< water radius) -> white (~water) -> green (> water radius)
        if metric == "electro":
            attr_name = "occupancy"
            if electro_scale == "01":
                palette = "0,red:0.5,white:1,blue"
            else:
                palette = "-1,red:0,white:1,blue"
        elif metric == "radii":
            attr_name = "occupancy"
            cmin, cmax = self._pdb_attr_minmax(pdb_path, attr_name)
            if cmin is None or cmax is None or cmax <= cmin:
                cmin, cmax = 0.1, 3.0
            # Anchor at water radius 1.4 A for continuity with prior behavior.
            palette = f"{cmin:.2f},red:1.40,white:{cmax:.2f},green"
        else:
            attr_name = "occupancy"
            cmin, cmax = self._pdb_attr_minmax(pdb_path, attr_name)
            if cmin is None or cmax is None or cmax <= cmin:
                cmin, cmax = -4.5, 4.5
            cmid = 0.5 * (cmin + cmax)
            palette = f"{cmin:.2f},cyan:{cmid:.2f},white:{cmax:.2f},yellow"

        cxrun(
            self.session,
            f'color byattribute a:{attr_name} #{mid_str} target cabs palette {palette}'
        )

    def _render_true_surface(self, density_map_path):
        """Open the pore density map and isosurface it at level 0, which is
        exactly the probe-accessible boundary by construction (see
        build_density_grid()/write_density_map() in holepy.py) -- a real
        molecular-surface-style rendering of the lumen, not a tube of
        profiled spheres."""
        pre = {getattr(m, 'id_string', getattr(m, 'id', '')) for m in self.session.models.list()}
        try:
            cxrun(self.session, f'open "{density_map_path}"')
        except Exception as e:
            self.session.logger.warning(f"pyHole: could not open density map: {e}")
            return

        new_models = [m for m in self.session.models.list()
                      if getattr(m, 'id_string', getattr(m, 'id', '')) not in pre]
        if not new_models:
            self.session.logger.warning("pyHole: density map model not found after open.")
            return
        model = new_models[-1]
        mid_str = getattr(model, 'id_string', getattr(model, 'id', ''))
        try:
            cxrun(self.session, f'volume #{mid_str} level 0 style surface color light sea green transparency 20')
        except Exception as e:
            self.session.logger.warning(f"pyHole: could not isosurface density map: {e}")

    def _install_model_triggers(self):
        self._trigger_handlers = []

        def _try(triggers, name):
            try:
                h = triggers.add_handler(name, self._models_changed)
                self._trigger_handlers.append(h)
                return True
            except KeyError:
                return False

        # First try the session trigger many tools use
        added = _try(self.session.triggers, 'models changed')

        # If that fails, try a few common variants
        if not added:
            for nm in ('add models', 'remove models', 'reorder models',
                       'add model', 'remove model', 'models change', 'model list changed'):
                _try(self.session.triggers, nm)

        # Also try triggers on the model list object, if present
        try:
            mt = self.session.models.triggers
            # Prefer explicit names if available
            for nm in getattr(mt, 'names', ()):
                if 'model' in nm:
                    _try(mt, nm)
        except Exception:
            pass
        
    def delete(self):
        # called when tool closes
        for h in getattr(self, '_trigger_handlers', []):
            try:
                h.remove()
            except Exception:
                pass
        super().delete()    
    
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
        if self.use_mid.isChecked() and self.mid.text().strip():
            args += ["--mid", self.mid.text().strip()]
        if self.noH.isChecked(): args+=["--noH"]
        if self.noHet.isChecked(): args+=["--noHet"]
        if self.occR.isChecked():
            args+=["--occupancy","radii"]
        elif self.occE.isChecked():
            args+=["--occupancy","electro","--electroscale",self.scale.currentText()]
        else:
            args+=["--occupancy","hydro","--hydroscale",self.scale.currentText()]
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
