
from pathlib import Path
import argparse, sys, re, math, json, csv
import numpy as np

ONE_LETTER={'ALA':'A','CYS':'C','ASP':'D','GLU':'E','PHE':'F','GLY':'G','HIS':'H','ILE':'I','LYS':'K','LEU':'L','MET':'M','ASN':'N','PRO':'P','GLN':'Q','ARG':'R','SER':'S','THR':'T','VAL':'V','TRP':'W','TYR':'Y','SEC':'U','PYL':'O'}
VDW_DEFAULT={'H':1.20,'C':1.70,'N':1.55,'O':1.52,'F':1.47,'P':1.80,'S':1.80,'CL':1.75,'BR':1.85,'I':1.98}
KD={'ILE':4.5,'VAL':4.2,'LEU':3.8,'PHE':2.8,'CYS':2.5,'MET':1.9,'ALA':1.8,'GLY':-0.4,'THR':-0.7,'SER':-0.8,'TRP':-0.9,'TYR':-1.3,'PRO':-1.6,'HIS':-3.2,'GLU':-3.5,'GLN':-3.5,'ASP':-3.5,'ASN':-3.5,'LYS':-3.9,'ARG':-4.5}
KD_MIN=min(KD.values()); KD_MAX=max(KD.values())
CHARGE={'ARG':+1.0,'LYS':+1.0,'HIS':+0.1,'ASP':-1.0,'GLU':-1.0}
ELEC_MIN,ELEC_MAX=-1.0,+1.0

def guess_element(atom_name, element_field):
    if element_field and element_field.strip(): return element_field.strip().upper()
    an=atom_name.strip()
    if not an: return 'C'
    if an[0].isdigit() and len(an)>=2:
        c=an[1]; 
        if len(an)>=3 and an[2].isalpha(): return (c+an[2]).upper()
        return c.upper()
    c=an[0]
    if len(an)>=2 and an[1].isalpha() and an[1].islower(): return (c+an[1]).upper()
    return c.upper()

def vdw_radius(element, custom): return custom.get(element.upper(), VDW_DEFAULT.get(element.upper(),1.7))

class Atom:
    __slots__=('x','y','z','name','resname','chain','resi','element')
    def __init__(self,x,y,z,name,resname,chain,resi,element):
        self.x=float(x); self.y=float(y); self.z=float(z); self.name=name.strip()
        self.resname=resname; self.chain=chain; self.resi=int(resi); self.element=element

def load_pdb_atoms(path, include_h=True):
    atoms=[]
    with open(path,'r') as f:
        for line in f:
            if line[:6] not in ('ATOM  ','HETATM'): continue
            name=line[12:16]; resname=line[17:20].strip(); chain=line[21].strip() or 'A'; resi_str=line[22:26].strip() or '0'
            try: resi=int(resi_str)
            except: continue
            x=float(line[30:38]); y=float(line[38:46]); z=float(line[46:54])
            elem_field=line[76:78] if len(line)>=78 else ''
            elem=guess_element(name,elem_field)
            if (not include_h) and elem.upper()=='H': continue
            atoms.append(Atom(x,y,z,name,resname,chain,resi,elem))
    return atoms

def parse_residue_tokens(s):
    toks=[]
    if not s: return toks
    import re
    for t in s.split(','):
        t=t.strip()
        if not t: continue
        m=re.match(r'^[A-Za-z]{0,3}?(\\d+)\\s*/\\s*([A-Za-z0-9])$',t)
        if m: toks.append((m.group(2),int(m.group(1)))); continue
        m=re.match(r'^([A-Za-z0-9])\\s*[:\\s]\\s*(\\d+)$',t)
        if m: toks.append((m.group(1),int(m.group(2)))); continue
        m=re.match(r'^\\s*(\\d+)\\s*$',t)
        if m: toks.append(('*',int(m.group(1)))); continue
        raise ValueError(f"Could not parse residue token: '{t}'.")
    return toks

def ca_positions_for(atoms, sel):
    out=[]
    for ch,resi in sel:
        for a in atoms:
            if a.name=='CA' and a.resi==resi and (ch=='*' or a.chain==ch): out.append([a.x,a.y,a.z])
    return np.array(out,float) if out else np.zeros((0,3),float)

def orthonormal_basis_from_axis(axis):
    u=axis/(np.linalg.norm(axis)+1e-12)
    a=np.array([1.0,0.0,0.0]) if abs(u[0])<0.9 else np.array([0.0,1.0,0.0])
    v=np.cross(u,a); n=np.linalg.norm(v)
    if n<1e-8: a=np.array([0.0,0.0,1.0]); v=np.cross(u,a); n=np.linalg.norm(v)
    v/= (n+1e-12); w=np.cross(u,v); w/= (np.linalg.norm(w)+1e-12)
    return v,w

def clearance_at_point(atom_xyz, atom_r, c):
    dv=atom_xyz-c; d=np.sqrt((dv*dv).sum(axis=1))-atom_r; return float(np.min(d))

def evaluate_slice(atom_xyz, atom_r, atom_meta, c, eps, hydroscale, electroscale):
    dv=atom_xyz-c; d=np.sqrt((dv*dv).sum(axis=1))-atom_r
    rmin=float(np.min(d)); mask=d<= (rmin+eps)
    seen=set(); tags=[]; H=[]; E=[]
    for i,ok in enumerate(mask):
        if not ok: continue
        chain,resname,resi=atom_meta[i]; key=(chain,resi,resname)
        if key in seen: continue
        seen.add(key)
        tags.append(f"{ONE_LETTER.get(resname.upper(),'?')}{resi}/{chain}")
        H.append(KD.get(resname.upper(),0.0)); E.append(CHARGE.get(resname.upper(),0.0))
    hydro=float(np.mean(H)) if H else 0.0; electro=float(np.mean(E)) if E else 0.0
    if hydroscale=='01': hydro=(hydro-KD_MIN)/(KD_MAX-KD_MIN) if KD_MAX>KD_MIN else 0.0
    if electroscale=='01': electro=(electro-(-1.0))/(2.0)
    return rmin, ';'.join(tags[:50]), hydro, electro

def profile_straight(atom_xyz, atom_r, c0, c1, step, eps, atom_meta, adaptive, slope_thresh, max_refine, hydroscale, electroscale, occ_metric):
    axis=c1-c0; L=np.linalg.norm(axis)
    if L<1e-6: raise ValueError("Top and bottom too close.")
    u=axis/L
    def ctr(s): return c0+u*s
    svals=list(np.linspace(0.0, L, max(1,int(round(L/step))+1)))
    def eval_rows(vals):
        out=[]
        for s in vals:
            c=ctr(s); rmin,tags,hyd,elec=evaluate_slice(atom_xyz,atom_r,atom_meta,c,eps,hydroscale,electroscale)
            out.append({'s_A':float(s),'x':float(c[0]),'y':float(c[1]),'z':float(c[2]),'radius_A':float(rmin),'hydro_index':float(hyd),'electro_index':float(elec),'contributors':tags})
        return out
    rows=eval_rows(svals)
    if adaptive:
        for _ in range(max_refine):
            rows.sort(key=lambda r:r['s_A']); ns=[]
            for i in range(len(rows)-1):
                s0,r0=rows[i]['s_A'], rows[i]['radius_A']; s1,r1=rows[i+1]['s_A'], rows[i+1]['radius_A']; ds=s1-s0
                if ds<=step/2: continue
                slope=abs((r1-r0)/ds) if ds>1e-9 else 0.0
                if slope> slope_thresh: ns.append(0.5*(s0+s1))
            if not ns: break
            rows+=eval_rows(ns)
    rows.sort(key=lambda r:r['s_A'])
    for r in rows: r['occ_value']=float(r['hydro_index'] if occ_metric=='hydro' else r['electro_index'])
    return rows, u, L

def profile_curved(atom_xyz, atom_r, c0, c1, step, eps, atom_meta, curve_radius, curve_iters, hydroscale, electroscale, occ_metric):
    axis=c1-c0; L=np.linalg.norm(axis)
    if L<1e-6: raise ValueError("Top and bottom too close.")
    u=axis/L; v,w=orthonormal_basis_from_axis(u)
    svals=list(np.linspace(0.0, L, max(1,int(round(L/step))+1)))
    centers=[]; c_prev=c0.copy()
    for idx,s in enumerate(svals):
        if s<=1e-9: c=c0.copy()
        elif abs(s-L)<=1e-9: c=c1.copy()
        else:
            ds=svals[idx]-svals[idx-1]; c=c_prev+u*ds; r=float(curve_radius)
            for _ in range(int(curve_iters)):
                best=c; best_cl=clearance_at_point(atom_xyz,atom_r,c)
                for dx,dy in [(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(1,-1),(-1,-1),(0,0)]:
                    cand=c+(dx*r)*v+(dy*r)*w; cl=clearance_at_point(atom_xyz,atom_r,cand)
                    if cl>best_cl: best_cl=cl; best=cand
                c=best; r*=0.5
        centers.append(c); c_prev=c
    rows=[]; s_acc=0.0
    for i,c in enumerate(centers):
        if i>0: s_acc+=float(np.linalg.norm(centers[i]-centers[i-1]))
        rmin,tags,hyd,elec=evaluate_slice(atom_xyz,atom_r,atom_meta,c,eps,hydroscale,electroscale)
        rows.append({'s_A':float(s_acc),'x':float(c[0]),'y':float(c[1]),'z':float(c[2]),'radius_A':float(rmin),'hydro_index':float(hyd),'electro_index':float(elec),'contributors':tags})
    for i in range(len(rows)):
        if i==0: t=np.array([rows[1]['x']-rows[0]['x'], rows[1]['y']-rows[0]['y'], rows[1]['z']-rows[0]['z']],float)
        elif i==len(rows)-1: t=np.array([rows[i]['x']-rows[i-1]['x'], rows[i]['y']-rows[i-1]['y'], rows[i]['z']-rows[i-1]['z']],float)
        else: t=np.array([rows[i+1]['x']-rows[i-1]['x'], rows[i+1]['y']-rows[i-1]['y'], rows[i+1]['z']-rows[i-1]['z']],float)
        n=np.linalg.norm(t); t=np.array([0.0,0.0,1.0]) if n<1e-9 else (t/n)
        rows[i]['tx']=float(t[0]); rows[i]['ty']=float(t[1]); rows[i]['tz']=float(t[2])
        rows[i]['occ_value']=float(rows[i]['hydro_index'] if occ_metric=='hydro' else rows[i]['electro_index'])
    return rows, u, L

def write_csv(path, rows):
    if not rows: return
    import csv
    with open(path,'w',newline='') as f:
        cols=list(rows[0].keys())
        if 'occ_value' not in cols: cols.append('occ_value')
        w=csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in rows: w.writerow(r)

def format_atom(serial, x,y,z, occ, b):
    return f"ATOM  {serial:5d}  C   ALA M   1   {x:8.3f}{y:8.3f}{z:8.3f}{occ:6.2f}{b:6.2f}          C  \r\n"

def mesh_pdb(path, rows, axis_u, rings=24):
    use_local=('tx' in rows[0])
    if not use_local:
        u=axis_u/(np.linalg.norm(axis_u)+1e-12)
        v,w=orthonormal_basis_from_axis(u)
    coords=[]; bvals=[]; occs=[]
    for r in rows:
        c=np.array([r['x'],r['y'],r['z']],float)
        rad=max(0.0,float(r['radius_A'])); occ=float(r.get('occ_value',0.0))
        if use_local:
            u_loc=np.array([r['tx'],r['ty'],r['tz']],float); v,w=orthonormal_basis_from_axis(u_loc)
        for k in range(rings):
            ang=2*np.pi*(k/rings); p=c+rad*(np.cos(ang)*v+np.sin(ang)*w); coords.append(p); bvals.append(rad); occs.append(occ)
    lines=[]; s0=1
    for i,p in enumerate(coords, start=s0):
        x,y,z=p; b=bvals[i-s0]; o=occs[i-s0]
        lines.append(format_atom(i,x,y,z,o,b))
    # connectivities (optional for vis)
    with open(path,'w',newline='') as f: f.writelines(lines)

def centerline_pdb(path, rows):
    lines=[]; s=1
    for i,r in enumerate(rows, start=1):
        x,y,z=r['x'],r['y'],r['z']; b=r['radius_A']; o=float(r.get('occ_value',0.0))
        lines.append(f"HETATM{s:5d}  O  PORE Z{i:4d}    {x:8.3f}{y:8.3f}{z:8.3f}{o:6.2f}{b:6.2f}          O \r\n"); s+=1
    with open(path,'w',newline='') as f: f.writelines(lines)

def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument("pdb"); p.add_argument("--top",default=""); p.add_argument("--bottom",default=""); p.add_argument("--interactive",action="store_true")
    p.add_argument("--step",type=float,default=1.0); p.add_argument("--eps",type=float,default=0.25); p.add_argument("--noH",action="store_true")
    p.add_argument("--vdwjson",type=str,default=""); p.add_argument("--rings",type=int,default=24); p.add_argument("--outprefix",type=str,default="holepy_out")
    p.add_argument("--probe",type=float,default=0.0); p.add_argument("--conductivity",type=float,default=1.5)
    p.add_argument("--occupancy",choices=["hydro","electro"],default="hydro"); p.add_argument("--hydroscale",choices=["raw","01"],default="raw"); p.add_argument("--electroscale",choices=["raw","01"],default="raw")
    p.add_argument("--passable_json",type=str,default="")
    p.add_argument("--centerline",choices=["straight","curved"],default="straight")
    p.add_argument("--adaptive",action="store_true"); p.add_argument("--slope_thresh",type=float,default=0.5); p.add_argument("--max_refine",type=int,default=3)
    p.add_argument("--curve_radius",type=float,default=2.0); p.add_argument("--curve_iters",type=int,default=3)
    ns=p.parse_args(argv)

    atoms=load_pdb_atoms(Path(ns.pdb), include_h=not ns.noH)
    if ns.interactive or (not ns.top and not ns.bottom):
        print("Enter TOP:", file=sys.stderr); ns.top=input().strip(); print("Enter BOTTOM:", file=sys.stderr); ns.bottom=input().strip()
    top=ca_positions_for(atoms, parse_residue_tokens(ns.top))
    bot=ca_positions_for(atoms, parse_residue_tokens(ns.bottom))
    if top.size==0 or bot.size==0: print("ERROR: could not find CA for selections", file=sys.stderr); return 3
    c_top=top.mean(axis=0); c_bot=bot.mean(axis=0)

    custom={}
    if ns.vdwjson:
        import json
        with open(ns.vdwjson,'r') as jf: custom=json.load(jf)
    coords=np.array([[a.x,a.y,a.z] for a in atoms],float)
    radii=np.array([vdw_radius(a.element, custom) for a in atoms],float)
    meta=[(a.chain, a.resname, a.resi) for a in atoms]

    if ns.centerline=='straight':
        rows,u,L=profile_straight(coords,radii,c_bot,c_top,ns.step,ns.eps,meta,ns.adaptive,ns.slope_thresh,ns.max_refine,ns.hydroscale,ns.electroscale,ns.occupancy)
    else:
        rows,u,L=profile_curved(coords,radii,c_bot,c_top,ns.step,ns.eps,meta,ns.curve_radius,ns.curve_iters,ns.hydroscale,ns.electroscale,ns.occupancy)

    outprefix=Path(ns.outprefix)
    if outprefix.suffix.lower()=='.csv': outprefix=outprefix.with_suffix('')
    csv_path=outprefix.with_suffix('.csv'); mesh_path=outprefix.with_name(outprefix.stem+'_mesh.pdb'); center_path=outprefix.with_name(outprefix.stem+'_centerline.pdb')
    write_csv(csv_path, rows); mesh_pdb(mesh_path, rows, axis_u=(c_top-c_bot), rings=ns.rings); centerline_pdb(center_path, rows)

    print(f"Wrote: {csv_path}")
    print(f"Wrote: {mesh_path}")
    print(f"Wrote: {center_path}")

if __name__=='__main__':
    sys.exit(main())
