
"""
L-DED second-law analysis: entropy generation + interface metric
Assumptions:
- Uniform cubic grid with dx=dy=dz specified
- CSV columns: 'X [ m ]',' Y [ m ]',' Z [ m ]',' Bulk Temperature [ K ]',
               ' Velocity u [ m s^-1 ]',' Velocity v [ m s^-1 ]',' Velocity w [ m s^-1 ]',
               ' Fluid.Liquid Fraction'
"""
import pandas as pd, numpy as np, re, os
from collections import defaultdict

# ---- User constants (edit as needed) ----
DX = DY = DZ = 2e-4       # m
V_CELL = DX*DY*DZ
K = 30.0                  # W/m/K
MU = 0.0073               # Pa*s
T_INF = 298.0             # K
DT = 2.5e-05              # s
CSV_GLOB = "step_*.csv"   # pattern

def add_indices(df):
    x = df['X [ m ]'].values
    y = df[' Y [ m ]'].values
    z = df[' Z [ m ]'].values
    x0, y0, z0 = x.min(), y.min(), z.min()
    ix = np.rint((x - x0)/DX).astype(np.int64)
    iy = np.rint((y - y0)/DY).astype(np.int64)
    iz = np.rint((z - z0)/DZ).astype(np.int64)
    df['_ix'] = ix; df['_iy'] = iy; df['_iz'] = iz
    mapping = defaultdict(lambda: -1)
    for i,(a,b,c) in enumerate(zip(ix,iy,iz)): mapping[(a,b,c)] = i
    return mapping

def central_diff(field, df, mapping):
    n = len(df); gx = np.zeros(n); gy = np.zeros(n); gz = np.zeros(n)
    ix = df['_ix'].values; iy = df['_iy'].values; iz = df['_iz'].values
    arr = df[field].values
    for i in range(n):
        a,b,c = ix[i],iy[i],iz[i]
        # x
        im = mapping[(a-1,b,c)]; ip = mapping[(a+1,b,c)]
        if im>=0 and ip>=0: gx[i] = (arr[ip]-arr[im])/(2*DX)
        elif ip>=0:         gx[i] = (arr[ip]-arr[i])/DX
        elif im>=0:         gx[i] = (arr[i]-arr[im])/DX
        else:               gx[i] = 0.0
        # y
        jm = mapping[(a,b-1,c)]; jp = mapping[(a,b+1,c)]
        if jm>=0 and jp>=0: gy[i] = (arr[jp]-arr[jm])/(2*DY)
        elif jp>=0:         gy[i] = (arr[jp]-arr[i])/DY
        elif jm>=0:         gy[i] = (arr[i]-arr[jm])/DY
        else:               gy[i] = 0.0
        # z
        km = mapping[(a,b,c-1)]; kp = mapping[(a,b,c+1)]
        if km>=0 and kp>=0: gz[i] = (arr[kp]-arr[km])/(2*DZ)
        elif kp>=0:         gz[i] = (arr[kp]-arr[i])/DZ
        elif km>=0:         gz[i] = (arr[i]-arr[km])/DZ
        else:               gz[i] = 0.0
    return gx,gy,gz

def process(path, step_idx):
    df = pd.read_csv(path)
    mapping = add_indices(df)
    # Gradients
    dTx,dTy,dTz = central_diff(' Bulk Temperature [ K ]', df, mapping)
    dux, duy, duz = central_diff(' Velocity u [ m s^-1 ]', df, mapping)
    dvx, dvy, dvz = central_diff(' Velocity v [ m s^-1 ]', df, mapping)
    dwx, dwy, dwz = central_diff(' Velocity w [ m s^-1 ]', df, mapping)
    glx, gly, glz = central_diff(' Fluid.Liquid Fraction', df, mapping)
    T = np.clip(df[' Bulk Temperature [ K ]'].values, 1.0, None)

    # Entropy generation
    gradT2 = dTx**2 + dTy**2 + dTz**2
    s_q = K * gradT2 / (T**2)
    Exx = dux; Eyy = dvy; Ezz = dwz
    Exy = 0.5*(duy + dvx); Exz = 0.5*(duz + dwx); Eyz = 0.5*(dvz + dwy)
    Phi = 2.0 * MU * (Exx**2 + Eyy**2 + Ezz**2 + Exy**2 + Exz**2 + Eyz**2)
    s_mu = Phi / T
    S_q = float(np.sum(s_q) * V_CELL)
    S_mu = float(np.sum(s_mu) * V_CELL)
    grad_alpha = np.sqrt(glx**2 + gly**2 + glz**2)
    iface_area_proxy = float(np.sum(grad_alpha) * V_CELL)
    return {
        "step": step_idx,
        "time_s": step_idx*DT,
        "Sgen_heat_W_per_K": S_q,
        "Sgen_visc_W_per_K": S_mu,
        "Sgen_total_W_per_K": S_q+S_mu,
        "iface_area_proxy_m2": iface_area_proxy,
        "Tmax_K": float(T.max()),
        "Tmean_K": float(T.mean()),
        "Ncells": len(df)
    }

if __name__ == "__main__":
    files = sorted([f for f in os.listdir(".") if re.match(r"step_\d+\.csv", f)], key=lambda s:int(re.findall(r"step_(\d+)\.csv", s)[0]))
    rows = []
    for f in files:
        step = int(re.findall(r"step_(\d+)\.csv", f)[0])
        rows.append(process(f, step))
    out = pd.DataFrame(rows).sort_values("time_s")
    out.to_csv("entropy_summary.csv", index=False)
    print(out)
