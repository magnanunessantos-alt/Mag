"""Junta a CABECA de um modelo (A) com o CORPO de outro (B), alinhando pelos olhos.
Usado no boneco de jaqueta amarela: A = geracao frente+costas (cabelo cacheado, mas 3 maos),
B = geracao so de frente (maos certas). Saida: .npz para o finalizar_3mf.py.

Uso: python3 juntar_cabeca_corpo.py A.glb B.glb saida.npz
"""
import sys, numpy as np, trimesh, manifold3d as mf
import scipy.sparse as sps, scipy.sparse.csgraph as csg

A_SRC, B_SRC, OUT = sys.argv[1:4]
H = 100.0

def to_lab(rgb):
    r = rgb / 255.0
    r = np.where(r > 0.04045, ((r + 0.055) / 1.055) ** 2.4, r / 12.92)
    M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = r @ M.T / np.array([0.9505, 1.0, 1.089])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[:, 1] - 16, 500 * (f[:, 0] - f[:, 1]), 200 * (f[:, 1] - f[:, 2])], 1)

def load(path):
    g = list(trimesh.load(path, process=False).dump())[0]
    tex = np.asarray(g.visual.material.baseColorTexture.convert("RGB")).astype(np.float32)
    th, tw = tex.shape[:2]; uv = g.visual.uv
    V = g.vertices.copy(); F = g.faces
    V = np.stack([-V[:, 2], -V[:, 0], V[:, 1]], 1)
    V -= [V[:, 0].mean(), V[:, 1].mean(), V[:, 2].min()]
    V *= H / V[:, 2].max()
    fu = uv[F].mean(1)
    x = np.clip((fu[:, 0] % 1) * (tw - 1), 0, tw - 1).round().astype(int)
    y = np.clip((1 - fu[:, 1] % 1) * (th - 1), 0, th - 1).round().astype(int)
    C = tex[y, x]
    m = trimesh.Trimesh(V, F, process=False)
    m.merge_vertices(merge_tex=True, merge_norm=True)
    comp = trimesh.graph.connected_component_labels(m.face_adjacency, node_count=len(m.faces))
    keep = np.bincount(comp)[comp] >= 0.002 * len(m.faces)
    m.update_faces(keep); m.remove_unreferenced_vertices()
    return m, C[keep]

def eyes(m, C):
    lab = to_lab(C); cen = m.triangles_center
    front = cen[:, 1] < np.median(cen[:, 1])
    dark = (lab[:, 0] < 22) & front & (cen[:, 2] > 0.62 * H)
    adj = m.face_adjacency
    both = dark[adj[:, 0]] & dark[adj[:, 1]]
    G = sps.coo_matrix((np.ones(both.sum()), (adj[both, 0], adj[both, 1])), shape=(len(cen),) * 2)
    _, comp = csg.connected_components(G, directed=False)
    best = {}
    for cid in np.unique(comp[dark]):
        s = dark & (comp == cid)
        A = m.area_faces[s].sum()
        if A < 4: continue
        p = cen[s]; ext = np.ptp(p[:, [0, 2]], 0)
        ny = np.average(m.face_normals[s][:, 1], weights=m.area_faces[s])
        if 0.55 < ext[0] / ext[1] < 1.8 and ny < -0.6 and (ext > 3.5).all() and (ext < 15).all():
            side = int(p[:, 0].mean() > 0)
            if A > best.get(side, (0,))[0]:
                best[side] = (A, (np.percentile(p, 1.5, 0) + np.percentile(p, 98.5, 0)) / 2, p[:, 1].min())
    assert len(best) >= 1, "nenhum olho"
    if len(best) == 1:   # espelha pelo nariz (ponto mais a frente abaixo dos olhos)
        (side, (A_, c_, y_)), = best.items()
        nz = front & (cen[:, 2] < c_[2]) & (cen[:, 2] > c_[2] - 12) & (np.abs(cen[:, 0]) < 8)
        nose = cen[nz][np.argmin(cen[nz, 1])]
        c2 = c_.copy(); c2[0] = 2 * nose[0] - c_[0]
        best[1 - side] = (A_, c2, y_)
        print("olho espelhado pelo nariz")
    return np.array([best[0][1], best[1][1]]), min(best[0][2], best[1][2])

mA, CA = load(A_SRC); mB, CB = load(B_SRC)
print("A", len(mA.faces), mA.is_watertight, "| B", len(mB.faces), mB.is_watertight)
eA, yA = eyes(mA, CA); eB, yB = eyes(mB, CB)
s = np.linalg.norm(eB[1, [0, 2]] - eB[0, [0, 2]]) / np.linalg.norm(eA[1, [0, 2]] - eA[0, [0, 2]])
cA, cB = eA.mean(0), eB.mean(0)
VA = (mA.vertices - cA) * s + cB
VA[:, 1] += yB - ((yA - cA[1]) * s + cB[1])   # superficie dos olhos no mesmo plano
print("escala A->B", round(s, 3), "olhos B", eB.round(1).tolist())

# queixo de B: pele mais baixa na linha do centro do rosto
labB = to_lab(CB); cenB = mB.triangles_center
band = (np.abs(cenB[:, 0] - cB[0]) < 3) & (cenB[:, 1] < cB[1] + 2) & (cenB[:, 2] < cB[2]) & (cenB[:, 2] > cB[2] - 30)
skin = band & (labB[:, 1] > 9) & (labB[:, 2] > 15) & (labB[:, 2] < 45) & (labB[:, 0] > 45)
chin = np.percentile(cenB[skin, 2], 1)
z_cut, ov = chin - 1.0, 2.5
print("queixo z", round(chin, 1), "corte", round(z_cut, 1))

import fast_simplification, pymeshfix
def M(V, F):
    V, F = fast_simplification.simplify(np.asarray(V, np.float64), np.asarray(F, np.int64), target_reduction=0.6)
    V, F = pymeshfix.clean_from_arrays(V, F)          # malha 100% fechada e manifold
    r = mf.Manifold(mf.Mesh(vert_properties=np.asarray(V, np.float32), tri_verts=np.asarray(F, np.uint32)))
    print("  parte:", len(F), "faces,", r.status())
    return r
head = M(VA, mA.faces).trim_by_plane([0, 0, 1], z_cut)
body = M(mB.vertices, mB.faces).trim_by_plane([0, 0, -1], -(z_cut + ov))
res = (head + body).to_mesh()
V = np.asarray(res.vert_properties)[:, :3]; F = np.asarray(res.tri_verts)
out = trimesh.Trimesh(V, F, process=False)
print("uniao:", len(F), "faces, estanque", out.is_watertight, "corpos", out.body_count)

tA = VA[mA.faces]; tB = mB.vertices[mB.faces]
kA = tA.mean(1)[:, 2] > z_cut - 0.5; kB = tB.mean(1)[:, 2] < z_cut + ov + 0.5
np.savez(OUT, V=V, F=F, SRC_T=np.concatenate([tA[kA], tB[kB]]).astype(np.float32),
         SRC_C=np.concatenate([CA[kA], CB[kB]]).astype(np.float32))
print("salvo", OUT)
