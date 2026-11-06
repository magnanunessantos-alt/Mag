"""Limpa a malha de um 3MF de uma cor: remove pedacinhos soltos e triangulos degenerados
e conserta arestas non-manifold. Uso: python3 limpar_malha.py arquivo.3mf (sobrescreve)"""
import sys, re, zipfile, numpy as np, trimesh, pymeshfix
f = sys.argv[1]
z = zipfile.ZipFile(f); names = z.namelist(); data = {n: z.read(n) for n in names}; z.close()
xml = data["3D/3dmodel.model"].decode()
V = np.array(re.findall(r'<vertex x="([-\d.]+)" y="([-\d.]+)" z="([-\d.]+)"/>', xml), float)
F = np.array(re.findall(r'<triangle v1="(\d+)" v2="(\d+)" v3="(\d+)"', xml), int)
m = trimesh.Trimesh(V, F, process=False); m.merge_vertices()
m.update_faces(m.area_faces > 1e-8); m.remove_unreferenced_vertices()
comps = m.split(only_watertight=False); m = max(comps, key=lambda c: len(c.faces))
def nm(mm):
    u, c = np.unique(mm.edges_sorted, axis=0, return_counts=True); return int((c > 2).sum())
if nm(m) or not m.is_watertight:
    Vf, Ff = pymeshfix.clean_from_arrays(np.ascontiguousarray(m.vertices, dtype=np.float64), np.ascontiguousarray(m.faces, dtype=np.int32))
    m = trimesh.Trimesh(Vf, Ff, process=False)
if m.volume < 0: m.invert()
m.vertices -= [0, 0, m.vertices[:, 2].min()]
vs = "".join(f'<vertex x="{a:.3f}" y="{b:.3f}" z="{c:.3f}"/>' for a, b, c in m.vertices)
ts = "".join(f'<triangle v1="{a}" v2="{b}" v3="{c}"/>' for a, b, c in m.faces)
new = re.sub(r"<vertices>.*</vertices>", "<vertices>" + vs + "</vertices>", xml, flags=re.S)
new = re.sub(r"<triangles>.*</triangles>", "<triangles>" + ts + "</triangles>", new, flags=re.S)
data["3D/3dmodel.model"] = new.encode()
with zipfile.ZipFile(f, "w", zipfile.ZIP_DEFLATED) as z2:
    for n in names: z2.writestr(n, data[n])
print(f.split("/")[-1], "faces", len(m.faces), "estanque", m.is_watertight, "non-manifold", nm(m), "partes", m.body_count, "vol", round(m.volume / 1000, 1))
