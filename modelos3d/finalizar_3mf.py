"""Converte um GLB texturizado (ex.: gerado no Magnific/Tripo) em 3MF multicolor
pronto para Bambu Studio / Orca / PrusaSlicer, e tambem OBJ+MTL.

Uso: python3 finalizar_3mf.py entrada.glb saida_sem_ext altura_mm n_cores faces_alvo
"""
import sys, zipfile, numpy as np, trimesh, fast_simplification
from scipy.spatial import cKDTree

src, out = sys.argv[1], sys.argv[2]
H_MM = float(sys.argv[3]); K = int(sys.argv[4]); TARGET = int(sys.argv[5])

scene = trimesh.load(src, process=False)
g = list(scene.dump()) if hasattr(scene, "dump") else [scene]
g = g[0]
tex = np.asarray(g.visual.material.baseColorTexture.convert("RGB")).astype(np.float32)
th, tw = tex.shape[:2]
uv = g.visual.uv
V0 = g.vertices.copy(); F0 = g.faces

# eixo Y-up (glTF, metros) -> Z-up (mm), frente para -Y
V0 = np.stack([-V0[:, 2], -V0[:, 0], V0[:, 1]], 1)  # Tripo: frente em +X -> girar para -Y
V0 -= [V0[:, 0].mean(), V0[:, 1].mean(), V0[:, 2].min()]
V0 *= H_MM / V0[:, 2].max()

def sample(uvs):
    x = np.clip((uvs[:, 0] % 1.0) * (tw - 1), 0, tw - 1)
    y = np.clip((1 - (uvs[:, 1] % 1.0)) * (th - 1), 0, th - 1)
    return tex[y.round().astype(int), x.round().astype(int)]

# cor de cada face original = media de 4 amostras (centro + perto dos vertices)
fu = uv[F0]
c = sample(fu.mean(1))
for w in ([.6, .2, .2], [.2, .6, .2], [.2, .2, .6]):
    c += sample(np.einsum("k,fkd->fd", np.array(w), fu))
col0 = c / 4.0

# --- simplificar malha
mm = trimesh.Trimesh(V0, F0, process=False)
mm.merge_vertices(merge_tex=True, merge_norm=True)   # fecha as costuras de UV
comp = trimesh.graph.connected_component_labels(mm.face_adjacency, node_count=len(mm.faces))
sizes = np.bincount(comp)
mm.update_faces(sizes[comp] >= 0.002 * len(mm.faces))   # descarta fragmentos soltos
mm.remove_unreferenced_vertices()
print("vertices", len(V0), "->", len(mm.vertices), "estanque original", mm.is_watertight)
V, F = fast_simplification.simplify(mm.vertices.astype(np.float64), mm.faces.astype(np.int64),
                                    target_reduction=max(0.0, 1 - TARGET / len(mm.faces)))
m = trimesh.Trimesh(V, F, process=False)
m.remove_unreferenced_vertices()
if not m.is_winding_consistent:
    trimesh.repair.fix_winding(m)
if m.volume < 0:
    m.invert()
openE = len(trimesh.grouping.group_rows(m.edges_sorted, require_count=1))
print("faces", len(F0), "->", len(m.faces), "estanque", m.is_watertight, "arestas abertas", openE,
      "winding ok", m.is_winding_consistent)

# cor de cada face nova = cor da face original mais proxima (amostras no triangulo)
tree = cKDTree(V0[F0].mean(1))
tri = m.triangles
pts = [tri.mean(1)] + [np.einsum("k,fkd->fd", np.array(w), tri) for w in ([.6, .2, .2], [.2, .6, .2], [.2, .2, .6])]
cols = np.zeros((len(m.faces), 3))
for p in pts:
    cols += col0[tree.query(p, workers=-1)[1]]
cols /= len(pts)

# --- k-means em espaco Lab aproximado, ponderado por area
def to_lab(rgb):
    r = rgb / 255.0
    r = np.where(r > 0.04045, ((r + 0.055) / 1.055) ** 2.4, r / 12.92)
    M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = r @ M.T / np.array([0.9505, 1.0, 1.089])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[:, 1] - 16, 500 * (f[:, 0] - f[:, 1]), 200 * (f[:, 1] - f[:, 2])], 1)

import os
if os.environ.get("RAW"):
    trimesh.Trimesh(m.vertices, m.faces, face_colors=np.c_[cols.astype(np.uint8), np.full(len(cols), 255, np.uint8)], process=False).export(out + "_raw.glb")
lab = to_lab(cols)
area = m.area_faces
# paleta fixa de filamentos (nome, cor da amostra na textura, cor exibida)
PALETTE = [
    ("Preto",         ["#1D1615", "#2A201E", "#1B0F0F", "#2B1A16"], "#1A1414"),
    ("Marrom habito", ["#3F251D", "#4F3124", "#3E2319", "#4F2F20"], "#4A2C1F"),
    ("Pele/cabelo do Menino", ["#673C25", "#6B3D27", "#8B5522", "#8C5823"], "#6E4128"),
    ("Dourado",       ["#A86D2D", "#C28336", "#D69841", "#E3AA50", "#EFBA5A", "#BE7F2C",
                       "#C88F3C", "#DAA950", "#E5B554", "#E8BF6A"], "#DCA246"),
    ("Branco",        ["#DDD8D7", "#C1B6B4", "#988882", "#D9D7D6", "#BEB6B0"], "#EEEBE8"),
    ("Pele Menino",   ["#E5AD7B", "#D0945B", "#DCA689"], "#E8B48A"),
][:K]
if len(sys.argv) > 6:  # paleta opcional em JSON: [[nome, [amostras...], cor_exibida], ...]
    import json
    PALETTE = [tuple(p) for p in json.load(open(sys.argv[6]))]
hx =lambda h: [int(h[i:i + 2], 16) for i in (1, 3, 5)]
refs, ref_lbl = [], []
for i, (_, samples, _) in enumerate(PALETTE):
    for h in samples:
        refs.append(hx(h)); ref_lbl.append(i)
R = to_lab(np.array(refs, float)); ref_lbl = np.array(ref_lbl)
d = ((lab[:, None] - R[None]) ** 2 * np.array([0.6, 1.0, 1.0])).sum(2)  # menos peso na luminosidade (sombras)
lbl = ref_lbl[np.argmin(d, 1)]
K = len(PALETTE)

# --- limpar pontinhos: voto da vizinhanca (3 passadas), sem apagar detalhes grandes
adj = m.face_adjacency
for _ in range(3):
    votes = np.zeros((len(lbl), K))
    np.add.at(votes, (adj[:, 0], lbl[adj[:, 1]]), 1)
    np.add.at(votes, (adj[:, 1], lbl[adj[:, 0]]), 1)
    best = votes.argmax(1)
    change = votes[np.arange(len(lbl)), best] >= 2.5  # >=3 vizinhos de outra cor
    lbl = np.where(change & (votes[np.arange(len(lbl)), lbl] <= 0.5), best, lbl)

hexes = [p[2] for p in PALETTE]
names = [p[0] for p in PALETTE]
rgbC = np.array([hx(h) for h in hexes], float)
print("cores:", dict(zip(names, np.bincount(lbl, minlength=K).tolist())))

# --- exportar 3MF
Vm, Fm = m.vertices, m.faces
def paint_code(i):
    s = i + 1
    return "4" if s == 1 else "8" if s == 2 else "%XC" % (s - 3)
codes = [paint_code(i) for i in range(K)]
base_xml = "".join(f'<base name="{names[i]}" displaycolor="{h}"/>' for i, h in enumerate(hexes))
vs = "".join(f'<vertex x="{a:.3f}" y="{b:.3f}" z="{c_:.3f}"/>' for a, b, c_ in Vm)
ts = "".join(f'<triangle v1="{a}" v2="{b}" v3="{c_}" pid="1" p1="{k}" paint_color="{codes[k]}" slic3rpe:mmu_segmentation="{codes[k]}"/>'
             for (a, b, c_), k in zip(Fm, lbl))
model = ('<?xml version="1.0" encoding="UTF-8"?>\n<model unit="millimeter" xml:lang="pt-BR" '
         'xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02" '
         'xmlns:slic3rpe="http://schemas.slic3r.org/3mf/2017/06">'
         '<metadata name="Title">Santo Antonio</metadata>'
         f'<resources><basematerials id="1">{base_xml}</basematerials>'
         f'<object id="2" name="santo" type="model" pid="1" pindex="0"><mesh><vertices>{vs}</vertices>'
         f'<triangles>{ts}</triangles></mesh></object></resources>'
         '<build><item objectid="2" transform="1 0 0 0 1 0 0 0 1 128 128 0"/></build></model>')
ct = ('<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
      '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
      '<Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/></Types>')
rels = ('<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Target="/3D/3dmodel.model" Id="rel0" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/></Relationships>')
with zipfile.ZipFile(out + ".3mf", "w", zipfile.ZIP_DEFLATED) as z:
    z.writestr("[Content_Types].xml", ct); z.writestr("_rels/.rels", rels); z.writestr("3D/3dmodel.model", model)

# --- OBJ + MTL (Bambu/Orca convertem materiais em filamentos)
name = out.split("/")[-1]
with open(out + ".mtl", "w") as f:
    for i, c_ in enumerate(rgbC):
        f.write(f"newmtl cor{i+1}\nKd {c_[0]/255:.4f} {c_[1]/255:.4f} {c_[2]/255:.4f}\n\n")
with open(out + ".obj", "w") as f:
    f.write(f"mtllib {name}.mtl\n" + "".join(f"v {a:.3f} {b:.3f} {c_:.3f}\n" for a, b, c_ in Vm))
    for i in range(K):
        sel = Fm[lbl == i]
        if len(sel):
            f.write(f"usemtl cor{i+1}\n" + "".join(f"f {a+1} {b+1} {c_+1}\n" for a, b, c_ in sel))

fc = np.c_[rgbC[lbl].astype(np.uint8), np.full(len(lbl), 255, np.uint8)]
pv = trimesh.Trimesh(Vm, Fm, face_colors=fc, process=False); pv.unmerge_vertices(); pv.export(out + "_previa.glb")
print("dimensoes mm:", np.ptp(Vm, 0).round(1).tolist())
