"""Troca o nariz de um boneco (3MF colorido gerado pelo finalizar_3mf.py) por um nariz
triangular pequeno e arredondado, no estilo vinil. Mantem as cores.

Uso: python3 ajustar_nariz.py entrada.3mf saida.3mf IDX_PELE IDX_OLHO [largura altura saliencia]
(largura/altura/saliencia relativas ao raio do olho; padrao 0.75 0.95 0.5)
"""
import sys, re, zipfile, numpy as np, trimesh, manifold3d as mf, pymeshfix
from scipy.spatial import cKDTree

src, dst = sys.argv[1], sys.argv[2]
PELE, OLHO = int(sys.argv[3]), int(sys.argv[4])
LARG, ALT, SAL = (float(v) for v in (sys.argv[5:8] if len(sys.argv) > 7 else (0.75, 0.95, 0.5)))

xml = zipfile.ZipFile(src).read("3D/3dmodel.model").decode()
V = np.array(re.findall(r'<vertex x="([-\d.]+)" y="([-\d.]+)" z="([-\d.]+)"/>', xml), float)
T = re.findall(r'<triangle v1="(\d+)" v2="(\d+)" v3="(\d+)" pid="1" p1="(\d+)"', xml)
T = np.array(T, int); F, L = T[:, :3], T[:, 3]
base_xml = re.search(r'<basematerials id="1">.*?</basematerials>', xml, re.S).group(0)
codes = dict(re.findall(r'p1="(\d+)" paint_color="([0-9A-F]+)"', xml[:200000]))
m = trimesh.Trimesh(V, F, process=False)
cen = m.triangles_center

# posicao dos olhos/nariz salva pelo finalizar_3mf.py (saida_rosto.json)
import json
info = json.load(open(src.replace(".3mf", "_rosto.json")))
(x1, z1, r1), (x2, z2, r2) = info["olhos"]
r = (r1 + r2) / 2; ec = np.array([(x1 + x2) / 2, 0.0, (z1 + z2) / 2])
front = cen[:, 1] < np.median(cen[:, 1])
print("olhos x", round(x1, 1), round(x2, 1), "z", round(ec[2], 1), "raio", round(r, 2))

# nariz atual: o que sai mais para a frente entre os olhos e abaixo deles
cand = front & (np.abs(cen[:, 0] - ec[0]) < 0.8 * r) & (cen[:, 2] < ec[2] + 0.5 * r) & (cen[:, 2] > ec[2] - 2.4 * r)
tip = np.array(info["nariz"])
dxz = np.linalg.norm(V[:, [0, 2]] - tip[[0, 2]], axis=1)
ring = (dxz > 1.0 * r) & (dxz < 1.4 * r) & (np.abs(V[:, 1] - tip[1]) < 2.5 * r)
def basis(x, z):
    x = np.asarray(x, float) - tip[0]; z = np.asarray(z, float) - tip[2]
    return np.stack([np.ones_like(x), x, z, x * x, z * z, x * z], -1)
cf = np.linalg.lstsq(basis(V[ring, 0], V[ring, 2]), V[ring, 1], rcond=None)[0]   # superficie curva do rosto
plane = lambda x, z: basis(x, z) @ cf
prot = plane(V[:, 0], V[:, 2]) - V[:, 1]
old = (dxz < 1.0 * r) & (prot > 0.15)
print("nariz antigo: sai", round(prot[old].max(), 2), "mm,", int(old.sum()), "vertices")
V2 = V.copy()
t = np.clip((1.0 * r - dxz[old]) / (0.25 * r), 0, 1)
V2[old, 1] += (plane(V2[old, 0], V2[old, 2]) + 0.05 - V2[old, 1]) * (t * t * (3 - 2 * t))   # achata o nariz antigo

# nariz novo: casco convexo de esferas (triangulo arredondado)
x0 = ec[0]; zt = ec[2] - 0.45 * r; zb = zt - ALT * r; hw = LARG * r / 2; h = SAL * r
def P(x, z, out_, rad):
    return mf.Manifold.sphere(rad, 24).translate([x, plane(x, z) - out_, z])
pts = [P(x0, zt, 0.05, 0.22 * r * 0.6),                       # topo fino
       P(x0 - hw, zb + 0.12 * r, 0.05, 0.16 * r), P(x0 + hw, zb + 0.12 * r, 0.05, 0.16 * r),
       P(x0, zb + 0.2 * r, h - 0.18 * r, 0.18 * r),            # ponta, mais embaixo
       P(x0, zt - 0.3 * ALT * r, 0.45 * h, 0.2 * r)]
inner = [mf.Manifold.sphere(0.3 * r, 16).translate([x0, plane(x0, z) + 0.8 * r, z]) for z in (zt, zb)]
nose = mf.Manifold.batch_hull(pts + inner)

Vf, Ff = pymeshfix.clean_from_arrays(V2, F)
body = mf.Manifold(mf.Mesh(vert_properties=np.asarray(Vf, np.float32), tri_verts=np.asarray(Ff, np.uint32)))
print("corpo:", body.status(), "| nariz:", nose.status())
res = (body + nose).to_mesh()
Vn = np.asarray(res.vert_properties)[:, :3].astype(float); Fn = np.asarray(res.tri_verts).astype(int)
mn = trimesh.Trimesh(Vn, Fn, process=False)
print("resultado:", len(Fn), "faces, estanque", mn.is_watertight)

# cores: da face antiga mais proxima; nariz novo = pele
cn = mn.triangles_center
Ln = L[cKDTree(cen).query(cn, workers=-1)[1]]
inside_nose = np.linalg.norm(cn[:, [0, 2]] - [x0, (zt + zb) / 2], axis=1) < 0.9 * r
Ln[inside_nose & (plane(cn[:, 0], cn[:, 2]) - cn[:, 1] > -0.3)] = PELE

def paint_code(i):
    s = i + 1
    return "4" if s == 1 else "8" if s == 2 else "%XC" % (s - 3)
vs = "".join(f'<vertex x="{a:.3f}" y="{b:.3f}" z="{c:.3f}"/>' for a, b, c in Vn)
ts = "".join(f'<triangle v1="{a}" v2="{b}" v3="{c}" pid="1" p1="{k}" paint_color="{paint_code(k)}" slic3rpe:mmu_segmentation="{paint_code(k)}"/>'
             for (a, b, c), k in zip(Fn, Ln))
head = xml[:xml.index("<resources>")]
model = (head + f"<resources>{base_xml}" +
         f'<object id="2" name="boneco" type="model" pid="1" pindex="0"><mesh><vertices>{vs}</vertices>'
         f"<triangles>{ts}</triangles></mesh></object></resources>"
         '<build><item objectid="2" transform="1 0 0 0 1 0 0 0 1 128 128 0"/></build></model>')
zin = zipfile.ZipFile(src)
with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as z:
    z.writestr("[Content_Types].xml", zin.read("[Content_Types].xml"))
    z.writestr("_rels/.rels", zin.read("_rels/.rels"))
    z.writestr("3D/3dmodel.model", model)
names = re.findall(r'<base name="([^"]+)" displaycolor="(#[0-9A-Fa-f]+)"', base_xml)
fc = np.array([[int(h[i:i + 2], 16) for i in (1, 3, 5)] + [255] for _, h in names], np.uint8)
pv = trimesh.Trimesh(Vn, Fn, face_colors=fc[Ln], process=False); pv.unmerge_vertices()
pv.export(dst.replace(".3mf", "_previa.glb"))
print("salvo", dst)
