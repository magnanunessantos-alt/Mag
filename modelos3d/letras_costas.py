"""Apaga as letras tortas que a IA esculpiu nas costas e coloca letras novas, limpas, feitas
a partir de uma fonte (em relevo solido ou em tachinhas), acompanhando a curva das costas.

Uso: python3 letras_costas.py entrada.3mf saida.3mf config.json
config: {"apagar": [xmin,xmax,zmin,zmax], "fundo": idx_cor, "letra": idx_cor,
         "estilo": "solido"|"tachas", "fonte": caminho.ttf,
         "linhas": [{"texto": "JONAS", "caixa": [xmin,xmax,zmin,zmax]}, ...],
         "relevo": 0.7, "tacha_raio": 0.45, "tacha_passo": 1.05}
"""
import sys, re, json, zipfile, numpy as np, trimesh, manifold3d as mf, pymeshfix
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import cKDTree
from skimage.measure import find_contours
from skimage.morphology import skeletonize

src, dst, cfgp = sys.argv[1:4]
cfg = json.load(open(cfgp))
RELEVO = cfg.get("relevo", 0.7)

xml = zipfile.ZipFile(src).read("3D/3dmodel.model").decode()
V = np.array(re.findall(r'<vertex x="([-\d.]+)" y="([-\d.]+)" z="([-\d.]+)"/>', xml), float)
T = np.array(re.findall(r'<triangle v1="(\d+)" v2="(\d+)" v3="(\d+)" pid="1" p1="(\d+)"', xml), int)
F, L = T[:, :3], T[:, 3].copy()
base_xml = re.search(r'<basematerials id="1">.*?</basematerials>', xml, re.S).group(0)
m = trimesh.Trimesh(V, F, process=False)
vn = m.vertex_normals

# --- 1) apagar as letras antigas: alisa a area das costas (borda com transicao suave)
x0, x1, z0, z1 = cfg["apagar"]
YMIN = cfg.get("y_min", 1.5)          # metade de tras do corpo
back = V[:, 1] > YMIN
def wbox(P, pad):
    tx = np.clip(np.minimum(P[:, 0] - x0, x1 - P[:, 0]) / pad, 0, 1)
    tz = np.clip(np.minimum(P[:, 2] - z0, z1 - P[:, 2]) / pad, 0, 1)
    t = np.minimum(tx, tz); return t * t * (3 - 2 * t)
w = wbox(V, 1.5) * back
move = w > 0
ed = m.edges_unique
V2 = V.copy()
for it in range(cfg.get("iter_apagar", 200)):
    acc = np.zeros_like(V2); cnt = np.zeros(len(V2))
    np.add.at(acc, ed[:, 0], V2[ed[:, 1]]); np.add.at(acc, ed[:, 1], V2[ed[:, 0]])
    np.add.at(cnt, ed[:, 0], 1); np.add.at(cnt, ed[:, 1], 1)
    V2[move] += 0.5 * w[move, None] * (acc[move] / cnt[move, None] - V2[move])
cenF = V2[F].mean(1)
inbox = (cenF[:, 0] > x0) & (cenF[:, 0] < x1) & (cenF[:, 2] > z0) & (cenF[:, 2] < z1) & (cenF[:, 1] > YMIN)
L[inbox] = cfg["fundo"]
print("apagado:", int(move.sum()), "vertices alisados,", int(inbox.sum()), "faces repintadas")

# superficie das costas (y em funcao de x,z) para as letras acompanharem a curva
bv = back & (V2[:, 0] > x0 - 3) & (V2[:, 0] < x1 + 3) & (V2[:, 2] > z0 - 3) & (V2[:, 2] < z1 + 3)
tree = cKDTree(V2[bv][:, [0, 2]]); ybv = V2[bv, 1]
def surf(x, z):
    d, i = tree.query(np.c_[x, z], k=12)
    yy = np.where(d < 0.8, ybv[i], -np.inf)          # camada mais externa (a de tras)
    return np.where(np.isfinite(yy.max(1)), yy.max(1), ybv[i].max(1))

# --- 2) texto a partir da fonte
PX = 40.0  # pixels por mm
def mask_line(texto, caixa):
    fnt = ImageFont.truetype(cfg["fonte"], 400)
    l, t, r, b = fnt.getbbox(texto)
    im = Image.new("L", (r - l + 40, b - t + 40), 0)
    ImageDraw.Draw(im).text((20 - l, 20 - t), texto, font=fnt, fill=255)
    a = np.asarray(im) > 127
    ys, xs = np.nonzero(a); a = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    bx0, bx1, bz0, bz1 = caixa
    s = min((bx1 - bx0) / a.shape[1], (bz1 - bz0) / a.shape[0])   # mm por pixel da fonte
    W, H = a.shape[1] * s, a.shape[0] * s
    cx, cz = (bx0 + bx1) / 2, (bz0 + bz1) / 2
    im2 = Image.fromarray(a.astype(np.uint8) * 255).resize((int(W * PX), int(H * PX)), Image.LANCZOS)
    a2 = np.asarray(im2) > 127
    # pixel (linha i, coluna j) -> mm: visto de tras, a direita do leitor e -x
    def to_mm(i, j):
        return cx + W / 2 - j / PX, cz + H / 2 - i / PX
    return a2, to_mm

parts = []
glyph_masks = []
for ln in cfg["linhas"]:
    a, to_mm = mask_line(ln["texto"], ln["caixa"])
    glyph_masks.append((a, to_mm))
    if cfg["estilo"] == "solido":
        pad = np.pad(a, 2)
        polys = []
        for c in find_contours(pad.astype(float), 0.5):
            xm, zm = to_mm(c[:, 0] - 2, c[:, 1] - 2)
            polys.append(np.c_[xm, zm][::4])
        cs = mf.CrossSection(polys, mf.FillRule.EvenOdd)
        ext = mf.Manifold.extrude(cs, 1.0)          # z de 0 a 1 (sera mapeado para a profundidade)
        mesh = ext.to_mesh()
        P = np.asarray(mesh.vert_properties)[:, :3].astype(float); TF = np.asarray(mesh.tri_verts)
        # subdividir para acompanhar a curva
        P, TF = trimesh.remesh.subdivide_to_size(P, TF, max_edge=0.6)
        ys = surf(P[:, 0], P[:, 1])
        out = np.c_[P[:, 0], ys - 1.2 + P[:, 2] * (1.2 + RELEVO), P[:, 1]]
        TF = TF[:, ::-1]  # troca y<->z inverte a orientacao
        parts.append(mf.Manifold(mf.Mesh(vert_properties=out.astype(np.float32), tri_verts=TF.astype(np.uint32))))
    else:  # tachas: bolinhas ao longo do esqueleto das letras
        sk = skeletonize(a)
        ii, jj = np.nonzero(sk)
        pts = np.c_[to_mm(ii, jj)]
        step = cfg.get("tacha_passo", 1.05); chosen = []
        order = np.random.default_rng(3).permutation(len(pts))
        for p in pts[order]:
            if chosen and np.min(np.linalg.norm(np.array(chosen) - p, axis=1)) < step: continue
            chosen.append(p)
        chosen = np.array(chosen)
        rr = cfg.get("tacha_raio", 0.45)
        ys = surf(chosen[:, 0], chosen[:, 1])
        sph = [mf.Manifold.sphere(rr, 20).translate([x, y + 0.25 * rr, z]) for (x, z), y in zip(chosen, ys)]
        parts.append(mf.Manifold.batch_boolean(sph, mf.OpType.Add))
        print(ln["texto"], len(chosen), "tachinhas")

letters = mf.Manifold.batch_boolean(parts, mf.OpType.Add)
Vf, Ff = pymeshfix.clean_from_arrays(V2, F)
body = mf.Manifold(mf.Mesh(vert_properties=np.asarray(Vf, np.float32), tri_verts=np.asarray(Ff, np.uint32)))
print("corpo", body.status(), "| letras", letters.status())
res = (body + letters).to_mesh()
Vn = np.asarray(res.vert_properties)[:, :3].astype(float); Fn = np.asarray(res.tri_verts).astype(int)
mn = trimesh.Trimesh(Vn, Fn, process=False)
print("resultado:", len(Fn), "faces, estanque", mn.is_watertight)

# --- 3) cores: da face antiga mais proxima; o que saltou acima da superficie na area = letra
cn = mn.triangles_center
Ln = L[cKDTree(cenF).query(cn, workers=-1)[1]]
inb = (cn[:, 0] > x0) & (cn[:, 0] < x1) & (cn[:, 2] > z0) & (cn[:, 2] < z1) & (cn[:, 1] > YMIN)
above = np.zeros(len(cn), bool)
above[inb] = cn[inb, 1] > surf(cn[inb, 0], cn[inb, 2]) + 0.08
Ln[inb] = cfg["fundo"]; Ln[above] = cfg["letra"]

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
