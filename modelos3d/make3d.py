import sys, zipfile, numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage as ndi

src, out, height_mm = sys.argv[1], sys.argv[2], float(sys.argv[3])
N = 512
img = Image.open(src).convert("RGB").resize((N, N), Image.LANCZOS)
a = np.asarray(img).astype(float)

# --- mask: background = near-white region connected to the border
near_white = a.min(2) > 238
lab, _ = ndi.label(near_white)
border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
bg = np.isin(lab, list(border))
mask = ~bg
mask = ndi.binary_opening(mask, iterations=1)
mask = ndi.binary_fill_holes(mask)
lab, n = ndi.label(mask)
sizes = ndi.sum(mask, lab, range(1, n + 1))
mask = lab == (np.argmax(sizes) + 1)
# remove diagonal-only pixel contacts (non-manifold vertices)
for _ in range(5):
    m = mask
    d1 = m[:-1, :-1] & m[1:, 1:] & ~m[:-1, 1:] & ~m[1:, :-1]
    d2 = m[:-1, 1:] & m[1:, :-1] & ~m[:-1, :-1] & ~m[1:, 1:]
    if not (d1.any() or d2.any()):
        break
    m = m.copy()
    m[:-1, 1:] |= d1; m[1:, :-1] |= d2  # bridge them
    mask = m

# --- depth from inflated distance field + luminance detail
d = ndi.distance_transform_edt(mask)
d = ndi.gaussian_filter(d, 2.0) * mask
R = 0.11 * N
dc = np.minimum(d, R)
infl = np.sqrt(dc * (2 * R - dc)) + 0.35 * np.maximum(d - R, 0)
lum = a.mean(2) / 255.0
detail = lum - ndi.gaussian_filter(lum, 4)
t0 = 0.012 * N
front = t0 + 0.85 * infl + 6.0 * detail * mask
back = t0 + 0.55 * infl
front = ndi.gaussian_filter(front, 0.8)
back = ndi.gaussian_filter(back, 1.2)

# --- color quantization (k-means, 6 colors)
pix = a[mask]
rng = np.random.default_rng(0)
K = 6
C = pix[rng.choice(len(pix), K, replace=False)]
for _ in range(25):
    lbl = np.argmin(((pix[:, None] - C[None]) ** 2).sum(2), 1)
    C = np.array([pix[lbl == k].mean(0) if (lbl == k).any() else C[k] for k in range(K)])
cimg = np.full(mask.shape, -1); cimg[mask] = lbl
cimg_f = ndi.median_filter(np.where(mask, cimg, 0), size=3); cimg = np.where(mask, cimg_f, -1)

# --- mesh: vertices at pixel corners
H, W = mask.shape
ys, xs = np.nonzero(mask)
cm = np.zeros((H + 1, W + 1), bool)
for dy in (0, 1):
    for dx in (0, 1):
        cm[ys + dy, xs + dx] = True
def corner_avg(f):
    s = np.zeros((H + 1, W + 1)); c = np.zeros((H + 1, W + 1))
    fm = np.where(mask, f, 0); mm = mask.astype(float)
    for dy in (0, 1):
        for dx in (0, 1):
            s[dy:dy + H, dx:dx + W] += fm; c[dy:dy + H, dx:dx + W] += mm
    return s / np.maximum(c, 1)
F = corner_avg(front); B = corner_avg(back)
cid = -np.ones((H + 1, W + 1), int)
cy, cx = np.nonzero(cm); nc = len(cy)
cid[cy, cx] = np.arange(nc)
scale = height_mm / (ys.max() - ys.min() + 1)
X = (cx - (xs.min() + xs.max()) / 2) * scale
Z = (ys.max() + 1 - cy) * scale
vf = np.stack([X, -F[cy, cx] * scale, Z], 1)   # front faces -Y
vb = np.stack([X, B[cy, cx] * scale, Z], 1)
V = np.vstack([vf, vb])
tris, mats = [], []
a00 = cid[ys, xs]; a01 = cid[ys, xs + 1]; a10 = cid[ys + 1, xs]; a11 = cid[ys + 1, xs + 1]
col = cimg[ys, xs]
backcol = np.full_like(col, -2)
# front (outward -Y): orientation chosen so normals point outward
tris += [np.stack([a00, a10, a01], 1), np.stack([a01, a10, a11], 1)]; mats += [col, col]
tris += [np.stack([a00 + nc, a01 + nc, a10 + nc], 1), np.stack([a01 + nc, a11 + nc, a10 + nc], 1)]; mats += [col, col]
# walls on boundary edges
def wall(p, q, colv):
    t1 = np.stack([p, q, q + nc], 1); t2 = np.stack([p, q + nc, p + nc], 1)
    tris.extend([t1, t2]); mats.extend([colv, colv])
pad = np.pad(mask, 1)
up = ~pad[ys, xs + 1]; dn = ~pad[ys + 2, xs + 1]; lf = ~pad[ys + 1, xs]; rt = ~pad[ys + 1, xs + 2]
wall(a00[up], a01[up], col[up])
wall(a11[dn], a10[dn], col[dn])
wall(a10[lf], a00[lf], col[lf])
wall(a01[rt], a11[rt], col[rt])
T = np.vstack(tris); M = np.concatenate(mats)

# --- base plate under the feet
zmin_mesh = 0.0
bw = (xs.max() - xs.min()) * scale * 0.75
yb0, yb1 = V[:, 1].min() - 4, V[:, 1].max() + 4
bh = 4.0
bx0, bx1 = -bw / 2, bw / 2
bv = np.array([[bx0, yb0, -bh], [bx1, yb0, -bh], [bx1, yb1, -bh], [bx0, yb1, -bh],
               [bx0, yb0, 0.6], [bx1, yb0, 0.6], [bx1, yb1, 0.6], [bx0, yb1, 0.6]])
bt = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
               [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]])

# fix orientation using trimesh, export
import trimesh
fig = trimesh.Trimesh(V, T, process=True)
print("figure watertight:", fig.is_watertight, "winding:", fig.is_winding_consistent, "tris:", len(T))
if fig.volume < 0:
    T = T[:, ::-1]
# darkest cluster for base
order = np.argsort(C.sum(1))
base_c = order[0]
allV = np.vstack([V, bv]); allT = np.vstack([T, bt + len(V)])
allM = np.concatenate([M, np.full(len(bt), base_c)])

def hexc(c): return "#%02X%02X%02X" % tuple(int(round(x)) for x in c)
mats_xml = "".join(f'<base name="cor{k+1}" displaycolor="{hexc(C[k])}"/>' for k in range(K))
def build(objid, Vx, Tx, Mx, name, colored):
    vs = "".join(f'<vertex x="{x:.3f}" y="{y:.3f}" z="{z:.3f}"/>' for x, y, z in Vx)
    if colored:
        ts = "".join(f'<triangle v1="{p}" v2="{q}" v3="{r}" pid="1" p1="{m}"/>' for (p, q, r), m in zip(Tx, Mx))
    else:
        ts = "".join(f'<triangle v1="{p}" v2="{q}" v3="{r}"/>' for p, q, r in Tx)
    return f'<object id="{objid}" name="{name}" type="model"{" pid=\"1\" pindex=\"0\"" if colored else ""}><mesh><vertices>{vs}</vertices><triangles>{ts}</triangles></mesh></object>'

allZ = allV[:, 2].min()
allV[:, 2] -= allZ
model = ('<?xml version="1.0" encoding="UTF-8"?>\n<model unit="millimeter" xml:lang="pt-BR" '
         'xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">'
         '<metadata name="Title">Imagem 3D</metadata>'
         f'<resources><basematerials id="1">{mats_xml}</basematerials>'
         + build(2, allV, allT, allM, "figura", True) +
         '</resources><build><item objectid="2" transform="1 0 0 0 1 0 0 0 1 100 100 0"/></build></model>')
ct = ('<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
      '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
      '<Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/></Types>')
rels = ('<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Target="/3D/3dmodel.model" Id="rel0" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/></Relationships>')
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    z.writestr("[Content_Types].xml", ct); z.writestr("_rels/.rels", rels); z.writestr("3D/3dmodel.model", model)
# also GLB for preview
cols = np.c_[C[allM].astype(np.uint8), np.full(len(allM), 255, np.uint8)]
pm = trimesh.Trimesh(allV, allT, face_colors=cols, process=False)
pm.export(out.replace(".3mf", ".glb"))
print("bbox mm:", allV.min(0).round(1), allV.max(0).round(1), "colors:", [hexc(c) for c in C])
