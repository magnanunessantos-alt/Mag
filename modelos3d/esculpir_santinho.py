"""Esculpe um mini santinho chibi (Santo Antonio com o Menino Jesus) usando
campos de distancia (SDF), no estilo 'sentado na beira do monitor'.
Gera 3MF colorido (basematerials + pintura Bambu/Prusa) e OBJ+MTL.

Uso: python3 esculpir_santinho.py <saida_sem_extensao> [voxel_mm]
Eixos: Z para cima, -Y e a frente, unidades em mm.
"""
import sys, zipfile, numpy as np
from skimage.measure import marching_cubes

OUT = sys.argv[1] if len(sys.argv) > 1 else "santinho"
VOX = float(sys.argv[2]) if len(sys.argv) > 2 else 0.4

# ---------------- paleta (cada cor = um filamento) ----------------
COLORS = {
    "habito":  ("#4B2A17", "Marrom habito"),
    "pele":    ("#6B3F26", "Pele"),
    "dourado": ("#D9AE4E", "Dourado"),
    "branco":  ("#F3F0EA", "Branco"),
    "escuro":  ("#1C120D", "Preto/cabelo"),
    "pele2":   ("#F2C9A0", "Pele menino"),
}
MAT = list(COLORS)
mid = {k: i for i, k in enumerate(MAT)}

# ---------------- primitivas SDF ----------------
def L(v): return np.sqrt((v * v).sum(-1))

def sphere(p, c, r): return L(p - c) - r

def ellipsoid(p, c, r):
    q = (p - c) / r
    k0 = L(q); k1 = L(q / r)
    return k0 * (k0 - 1.0) / np.maximum(k1, 1e-6)

def capsule(p, a, b, ra, rb=None):
    rb = ra if rb is None else rb
    a = np.asarray(a, float); b = np.asarray(b, float)
    pa = p - a; ba = b - a
    h = np.clip((pa @ ba) / (ba @ ba), 0, 1)[..., None]
    return L(pa - ba * h) - (ra + (rb - ra) * h[..., 0])

def rbox(p, c, half, r):
    q = np.abs(p - c) - (np.asarray(half) - r)
    return L(np.maximum(q, 0)) + np.minimum(q.max(-1), 0) - r

def torus_xy(p, c, R, r):  # anel horizontal (eixo Z)
    q = p - c
    return np.sqrt((np.sqrt(q[..., 0]**2 + q[..., 1]**2) - R)**2 + q[..., 2]**2) - r

def torus_xz(p, c, R, r):  # anel de frente (eixo Y)
    q = p - c
    return np.sqrt((np.sqrt(q[..., 0]**2 + q[..., 2]**2) - R)**2 + q[..., 1]**2) - r

def smin(a, b, k):
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0, 1)
    return b * (1 - h) + a * h - k * h * (1 - h)

# ---------------- geometria ----------------
HEAD_C = np.array([0.0, 6.0, 52.0]); HEAD_R = np.array([17.0, 15.0, 16.5])

def head_front_y(x, z, off=0.0):
    """y da superficie frontal da cabeca em (x,z)."""
    q = 1 - ((x - HEAD_C[0]) / HEAD_R[0])**2 - ((z - HEAD_C[2]) / HEAD_R[2])**2
    return HEAD_C[1] - HEAD_R[1] * np.sqrt(max(q, 0)) - off

def halo(p):
    c = np.array([0.0, 19.5, 57.0])
    q = p - c
    x, z = q[..., 0], q[..., 2]
    r = np.sqrt(x * x + z * z); th = np.arctan2(z, x)
    d2 = r - 21.0                                   # disco central
    n = 28
    for i in range(n):
        a = np.pi * 2 * i / n
        if np.sin(a) < -0.35: continue          # sem raios atras do corpo
        long = i % 2 == 0
        r1 = 33.0 if long else 28.0
        a0 = np.array([np.cos(a) * 17, np.sin(a) * 17])
        a1 = np.array([np.cos(a) * r1, np.sin(a) * r1])
        pa = np.stack([x - a0[0], z - a0[1]], -1); ba = a1 - a0
        h = np.clip((pa @ ba) / (ba @ ba), 0, 1)
        dd = np.sqrt(((pa - ba * h[..., None])**2).sum(-1)) - (2.3 - 1.5 * h)
        d2 = np.minimum(d2, dd)
    thick = 1.6
    d = np.maximum(d2, np.abs(q[..., 1]) - thick)
    # anel de bolinhas ornamentais na frente
    ang = np.round(th / (2 * np.pi / 20)) * (2 * np.pi / 20)
    bc = np.stack([np.cos(ang) * 19, np.full_like(ang, -thick), np.sin(ang) * 19], -1)
    d = smin(d, L(q - bc) - 1.5, 0.6)
    return d

def scene(p):
    """retorna (distancia, indice de material) para pontos p (...,3)."""
    parts = []  # (dist, material)
    add = lambda d, m: parts.append((d, mid[m]))

    # --- corpo / habito ---
    tor = capsule(p * [1, 1.4, 1], [0, 8 * 1.4, 14], [0, 8 * 1.4, 33], 13.0, 11.0) / 1.4
    lap = rbox(p, [0, -4, 6], [14, 12.5, 6.0], 4.0)            # colo/coxas
    panel = rbox(p, [0, -11, -6], [13.5, 5.0, 14], 4.5)        # pano caindo nas pernas
    hood = ellipsoid(p, [0, 14, 36], [12, 6, 7])              # capuz nas costas
    body = smin(smin(smin(tor, lap, 4), panel, 4), hood, 3)
    # manga direita (lado esquerdo de quem olha) segurando a cruz
    s1 = smin(capsule(p, [-11, 9, 31], [-17, 5, 19], 5.5, 6.5),
              capsule(p, [-17, 5, 19], [-18.5, -4.5, 23], 6.5, 6.0), 3)
    # manga esquerda segurando o menino
    s2 = smin(capsule(p, [11, 9, 31], [16, 5, 19], 5.5, 6.5),
              capsule(p, [16, 5, 19], [9.5, -6.5, 16], 6.5, 6.0), 3)
    body = smin(body, smin(s1, s2, 2), 3)
    add(body, "habito")

    # gola dourada e cordao
    add(torus_xy(p, [0, 8.5, 36.5], 9.5, 2.8), "dourado")
    belt = np.maximum(tor - 1.3, np.abs(p[..., 2] - 15.5) - 1.3)  # cordao abracando a cintura
    cord = smin(belt, capsule(p, [-5, -10, 15], [-6, -16.8, 6], 1.4), 1)
    cord = np.minimum(cord, capsule(p, [-6, -16.6, 4], [-6.5, -16.6, -10], 1.3))
    cord = np.minimum(cord, capsule(p, [-6.5, -16.8, -10], [-6.5, -16.8, -15], 1.8, 2.6))
    add(cord, "dourado")
    # punhos dourados das mangas
    add(np.maximum(s1 - 0.7, p[..., 1] + 8.3), "dourado")    # punhos
    add(np.maximum(s2 - 0.7, np.minimum(p[..., 0] - 7.5, p[..., 1] + 10.0)), "dourado")

    # --- pescoco, cabeca, cabelo ---
    add(capsule(p, [0, 7, 33], [0, 7, 41], 6.5), "pele")
    head = ellipsoid(p, HEAD_C, HEAD_R)
    ears = np.minimum(ellipsoid(p, [-16.3, 7, 50], [3, 2.2, 4]),
                      ellipsoid(p, [16.3, 7, 50], [3, 2.2, 4]))
    cheeks = np.minimum(sphere(p, [-8, head_front_y(-8, 45) + 4.2, 45], 4.5),
                        sphere(p, [8, head_front_y(8, 45) + 4.2, 45], 4.5))
    add(smin(smin(head, ears, 1.5), cheeks, 2.5), "pele")
    add(sphere(p, [0, head_front_y(0, 47.5) + 0.6, 47.5], 1.6), "pele")    # nariz
    hair = ellipsoid(p, [0, 8.2, 54.5], [18.3, 16.3, 17.0])
    hair -= 0.55 * (np.sin(p[..., 0] * 1.3) * np.sin(p[..., 1] * 1.3) * np.sin(p[..., 2] * 1.3))
    hairline = -((p[..., 2] - 59.5) + 0.55 * (p[..., 1] + 10)) / 1.14
    hair = np.maximum(hair, hairline)
    add(hair, "escuro")
    # olhos grandes, sobrancelhas, sorriso
    for sx in (-1, 1):
        ex, ez = 6.3 * sx, 50.0
        ey = head_front_y(ex, ez)
        add(ellipsoid(p, [ex, ey + 0.55, ez], [2.7, 1.3, 3.5]), "escuro")
        add(sphere(p, [ex - 0.9 * sx + 0.0, ey - 0.65, ez + 1.2], 0.9), "branco")
        bx0, bx1 = 4.0 * sx, 8.8 * sx
        add(capsule(p, [bx0, head_front_y(bx0, 55.5) + 0.25, 55.6],
                    [bx1, head_front_y(bx1, 55.0) + 0.25, 55.0], 0.75), "escuro")
    my = head_front_y(0, 44.5)
    smile = torus_xz(p, [0, my + 0.35, 45.3], 2.6, 0.6)
    add(np.maximum(smile, p[..., 2] - 44.6), "escuro")

    # --- maos ---
    add(ellipsoid(p, [-18.8, -12.4, 23.4], [3.4, 3.3, 3.6]), "pele")
    add(ellipsoid(p, [5.6, -12.6, 17.6], [3.2, 3.0, 3.0]), "pele")

    # --- cruz ---
    cx, cy = -18.8, -13.0
    cross = np.minimum(rbox(p, [cx, cy, 31], [1.4, 1.4, 19], 0.6),
                       rbox(p, [cx, cy, 43], [5.8, 1.4, 1.4], 0.6))
    add(cross, "habito")
    corpus = np.minimum(capsule(p, [cx, cy - 1.4, 44.5], [cx, cy - 1.4, 34], 0.8, 0.6),
                        capsule(p, [cx - 4.6, cy - 1.4, 43.2], [cx + 4.6, cy - 1.4, 43.2], 0.55))
    corpus = np.minimum(corpus, sphere(p, [cx, cy - 1.5, 45.6], 1.0))
    add(corpus, "dourado")

    # --- Menino Jesus ---
    kc = np.array([11.0, -10.5, 0.0])
    dress = smin(capsule(p * [1, 1.15, 1], [11, -10.5 * 1.15, 15], [11, -10.5 * 1.15, 25], 6.8, 4.2) / 1.15,
                 ellipsoid(p, [11, -11.5, 15], [7.2, 5.5, 3.5]), 2)
    add(dress, "branco")
    add(capsule(p, [11, -10.5, 25], [11, -10.5, 28], 2.4), "pele2")
    khead = ellipsoid(p, [11, -11, 32.5], [6.4, 6.0, 6.2])
    add(khead, "pele2")
    khair = ellipsoid(p, [11, -10.2, 34], [6.9, 6.4, 6.0])
    khair -= 0.3 * np.sin(p[..., 0] * 2.2) * np.sin(p[..., 2] * 2.2)
    khair = np.maximum(khair, -((p[..., 2] - 34.5) + 0.7 * (p[..., 1] + 11)))
    add(khair, "habito")
    for sx in (-1, 1):
        add(sphere(p, [11 + 2.3 * sx, -16.55, 32.0], 0.85), "escuro")
    add(np.maximum(torus_xz(p, [11, -16.6, 30.2], 1.0, 0.35), p[..., 2] - 30.0), "escuro")
    # bracinho abencoando e mao
    add(capsule(p, [7.5, -11.5, 23], [6.0, -14.0, 28], 1.7, 1.5), "branco")
    add(sphere(p, [5.8, -14.6, 29.2], 1.6), "pele2")
    # globo dourado com cruzinha
    add(sphere(p, [15.6, -14.5, 22.5], 2.6), "dourado")
    add(np.minimum(capsule(p, [15.6, -14.5, 24.5], [15.6, -14.5, 27.6], 0.55),
                   capsule(p, [14.4, -14.5, 26.4], [16.8, -14.5, 26.4], 0.55)), "dourado")

    # --- pes com sandalias ---
    for sx in (-1, 1):
        fx = 6.2 * sx
        add(ellipsoid(p, [fx, -16.5, -22.5], [4.6, 6.8, 3.2]), "pele")
        add(rbox(p, [fx, -16.8, -25.4], [5.0, 7.2, 1.0], 0.8), "escuro")
        add(np.maximum(torus_xz(p * [1, 1, 1], [fx, -19.0, -22.8], 4.4, 0.9), -(p[..., 2] + 23.5)), "escuro")
        add(np.maximum(torus_xz(p, [fx, -14.0, -22.8], 4.4, 0.9), -(p[..., 2] + 23.5)), "escuro")
        # dedinhos
        for t in range(4):
            add(sphere(p, [fx - 2.6 * sx * 0 + (t - 1.5) * 1.9, -22.6, -23.6], 1.1), "pele")

    # --- aureola ---
    add(halo(p), "dourado")

    D = np.stack([d for d, _ in parts]); M = np.array([m for _, m in parts])
    k = np.argmin(D, 0)
    # uniao suave so do corpo principal: min simples entre partes de cor diferente
    d = D.min(0)
    return d, M[k]

# ---------------- avaliar em grade ----------------
lo = np.array([-36.0123, -27.0071, -29.0137]); hi = np.array([36.0, 26.0, 93.0])
nx, ny, nz = (np.ceil((hi - lo) / VOX)).astype(int) + 1
xs = lo[0] + np.arange(nx) * VOX; ys = lo[1] + np.arange(ny) * VOX; zs = lo[2] + np.arange(nz) * VOX
vol = np.empty((nx, ny, nz), np.float32)
YY, ZZ = np.meshgrid(ys, zs, indexing="ij")
for i, x in enumerate(xs):
    P = np.stack([np.full_like(YY, x), YY, ZZ], -1)
    vol[i] = scene(P)[0]
print("grade", vol.shape)
verts, faces, _, _ = marching_cubes(vol, 0.0, spacing=(VOX, VOX, VOX))
verts += lo
faces = faces[:, ::-1]  # normais para fora

import fast_simplification, trimesh
target = 0.55 if VOX < 0.35 else 0.35
verts, faces = fast_simplification.simplify(verts.astype(np.float32), faces.astype(np.int32), target_reduction=target)
m = trimesh.Trimesh(verts, faces, process=True)
# manter so a maior parte conectada
comps = m.split(only_watertight=False)
m = max(comps, key=lambda c: len(c.faces))
if m.volume < 0: m.invert()
print("tris", len(m.faces), "estanque", m.is_watertight, "vol cm3", round(m.volume / 1000, 1))

# material por face: amostra no centroide levemente para dentro
cen = m.triangles_center - m.face_normals * 0.15
_, fm = scene(cen)

# padrao de florzinhas douradas no habito + barra dourada
x, y, z = cen[:, 0], cen[:, 1], cen[:, 2]
sp = 8.0
gx = np.round(x / sp) * sp; gz = np.round(z / sp) * sp
gx = np.where((np.round(z / sp) % 2) == 0, gx, np.round((x - sp / 2) / sp) * sp + sp / 2)
dx, dz = x - gx, z - gz
r = np.sqrt(dx * dx + dz * dz); th = np.arctan2(dz, dx)
flower = r < (1.3 + 1.0 * np.cos(3 * th) ** 2)
hab = fm == mid["habito"]
is_cross = (np.abs(x + 18.8) < 6.5) & (np.abs(y + 13.0) < 2.5)
is_khair = (z > 30) & (x > 3)
fm[hab & flower & (y < 4) & ~is_cross & ~is_khair & (z < 34)] = mid["dourado"]
hem = hab & (z < -14.0) & (z > -17.5) & (y < -8)
fm[hem] = mid["dourado"]
fm[hab & (z < -18.5) & (z > -19.5) & (y < -8)] = mid["dourado"]

# ---------------- exportar ----------------
m.vertices -= [0, 0, m.bounds[0, 2]]
V, T = m.vertices, m.faces

def paint_code(i):  # codigo de pintura do TriangleSelector (Prusa/Bambu), extrusora i+1
    s = i + 1
    return "4" if s == 1 else "8" if s == 2 else "%XC" % (s - 3)

base_xml = "".join(f'<base name="{COLORS[k][1]}" displaycolor="{COLORS[k][0]}"/>' for k in MAT)
vs = "".join(f'<vertex x="{a:.3f}" y="{b:.3f}" z="{c:.3f}"/>' for a, b, c in V)
ts = "".join(
    f'<triangle v1="{a}" v2="{b}" v3="{c}" pid="1" p1="{k}" paint_color="{paint_code(k)}" slic3rpe:mmu_segmentation="{paint_code(k)}"/>'
    for (a, b, c), k in zip(T, fm))
model = ('<?xml version="1.0" encoding="UTF-8"?>\n'
         '<model unit="millimeter" xml:lang="pt-BR" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02" '
         'xmlns:slic3rpe="http://schemas.slic3r.org/3mf/2017/06" requiredextensions="">'
         '<metadata name="Title">Mini Santo Antonio</metadata>'
         f'<resources><basematerials id="1">{base_xml}</basematerials>'
         f'<object id="2" name="santinho" type="model" pid="1" pindex="0"><mesh><vertices>{vs}</vertices>'
         f'<triangles>{ts}</triangles></mesh></object></resources>'
         '<build><item objectid="2" transform="1 0 0 0 1 0 0 0 1 128 128 0"/></build></model>')
ct = ('<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
      '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
      '<Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/></Types>')
rels = ('<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Target="/3D/3dmodel.model" Id="rel0" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/></Relationships>')
with zipfile.ZipFile(OUT + ".3mf", "w", zipfile.ZIP_DEFLATED) as zf:
    zf.writestr("[Content_Types].xml", ct); zf.writestr("_rels/.rels", rels); zf.writestr("3D/3dmodel.model", model)

# OBJ + MTL (Bambu Studio / Orca convertem as cores em filamentos ao importar)
base = OUT.split("/")[-1]
with open(OUT + ".mtl", "w") as f:
    for k in MAT:
        h = COLORS[k][0]; r_, g_, b_ = (int(h[i:i + 2], 16) / 255 for i in (1, 3, 5))
        f.write(f"newmtl {k}\nKd {r_:.4f} {g_:.4f} {b_:.4f}\n\n")
with open(OUT + ".obj", "w") as f:
    f.write(f"mtllib {base}.mtl\n")
    f.write("".join(f"v {a:.3f} {b:.3f} {c:.3f}\n" for a, b, c in V))
    for k in range(len(MAT)):
        sel = T[fm == k]
        if len(sel):
            f.write(f"usemtl {MAT[k]}\n" + "".join(f"f {a+1} {b+1} {c+1}\n" for a, b, c in sel))

cols = np.array([[int(COLORS[k][0][i:i + 2], 16) for i in (1, 3, 5)] + [255] for k in MAT], np.uint8)
trimesh.Trimesh(V, T, face_colors=cols[fm], process=False).export(OUT + ".glb")
print("altura mm", round(V[:, 2].max(), 1), "larg", round(np.ptp(V[:, 0]), 1), "prof", round(np.ptp(V[:, 1]), 1))
print("faces por cor", {MAT[k]: int((fm == k).sum()) for k in range(len(MAT))})
