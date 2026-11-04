"""Versao branca (uma cor) com LINHAS-GUIA para pintar a mao: afunda levemente (0,3 mm) os
vertices que ficam na divisa entre duas cores do 3MF colorido. Nao muda a topologia.

Uso: python3 branco_com_guias.py colorido.3mf saida.3mf [profundidade_mm] [area_min_mm2]
"""
import sys, re, zipfile, numpy as np, trimesh

src, dst = sys.argv[1], sys.argv[2]
DEPTH = float(sys.argv[3]) if len(sys.argv) > 3 else 0.3
AMIN = float(sys.argv[4]) if len(sys.argv) > 4 else 1.5   # ignora manchinhas menores que isso

xml = zipfile.ZipFile(src).read("3D/3dmodel.model").decode()
V = np.array(re.findall(r'<vertex x="([-\d.]+)" y="([-\d.]+)" z="([-\d.]+)"/>', xml), float)
T = np.array(re.findall(r'<triangle v1="(\d+)" v2="(\d+)" v3="(\d+)" pid="1" p1="(\d+)"', xml), int)
F, L = T[:, :3], T[:, 3]
m = trimesh.Trimesh(V, F, process=False)

# remove manchas pequenas (pontinhos) para nao virar ruido de sulcos
import scipy.sparse as sps, scipy.sparse.csgraph as csg
adj = m.face_adjacency
same = L[adj[:, 0]] == L[adj[:, 1]]
G = sps.coo_matrix((np.ones(same.sum()), (adj[same, 0], adj[same, 1])), shape=(len(F),) * 2)
_, comp = csg.connected_components(G, directed=False)
area = np.bincount(comp, weights=m.area_faces)
small = area[comp] < AMIN
Lc = L.copy()
for _ in range(6):   # pinta as manchinhas com a cor vizinha dominante
    if not small.any(): break
    votes = np.zeros((len(F), L.max() + 1))
    ok0 = ~small[adj[:, 1]]; ok1 = ~small[adj[:, 0]]
    np.add.at(votes, (adj[ok0, 0], Lc[adj[ok0, 1]]), 1)
    np.add.at(votes, (adj[ok1, 1], Lc[adj[ok1, 0]]), 1)
    fix = small & (votes.sum(1) > 0)
    Lc[fix] = votes[fix].argmax(1); small &= ~fix

# vertices de divisa: arestas entre faces de cores diferentes
diff = Lc[adj[:, 0]] != Lc[adj[:, 1]]
ev = m.face_adjacency_edges[diff]
bnd = np.zeros(len(V), bool); bnd[ev.ravel()] = True
n = m.vertex_normals
V2 = V.copy(); V2[bnd] -= n[bnd] * DEPTH
print(src.split("/")[-1], "->", int(bnd.sum()), "vertices de divisa afundados", DEPTH, "mm")

vs = "".join(f'<vertex x="{a:.3f}" y="{b:.3f}" z="{c:.3f}"/>' for a, b, c in V2)
ts = "".join(f'<triangle v1="{a}" v2="{b}" v3="{c}"/>' for a, b, c in F)
head = xml[:xml.index("<resources>")]
model = (head + f'<resources><object id="2" name="branco_guias" type="model"><mesh><vertices>{vs}</vertices>'
         f"<triangles>{ts}</triangles></mesh></object></resources>"
         '<build><item objectid="2" transform="1 0 0 0 1 0 0 0 1 128 128 0"/></build></model>')
zin = zipfile.ZipFile(src)
with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as z:
    z.writestr("[Content_Types].xml", zin.read("[Content_Types].xml"))
    z.writestr("_rels/.rels", zin.read("_rels/.rels"))
    z.writestr("3D/3dmodel.model", model)
