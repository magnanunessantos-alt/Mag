"""Gera um 3MF de uma cor so (somente geometria) a partir de um 3MF colorido.
Uso: python3 um_cor_3mf.py entrada.3mf saida.3mf"""
import sys, re, zipfile
src, dst = sys.argv[1], sys.argv[2]
x = zipfile.ZipFile(src).read("3D/3dmodel.model").decode()
x = re.sub(r'<basematerials id="1">.*?</basematerials>', "", x, flags=re.S)
x = re.sub(r' pid="1" pindex="0"', "", x)
x = re.sub(r' pid="1" p1="\d+" paint_color="[0-9A-F]+" slic3rpe:mmu_segmentation="[0-9A-F]+"', "", x)
assert "paint_color" not in x and "basematerials" not in x
ct = ('<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
      '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
      '<Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/></Types>')
rels = ('<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Target="/3D/3dmodel.model" Id="rel0" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/></Relationships>')
with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as z:
    z.writestr("[Content_Types].xml", ct); z.writestr("_rels/.rels", rels); z.writestr("3D/3dmodel.model", x)
print(dst, x.count("<triangle "), "triangulos")
