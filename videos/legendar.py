"""Gera as imagens (PNG transparente) da faixa de datas e das legendas e monta os videos
com ffmpeg. Uso: python3 legendar.py video_9x16.mp4 audio.wav pasta_saida"""
import sys, os, subprocess
from PIL import Image, ImageDraw, ImageFont, ImageFilter

SRC, AUDIO, OUT = sys.argv[1:4]
TMP = os.path.join(OUT, "_overlays"); os.makedirs(TMP, exist_ok=True)
FB = "/usr/share/fonts/opentype/inter/Inter-ExtraBold.otf"
FS = "/usr/share/fonts/opentype/inter/Inter-Bold.otf"
FCAP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fontes", "Poppins_600SemiBold.ttf")
FSERIF = "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"

LEG = [
    (3.70, 5.85, "Oi, eu sou a Imaculadinha!"),
    (6.15, 7.75, "E eu sou o Beneditinho!"),
    (8.00, 10.80, "Somos da Paróquia Nossa Senhora da Conceição."),
    (11.25, 14.25, "Venham para a festa da Imaculada Conceição!"),
    (14.45, 17.65, "A festa começa no dia 28 de novembro."),
    (18.10, 21.85, "E logo depois, no dia 13/12, começam os festejos de São Benedito."),
    (22.00, 25.00, "Este ano eu completo 500 anos!"),
    (25.50, 27.05, "Venham todos!"),
    (27.20, 29.20, "Esperamos vocês!"),
]
GOLD, BROWN, CREAM = (242, 193, 78), (52, 30, 18), (255, 248, 235)

def wrap(d, text, font, maxw):
    words, lines, cur = text.split(), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if d.textlength(t, font=font) <= maxw: cur = t
        else: lines.append(cur); cur = w
    lines.append(cur)
    n = len(lines)
    if n in (2, 3):   # equilibra as linhas (evita palavra sozinha)
        from itertools import combinations
        best = None
        for cut in combinations(range(1, len(words)), n - 1):
            idx = (0,) + cut + (len(words),)
            ls = [" ".join(words[idx[i]:idx[i + 1]]) for i in range(n)]
            mw = max(d.textlength(l, font=font) for l in ls)
            if mw <= maxw and (best is None or mw < best[0]): best = (mw, ls)
        if best: lines = best[1]
    return lines

def caption_png(text, W, H, y_center, size, maxw, path):
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    f = ImageFont.truetype(FCAP, size)
    lines = wrap(d, text, f, maxw)
    lh = int(size * 1.30); th = lh * len(lines)
    bw = max(d.textlength(l, font=f) for l in lines) + size * 1.1
    bx0 = (W - bw) / 2; by0 = y_center - th / 2 - size * 0.35
    d.rounded_rectangle([bx0, by0, bx0 + bw, by0 + th + size * 0.7], radius=size * 0.45, fill=(20, 12, 8, 165))
    for i, l in enumerate(lines):
        tw = d.textlength(l, font=f); x = (W - tw) / 2; y = by0 + size * 0.35 + i * lh
        d.text((x, y - size * 0.12), l, font=f, fill=(255, 255, 255, 255), stroke_width=max(2, size // 18), stroke_fill=(0, 0, 0, 255))
    im.save(path)

def banner_png(W, H, top, bw, scale, path):
    """faixa: titulo + 3 colunas de datas"""
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    s = scale
    h = int(250 * s); x0 = (W - bw) / 2; x1 = x0 + bw
    sh = Image.new("RGBA", (W, H), (0, 0, 0, 0)); ImageDraw.Draw(sh).rounded_rectangle([x0, top + 8 * s, x1, top + h + 8 * s], radius=34 * s, fill=(0, 0, 0, 110))
    im = Image.alpha_composite(im, sh.filter(ImageFilter.GaussianBlur(10 * s))); d = ImageDraw.Draw(im)
    d.rounded_rectangle([x0, top, x1, top + h], radius=34 * s, fill=BROWN + (228,), outline=GOLD + (255,), width=max(3, int(4 * s)))
    ft = ImageFont.truetype(FSERIF, int(36 * s))
    title = "FESTA DA IMACULADA CONCEIÇÃO E DE SÃO BENEDITO"
    while d.textlength(title, font=ft) > bw - 50 * s: ft = ImageFont.truetype(FSERIF, ft.size - 1)
    tw = d.textlength(title, font=ft); d.text(((W - tw) / 2, top + 22 * s), title, font=ft, fill=GOLD + (255,))
    cols = [("28/11", "Abertura"), ("29/11 a 07/12", "Novena"), ("08/12", "Festa da Imaculada"), ("13/12", "São Benedito\nCortada do Mastro")]
    fd = ImageFont.truetype(FB, int(42 * s)); fl = ImageFont.truetype(FS, int(25 * s))
    widths = [0.9, 1.55, 1.05, 1.1]; tot = sum(widths); cx = x0 + 18 * s; inner = bw - 36 * s
    yd, yl = top + 90 * s, top + 158 * s
    for i, ((dt, lb), wf) in enumerate(zip(cols, widths)):
        cw = inner * wf / tot; mid = cx + cw / 2
        f2 = fd
        while d.textlength(dt, font=f2) > cw - 14 * s: f2 = ImageFont.truetype(FB, f2.size - 2)
        d.text((mid - d.textlength(dt, font=f2) / 2, yd + (fd.size - f2.size) / 2), dt, font=f2, fill=CREAM + (255,))
        f3 = fl
        parts = lb.split("\n")
        if len(parts) == 1 and d.textlength(lb, font=f3) > cw - 10 * s and " " in lb:   # rotulo longo: duas linhas
            ws = lb.split(); k = min(range(1, len(ws)), key=lambda k: max(d.textlength(" ".join(ws[:k]), font=f3), d.textlength(" ".join(ws[k:]), font=f3)))
            parts = [" ".join(ws[:k]), " ".join(ws[k:])]
        while max(d.textlength(p, font=f3) for p in parts) > cw - 10 * s: f3 = ImageFont.truetype(FS, f3.size - 1)
        y0 = yl - (f3.size * 0.55 if len(parts) == 2 else 0)
        for j, p in enumerate(parts):
            d.text((mid - d.textlength(p, font=f3) / 2, y0 + j * f3.size * 1.12), p, font=f3, fill=GOLD + (255,))
        if i < len(cols) - 1: d.line([cx + cw, top + 92 * s, cx + cw, top + h - 26 * s], fill=GOLD + (170,), width=max(2, int(2 * s)))
        cx += cw
    im.save(path)

def build(fmt):
    if fmt == "9x16":
        W, H = 1080, 1920
        base = "[0:v]scale=1080:1920:flags=lanczos,setsar=1[b]"
        banner_png(W, H, top=150, bw=1010, scale=1.0, path=f"{TMP}/banner_{fmt}.png")
        cy, size, maxw = 1520, 58, 900
        out = f"{OUT}/convite_festa_legendado_stories_reels_9x16.mp4"
    else:
        W, H = 1080, 1350
        base = ("[0:v]scale=1080:1350:force_original_aspect_ratio=increase,crop=1080:1350,boxblur=28:2,eq=brightness=-0.04[bg];"
                "[0:v]scale=-2:1350:flags=lanczos[fg];[bg][fg]overlay=(W-w)/2:0,setsar=1[b]")
        banner_png(W, H, top=22, bw=1040, scale=0.80, path=f"{TMP}/banner_{fmt}.png")
        cy, size, maxw = 1190, 46, 700
        out = f"{OUT}/convite_festa_legendado_feed_4x5.mp4"
    inputs = ["-i", SRC, "-i", AUDIO, "-loop", "1", "-i", f"{TMP}/banner_{fmt}.png"]
    chain = base + ";[b][2:v]overlay=0:0:shortest=1[v0]"
    for k, (a, b, t) in enumerate(LEG):
        p = f"{TMP}/leg_{fmt}_{k}.png"; caption_png(t, W, H, cy, size, maxw, p)
        inputs += ["-loop", "1", "-i", p]
        chain += f";[v{k}][{k + 3}:v]overlay=0:0:shortest=1:enable='between(t,{a},{b})'[v{k + 1}]"
    chain += f";[v{len(LEG)}]format=yuv420p[vout]"
    cmd = ["ffmpeg", "-v", "error", "-y", *inputs, "-filter_complex", chain, "-map", "[vout]", "-map", "1:a",
           "-t", "30.05", "-c:v", "libx264", "-preset", "slow", "-crf", "19", "-r", "24", "-c:a", "aac", "-b:a", "160k",
           "-movflags", "+faststart", out]
    subprocess.run(cmd, check=True); print("ok", out)

for fmt in ("9x16", "4x5"): build(fmt)
