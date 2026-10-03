"""Reference vs digital-twin comparison sheet (photo-matched cameras) + silhouette IoU."""
import os, sys, json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi

HERE = os.path.dirname(os.path.abspath(__file__))
RD = os.path.join(HERE, sys.argv[1] if len(sys.argv) > 1 else 'renders')
OUT = os.path.join(RD, 'comparacion_referencia_vs_render.jpg')
FONT = os.path.join(HERE, 'fonts', 'montserrat-latin-600-normal.ttf')
FONT2 = os.path.join(HERE, 'fonts', 'montserrat-latin-500-normal.ttf')

pairs = [('match_lat', 1, 'Lateral'), ('match_med', 2, 'Medial'), ('match_top', 4, 'Cenital')]
rows = []
metrics = {}
for v, i, label in pairs:
    rp = os.path.join(RD, f'{v}.png')
    ra = os.path.join(RD, f'{v}_rgba.png')
    if not os.path.exists(rp):
        continue
    # cut-out like the reference photos: drop the semi-transparent contact shadow
    rgba = np.asarray(Image.open(ra).convert('RGBA')).astype(np.float32) / 255.0
    a = np.clip((rgba[..., 3:4] - 0.80) / 0.18, 0, 1)
    comp = rgba[..., :3] * a + (1 - a)
    r = Image.fromarray((np.clip(comp, 0, 1) * 255 + 0.5).astype(np.uint8))
    r.save(os.path.join(RD, f'{v}_recorte.png'))
    w, h = r.size
    ph = Image.open(os.path.join(HERE, 'ref', f'652LUMILILA_{i}.jpg')).convert('RGB').resize((w, h), Image.LANCZOS)
    fg = np.load(os.path.join(HERE, 'ref', f'fg_{i}.npy'))
    fgs = np.asarray(Image.fromarray(fg.astype(np.uint8) * 255).resize((w, h), Image.BILINEAR)) > 127
    al = np.asarray(Image.open(ra))[..., 3]
    # the shoe itself is opaque; the contact shadow is semi-transparent
    shoe = al > 250
    shoe = ndi.binary_opening(shoe, iterations=1)
    iou = (fgs & shoe).sum() / max((fgs | shoe).sum(), 1)
    metrics[label] = round(float(iou), 4)
    rows.append((label, ph, r, iou))

if rows:
    tw = 1500
    pad = 36
    head = 150
    blocks = []
    for label, ph, r, iou in rows:
        s = (tw / 2 - pad * 1.5) / ph.width
        a = ph.resize((int(ph.width * s), int(ph.height * s)), Image.LANCZOS)
        b = r.resize(a.size, Image.LANCZOS)
        blocks.append((label, a, b, iou))
    H = head + sum(bl[1].height + 90 for bl in blocks) + pad
    sheet = Image.new('RGB', (tw, H), (250, 249, 247))
    d = ImageDraw.Draw(sheet)
    f1 = ImageFont.truetype(FONT, 40)
    f2 = ImageFont.truetype(FONT2, 22)
    f3 = ImageFont.truetype(FONT, 22)
    d.text((pad, 40), 'Olympikus Lumi · referencia y gemelo digital', font=f1, fill=(40, 38, 60))
    d.text((pad, 98), 'Fotos de estudio originales (izq.) y renders con cámara calibrada a cada foto (der.)', font=f2, fill=(110, 108, 130))
    y = head
    for label, a, b, iou in blocks:
        d.text((pad, y + 8), f'{label}', font=f3, fill=(40, 38, 60))
        d.text((pad + 160, y + 8), 'Coincidencia de silueta (IoU): ' + f'{iou * 100:.1f}'.replace('.', ',') + ' %', font=f2, fill=(110, 108, 130))
        y += 48
        sheet.paste(a, (pad, y))
        sheet.paste(b, (tw - pad - b.width, y))
        d.text((pad + 8, y + a.height - 34), 'Foto de referencia', font=f2, fill=(150, 148, 165))
        d.text((tw - pad - b.width + 8, y + b.height - 34), 'Render del modelo 3D', font=f2, fill=(150, 148, 165))
        y += a.height + 42
    sheet.save(OUT, quality=92)
    json.dump(metrics, open(os.path.join(RD, 'metricas_silueta.json'), 'w'), indent=2)
    print('sheet', OUT, metrics)
