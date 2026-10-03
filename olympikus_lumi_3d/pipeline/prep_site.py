"""Prepare web assets for the presentation page (WebP renders, comparison pairs, data blobs)."""
import os, json, shutil
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
RD = os.path.join(HERE, 'renders')
SITE = os.path.join(HERE, 'site')
A = os.path.join(SITE, 'assets')
os.makedirs(A, exist_ok=True)

FINAL = ['side_lateral', 'q34_front', 'q34_rear', 'top_down', 'sole', 'macro_logo', 'macro_tongue',
         'macro_knit', 'side_medial', 'macro_heel']
for v in FINAL:
    src = os.path.join(RD, f'{v}_rgba.png')
    if not os.path.exists(src):
        print('missing', v); continue
    im = Image.open(src).convert('RGBA')
    im.save(os.path.join(A, f'{v}.webp'), 'WEBP', quality=88, method=6)
    print(v, im.size, os.path.getsize(os.path.join(A, f'{v}.webp')) // 1024, 'KB')

for v, i in (('lat', 1), ('med', 2), ('top', 4)):
    mp = os.path.join(RD, f'match_{v}_recorte.png')
    if not os.path.exists(mp):
        print('missing match', v); continue
    r = Image.open(mp).convert('RGB')
    w = 1600
    h = int(round(r.height * w / r.width))
    r.resize((w, h), Image.LANCZOS).save(os.path.join(A, f'match_{v}.webp'), 'WEBP', quality=88, method=6)
    ph = Image.open(os.path.join(HERE, 'ref', f'652LUMILILA_{i}.jpg')).convert('RGB').resize((w, h), Image.LANCZOS)
    ph.save(os.path.join(A, f'ref_{v}.webp'), 'WEBP', quality=86, method=6)
    print('pair', v, (w, h))

iou = json.load(open(os.path.join(RD, 'metricas_silueta.json'))) if os.path.exists(os.path.join(RD, 'metricas_silueta.json')) else {}
mats = [
    ['Upper', 'Punto jacquard', '#DCD9E8', '0.78', 'Sheen 0.55', 'Textura proyectada desde las fotos lateral, medial y cenital; relieve de pasadas de 1,1 mm'],
    ['Entresuela', 'Espuma EVA mate', '#EBE5DA', '0.58', 'Subsuperficie 0.05', 'Ondas esculpidas en la geometría; grano de espuma y dibujo de suela por mapa de altura'],
    ['Ojalera', 'TPU satinado', '#C4C1E3', '0.42', 'Clear coat 0.12', 'Escudo en relieve y filete menta #86DCC6'],
    ['Logotipo', 'TPU inyectado', '#8B8AB4', '0.33', 'Clear coat 0.25', 'Relieve de 1,05 mm generado desde el trazado vectorial, filete menta #7ED9C3'],
    ['Cordones', 'Cordón plano tejido', '#A9A8C6', '0.74', 'Sheen 0.60', 'Sección redondeada 6,9 × 2,5 mm con relieve de tejido en espiga'],
    ['Cuello y forro', 'Punto de forro', '#A2A9C8', '0.72', 'Sheen 0.70', 'Ribete enrollado sobre el borde y acolchado interior'],
    ['Lengüeta', 'Malla de aire y punto menta', '#A0E2D0', '0.70', 'Sheen 0.50', 'Etiqueta LUMI de 26 × 16 mm con la marca caligráfica'],
    ['Tirador', 'Cinta tejida', '#8F6DD9', '0.60', 'Sheen 0.50', 'Cara exterior lila claro #CFCBE2, cara interior morada'],
    ['Plantilla', 'Tela impresa', '#EFEFEC', '0.85', 'Sheen 0.30', 'Sello FEITO ◇ POR BRASILEIROS de 35,5 × 20,7 mm'],
    ['Impresión medial', 'Tinta serigráfica', '#1E1E20', '0.50', 'Alfa', 'EVASENSE con el icono de la marca'],
]
blob = '<script>window.__IOU__ = %s; window.__MATS__ = %s;</script>\n' % (json.dumps(iou), json.dumps(mats, ensure_ascii=False))
html = open(os.path.join(SITE, 'index.html')).read()
marker = '<div class="lb" id="lb" hidden>'
if 'window.__IOU__ =' in html:
    import re
    html = re.sub(r'<script>window.__IOU__ = .*?</script>\n', blob, html, flags=re.S)
else:
    html = html.replace(marker, blob + marker, 1)
open(os.path.join(SITE, 'index.html'), 'w').write(html)
glb = os.path.join(HERE, 'out', 'olympikus_lumi.glb')
if os.path.exists(glb):
    shutil.copy2(glb, os.path.join(A, 'olympikus_lumi.glb'))
tot = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(SITE) for f in fs)
print('site total MB', round(tot / 1e6, 2), 'iou', iou)
