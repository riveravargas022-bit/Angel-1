"""Copy final deliverables into the repository folder with Spanish file names."""
import os, json, shutil
HERE = os.path.dirname(os.path.abspath(__file__))
RD = os.path.join(HERE, 'renders')
REPO = '/home/user/Angel-1/olympikus_lumi_3d'
NAMES = {
    'side_lateral': 'perfil_lateral', 'side_medial': 'perfil_medial',
    'q34_front': 'tres_cuartos_delantera', 'q34_rear': 'tres_cuartos_trasera',
    'top_down': 'cenital_lengueta_interior', 'sole': 'suela',
    'macro_logo': 'macro_logotipo', 'macro_tongue': 'macro_etiqueta_lumi',
    'macro_knit': 'macro_tejido_jacquard', 'macro_heel': 'macro_talon',
}
os.makedirs(os.path.join(REPO, 'renders', 'transparente'), exist_ok=True)
os.makedirs(os.path.join(REPO, 'comparacion'), exist_ok=True)
for k, n in NAMES.items():
    shutil.copy2(os.path.join(RD, f'{k}.png'), os.path.join(REPO, 'renders', f'{n}.png'))
    shutil.copy2(os.path.join(RD, f'{k}_rgba.png'), os.path.join(REPO, 'renders', 'transparente', f'{n}.png'))
shutil.copy2(os.path.join(RD, 'comparacion_referencia_vs_render.jpg'), os.path.join(REPO, 'comparacion', 'comparacion_referencia_vs_render.jpg'))
shutil.copy2(os.path.join(RD, 'metricas_silueta.json'), os.path.join(REPO, 'comparacion', 'metricas_silueta.json'))
for v, n in (('lat', 'lateral'), ('med', 'medial'), ('top', 'cenital')):
    shutil.copy2(os.path.join(RD, f'match_{v}_recorte.png'), os.path.join(REPO, 'comparacion', f'render_camara_calibrada_{n}.png'))
m = json.load(open(os.path.join(RD, 'metricas_silueta.json')))
rows = '\n'.join(f'| {k} | {v * 100:.1f} %'.replace('.', ',') + ' |' for k, v in m.items())
table = '| Vista | IoU de silueta |\n|---|---|\n' + rows
readme = open(os.path.join(HERE, 'README_repo.md')).read().replace('METRICAS_PLACEHOLDER', table)
open(os.path.join(REPO, 'README.md'), 'w').write(readme)
print('ok', m)
