import numpy as np, json
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from scipy.signal import find_peaks
P = json.load(open('ref/profiles.json'))
for i in ['1','2']:
    im = np.asarray(Image.open(f'ref/652LUMILILA_{i}.jpg').convert('L')).astype(float)
    fg = np.load(f'ref/fg_{i}.npy')
    sm = ndi.gaussian_filter(im, 6)
    p = P[i]
    pim = Image.open(f'ref/652LUMILILA_{i}.jpg').convert('RGB')
    d = ImageDraw.Draw(pim)
    from scipy.ndimage import median_filter
    mid = median_filter(np.array(p['mid']),21)
    pts=[]
    for c,b,m in zip(p['cols'][::3], p['bot'][::3], mid[::3]):
        seg = sm[m+15:b-15, c]
        if len(seg)<30: continue
        # minima of luminance = creases; use prominence
        pk,_ = find_peaks(-seg, prominence=4, distance=25)
        for k in pk:
            pts.append((int(c), int(m+15+k)))
            d.ellipse([c-7, m+15+k-7, c+7, m+15+k+7], fill=(255,0,0))
    json.dump(pts, open(f'ref/creases_{i}.json','w'))
    pim.crop((300,1600,4700,2750)).resize((1600, int(1150*1600/4400))).save(f'ref/creases_{i}.png')
