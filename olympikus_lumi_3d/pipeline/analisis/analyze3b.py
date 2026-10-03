import numpy as np, json
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from scipy.ndimage import median_filter
P = json.load(open('ref/profiles.json'))
for i in ['1','2']:
    im = np.asarray(Image.open(f'ref/652LUMILILA_{i}.jpg').convert('RGB')).astype(float)
    fg = np.load(f'ref/fg_{i}.npy')
    R,G,B = im[...,0],im[...,1],im[...,2]
    blue = B - (R+G)/2
    bl = ((blue >= 6) & fg).astype(float)
    dens = ndi.uniform_filter(bl, 21)
    p = P[i]
    mid=[]
    for c,b in zip(p['cols'], p['bot']):
        rr = b-60
        while rr>0 and fg[rr,c] and dens[rr,c] < 0.08: rr-=1
        mid.append(int(rr))
    p['mid'] = mid
    pim = Image.open(f'ref/652LUMILILA_{i}.jpg').convert('RGB')
    pim = Image.blend(pim, Image.new('RGB', pim.size, (255,255,255)), 0.4)
    d = ImageDraw.Draw(pim)
    m2 = median_filter(np.array(mid), 21)
    d.line(list(zip(p['cols'], p['top'])), fill=(255,0,0), width=12)
    d.line(list(zip(p['cols'], p['bot'])), fill=(0,160,0), width=12)
    d.line(list(zip(p['cols'], m2.tolist())), fill=(0,0,255), width=12)
    pim.resize((pim.width//3, pim.height//3)).save(f'ref/prof_{i}.png')
json.dump(P, open('ref/profiles.json','w'))
