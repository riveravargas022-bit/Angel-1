import numpy as np, json
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
im = np.asarray(Image.open('ref/652LUMILILA_4.jpg').convert('RGB')).astype(float)
fg = np.load('ref/fg_4.npy')
R,G,B = im[...,0],im[...,1],im[...,2]
blue = B - (R+G)/2
lum = (R+G+B)/3
# classes: midsole off-white neutral: blue<5 & lum>200 ; insole: neutral/white inside; lining: bluish darker
lab = np.zeros(fg.shape, np.uint8)
lab[fg] = 1
lab[fg & (blue>=6)] = 2
dens = ndi.uniform_filter((lab==2).astype(float), 31)
up = (dens>0.10) & fg
up = ndi.binary_closing(up, iterations=5)
up = ndi.binary_fill_holes(up)
lab2, n = ndi.label(up); sizes = ndi.sum(up, lab2, range(1,n+1)); up = lab2==(np.argmax(sizes)+1)
np.save('ref/up_4.npy', up)
ys,xs = np.nonzero(up)
print('upper bbox', xs.min(), xs.max(), ys.min(), ys.max())
rows = np.arange(ys.min(), ys.max()+1, 4)
L=[];Rr=[]
for r in rows:
    c = np.nonzero(up[r])[0]; L.append(int(c.min())); Rr.append(int(c.max()))
T = json.load(open('ref/profiles_top.json'))
T['4u'] = dict(rows=rows.tolist(), left=L, right=Rr)
json.dump(T, open('ref/profiles_top.json','w'))
pim = Image.open('ref/652LUMILILA_4.jpg').convert('RGB')
pim = Image.blend(pim, Image.new('RGB', pim.size, (255,255,255)), 0.3)
d = ImageDraw.Draw(pim)
d.line(list(zip(L, rows.tolist())), fill=(255,0,0), width=10)
d.line(list(zip(Rr, rows.tolist())), fill=(0,0,255), width=10)
t=T['4']; d.line(list(zip(t['left'], t['rows'])), fill=(0,150,0), width=10); d.line(list(zip(t['right'], t['rows'])), fill=(0,150,0), width=10)
pim.resize((pim.width//3, pim.height//3)).save('ref/prof_4u.png')
