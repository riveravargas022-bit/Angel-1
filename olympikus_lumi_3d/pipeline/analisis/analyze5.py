import numpy as np, json
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
res={}
for i in [4,5]:
    im = np.asarray(Image.open(f'ref/652LUMILILA_{i}.jpg').convert('RGB')).astype(float)
    h,w,_ = im.shape
    fg = (765 - im.sum(2)) > 8
    fg = ndi.binary_opening(fg, iterations=2)
    lab, n = ndi.label(fg)
    sizes = ndi.sum(fg, lab, range(1,n+1))
    big = np.argmax(sizes)+1
    fg = ndi.binary_fill_holes(lab==big)
    np.save(f'ref/fg_{i}.npy', fg)
    ys,xs = np.nonzero(fg)
    print(i,'bbox x',xs.min(),xs.max(),'y',ys.min(),ys.max())
    rows = np.arange(ys.min(), ys.max()+1, 4)
    L=[];Rr=[]
    for r in rows:
        c = np.nonzero(fg[r])[0]
        L.append(int(c.min())); Rr.append(int(c.max()))
    res[i]=dict(rows=rows.tolist(),left=L,right=Rr,y0=int(ys.min()),y1=int(ys.max()))
    pim = Image.open(f'ref/652LUMILILA_{i}.jpg').convert('RGB')
    pim = Image.blend(pim, Image.new('RGB', pim.size, (255,255,255)), 0.4)
    d = ImageDraw.Draw(pim)
    d.line(list(zip(L, rows.tolist())), fill=(255,0,0), width=12)
    d.line(list(zip(Rr, rows.tolist())), fill=(0,0,255), width=12)
    pim.resize((pim.width//3, pim.height//3)).save(f'ref/prof_{i}.png')
json.dump(res, open('ref/profiles_top.json','w'))
