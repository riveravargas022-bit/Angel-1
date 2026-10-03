import numpy as np, json
from PIL import Image
from scipy import ndimage as ndi
res = {}
for i in [1,2]:
    im = np.asarray(Image.open(f'ref/652LUMILILA_{i}.jpg').convert('RGB')).astype(float)
    fg = np.load(f'ref/fg_{i}.npy')
    h,w = fg.shape
    R,G,B = im[...,0],im[...,1],im[...,2]
    blue = B - (R+G)/2
    neutral = (fg & (blue < 6)).astype(float)
    dens = ndi.uniform_filter(neutral, 25)
    solid = (dens > 0.97) & fg
    ys,xs = np.nonzero(fg)
    x0,x1 = xs.min(), xs.max()
    cols = np.arange(x0, x1+1, 4)
    top=[];bot=[];mid=[]
    for c in cols:
        col = fg[:,c]
        r = np.nonzero(col)[0]
        top.append(int(r.min())); bot.append(int(r.max()))
        s = solid[:,c]
        # walk up from bottom while solid
        rr = r.max()-3
        while rr>r.min() and s[rr]: rr-=1
        mid.append(int(rr))
    res[i] = dict(x0=int(x0),x1=int(x1),cols=cols.tolist(),top=top,bot=bot,mid=mid)
    # visualize
    vis = (im*0.5+127).astype(np.uint8)
    for c,t,b,m in zip(cols,top,bot,mid):
        vis[max(t-3,0):t+3,c]=(255,0,0); vis[b-3:b+3,c]=(0,160,0); vis[m-3:m+3,c]=(0,0,255)
    Image.fromarray(vis).resize((w//3,h//3)).save(f'ref/prof_{i}.png')
json.dump(res, open('ref/profiles.json','w'))
print('ok')
