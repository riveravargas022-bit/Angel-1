import numpy as np
from PIL import Image
from scipy import ndimage as ndi
for i in [1,2]:
    im = np.asarray(Image.open(f'ref/652LUMILILA_{i}.jpg').convert('RGB')).astype(float)
    h,w,_ = im.shape
    R,G,B = im[...,0],im[...,1],im[...,2]
    fg = (765 - im.sum(2)) > 8
    fg = ndi.binary_opening(fg, iterations=2)
    lab, n = ndi.label(fg)
    sizes = ndi.sum(fg, lab, range(1,n+1))
    big = np.argmax(sizes)+1
    fg = ndi.binary_fill_holes(lab==big)
    blue = B - (R+G)/2
    # label image: 0 bg, 1 midsole (neutral), 2 upper (bluish)
    out = np.zeros((h,w,3),np.uint8)+255
    neutral = fg & (blue < 6)
    bluish = fg & (blue >= 6)
    out[neutral] = (255,140,0)
    out[bluish] = (60,60,200)
    Image.fromarray(out).resize((w//3,h//3)).save(f'ref/seg_{i}.png')
    ys,xs = np.nonzero(fg)
    print(i,'bbox x',xs.min(),xs.max(),'y',ys.min(),ys.max(), 'len px', xs.max()-xs.min(), 'height', ys.max()-ys.min())
    np.save(f'ref/fg_{i}.npy', fg)
    # sample color stats
    print(' midsole mean', im[neutral].mean(0).round(1), ' upper mean', im[bluish].mean(0).round(1))
