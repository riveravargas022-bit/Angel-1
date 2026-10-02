import numpy as np
from PIL import Image
for i in [1,2,3,4,5]:
    im = np.asarray(Image.open(f'ref/652LUMILILA_{i}.jpg').convert('RGB')).astype(int)
    h,w,_ = im.shape
    # corners
    print(i, (w,h), 'corner', im[5,5], im[5,w-5], im[h-5,5], im[h-5,w-5])
    d = 255*3 - im.sum(axis=2)
    for t in [3,6,10,20]:
        print('   thr',t,'frac nonwhite', (d>t).mean().round(4))
