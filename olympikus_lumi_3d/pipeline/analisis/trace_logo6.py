import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.morphology import skeletonize
mint = np.load('tex/logo_mint.npy')
m = ndi.binary_closing(mint, iterations=2)
lab, n = ndi.label(m); sizes = ndi.sum(m, lab, range(1,n+1))
m = np.isin(lab, [k+1 for k,s in enumerate(sizes) if s>150])
sk = skeletonize(m)
# endpoints & junctions
nb = ndi.convolve(sk.astype(int), np.ones((3,3),int), mode='constant') - sk
ends = np.argwhere(sk & (nb==1)); junc = np.argwhere(sk & (nb>=3))
print('skeleton px', sk.sum(), 'ends', len(ends), 'junc', len(junc))
print('ends', ends.tolist())
print('junc', junc[::3].tolist())
im = np.asarray(Image.open('out/logo_full.png').convert('RGB')).copy()
im[ndi.binary_dilation(sk)] = (255,0,0)
for y,x in ends: im[max(y-5,0):y+5, max(x-5,0):x+5] = (0,0,255)
for y,x in junc: im[max(y-3,0):y+3, max(x-3,0):x+3] = (255,255,0)
Image.fromarray(im).save('out/logo_skel.png')
np.save('tex/mint_skel.npy', sk)
