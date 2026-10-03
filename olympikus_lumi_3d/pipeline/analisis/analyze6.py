import numpy as np, json
from scipy.ndimage import median_filter
P = json.load(open('ref/profiles.json'))
T = json.load(open('ref/profiles_top.json'))
# lateral (img1): heel at left (x0), toe at right (x1)
for i,flip in [('1',False),('2',True)]:
    p=P[i]; cols=np.array(p['cols'],float); top=np.array(p['top'],float); bot=np.array(p['bot'],float); mid=median_filter(np.array(p['mid'],float),21)
    x0,x1=p['x0'],p['x1']; Lp=x1-x0
    ground = bot.max()
    u=(cols-x0)/Lp
    if flip: u=1-u
    o=np.argsort(u); u=u[o]; top=top[o]; bot=bot[o]; mid=mid[o]
    print('img',i,'Lpx',Lp,'ground',ground)
    for q in np.linspace(0,1,21):
        k=np.argmin(abs(u-q))
        print(f'  u={q:.2f}  top={(ground-top[k])/Lp:.4f}  mid={(ground-mid[k])/Lp:.4f}  bot={(ground-bot[k])/Lp:.4f}')
for i in ['4','5']:
    t=T[i]; rows=np.array(t['rows'],float); L=np.array(t['left'],float); R=np.array(t['right'],float)
    Lp=t['y1']-t['y0']
    print('img',i,'Lpx',Lp)
    for q in np.linspace(0,1,21):
        r=t['y0']+q*Lp; k=np.argmin(abs(rows-r))
        print(f'  v={q:.2f}  left={(L[k]-t["y0"]*0)/Lp:.4f} right={R[k]/Lp:.4f} width={(R[k]-L[k])/Lp:.4f} center={(R[k]+L[k])/2/Lp:.4f}')
