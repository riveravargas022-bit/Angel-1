import numpy as np, json
P = json.load(open('ref/profiles.json'))
for i,flip in [('1',False),('2',True)]:
    pts = json.load(open(f'ref/creases_{i}.json'))
    p=P[i]; x0,x1=p['x0'],p['x1']; Lp=x1-x0; ground=max(p['bot'])
    cols = sorted(set(c for c,_ in pts))
    bycol = {c:[r for cc,r in pts if cc==c] for c in cols}
    tracks=[]  # list of lists of (c,r)
    active=[]
    for c in cols:
        rs = bycol[c]; used=set(); newactive=[]
        for tr in active:
            lc, lr = tr[-1]
            if c-lc>40: continue
            best=None
            for j,r in enumerate(rs):
                if j in used: continue
                if abs(r-lr)<=14 and (best is None or abs(r-lr)<abs(rs[best]-lr)): best=j
            if best is not None:
                used.add(best); tr.append((c,rs[best])); newactive.append(tr)
        for j,r in enumerate(rs):
            if j not in used:
                tr=[(c,r)]; tracks.append(tr); newactive.append(tr)
        active=newactive
    tracks=[t for t in tracks if (t[-1][0]-t[0][0])>250]
    print('img',i,'tracks',len(tracks))
    out=[]
    for t in tracks:
        a=np.array(t,float); u=(a[:,0]-x0)/Lp; h=(ground-a[:,1])/Lp
        if flip: u=1-u
        o=np.argsort(u); u=u[o]; h=h[o]
        s = ' '.join(f'({uu:.2f},{hh:.3f})' for uu,hh in zip(u[::max(1,len(u)//8)], h[::max(1,len(u)//8)]))
        print('  ', f'u {u.min():.2f}-{u.max():.2f}:', s)
        out.append(list(zip(u.tolist(),h.tolist())))
    json.dump(out, open(f'ref/tracks_{i}.json','w'))
