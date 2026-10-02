import numpy as np, json
from scipy.ndimage import median_filter, gaussian_filter1d
P = json.load(open('ref/profiles.json'))
T = json.load(open('ref/profiles_top.json'))
U = np.linspace(0,1,401)
spec = {'u': U.tolist()}
def side(i, flip):
    p=P[i]; cols=np.array(p['cols'],float); x0,x1=p['x0'],p['x1']; Lp=x1-x0
    ground=max(p['bot'])
    u=(cols-x0)/Lp
    if flip: u=1-u
    o=np.argsort(u); u=u[o]
    top=(ground-np.array(p['top'],float)[o])/Lp
    bot=(ground-np.array(p['bot'],float)[o])/Lp
    mid=(ground-median_filter(np.array(p['mid'],float),21)[o])/Lp
    return u, top, bot, mid
for i,flip,name in [('1',False,'lat'),('2',True,'med')]:
    u,top,bot,mid = side(i,flip)
    spec['top_'+name] = gaussian_filter1d(np.interp(U,u,top),1.5).tolist()
    spec['bot_'+name] = np.interp(U,u,bot).tolist()
    spec['mid_'+name] = gaussian_filter1d(np.interp(U,u,mid),3).tolist()
# smooth bottom envelope (lower hull, remove tread notches): rolling min of height then smooth
b = np.minimum(np.array(spec['bot_lat']), np.array(spec['bot_med']))
spec['bot_env'] = gaussian_filter1d(b, 4).tolist()
# top view
t=T['4']; rows=np.array(t['rows'],float); L=np.array(t['left'],float); R=np.array(t['right'],float)
from scipy.interpolate import PchipInterpolator
bad = (rows>3795)&(rows<3891)
good = ~bad
L = PchipInterpolator(rows[good], L[good])(rows); R = PchipInterpolator(rows[good], R[good])(rows)
W=R-L
k = np.argmax((W > 200) & (rows > 300))
row_heel = rows[k]-6; row_toe = rows[-1]+2
print('row_heel', row_heel, 'row_toe', row_toe)
Lp = row_toe-row_heel
u = (rows-row_heel)/Lp
# axis: center at heel(u~0.08) and toe (u~0.97)
c = (L+R)/2
ch = np.interp(0.08,u,c); ct = np.interp(0.985,u,c)
axis = ch + (ct-ch)*(u-0.08)/(0.985-0.08)
wl = (axis-L)/Lp; wm=(R-axis)/Lp
m = u>=0
spec['w_lat'] = np.interp(U,u[m],wl[m]).tolist()
spec['w_med'] = np.interp(U,u[m],wm[m]).tolist()
t=T['4u']; rows2=np.array(t['rows'],float); L2=np.array(t['left'],float); R2=np.array(t['right'],float)
u2=(rows2-row_heel)/Lp; axis2 = ch + (ct-ch)*(u2-0.08)/(0.985-0.08)
spec['wu_lat'] = gaussian_filter1d(np.interp(U,u2,(axis2-L2)/Lp),3).tolist()
spec['wu_med'] = gaussian_filter1d(np.interp(U,u2,(R2-axis2)/Lp),3).tolist()
spec['topview'] = dict(row_heel=float(row_heel), row_toe=float(row_toe), axis_c_heel=float(ch), axis_c_toe=float(ct))
spec['side1'] = dict(x0=P['1']['x0'], x1=P['1']['x1'], ground=max(P['1']['bot']))
spec['side2'] = dict(x0=P['2']['x0'], x1=P['2']['x1'], ground=max(P['2']['bot']))
json.dump(spec, open('spec.json','w'))
for q in np.linspace(0,1,21):
    j=np.argmin(abs(U-q))
    print(f"u={q:.2f} wl={spec['w_lat'][j]:.4f} wm={spec['w_med'][j]:.4f} wul={spec['wu_lat'][j]:.4f} wum={spec['wu_med'][j]:.4f} botenv={spec['bot_env'][j]:.4f} topL={spec['top_lat'][j]:.3f} midL={spec['mid_lat'][j]:.3f} midM={spec['mid_med'][j]:.3f}")
