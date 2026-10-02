"""Builds every texture used by the digital twin.

knit_*      : de-lit, inpainted photo projections of the jacquard knit (albedo + height)
knit_tile   : seamless knit patch for areas no photo sees (heel back)
label       : tongue label (traced calligraphic mark + LUMI)
badge_*     : height + mint masks of the raised side logo
insole      : FEITO<>POR BRASILEIROS insole print
evasense    : midsole print decal
outsole_*   : tread height map from the sole photo
"""
import json, os, sys
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from scipy import ndimage as ndi

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.makedirs(os.path.join(HERE, 'tex'), exist_ok=True)
T = lambda n: os.path.join(HERE, 'tex', n)
REF = lambda i: os.path.join(HERE, 'ref', f'652LUMILILA_{i}.jpg')
FONTS = os.path.join(HERE, 'fonts')
S = json.load(open(os.path.join(HERE, 'spec.json')))
L_MM = 265.0

only = sys.argv[1:]


def want(name):
    return not only or name in only


def load(i):
    return cv2.imread(REF(i), cv2.IMREAD_COLOR)[:, :, ::-1].astype(np.float32)


def save_rgb(path, a):
    Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).save(path)


def save16(path, a):
    a = np.clip(a, 0, 1)
    cv2.imwrite(path, (a * 65535).astype(np.uint16))


# ---------------------------------------------------------------------------
# Olympikus mark (vector paths traced from the mint inlay of the side badge)
# ---------------------------------------------------------------------------
LOGO = json.load(open(T('logo_paths.json')))
LPATHS = [np.array(p, float) for p in LOGO['paths']]
LALL = np.concatenate(LPATHS)
LBX0, LBY0 = LALL.min(0)
LBX1, LBY1 = LALL.max(0)


def draw_mark(size_px, stroke_r, pad, rot_deg=0.0, flip=False, canvas=None, center=None, scale=None):
    """render the mark: returns L image. coordinates in crop px (1 px = 265/4207 mm)."""
    if canvas is None:
        w = (LBX1 - LBX0 + 2 * pad)
        h = (LBY1 - LBY0 + 2 * pad)
        s = size_px / w
        im = Image.new('L', (int(w * s), int(h * s)), 0)
        cx, cy = (LBX0 + LBX1) / 2, (LBY0 + LBY1) / 2
        ox, oy = im.width / 2, im.height / 2
    else:
        im = canvas
        s = scale
        cx, cy = (LBX0 + LBX1) / 2, (LBY0 + LBY1) / 2
        ox, oy = center
    d = ImageDraw.Draw(im)
    a = np.deg2rad(rot_deg)
    ca, sa = np.cos(a), np.sin(a)
    for p in LPATHS:
        q = p - [cx, cy]
        if flip:
            q[:, 1] *= -1
        q = np.stack([q[:, 0] * ca - q[:, 1] * sa, q[:, 0] * sa + q[:, 1] * ca], 1) * s + [ox, oy]
        pts = [tuple(v) for v in q]
        d.line(pts, fill=255, width=max(1, int(round(2 * stroke_r * s))), joint='curve')
        for v in (pts[0], pts[-1]):
            r = stroke_r * s
            d.ellipse([v[0] - r, v[1] - r, v[0] + r, v[1] + r], fill=255)
    return im, s


def smooth_mask(im, blur):
    m = im.filter(ImageFilter.GaussianBlur(blur))
    m = m.point(lambda v: 255 if v > 127 else 0)
    return m.filter(ImageFilter.GaussianBlur(max(blur * 0.35, 0.6)))


# ---------------------------------------------------------------------------
# 1. side badge masks (for the 3D raised logo)
# ---------------------------------------------------------------------------
if want('badge'):
    PX_PER_CROP = 8  # supersampling: 8 px per photo px -> 0.0079 mm/px
    pad = 30
    w_crop = LBX1 - LBX0 + 2 * pad
    size = int(w_crop * PX_PER_CROP)
    stroke, s = draw_mark(size, 21, pad)
    stroke = smooth_mask(stroke, 1.4 * PX_PER_CROP)
    mint, _ = draw_mark(size, 2.5, pad)
    mint = smooth_mask(mint, 0.6 * PX_PER_CROP)
    sm = np.asarray(stroke).astype(np.float32) / 255
    # height: rounded bevel + flat top + shallow channel along the mint line
    mm_per_px = (L_MM / 4207.0) / PX_PER_CROP
    din = ndi.distance_transform_edt(sm > 0.5) * mm_per_px
    bevel = 0.55
    h = np.where(din < bevel, np.sqrt(np.clip(1 - (1 - din / bevel) ** 2, 0, 1)), 1.0)
    dmint = ndi.distance_transform_edt(np.asarray(mint) < 128) * mm_per_px
    channel = 0.10 * np.exp(-(dmint / 0.22) ** 2)
    h = h - channel / 1.0
    h = ndi.gaussian_filter(h * (sm > 0.02), 1.0) * (sm > 0.02)
    save16(T('badge_height.png'), np.clip(h, 0, 1))
    Image.fromarray(np.asarray(mint)).save(T('badge_mint.png'))
    Image.fromarray((sm * 255).astype(np.uint8)).save(T('badge_mask.png'))
    # geometry info: crop px -> photo full res px
    ox, oy = LOGO['crop_origin']
    info = dict(px_per_crop=PX_PER_CROP, pad=pad, crop_x0=float(LBX0 - pad + ox), crop_y0=float(LBY0 - pad + oy),
                w=stroke.width, h=stroke.height, mm_per_px=mm_per_px)
    json.dump(info, open(T('badge_info.json'), 'w'))
    print('badge', stroke.size, info)

# ---------------------------------------------------------------------------
# 2. tongue label (rectified photo trace of the calligraphic mark + LUMI)
# ---------------------------------------------------------------------------
PURPLE = (139, 121, 216)
MINT = (160, 228, 212)
if want('label'):
    im = load(4)
    src = np.float32([[1332.5, 1711], [1705, 1700], [1696, 1870], [1350, 1882]])
    Wd, Hd = 3200, 2000
    dst = np.float32([[0, 0], [Wd, 0], [Wd, Hd], [0, Hd]])
    M = cv2.getPerspectiveTransform(src, dst)
    rect = cv2.warpPerspective(im, M, (Wd, Hd), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    R, G, B = rect[..., 0], rect[..., 1], rect[..., 2]
    mint = ((G - R) > 28).astype(np.float32)
    # keep the label area (exclude the mint knit band above the label)
    yy, xx = np.mgrid[0:Hd, 0:Wd]
    top_edge = 300 - 120 * (xx / Wd)          # label top border curve (rectified)
    inside = (yy > top_edge + 40) & (yy < 1980)
    mint *= inside
    # remove traced LUMI letters (re-typeset below)
    mint[(yy > 1330) & (xx > 1580) & (xx < 2800)] = 0
    mint[(yy > 1700) & (xx > 2900)] = 0                 # mint band below label (bottom-right)
    mint = cv2.GaussianBlur(mint, (0, 0), 14)
    mint = (mint > 0.5).astype(np.uint8)
    n, labm, stats, _ = cv2.connectedComponentsWithStats(mint, 8)
    for k in range(1, n):
        if stats[k, cv2.CC_STAT_AREA] < 20000:
            mint[labm == k] = 0
    mint = cv2.GaussianBlur(mint.astype(np.float32), (0, 0), 2.0)
    lab = np.zeros((Hd, Wd, 3), np.float32) + PURPLE
    lab = lab * (1 - mint[..., None]) + np.array(MINT, np.float32) * mint[..., None]
    img = Image.fromarray(np.clip(lab, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(img)
    font = ImageFont.truetype(os.path.join(FONTS, 'montserrat-latin-500-normal.ttf'), 470)
    x = 1640
    for ch in 'LUMI':
        d.text((x, 1390), ch, font=font, fill=MINT)
        x += d.textlength(ch, font=font) + 95
    img = img.resize((2400, 1500), Image.LANCZOS)
    img.save(T('label.png'))
    print('label ok')

# ---------------------------------------------------------------------------
# 3. insole print
# ---------------------------------------------------------------------------
if want('insole'):
    # canvas: columns = x from the insole start (heel) 0..240 mm ; rows = y from +50 (medial, row 0) to -50 (lateral)
    PPM = 20
    Wc, Hc = 240 * PPM, 100 * PPM
    img = Image.new('L', (Wc, Hc), 255)
    box_w, box_h = 35.5, 20.7                 # mm (along x, along y) measured on the top view
    cx_mm, cy_row = 52.6, 50 + 5.8            # centre: 52.6 mm from the insole start, y = -5.8 mm (lateral)
    SS = 4
    box = Image.new('L', (int(box_w * PPM * SS), int(box_h * PPM * SS)), 255)
    bd = ImageDraw.Draw(box)
    bw = int(1.15 * PPM * SS)
    bd.rectangle([bw // 2, bw // 2, box.width - 1 - bw // 2, box.height - 1 - bw // 2], outline=0, width=bw)
    f = ImageFont.truetype(os.path.join(FONTS, 'oswald-latin-700-normal.ttf'), int(9.6 * PPM * SS))
    pad = int(1.2 * PPM * SS)
    inner_w = box.width - 2 * bw - 2 * pad
    line_h = int((box.height - 2 * bw - 3 * pad // 2) / 2)

    def stamp(text, width_target):
        tw = int(bd.textlength(text, font=f)) + 4
        asc, desc = f.getmetrics()
        t = Image.new('L', (tw, asc + desc), 255)
        ImageDraw.Draw(t).text((0, 0), text, font=f, fill=0)
        bb = Image.eval(t, lambda v: 255 - v).getbbox()
        t = t.crop(bb)
        return t.resize((int(width_target), line_h), Image.LANCZOS)
    x0 = bw + pad
    yA = bw + pad // 2 + int(0.2 * PPM * SS)
    yB = yA + line_h + pad // 2
    feito = stamp('FEITO', inner_w * 0.37)
    por = stamp('POR', inner_w * 0.27)
    bras = stamp('BRASILEIROS', inner_w)
    box.paste(feito, (x0, yA))
    box.paste(por, (box.width - x0 - por.width, yA))
    dx0 = x0 + feito.width + int(0.5 * PPM * SS)
    dx1 = box.width - x0 - por.width - int(0.5 * PPM * SS)
    cyd = yA + line_h // 2
    mid = (dx0 + dx1) / 2
    hw = (dx1 - dx0) / 2
    hh = line_h * 0.30
    bd.polygon([(dx0, cyd), (mid, cyd - hh), (dx1, cyd), (mid, cyd + hh)], fill=0)
    ihw, ihh = hw * 0.42, hh * 0.42
    bd.polygon([(mid - ihw, cyd), (mid, cyd - ihh), (mid + ihw, cyd), (mid, cyd + ihh)], fill=255)
    bd.line([(dx0 - int(0.5 * PPM * SS), cyd), (dx0, cyd)], fill=0, width=int(0.5 * PPM * SS))
    bd.line([(dx1, cyd), (dx1 + int(0.5 * PPM * SS), cyd)], fill=0, width=int(0.5 * PPM * SS))
    box.paste(bras, (x0, yB))
    box = box.resize((int(box_w * PPM), int(box_h * PPM)), Image.LANCZOS)
    img.paste(box, (int((cx_mm - box_w / 2) * PPM), int((cy_row - box_h / 2) * PPM)))
    # partial mark outline just after the box (toward the toe), medial half
    side_px = int(30 * PPM)
    cv_ = Image.new('L', (side_px, side_px), 0)
    mk, _ = draw_mark(None, 30, 0, rot_deg=90, canvas=cv_, center=(side_px / 2, side_px / 2), scale=side_px / (LBX1 - LBX0 + 60) * 0.8)
    mkA = np.asarray(mk) > 127
    er = ndi.binary_erosion(mkA, iterations=8)
    outline = (mkA & ~er).astype(np.uint8) * 255
    om = Image.fromarray(outline).filter(ImageFilter.GaussianBlur(1.0))
    bb = om.getbbox(); om = om.crop(bb)
    ox = int((cx_mm + box_w / 2 + 2.5) * PPM)
    oy = int((cy_row - box_h / 2 - 2.0) * PPM)
    img.paste(Image.new('L', om.size, 0), (ox, oy), om)
    img.save(T('insole_print.png'))
    json.dump(dict(x0_mm=0.0, x1_mm=240.0, y0_mm=-50.0, y1_mm=50.0, start_u=0.048), open(T('insole_info.json'), 'w'))
    print('insole ok')

# ---------------------------------------------------------------------------
# 4. EVASENSE decal (alpha)
# ---------------------------------------------------------------------------
if want('evasense'):
    PPM = 120
    Wc, Hc = int(16.5 * PPM), int(3.6 * PPM)
    a = Image.new('L', (Wc, Hc), 0)
    d = ImageDraw.Draw(a)
    f = ImageFont.truetype(os.path.join(FONTS, 'barlow-condensed-latin-600-normal.ttf'), int(3.05 * PPM))
    txt = 'EVASENSE'
    tw = d.textlength(txt, font=f)
    tx = int(4.2 * PPM)
    tmp = Image.new('L', (int(tw + 40), Hc), 0)
    ImageDraw.Draw(tmp).text((0, -int(0.42 * PPM)), txt, font=f, fill=255)
    tmp = tmp.resize((int(11.6 * PPM), Hc), Image.LANCZOS)
    a.paste(tmp, (tx, int(0.05 * PPM)), tmp)
    # icon: rounded square outline with two loops (mark rotated 90deg)
    side = int(9 * PPM)
    cv_ = Image.new('L', (side, side), 0)
    mk, _ = draw_mark(None, 17, 0, rot_deg=90, canvas=cv_, center=(side / 2, side / 2), scale=side / (LBX1 - LBX0 + 60))
    bb = mk.getbbox(); mk = mk.crop(bb)
    mk = mk.resize((int(2.9 * PPM), int(2.35 * PPM)), Image.LANCZOS)
    mk = smooth_mask(mk, 3)
    a.paste(mk, (int(0.35 * PPM), int(0.62 * PPM)), mk)
    a = a.filter(ImageFilter.GaussianBlur(0.8))
    a.save(T('evasense_alpha.png'))
    print('evasense ok', a.size)

# ---------------------------------------------------------------------------
# 5. knit projections (lateral, medial, top) : de-lit + inpainted albedo, height
# ---------------------------------------------------------------------------


def delight(img, mask, sigma, target):
    lum = img.mean(2)
    w = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigma)
    lo = cv2.GaussianBlur(lum * mask, (0, 0), sigma) / np.maximum(w, 1e-3)
    gain = target / np.maximum(lo, 1.0)
    out = img * gain[..., None]
    return out, lo


def knit_height(img):
    lum = img.mean(2)
    hp = lum - cv2.GaussianBlur(lum, (0, 0), 3.0)
    hp = cv2.GaussianBlur(hp, (0, 0), 0.7)
    return np.clip(0.5 + hp / 60.0, 0, 1)


def side_knit(i, flip):
    img = load(i)
    side = S['side1'] if i == 1 else S['side2']
    x0, x1, g = side['x0'], side['x1'], side['ground']
    Lp = x1 - x0
    fg = np.load(os.path.join(HERE, 'ref', f'fg_{i}.npy'))
    H, W = fg.shape
    R, G, B = img[..., 0], img[..., 1], img[..., 2]
    lum = img.mean(2)
    yy, xx = np.mgrid[0:H, 0:W]
    u = (xx - x0) / Lp
    if flip:
        u = 1 - u
    h = (g - yy) / Lp
    mid = np.interp(u, np.array(S['u']), np.array(S['mid_lat' if i == 1 else 'mid_med']))
    top = np.interp(u, np.array(S['u']), np.array(S['top_lat' if i == 1 else 'top_med']))
    bad = ~fg
    bad |= h < mid - 0.002                          # midsole
    # eyestay / laces / tongue zone (above the eyestay lower edge)
    es_lo = np.interp(u, [0.33, 0.37, 0.45, 0.55, 0.65, 0.70, 0.74],
                      [0.60, 0.338, 0.322, 0.293, 0.262, 0.245, 0.60])
    bad |= (u > 0.33) & (u < 0.74) & (h > es_lo)
    # collar roll: within ~4.5mm of the silhouette top in the collar zone
    bad |= (u < 0.37) & (h > top - 0.020)
    # heel tab strip
    bad |= (u < 0.075) & (((R - G) > 8) & ((B - G) > 25))
    bad |= (u < 0.088)
    # side logo (lateral only)
    if i == 1:
        bm = np.asarray(Image.open(T('badge_mask.png'))).astype(np.float32) / 255
        info = json.load(open(T('badge_info.json')))
        s = 1.0 / info['px_per_crop']
        small = cv2.resize(bm, (int(bm.shape[1] * s), int(bm.shape[0] * s)), interpolation=cv2.INTER_AREA)
        bx, by = int(round(info['crop_x0'])), int(round(info['crop_y0']))
        lm = np.zeros((H, W), np.float32)
        lm[by:by + small.shape[0], bx:bx + small.shape[1]] = small[:H - by, :W - bx]
        bad |= cv2.dilate((lm > 0.1).astype(np.uint8), np.ones((49, 49), np.uint8)) > 0
    good = ~bad
    good = cv2.erode(good.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    # crop to the upper zone
    rows = np.nonzero(good.any(1))[0]
    cols = np.nonzero(good.any(0))[0]
    cy0, cy1 = max(rows.min() - 80, 0), min(rows.max() + 120, H)
    cx0, cx1 = max(cols.min() - 120, 0), min(cols.max() + 120, W)
    img = img[cy0:cy1, cx0:cx1]
    good = good[cy0:cy1, cx0:cx1]
    target = 205.0
    dl, lo = delight(img, good, 220, target)
    # inpaint bad zones (texture-agnostic, hidden under overlays)
    inp = np.zeros_like(dl)
    m8 = (~good).astype(np.uint8) * 255
    small = cv2.resize(np.clip(dl, 0, 255).astype(np.uint8), None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    msmall = cv2.resize(m8, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_NEAREST)
    fill = cv2.inpaint(small, msmall, 12, cv2.INPAINT_TELEA)
    fill = cv2.resize(fill, (dl.shape[1], dl.shape[0]), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    # add knit-like noise in filled zones: borrow texture from a shifted copy of good pixels
    # texture-preserving fill: copy knit from the same rows, shifted along the length
    hraw = knit_height(img)
    tex_fill = fill.copy()
    hfill = np.full(hraw.shape, 0.5, np.float32)
    todo = ~good
    for sh in (520, -520, 1040, -1040, 260, -260, 1560, -1560, 780, -780):
        src_ok = np.roll(good, sh, axis=1)
        take = todo & src_ok
        tex_fill[take] = np.roll(dl, sh, axis=1)[take]
        hfill[take] = np.roll(hraw, sh, axis=1)[take]
        todo &= ~take
    fill2 = tex_fill
    soft = cv2.GaussianBlur(good.astype(np.float32), (0, 0), 4)[..., None]
    out = dl * soft + fill2 * (1 - soft)
    hgt = hraw * soft[..., 0] + hfill * (1 - soft[..., 0])
    name = 'lat' if i == 1 else 'med'
    save_rgb(T(f'knit_{name}.jpg'.replace('.jpg', '.png')), out)
    save16(T(f'knit_{name}_h.png'), hgt)
    cv2.imwrite(T(f'knit_{name}_mask.png'), (soft[..., 0] * 255).astype(np.uint8))
    meta = dict(cx0=int(cx0), cy0=int(cy0), cx1=int(cx1), cy1=int(cy1), x0=x0, Lp=Lp, ground=g, flip=flip)
    print(name, meta)
    return meta


if want('knit'):
    metas = {}
    metas['lat'] = side_knit(1, False)
    metas['med'] = side_knit(2, True)
    # ---- top view
    img = load(4)
    fg = np.load(os.path.join(HERE, 'ref', 'fg_4.npy'))
    H, W = fg.shape
    R, G, B = img[..., 0], img[..., 1], img[..., 2]
    tv = S['topview']
    rh, rt = tv['row_heel'], tv['row_toe']
    Lt = rt - rh
    yy, xx = np.mgrid[0:H, 0:W]
    u = (yy - rh) / Lt
    blue = B - (R + G) / 2
    upm = np.load(os.path.join(HERE, 'ref', 'up_4.npy'))
    bad = ~upm
    # exclude collar/lining/insole/tongue/laces/eyestay: everything inside the lacing+opening zone
    axis = tv['axis_c_heel'] + (tv['axis_c_toe'] - tv['axis_c_heel']) * (u - 0.08) / (0.985 - 0.08)
    dy = (xx - axis) / Lt
    hw = np.interp(u, [-0.1, 0.0, 0.08, 0.20, 0.33, 0.40, 0.50, 0.60, 0.66, 0.70, 0.72, 1.2],
                   [0.30, 0.30, 0.19, 0.17, 0.165, 0.150, 0.130, 0.110, 0.095, 0.060, 0.0, 0.0])
    bad |= np.abs(dy - 0.004) < hw
    bad |= (blue > 22) & (img.mean(2) < 175)      # lining blue-gray, laces shadows
    good = ~bad
    good = cv2.erode(good.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    rows = np.nonzero(good.any(1))[0]
    cols = np.nonzero(good.any(0))[0]
    cy0, cy1 = max(rows.min() - 80, 0), min(rows.max() + 80, H)
    cx0, cx1 = max(cols.min() - 80, 0), min(cols.max() + 80, W)
    img = img[cy0:cy1, cx0:cx1]
    good = good[cy0:cy1, cx0:cx1]
    dl, lo = delight(img, good, 200, 205.0)
    m8 = (~good).astype(np.uint8) * 255
    small = cv2.resize(np.clip(dl, 0, 255).astype(np.uint8), None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    msmall = cv2.resize(m8, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_NEAREST)
    fill = cv2.inpaint(small, msmall, 12, cv2.INPAINT_TELEA)
    fill = cv2.resize(fill, (dl.shape[1], dl.shape[0]), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    soft = cv2.GaussianBlur(good.astype(np.float32), (0, 0), 6)[..., None]
    out = dl * soft + fill * (1 - soft)
    hgt = knit_height(img) * soft[..., 0] + 0.5 * (1 - soft[..., 0])
    save_rgb(T('knit_top.png'), out)
    save16(T('knit_top_h.png'), hgt)
    cv2.imwrite(T('knit_top_mask.png'), (soft[..., 0] * 255).astype(np.uint8))
    metas['top'] = dict(cx0=int(cx0), cy0=int(cy0), cx1=int(cx1), cy1=int(cy1), row_heel=rh, row_toe=rt,
                        axis_c_heel=tv['axis_c_heel'], axis_c_toe=tv['axis_c_toe'], W=W, H=H)
    print('top', metas['top'])
    # ---- seamless tile from the lateral heel/quarter area (dot jacquard)
    img = load(1)
    patch = img[1430:1430 + 512, 760:760 + 768].copy()
    dlp, _ = delight(patch, np.ones(patch.shape[:2], bool), 120, 205.0)
    # make tileable by mirror-free cross-fade (offset & blend)
    def seamless(a, f=64):
        h, w = a.shape[:2]
        out = a.copy()
        r = np.roll(np.roll(a, w // 2, 1), h // 2, 0)
        wx = np.minimum(np.arange(w), np.arange(w)[::-1]).astype(np.float32)
        wy = np.minimum(np.arange(h), np.arange(h)[::-1]).astype(np.float32)
        mw = np.clip(np.minimum(wx[None, :], wy[:, None]) / f, 0, 1)[..., None]
        return a * mw + r * (1 - mw)
    tile = seamless(dlp)
    save_rgb(T('knit_tile.png'), tile)
    save16(T('knit_tile_h.png'), knit_height(tile))
    json.dump(metas, open(T('knit_meta.json'), 'w'))

# ---------------------------------------------------------------------------
# 6. outsole tread height from the sole photo
# ---------------------------------------------------------------------------
if want('outsole'):
    # The sole photo is shot straight on, like the sole render: project it.
    #  outsole_ao.png : de-lit luminance ratio (channel shadows, lug walls, debossed logo) -> albedo multiplier
    #  outsole_h.png  : band-passed luminance -> bump (grip dots + lug edges)
    img = load(5)
    fg = np.load(os.path.join(HERE, 'ref', 'fg_5.npy'))
    lum = img.mean(2)
    H, W = lum.shape
    sole = fg.copy()
    sole[3925:, :] = False                      # acrylic stand under the heel
    sole = cv2.erode(sole.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    rows = np.nonzero(sole.any(1))[0]
    cols = np.nonzero(sole.any(0))[0]
    y0, y1 = rows.min(), rows.max()
    x0, x1 = cols.min(), cols.max()
    w = cv2.GaussianBlur(sole.astype(np.float32), (0, 0), 30)
    lo = cv2.GaussianBlur(lum * sole, (0, 0), 30) / np.maximum(w, 1e-3)
    ratio = np.where(sole, lum / np.maximum(lo, 1.0), 1.0)
    ratio = np.clip(ratio, 0.72, 1.12)
    # fade to neutral at the outline (avoid the photo's rim shading on the bevel)
    d_edge = cv2.distanceTransform(sole.astype(np.uint8), cv2.DIST_L2, 5)
    fade = np.clip(d_edge / 40.0, 0, 1)
    ratio = 1.0 + (ratio - 1.0) * fade
    band = cv2.GaussianBlur(lum, (0, 0), 1.0) - cv2.GaussianBlur(lum, (0, 0), 10.0)
    hgt = 0.5 + 0.5 * np.clip(band / 22.0, -1, 1) * fade
    hgt = np.where(sole, hgt, 0.5)
    c_ao = ratio[y0:y1 + 1, x0:x1 + 1] / 1.25
    c_h = hgt[y0:y1 + 1, x0:x1 + 1]
    save16(T('outsole_ao.png'), c_ao)
    save16(T('outsole_h.png'), c_h)
    json.dump(dict(x0=int(x0), x1=int(x1), y0=int(y0), y1=int(y1), ao_scale=1.25), open(T('outsole_info.json'), 'w'))
    cv2.imwrite(os.path.join(HERE, 'out', 'outsole_h_prev.png'), cv2.resize((c_ao / c_ao.max() * 255).astype(np.uint8), None, fx=0.3, fy=0.3))
    print('outsole', c_h.shape)


# ---------------------------------------------------------------------------
# 7. tongue (front): mint ribbed top band, LUMI label, open air-mesh below
#    UV: U = 0.5 + y / 60mm ; V = 1 - rem / 110mm   (rem = distance from the top edge)
# ---------------------------------------------------------------------------
if want('tongue'):
    PPM = 24
    Wt, Ht = 60 * PPM, 110 * PPM
    X = (np.arange(Wt) + 0.5) / PPM - 30.0          # mm across (y)
    R = (np.arange(Ht) + 0.5) / PPM                  # mm from the top edge (row 0 = top)
    XX, RR = np.meshgrid(X, R)
    col = np.zeros((Ht, Wt, 3), np.float32)
    hgt = np.zeros((Ht, Wt), np.float32)
    # air mesh: staggered diamond holes
    px, py = 2.7, 2.3
    row = np.floor(RR / py)
    xs = XX + (row % 2) * px / 2
    fx = (xs / px) - np.floor(xs / px) - 0.5
    fy = (RR / py) - np.floor(RR / py) - 0.5
    dia = np.abs(fx) / 0.30 + np.abs(fy) / 0.27
    hole = np.clip((1.0 - dia) * 6, 0, 1)
    mesh_c = np.array([196, 194, 212], np.float32) * (1 - hole[..., None]) + np.array([70, 68, 92], np.float32) * hole[..., None]
    yarn = 0.5 + 0.5 * np.cos(2 * np.pi * (XX / 0.55 + RR / 0.9))
    mesh_c *= (0.93 + 0.07 * yarn)[..., None]
    mesh_h = (1 - hole) * (0.6 + 0.4 * (1 - np.clip(dia, 0, 2) / 2))
    # mint ribbed knit band (top 0..31 mm)
    course = 1.05
    vv = (RR / course) - np.floor(RR / course)
    chev = np.abs(((XX / 0.9) % 1.0) - 0.5)
    rib = np.clip(np.sin(np.pi * vv) * (0.75 + 0.5 * chev), 0, 1)
    mint_c = np.array([160, 226, 210], np.float32) * (0.86 + 0.16 * rib)[..., None]
    mint_h = 0.4 + 0.6 * rib
    band = (RR < 31.0).astype(np.float32)
    band = cv2.GaussianBlur(band, (0, 0), 0.6 * PPM)
    col = mesh_c * (1 - band[..., None]) + mint_c * band[..., None]
    hgt = mesh_h * (1 - band) + mint_h * band
    # label 26 x 16.25 mm, top edge at 8.5 mm from the top
    lab = Image.open(T('label.png')).convert('RGB')
    lw, lh = int(26 * PPM), int(16.25 * PPM)
    lab = np.asarray(lab.resize((lw, lh), Image.LANCZOS)).astype(np.float32)
    x0 = int((30 - 13) * PPM); y0 = int(5.5 * PPM)
    rr_ = int(1.2 * PPM)
    lm = np.zeros((lh, lw), np.float32)
    cv2.rectangle(lm, (rr_, 0), (lw - 1 - rr_, lh - 1), 1.0, -1)
    cv2.rectangle(lm, (0, rr_), (lw - 1, lh - 1 - rr_), 1.0, -1)
    for cx_, cy_ in ((rr_, rr_), (lw - 1 - rr_, rr_), (rr_, lh - 1 - rr_), (lw - 1 - rr_, lh - 1 - rr_)):
        cv2.circle(lm, (cx_, cy_), rr_, 1.0, -1)
    lm = cv2.GaussianBlur(lm, (0, 0), 0.8)
    sl = (slice(y0, y0 + lh), slice(x0, x0 + lw))
    col[sl] = col[sl] * (1 - lm[..., None]) + lab * lm[..., None]
    hgt[sl] = hgt[sl] * (1 - lm) + (0.85 + 0.0 * lm) * lm
    # dark seam shadow along the label's top edge
    shadow = np.zeros((Ht, Wt), np.float32)
    shadow[y0 - int(0.5 * PPM):y0 + int(0.25 * PPM), x0:x0 + lw] = 1
    shadow = cv2.GaussianBlur(shadow, (0, 0), 0.25 * PPM)
    col *= (1 - 0.55 * shadow)[..., None]
    save_rgb(T('tongue.png'), col)
    save16(T('tongue_h.png'), np.clip(hgt, 0, 1))
    print('tongue tex', col.shape)

# ---------------------------------------------------------------------------
# 8. eyestay (TPU): embossed shield ridge + mint inlay line + mint accents
#    UV: U = s (along the edge loop), V = t (0 at the throat edge -> 1 at the lower boundary)
#    R channel = mint mask, G channel = height
# ---------------------------------------------------------------------------
if want('eyestay'):
    import shoe_parts as sp
    up = sp.Upper()
    V_, F_, es = sp.eyestay_mesh(up)
    E, low, uE, sideE, s_ = es['E'], es['low'], es['uE'], es['side'], es['s']
    seg = np.linalg.norm(np.diff(E, axis=0), axis=1)
    along = np.concatenate([[0], np.cumsum(seg)]) * 1000          # mm
    width = np.linalg.norm(low - E, axis=1) * 1000                  # mm
    PPM = 12
    Lmm = along[-1]
    Wmax = float(width.max()) + 1
    nA, nW = int(Lmm * PPM), int(Wmax * PPM)
    A = (np.arange(nA) + 0.5) / PPM
    Wd = (np.arange(nW) + 0.5) / PPM
    wA = np.interp(A, along, width)
    uA = np.interp(A, along, uE)
    sdA = np.interp(A, along, sideE)
    AA, WW = np.meshgrid(A, Wd)                                     # (nW, nA)
    inside = WW <= wA[None, :]
    shield = np.zeros_like(inside)
    mint = np.zeros(inside.shape, np.float32)
    accent = np.zeros(inside.shape, np.float32)
    for side, u0, u1 in (('lat', 0.372, 0.565), ('med', 0.388, 0.540)):
        sel = (sdA < 0.5) if side == 'lat' else (sdA >= 0.5)
        inr = sel & (uA >= u0) & (uA <= u1)
        cols = np.nonzero(inr)[0]
        a0, a1 = A[cols.min()], A[cols.max()]
        reg = inside & (AA >= a0) & (AA <= a1) & (WW >= 2.6) & (WW <= wA[None, :] - 3.0)
        # rounded corners (radius 5 mm) by opening
        k = int(5 * PPM)
        reg8 = cv2.morphologyEx(reg.astype(np.uint8), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1)))
        shield |= reg8.astype(bool)
        # mint inlay line: 3.4 mm inside the shield edge, open toward the toe
        d_in = cv2.distanceTransform(reg8, cv2.DIST_L2, 5) / PPM
        line = np.exp(-((d_in - 3.6) / 0.28) ** 2) * (d_in > 0)
        toe_side = (AA > a1 - 7.0) if side == 'lat' else (AA < a0 + 7.0)
        line *= ~toe_side
        mint = np.maximum(mint, line.astype(np.float32))
    # ridge = band 1.6 mm inside the shield outline, rounded profile
    sh8 = shield.astype(np.uint8)
    d_in = cv2.distanceTransform(sh8, cv2.DIST_L2, 5) / PPM
    ridge_in = np.clip(np.sin(np.clip(d_in / 1.7, 0, 1) * np.pi), 0, 1)
    d_out = cv2.distanceTransform(1 - sh8, cv2.DIST_L2, 5) / PPM
    ridge = np.where(sh8 > 0, ridge_in, np.exp(-(d_out / 0.30) ** 2) * 0.5)
    # small mint accents (swoosh lines) below eyelets 4 & 6 (lateral) / 3 & 5 (medial)
    for side, ulist in (('lat', (0.575, 0.675)), ('med', (0.585, 0.69))):
        sel = (sdA < 0.5) if side == 'lat' else (sdA >= 0.5)
        for uc in ulist:
            cols = np.nonzero(sel & (np.abs(uA - uc) < 0.004))[0]
            if len(cols) == 0:
                continue
            ac = A[cols[len(cols) // 2]]
            wc = wA[cols[len(cols) // 2]]
            tline = 0.70 * wc + 0.10 * (AA - ac) * (1 if side == 'lat' else -1)
            m = np.exp(-((WW - tline) / 0.30) ** 2) * (np.abs(AA - ac) < 4.5)
            accent = np.maximum(accent, m.astype(np.float32))
    # resample to UV image: U = s (proportional to along), V = t = W / w(A)
    Uimg, Vimg = 4096, 512
    us = np.linspace(0, 1, Uimg)
    Aq = np.interp(us, s_, along)
    wq = np.interp(Aq, along, width)
    tq = (np.arange(Vimg) + 0.5) / Vimg
    mapx = (Aq[None, :] * PPM - 0.5).astype(np.float32).repeat(Vimg, 0)
    mapy = ((tq[:, None] * wq[None, :]) * PPM - 0.5).astype(np.float32)
    mint_uv = cv2.remap(mint, mapx, mapy, cv2.INTER_LINEAR)
    ridge_uv = cv2.remap(ridge.astype(np.float32), mapx, mapy, cv2.INTER_LINEAR)
    img = np.zeros((Vimg, Uimg, 3), np.float32)
    img[..., 0] = mint_uv * 255
    img[..., 1] = ridge_uv * 255
    img = img[::-1]       # V up
    save_rgb(T('eyestay.png'), img)
    print('eyestay tex', img.shape, 'loop length mm', Lmm)
