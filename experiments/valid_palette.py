"""Portage Python des controles du validateur dataviz (node absent sur la machine)."""
import math

BAND = {'light': (0.43, 0.77), 'dark': (0.48, 0.67)}
CHROMA_FLOOR, CVD_TARGET, CVD_FLOOR, NORMAL_FLOOR, CONTRAST_MIN = 0.10, 8.0, 6.0, 15.0, 3.0
MACHADO = {
    'protan': [[0.152286, 1.052583, -0.204868], [0.114503, 0.786281, 0.099216], [-0.003882, -0.048116, 1.051998]],
    'deutan': [[0.367322, 0.860646, -0.227968], [0.280085, 0.672501, 0.047413], [-0.011820, 0.042940, 0.968881]],
    'tritan': [[1.255528, -0.076749, -0.178779], [-0.078411, 0.930809, 0.147602], [0.004733, 0.691367, 0.303900]],
}
s2lin = lambda c: c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
def lin(h):
    h = h.lstrip('#')
    return [s2lin(int(h[i:i+2], 16) / 255) for i in (0, 2, 4)]
def rel_lum(h):
    r, g, b = lin(h); return 0.2126*r + 0.7152*g + 0.0722*b
def contrast(a, b):
    hi, lo = sorted([rel_lum(a), rel_lum(b)], reverse=True); return (hi + .05) / (lo + .05)
def oklab_from_lin(rgb):
    r, g, b = rgb
    l = (0.4122214708*r + 0.5363325363*g + 0.0514459929*b) ** (1/3)
    m = (0.2119034982*r + 0.6806995451*g + 0.1073969566*b) ** (1/3)
    s = (0.0883024619*r + 0.2817188376*g + 0.6299787005*b) ** (1/3)
    return [0.2104542553*l + 0.7936177850*m - 0.0040720468*s,
            1.9779984951*l - 2.4285922050*m + 0.4505937099*s,
            0.0259040371*l + 0.7827717662*m - 0.8086757660*s]
def oklch(h):
    L, a, b = oklab_from_lin(lin(h)); return L, math.hypot(a, b)
def simulate(h, kind):
    r, g, b = lin(h); M = MACHADO[kind]
    return [min(1, max(0, M[i][0]*r + M[i][1]*g + M[i][2]*b)) for i in range(3)]
def deltaE(h1, h2, kind=None):
    a = oklab_from_lin(simulate(h1, kind) if kind else lin(h1))
    b = oklab_from_lin(simulate(h2, kind) if kind else lin(h2))
    return 100 * math.dist(a, b)

def valider(pal, mode, surface):
    lo, hi = BAND[mode]; ok = True
    off = [(c, round(oklch(c)[0], 3)) for c in pal if not (lo <= oklch(c)[0] <= hi)]
    print(f'  bande de clarte      {"PASS" if not off else "FAIL " + str(off)}')
    ok &= not off
    lowc = [(c, round(oklch(c)[1], 3)) for c in pal if oklch(c)[1] < CHROMA_FLOOR]
    print(f'  plancher de chroma   {"PASS" if not lowc else "FAIL " + str(lowc)}')
    ok &= not lowc
    pairs = [(i, i+1) for i in range(len(pal)-1)]
    worst = min(((deltaE(pal[i], pal[j], k), k, pal[i], pal[j]) for k in ('protan', 'deutan') for i, j in pairs))
    tri = min(deltaE(pal[i], pal[j], 'tritan') for i, j in pairs)
    etat = 'PASS' if worst[0] >= CVD_TARGET else ('PLANCHER' if worst[0] >= CVD_FLOOR else 'FAIL')
    print(f'  separation DALTONIEN {etat}  dE {worst[0]:.1f} ({worst[1]}) · tritan {tri:.1f}')
    ok &= worst[0] >= CVD_FLOOR
    nw = min((deltaE(pal[i], pal[j]), pal[i], pal[j]) for i, j in pairs)
    print(f'  vision normale       {"PASS" if nw[0] >= NORMAL_FLOOR else "FAIL"}  dE {nw[0]:.1f} (plancher {NORMAL_FLOOR:.0f})')
    ok &= nw[0] >= NORMAL_FLOOR
    low = [(c, round(contrast(c, surface), 2)) for c in pal if contrast(c, surface) < CONTRAST_MIN]
    print(f'  contraste / fond     {"PASS" if not low else "RELIEF requis " + str(low)}')
    return ok

print('MODE CLAIR  (fond #FFFFFF)')
a = valider(['#2a78d6', '#eb6834'], 'light', '#FFFFFF')
print('MODE SOMBRE (fond #131C1F)')
b = valider(['#3987e5', '#d95926'], 'dark', '#131C1F')
print('\nVERDICT :', 'palette valide' if a and b else 'A CORRIGER')
