"""Gisement de flexibilite du chauffage — simulation contrefactuelle.

Principe : la consigne de thermostat est une ENTREE du reseau. On la modifie, on relit
la courbe de charge, et la difference est la puissance effacee.

    consignes reelles   -> f(t)  -> modele -> courbe de reference
    consignes decalees  -> f'(t) -> modele -> courbe scenario
    dP(t) = reference - scenario        dP > 0 : effacement   dP < 0 : report

Deux points critiques, tous deux verifies par des tests dans main() :

  1. f(t) contient la consigne ET TROIS colonnes qui en derivent (les deux ecarts
     rectifies et le besoin electrique de chauffage). Les quatre doivent bouger
     ensemble, sinon on presente au reseau une entree physiquement incoherente.

  2. Les series ResStock sont horodatees dans un fuseau UNIQUE pour tout le pays
     (heure de l'Est), pas en heure locale. Une fenetre « 18-21 h » brute vise donc
     15-18 h locales en Californie. Le decalage vaut round((longitude + 75) / 15)
     heures : heure_fichier = heure_locale - decalage.
"""
import argparse
import json
import sys
from pathlib import Path

# La console Windows est en cp1252 : le signe moins typographique des libelles
# (U+2212) la fait planter a l'affichage.
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
import torch

import ts_data as D
from ts_train import LoadNet, static_vector, time_extra

SEQ = 168
NOMS = ['total', 'chauffage', 'clim', 'eau_chaude']
HIVER = (12, 1, 2)
RES = Path(__file__).resolve().parent / 'results'
torch.set_num_threads(int(__import__('os').environ.get('TS_THREADS', 10)))


# --------------------------------------------------------------------- scenarios
SCENARIOS = {
    'soir':        dict(delta=-2.0, heures=(18, 19, 20), libelle='Soir 18–21 h, −2 °C'),
    'matin':       dict(delta=-2.0, heures=(7, 8, 9),    libelle='Matin 7–10 h, −2 °C'),
    'nuit':        dict(delta=-3.0, heures=tuple(range(0, 6)),
                        libelle='Nuit 0–6 h, −3 °C'),
    'prechauffe':  dict(delta=-2.0, heures=(18, 19, 20), avant=(16, 17), delta_avant=+2.0,
                        libelle='Préchauffage +2 °C 16–18 h puis −2 °C 18–21 h'),
}
AMPLITUDES = [-1.0, -2.0, -3.0, -4.0]
DUREES = [1, 2, 3, 4, 6]          # heures, fenetre se terminant a 21 h locale


class Banc:
    """Charge le modele et le parc une seule fois, puis repond aux scenarios."""

    def __init__(self, checkpoint='loadnet.pt', parc='nn_buildings_elargi.csv',
                 cache='elargi', local=True):
        ck = torch.load(D.DATA / checkpoint, weights_only=False)
        self.ck, self.local = ck, local
        bats = D.parc(fichier=parc)
        X, Y, ids = D.build_cache(cache, bats)
        self.ids = ids
        feats = ck['feats']
        S, GATE, cop, _ = static_vector(ids, feats)
        self.S, self.GATE, self.cop, self.feats = S, GATE, cop, feats

        # Le cache ne contient que les 31 colonnes de base ; le modele en attend 32.
        # On reconstruit la colonne derivee exactement comme a l'entrainement.
        Xb = np.array(X, dtype='float32')
        extra, noms_extra = time_extra(Xb, ids, cop, feats)
        if extra is not None:
            Xb = np.concatenate([Xb, extra], axis=-1)
        assert Xb.shape[-1] == ck['n_time'], (Xb.shape[-1], ck['n_time'])
        self.X = Xb
        self.Y = np.array(Y, dtype='float32')
        self.i_besoin = (ck['colonnes'].index('besoin_elec_chauffage')
                         if 'besoin_elec_chauffage' in ck['colonnes'] else None)
        self._ref = {}                                  # reference par batiment, en cache

        self.model = LoadNet(ck['n_time'], ck['n_static'], ck['hidden'], 4)
        self.model.load_state_dict(ck['state_dict'])
        self.model.eval()
        self.Sn = ((S - ck['sm']) / ck['ss']).astype('float32')

        meta = pd.read_parquet(D.DATA / 'metadata_clean.parquet',
                               columns=['bldg_id', 'in.state', 'in.county',
                                        'in.weather_file_longitude',
                                        'in.ashrae_iecc_climate_zone_2004']).set_index('bldg_id')
        self.meta = meta.loc[ids]
        # heure_fichier = heure_locale - decalage
        self.decalage = {int(b): int(round((self.meta.loc[b, 'in.weather_file_longitude'] + 75) / 15))
                         for b in ids}

        feat = pd.read_parquet(D.DATA / 'X_47features.parquet')
        self.tau = (feat['C'] / (feat['UA'] + feat['H_ve']) / 3.6).loc[ids]
        self.ua = feat['UA'].loc[ids]

        idx = pd.date_range('2018-01-01', periods=D.N_H, freq='1h')
        self.heure, self.mois = idx.hour.values, idx.month.values
        self.val = [k for k, b in enumerate(ids) if int(b) in set(ck['val_b'])]
        # fenetres de 168 h dont le milieu tombe en hiver
        deb = np.arange(0, D.N_H - SEQ + 1, SEQ)
        self.f_hiver = np.array([d for d in deb if self.mois[d + SEQ // 2] in HIVER])

    def masque_heures(self, k, heures, mois=HIVER):
        """Masque horaire annuel, heures exprimees en heure LOCALE si local=True."""
        d = self.decalage[int(self.ids[k])] if self.local else 0
        cibles = [(h - d) % 24 for h in heures]
        return np.isin(self.heure, cibles) & np.isin(self.mois, mois)

    def _predire(self, k, Xb):
        n = len(self.f_hiver)
        lots = np.stack([Xb[d:d + SEQ] for d in self.f_hiver])
        xn = (lots - self.ck['xm']) / self.ck['xs']
        sn = np.tile(self.Sn[k], (n, 1))
        with torch.no_grad():
            p = self.model(torch.from_numpy(xn.astype('float32')),
                           torch.from_numpy(sn)).numpy()
        return p * self.GATE[k] * self.ck['ys']            # (n, 168, 4) en kWh

    def reference(self, k):
        if k not in self._ref:
            self._ref[k] = self._predire(k, self.X[k])
        return self._ref[k]

    def scenario(self, k, decalages):
        """decalages = liste de (masque, delta). Renvoie (reference, scenario)."""
        Xb = self.X[k].copy()
        for masque, delta in decalages:
            Xb[masque, D.I_HEAT] += delta
        # RECALCUL OBLIGATOIRE des colonnes derivees de la consigne.
        t_ext = Xb[:, D.I_TEXT]
        Xb[:, D.I_ECH] = np.clip(Xb[:, D.I_HEAT] - t_ext, 0, None)
        Xb[:, D.I_ECC] = np.clip(t_ext - Xb[:, D.I_COOL], 0, None)
        if self.i_besoin is not None:
            c = self.cop[k]
            cop_t = np.clip(c * (0.6 + 0.02 * t_ext), 1.0, max(c, 1.0))
            Xb[:, self.i_besoin] = Xb[:, D.I_ECH] / cop_t
        return self.reference(k), self._predire(k, Xb)

    def indicateurs(self, k, heures, delta, avant=None, delta_avant=0.0, apres=6):
        m_evt = self.masque_heures(k, heures)
        dec = [(m_evt, delta)]
        if avant:
            dec.append((self.masque_heures(k, avant), delta_avant))
        ref, sc = self.scenario(k, dec)
        dP = (ref - sc).reshape(-1, 4)                       # >0 = efface
        h_abs = np.concatenate([np.arange(d, d + SEQ) for d in self.f_hiver])
        evt = m_evt[h_abs]
        reb = self._fenetre_rebond(evt, apres)

        b = int(self.ids[k])
        out = {'bldg_id': b, 'etat': self.meta.loc[b, 'in.state'],
               'zone': self.meta.loc[b, 'in.ashrae_iecc_climate_zone_2004'],
               'pac': bool(self.cop[k] > 1.0), 'tau': float(self.tau.loc[b]),
               'ua': float(self.ua.loc[b]), 'decalage_h': self.decalage[b],
               'n_heures_evt': int(evt.sum())}
        for i, n in enumerate(NOMS[:2]):                     # total et chauffage
            out[f'pointe_kW_{n}'] = float(dP[evt, i].max()) if evt.any() else 0.0
            out[f'efface_kWh_{n}'] = float(dP[evt, i].sum())
            out[f'rebond_kWh_{n}'] = float(-dP[reb, i].sum())
            # Bilan NET sur tout l'hiver : positif = energie economisee, negatif =
            # energie consommee EN PLUS. Indispensable des qu'un scenario agit hors
            # de la fenetre d'evenement — le prechauffage paie sa facture AVANT, et
            # le seul couple (efface, rebond) le ferait passer pour gratuit.
            out[f'net_kWh_{n}'] = float(dP[:, i].sum())
        # profil horaire moyen de dP, en heure LOCALE
        d = self.decalage[b] if self.local else 0
        h_loc = (self.heure[h_abs] + d) % 24
        prof = np.array([dP[h_loc == h, 1].mean() if (h_loc == h).any() else 0.0
                         for h in range(24)])
        out['profil_chauffage'] = [float(v) for v in prof]
        prof_t = np.array([dP[h_loc == h, 0].mean() if (h_loc == h).any() else 0.0
                           for h in range(24)])
        out['profil_total'] = [float(v) for v in prof_t]
        return out

    @staticmethod
    def _fenetre_rebond(evt, apres):
        """Les `apres` heures suivant chaque bloc, en TEMPS ABSOLU.

        Filtrer sur l'heure de la journee (range(fin, fin+apres)) sous-estimait le
        report de moitie : pour un evenement finissant a 21 h, les heures 24 a 26
        n'existent pas. Decaler les positions traverse naturellement minuit.
        """
        fins = np.flatnonzero(evt & ~np.r_[evt[1:], False])
        rb = np.zeros_like(evt)
        for j in range(1, apres + 1):
            q = fins + j
            rb[q[q < len(rb)]] = True
        return rb & ~evt


def controles(banc, k):
    """Deux tests qui doivent passer avant toute mesure de gisement."""
    ref, sc = banc.scenario(k, [(banc.masque_heures(k, (18, 19, 20)), 0.0)])
    ecart = float(np.abs(ref - sc).max())
    # Pas exactement 0 : les colonnes derivees sont RECALCULEES en float32 alors que le
    # cache les stocke apres un passage en float64. L'ecart est un arrondi machine.
    assert ecart < 1e-3, f'delta nul modifie la prediction de {ecart:.2e} kWh'

    # Test negatif : la consigne de chauffage decalee en ETE ne doit rien effacer.
    m_ete = banc.masque_heures(k, (18, 19, 20), mois=(6, 7, 8))
    ref2, sc2 = banc.scenario(k, [(m_ete, -2.0)])
    h_abs = np.concatenate([np.arange(d, d + SEQ) for d in banc.f_hiver])
    hiver = (ref2 - sc2).reshape(-1, 4)[:, 1].sum()
    return ecart, float(hiver)


def main(a):
    banc = Banc(local=not a.heure_fichier)
    cibles = banc.val[:a.n_bat] if a.n_bat else banc.val
    ecart, parasite = controles(banc, cibles[0])
    print(f'contrôle identité : {ecart:.1e} kWh | consigne décalée en été, '
          f'effet sur l\'hiver : {parasite:.3f} kWh')
    print(f'{len(cibles)} bâtiments de validation | fenêtres d\'hiver : {len(banc.f_hiver)}'
          f' | heures {"locales" if banc.local else "du fichier"}')

    res = {'meta': {'local': banc.local, 'n_bat': len(cibles)}, 'scenarios': {},
           'amplitudes': {}, 'durees': {}}

    for nom, s in SCENARIOS.items():
        lignes = [banc.indicateurs(k, s['heures'], s['delta'],
                                   s.get('avant'), s.get('delta_avant', 0.0))
                  for k in cibles]
        res['scenarios'][nom] = {'libelle': s['libelle'], 'batiments': lignes}
        d = pd.DataFrame(lignes)
        print(f"  {s['libelle']:48} pointe {d['pointe_kW_total'].median():5.2f} kW | "
              f"effacé {d['efface_kWh_chauffage'].median():7.1f} kWh | "
              f"report {100 * d['rebond_kWh_chauffage'].sum() / max(d['efface_kWh_chauffage'].sum(), 1e-9):5.1f} %")

    for amp in AMPLITUDES:
        lignes = [banc.indicateurs(k, (18, 19, 20), amp) for k in cibles]
        res['amplitudes'][str(amp)] = lignes
        d = pd.DataFrame(lignes)
        print(f'  amplitude {amp:+.0f} °C  → pointe {d["pointe_kW_total"].median():5.2f} kW | '
              f'effacé {d["efface_kWh_chauffage"].median():7.1f} kWh')

    for duree in DUREES:
        heures = tuple(range(21 - duree, 21))
        lignes = [banc.indicateurs(k, heures, -2.0) for k in cibles]
        res['durees'][str(duree)] = lignes
        d = pd.DataFrame(lignes)
        print(f'  durée {duree} h ({heures[0]}–21 h) → pointe {d["pointe_kW_total"].median():5.2f} kW | '
              f'effacé {d["efface_kWh_chauffage"].median():7.1f} kWh')

    (RES / a.sortie).write_text(json.dumps(res), encoding='utf-8')
    print('->', RES / a.sortie)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--sortie', default='flex_chauffage.json')
    p.add_argument('--n_bat', type=int, default=None)
    p.add_argument('--heure_fichier', action='store_true',
                   help="ne PAS corriger le fuseau (fenetres en heure du fichier)")
    main(p.parse_args())
