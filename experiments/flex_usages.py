"""Gisement de flexibilite de la climatisation.

Protocole symetrique de celui du chauffage : la consigne de thermostat est une ENTREE
du reseau, on la decale vers le HAUT en ete, on relit la courbe, la difference est
l'effacement.

L'eau chaude a ete ecartee volontairement. Le reseau ne voit NI la temperature du
ballon NI sa consigne : f(t) ne contient que les schedules de PUISAGE. On ne peut donc
pas simuler un ballon pilote, qui est la vraie flexibilite ECS — couper la resistance
pendant la pointe et laisser le stock tenir. Toute mesure produite ici aurait porte sur
un changement d'usage de l'occupant, pas sur un pilotage, et n'aurait pas eu le meme
statut que le chauffage et la climatisation. Le prerequis est d'ajouter la consigne de
ballon aux entrees du modele.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import ts_data as D
from flex_chauffage import Banc, SEQ, NOMS, RES

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

ETE = (6, 7, 8)

SCENARIOS_CLIM = {
    'pointe_ete':  dict(delta=+2.0, heures=(15, 16, 17, 18), mois=ETE,
                        libelle='Pointe d\'été 15–19 h, +2 °C'),
    'soir_ete':    dict(delta=+2.0, heures=(18, 19, 20), mois=ETE,
                        libelle='Soir d\'été 18–21 h, +2 °C'),
    'prerefroid':  dict(delta=+3.0, heures=(15, 16, 17, 18), mois=ETE,
                        avant=(10, 11, 12, 13), delta_avant=-2.0,
                        libelle='Pré-refroidissement −2 °C 10–14 h puis +3 °C 15–19 h'),
}
AMPLITUDES_CLIM = [1.0, 2.0, 3.0, 4.0]


class BancUsages(Banc):
    """Ajoute la climatisation au banc du chauffage."""

    def scenario_col(self, k, decalages, colonne):
        """Comme `scenario`, mais sur une colonne de consigne au choix."""
        Xb = self.X[k].copy()
        for masque, delta in decalages:
            Xb[masque, colonne] += delta
        t_ext = Xb[:, D.I_TEXT]
        Xb[:, D.I_ECH] = np.clip(Xb[:, D.I_HEAT] - t_ext, 0, None)
        Xb[:, D.I_ECC] = np.clip(t_ext - Xb[:, D.I_COOL], 0, None)
        if self.i_besoin is not None:
            c = self.cop[k]
            cop_t = np.clip(c * (0.6 + 0.02 * t_ext), 1.0, max(c, 1.0))
            Xb[:, self.i_besoin] = Xb[:, D.I_ECH] / cop_t
        return self.reference(k), self._predire(k, Xb)

    def indicateurs_clim(self, k, heures, delta, mois, avant=None,
                         delta_avant=0.0, apres=6):
        m_evt = self.masque_heures(k, heures, mois)
        i_cible = 2

        dec = [(m_evt, delta)]
        if avant:
            dec.append((self.masque_heures(k, avant, mois), delta_avant))
        ref, sc = self.scenario_col(k, dec, D.I_COOL)

        dP = (ref - sc).reshape(-1, 4)
        h_abs = np.concatenate([np.arange(d, d + SEQ) for d in self.f_hiver])
        evt = m_evt[h_abs]
        reb = self._fenetre_rebond(evt, apres)

        b = int(self.ids[k])
        out = {'bldg_id': b, 'etat': self.meta.loc[b, 'in.state'],
               'zone': self.meta.loc[b, 'in.ashrae_iecc_climate_zone_2004'],
               'pac': bool(self.cop[k] > 1.0), 'tau': float(self.tau.loc[b]),
               'n_heures_evt': int(evt.sum())}
        for nom, i in [('total', 0), ('chauffage', i_cible)]:
            out[f'pointe_kW_{nom}'] = float(dP[evt, i].max()) if evt.any() else 0.0
            out[f'efface_kWh_{nom}'] = float(dP[evt, i].sum())
            out[f'rebond_kWh_{nom}'] = float(-dP[reb, i].sum())
            out[f'net_kWh_{nom}'] = float(dP[:, i].sum())
        d = self.decalage[b] if self.local else 0
        h_loc = (self.heure[h_abs] + d) % 24
        out['profil_chauffage'] = [float(dP[h_loc == h, i_cible].mean())
                                   if (h_loc == h).any() else 0.0 for h in range(24)]
        out['profil_total'] = [float(dP[h_loc == h, 0].mean())
                               if (h_loc == h).any() else 0.0 for h in range(24)]
        return out


def main(a):
    banc = BancUsages(local=True)
    # les fenetres d'ETE remplacent celles d'hiver pour la climatisation
    deb = np.arange(0, D.N_H - SEQ + 1, SEQ)
    f_ete = np.array([d for d in deb if banc.mois[d + SEQ // 2] in ETE])

    cibles = banc.val[:a.n_bat] if a.n_bat else banc.val
    print(f'{len(cibles)} bâtiments de validation | fenêtres d\'été {len(f_ete)}')
    res = {'meta': {'n_bat': len(cibles)}, 'clim': {}, 'amplitudes_clim': {}}

    banc.f_hiver = f_ete
    banc._ref.clear()
    for nom, s in SCENARIOS_CLIM.items():
        lignes = [banc.indicateurs_clim(k, s['heures'], s['delta'], s['mois'],
                                        s.get('avant'), s.get('delta_avant', 0.0))
                  for k in cibles]
        res['clim'][nom] = {'libelle': s['libelle'], 'batiments': lignes}
        d = pd.DataFrame(lignes)
        print(f"  {s['libelle']:52} pointe {d['pointe_kW_total'].median():5.2f} kW | "
              f"effacé {d['efface_kWh_chauffage'].median():7.1f} kWh | "
              f"report {100*d['rebond_kWh_chauffage'].sum()/max(d['efface_kWh_chauffage'].sum(),1e-9):5.1f} %")

    for amp in AMPLITUDES_CLIM:
        lignes = [banc.indicateurs_clim(k, (15, 16, 17, 18), amp, ETE)
                  for k in cibles]
        res['amplitudes_clim'][str(amp)] = lignes
        d = pd.DataFrame(lignes)
        print(f'  amplitude +{amp:.0f} °C → pointe {d["pointe_kW_total"].median():5.2f} kW | '
              f'effacé {d["efface_kWh_chauffage"].median():7.1f} kWh')

    (RES / a.sortie).write_text(json.dumps(res), encoding='utf-8')
    print('->', RES / a.sortie)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--sortie', default='flex_usages.json')
    p.add_argument('--n_bat', type=int, default=None)
    main(p.parse_args())
