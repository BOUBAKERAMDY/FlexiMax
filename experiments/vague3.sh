#!/bin/bash
# Vague 3 : les gagnants (cop + copt) croises avec les trois leviers de DONNEES.
# 2 jobs a la fois seulement : le cache des 2005 batiments pese 2,3 Go par job.
cd "$(dirname "$0")"
mkdir -p logs
export TS_THREADS=9
ELARGI="--parc nn_buildings_elargi.csv --cache elargi --val_ref"

run () { python ts_train.py --nom "$1" "${@:2}" > "logs/$1.log" 2>&1; echo "fini $1"; }

# A : effet des niveaux annuels hors echantillon, a parc et stride constants
run oos      --feats cop,copt,oos                                          &
# B : effet du parc elargi, memes 101 batiments de validation qu'a 503
run parc2005 --feats cop,copt,oos $ELARGI                                  &
wait
echo "=== A/B donnees termines ==="

# C : effet des fenetres glissantes (7x plus de fenetres pour le meme parc)
run stride24 --feats cop,copt,oos --stride 24 --epochs 12 --patience 4     &
# D : tout cumule
run tout     --feats cop,copt,oos $ELARGI --stride 48 --epochs 12 --patience 4 &
wait
echo "=== vague 3 terminee ==="
