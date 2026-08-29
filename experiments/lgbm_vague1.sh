#!/bin/bash
# LightGBM : un levier a la fois, en sequentiel, 3 threads pour laisser tourner le reseau.
cd "$(dirname "$0")"
mkdir -p logs
J="--jobs 3 --arbres 3000"

python lgbm_exp.py --nom base    $J                        > logs/lgbm_base.log    2>&1
python lgbm_exp.py --nom log1p   $J --log1p                > logs/lgbm_log1p.log   2>&1
python lgbm_exp.py --nom cop     $J --feats cop            > logs/lgbm_cop.log     2>&1
python lgbm_exp.py --nom climat  $J --feats clim           > logs/lgbm_climat.log  2>&1
python lgbm_exp.py --nom cc      $J --feats cop,clim       > logs/lgbm_cc.log      2>&1
python lgbm_exp.py --nom tweedie $J --feats cop,clim --objectif tweedie > logs/lgbm_tweedie.log 2>&1
echo "=== lgbm vague 1 terminee ==="
