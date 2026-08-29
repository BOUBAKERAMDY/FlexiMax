#!/bin/bash
# Vague 1 : reference + les 3 leviers d'information mesures dans le plan.
# 4 jobs en parallele, 5 threads chacun (le GRU CPU ne monte pas au-dela).
cd "$(dirname "$0")"
export TS_THREADS=5

run () { python ts_train.py --nom "$1" "${@:2}" > "logs/$1.log" 2>&1; echo "fini $1"; }
mkdir -p logs

run base                                   &
run cop   --feats cop                      &
run hdd   --feats hdd                      &
run env   --feats env                      &
wait
echo "=== vague 1 terminee ==="

run copt  --feats cop,copt                 &
run film  --arch film                      &
run huber --loss huber --sched plateau     &
run gru2  --couches 2 --dropout 0.2        &
wait
echo "=== vague 2 terminee ==="
