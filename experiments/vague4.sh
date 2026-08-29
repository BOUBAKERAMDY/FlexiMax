#!/bin/bash
cd "$(dirname "$0")"
export TS_THREADS=7
run () { python ts_train.py --nom "$1" "${@:2}" > "logs/$1.log" 2>&1; echo "fini $1"; }

run final --feats cop,copt,oos --parc nn_buildings_elargi.csv --cache elargi --val_ref \
          --save loadnet.pt                                                            &
run lag   --feats cop,copt,oos,lag                                                     &
run solar --feats cop,copt,oos,solar                                                   &
wait
echo "=== vague 4 terminee ==="
