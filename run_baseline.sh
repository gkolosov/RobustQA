#!/bin/bash

if [ "$1" = "train_indomain" ]; then
	python3 baseline_trainer.py --do-train --dis-lambda 1e-2 --batch-size 32 --run-name baseline_0310 --save-dir save
elif [ "$1" = "train_oodomain" ]; then
  python3 baseline_trainer.py --do-train --dis-lambda 1e-2 --eval-every 10 --run-name baseline_0310_finetune --load-dir save/baseline_0310-01 --save-dir save --train-dir datasets/oodomain_train --train-datasets duorc,race,relation_extraction --val-dir datasets/oodomain_val
elif [ "$1" = "eval_oodomain" ]; then
    python3 baseline_trainer.py --do-eval --eval-dir datasets/oodomain_val --sub-file submission.csv --save-dir save/baseline_0310_finetune-01 --visualize-predictions
elif [ "$1" = "debug" ]; then
  python3 baseline_trainer.py --do-train --debug 500 --run-name augment --save-dir save --eval-every 5 --augment --recompute-features
else
	echo "Invalid Option Selected"
fi