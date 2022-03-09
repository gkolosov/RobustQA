#!/bin/bash

if [ "$1" = "train_indomain" ]; then
	python3 adversarial_trainer.py --do-train --dis-lambda 1e-2 --batch-size 32 --run-name adversarial_0303 --save-dir save
elif [ "$1" = "train_oodomain" ]; then
  python3 adversarial_trainer.py --do-train --dis-lambda 1e-2 --eval-every 100 --run-name adversarial_0303_finetune --load-dir save/adversarial_0303-02 --save-dir save --train-dir datasets/oodomain_train --train-datasets duorc,race,relation_extraction --val-dir datasets/oodomain_val --visualize-predictions --freeze-bert
elif [ "$1" = "eval_oodomain" ]; then
    python3 adversarial_trainer.py --do-eval --eval-dir datasets/oodomain_val --sub-file submission.csv --save-dir save/adversarial_0303_finetune-02 --visualize-predictions
else
	echo "Invalid Option Selected"
fi