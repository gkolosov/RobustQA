#!/bin/bash

if [ "$1" = "train_indomain" ]; then
	python3 adversarial_trainer.py --do-train --dis-lambda 1e-2 --run-name adversarial_0303 --save-dir save --recompute-features
elif [ "$1" = "train_oodomain" ]; then
  python3 adversarial_trainer.py --do-train --dis-lambda 1e-2 --run-name adversarial_0303_finetune --load-dir save/adversarial_0303 --save-dir save --recompute-features --train-dir datasets/oodomain_train --train-datasets duorc,race,relation_extraction --val-dir datasets/oodomain_val