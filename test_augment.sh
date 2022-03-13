#!/bin/bash

python3 adversarial_trainer.py --do-train --dis-lambda 1e-2 --eval-every 10 --run-name adversarial_0310_fewshot_test_augment --load-dir save/adversarial_0310-01 --save-dir save --train-dir datasets/oodomain_train --train-datasets duorc,race,relation_extraction --val-dir datasets/oodomain_val --recompute-features --augment