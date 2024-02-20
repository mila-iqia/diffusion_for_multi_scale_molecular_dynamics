cd-train --data ../data --output output --config config.yaml --start-from-scratch
cd-eval --data ../data --config config.yaml --ckpt-path output/best_model/model.ckpt
