# STTABT: Sparse Token Transformer with Attention Back-Tracking

Sparse Token Transformer with Attention Back-Tracking

# Experiments

## ViT Concrete Masking

```sh
#training
python -m main.vit_concrete_end2end --n-gpus $NGPU --imagenet-root /path/to/ILSVRC2012/
#ploting
python -m main.plot.vit_concrete_with_dyvit
python -m main.plot.vit_concrete_flops
python -m main.visualize.vit
```

lvvit


python -m main.vit_concrete_end2end --factor 4 --n-gpus 3 --model lvvit-small --master-port 14431 --auto-resume --p-logits "-1.5 -1.0 -0.5 0.0 1.0"

python -m main.vit_concrete_end2end --factor 4 --n-gpus 1 --model lvvit-small --master-port 14431 --auto-resume --p-logits "-1.5 -1.0 -0.5 0.0 1.0" --skip-approx --batch-size 32

PYTHONPATH=./ python -m torch.distributed.launch --nproc_per_node=2 --use_env trainer/deit_trainer_mvit.py --batch-size 48 --model mvit-tiny --output_dir ./saves/mvit-tiny-deit/

PYTHONPATH=./ python -m torch.distributed.launch --master_port 4432 --nproc_per_node=1 --use_env trainer/deit_trainer_mvit.py --batch-size 8 --model mvit-tiny-approx --output_dir ./saves/mvit-tiny-deit-approx/ --warmup-epochs 0 --epochs 30 --lr 1e-3

python -m main.vit_concrete_end2end --factor 4 --n-gpus 1 --model lvvit-small --master-port 14431 --auto-resume --p-logits "-2.0 -0.5 1.0" --skip-approx --batch-size 16


## GLUE Tasks

WIP...
```
main.approx_glue_plot
main.concrete_glue_plot
main.ltp_glue_plot
```