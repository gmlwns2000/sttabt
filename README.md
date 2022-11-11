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

### LVViT concrete samples

End2end.
```sh
python -m main.vit_concrete_end2end --factor 4 --n-gpus 3 --model lvvit-small --master-port 14431 --auto-resume --p-logits "-1.5 -1.0 -0.5 0.0 1.0"
```

Skip approx.
```sh
python -m main.vit_concrete_end2end --factor 4 --n-gpus 1 --model lvvit-small --master-port 14431 --auto-resume --p-logits "-1.5 -1.0 -0.5 0.0 1.0" --skip-approx --batch-size 32
```

### MViT pretrain

Pretrain MViT from scratch. (This process is required for train MViT that compatible to STTABT)

```bash
PYTHONPATH=./ python -m torch.distributed.launch --nproc_per_node=2 --use_env trainer/deit_trainer_mvit.py --batch-size 48 --model mvit-tiny --output_dir ./saves/mvit-tiny-deit/
```

### MViT approx net

Approx net train for MViT.

```bash
PYTHONPATH=./ python -m torch.distributed.launch --master_port 4432 --nproc_per_node=1 --use_env trainer/deit_trainer_mvit.py --batch-size 8 --model mvit-tiny-approx --output_dir ./saves/mvit-tiny-deit-approx/ --warmup-epochs 0 --epochs 30 --lr 1e-3
```

### MViT concrete samples

```bash
python -m main.vit_concrete_end2end --factor 4 --n-gpus 1 --model mvit-tiny --master-port 14431 --auto-resume --p-logits "-2.0 -0.5 1.0" --skip-approx --batch-size 16
```

 - per epochs

mvit 1hr (2 3090) = 2gpuhr

approxnet 5hr (1 TITAN RTX) = 3.75gpuhr

concrete 7hr (1 4090) = 14gpuhr

 - per experiment (on 8 TITAN RTX machine)

mvit 100 epochs = 33.3hr

approxnet 30 epochs = 18.75hr

one concrete 20 epochs = 46.6hr

 - we have...

about 140 hours

we have 92 hours for concrete

which is **two data point for concrete** model

 - gpu to gpuhr

TITAN RTX = 0.75

3090 = 1

4090 = 2

TITAN RTXx8 = 6




## GLUE Tasks

WIP...
```
main.approx_glue_plot
main.concrete_glue_plot
main.ltp_glue_plot
```