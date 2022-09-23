# Sparse Token Transformer with Attention Back-Tracking

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

## GLUE Tasks

WIP...
```
main.approx_glue_plot
main.concrete_glue_plot
main.ltp_glue_plot
```