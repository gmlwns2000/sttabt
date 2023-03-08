# STTABT: Sparse Token Transformer with Attention Back-Tracking [[Paper]](https://openreview.net/forum?id=VV0hSE8AxCw)

![image](https://user-images.githubusercontent.com/4879345/223654639-82b4c170-6e18-4ee3-9f9a-16f409df0b24.png)

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

## GLUE Tasks

WIP...
```
main.approx_glue_plot
main.concrete_glue_plot
main.ltp_glue_plot
```
