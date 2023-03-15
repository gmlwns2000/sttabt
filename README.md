# STTABT: Sparse Token Transformer with Attention Back-Tracking [[Paper]](https://openreview.net/forum?id=VV0hSE8AxCw)

![image](https://user-images.githubusercontent.com/4879345/223654639-82b4c170-6e18-4ee3-9f9a-16f409df0b24.png)

This repository inlcudes official implementations and model weights for [STTABT](https://openreview.net/forum?id=VV0hSE8AxCw).

[[`OpenReview`](https://openreview.net/forum?id=VV0hSE8AxCw)] [[`BibTeX`](#CitingSTTABT)]
 
> **[Sparse Token Transformer with Attention Back-Tracking](https://openreview.net/forum?id=VV0hSE8AxCw)**<br>
> :school::robot:[Heejun Lee](https://github.com/gmlwns2000), :school::alien:[Minki Kang](https://nardien.github.io/), :school::classical_building:[Youngwan Lee](https://youngwanlee.github.io/), :school:[Sung Ju Hwang](http://www.sungjuhwang.com/) <br>
> KAIST:school:, [DeepAuto.ai](http://deepauto.ai/):robot:, AITRICS:alien:, ETRI:classical_building:<br>
> Internation Conference on Learning Representation (ICLR) 2023

# Abstract

Despite the success of Transformers in various applications from text, vision, and speech domains, they are yet to become standard architectures for mobile and edge device applications due to their heavy memory and computational requirements. While there exist many different approaches to reduce the complexities of the Transformers, such as the pruning of the weights/attentions/tokens, quantization, and distillation, we focus on token pruning, which reduces not only the complexity of the attention operations, but also the linear layers, which have non-negligible computational costs. However, previous token pruning approaches often remove tokens during the feed-forward stage without consideration of their impact on later layers' attentions, which has a potential risk of dropping out important tokens for the given task. To tackle this issue, we propose an attention back-tracking method that tracks the importance of each attention in a Transformer architecture from the outputs to the inputs, to preserve the tokens that have a large impact on the final predictions. We experimentally validate the effectiveness of the method on both NLP and CV benchmarks, using Transformer architectures for both domains, and the results show that the proposed attention back-tracking allows the model to better retain the full models' performance even at high sparsity rates, significantly outperforming all baselines. Qualitative analysis of the examples further shows that our method does preserve semantically meaningful tokens.

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

WIP... Please check [`trainer`](https://github.com/gmlwns2000/sttabt/tree/master/trainer) folder.
```
main.approx_glue_plot
main.concrete_glue_plot
main.ltp_glue_plot
```

# <a name="CitingSTTABT"></a>CitingSTTABT

```BibTeX
@inproceedings{
    lee2023sttabt,
    title={Sparse Token Transformer with Attention Back Tracking},
    author={Heejun Lee and Minki Kang and Youngwan Lee and Sung Ju Hwang},
    booktitle={International Conference on Learning Representations},
    year={2023},
    url={https://openreview.net/forum?id=VV0hSE8AxCw}
}
```
