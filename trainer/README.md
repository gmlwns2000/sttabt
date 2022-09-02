# Trainers

Our training flow is like this:  [Train classifier or download finetuned network] -> [Train attention approx.] -> [Evalute ABT or Train Concrete masking]. 

**So you should train attention approximation first**. Then you can run further methods.

**If you try to evalute attention backtracking method**, and manual top-k method, then check out `main` scripts.

**If you try to evalute concrete masking method**, you should train concrete masking method first. This training steps requires hyperparameter tunings, so you should checkout `main` scripts. However you can train single concrete masking setting with trainer that exists in this folder.

**How to execute? Where is the results?**
Following trainers may stores checkpoint into `./saves/`, and may stores results into `./saves_plot/`. However following trainer classes are not designed to run with command line, so they are return python objects instead of store checkpoint and results into file system.

## NLP Trainers

NLP trainers are designed to evalute GLUE subsets. You may could train external dataset such as `AG_NEWS` with `classification.py`. However this script is deprecated.

### Attention Approximation Trainer: `glue_base.py`

This script contains `GlueAttentionApproxTrainer`, and various training hyperparameters for `GLUE` attentin approximation.

- Usage
```sh
# from root of repository

# for train bert, you should approx bert before train subsets.
python -m trainer.glue_base --subset bert --factor 4

# for train glue subset
python -m trainer.glue_base --subset mrpc --factor 4
```

### Concrete Masking Trainer: `concrete_trainer.py`

This script conatains `ConcreteTrainer`, and various training hyperparameter for concrete masking.

- Usage
```sh
# from root of repository
python -m trainer.concrete_trainer --subset mrpc --factor 4 --p-logit 0.0
```

## CV Trainers

In CV, we used `ViT` and `DeiT` for baselines. You can train our model with them.

### Classifier Trainer: `vit_trainer.py`

- Usage
```sh
# from root of repository
# 'base' subset means imagenet-1k
export IMAGENET_ROOT="/d1/dataset/ILSVRC2012/"  # you can change the root path.
python -m trainer.vit_trainer --subset base
```

### Attention Approximation Trainer: `vit_approx_trainer.py`

This script train attention approximation for attention backtracking method and concrete masking method. For evalute ABT, you should run `main.vit_approx_plot`. This scripts supports DDP, and this script will use all available GPUs.

- Usage
```sh
# from root of repository
# 'base' subset means imagenet-1k
export IMAGENET_ROOT="/d1/dataset/ILSVRC2012/"  # you can change the root path.

# for train
python -m trainer.vit_approx_trainer --subset base --model deit-small --factor 4

# for evalute
python -m trainer.vit_approx_trainer --eval --subset base --model deit-small --factor 4
```

### Concrete Masking Trainer: `vit_concrete_trainer.py`

This scripts supports DDP, and this script will use all available GPUs. I don't suggest you to use this script for hyperparameter searching, because you have to disable checkpointing (which is evalutation on every epoch). 

- Disable evaluation per training epoch

`trainer.enable_checkpointing = False`

- Usage
```sh
# from root of repository
# 'base' subset means imagenet-1k
export IMAGENET_ROOT="/d1/dataset/ILSVRC2012/"  # you can change the root path.

# for train
python -m trainer.vit_concrete_trainer --subset base --model deit-small --factor 4 --batch-size -1 --epochs 30 --p-logit -0.0 --json-prefix ""

# for train with validation set, the json result will be calculated with subset of train set.
python -m trainer.vit_concrete_trainer --subset base --model deit-small --factor 4 --batch-size -1 --epochs 30 --p-logit -0.0 --json-prefix "" --enable-valid
```