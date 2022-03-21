#!/bin/bash
tmux new-session -d -s ainl1 'python -m trainer.glue_base --subset cola --device 0'
tmux new-session -d -s ainl2 'python -m trainer.glue_base --subset mnli --device 1'
tmux new-session -d -s ainl3 'python -m trainer.glue_base --subset mrpc --device 2'
tmux new-session -d -s ainl4 'python -m trainer.glue_base --subset qnli --device 3'
tmux new-session -d -s ainl5 'python -m trainer.glue_base --subset qqp  --device 4'
tmux new-session -d -s ainl6 'python -m trainer.glue_base --subset rte  --device 5'
tmux new-session -d -s ainl7 'python -m trainer.glue_base --subset sst2 --device 6'
tmux new-session -d -s ainl8 'python -m trainer.glue_base --subset stsb --device 7 && python -m trainer.glue_base --subset wnli --device 7'