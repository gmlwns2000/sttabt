COMMON_ARGS="--not-wiki --factor 4 --init-checkpoint ./saves/glue-bert-4-wiki-b200.pth"
python -m trainer.glue_base --subset cola $COMMON_ARGS &&\
python -m trainer.glue_base --subset mnli $COMMON_ARGS &&\
python -m trainer.glue_base --subset mrpc $COMMON_ARGS &&\
python -m trainer.glue_base --subset qnli $COMMON_ARGS &&\
python -m trainer.glue_base --subset qqp  $COMMON_ARGS &&\
python -m trainer.glue_base --subset rte  $COMMON_ARGS &&\
python -m trainer.glue_base --subset sst2 $COMMON_ARGS &&\
python -m trainer.glue_base --subset stsb $COMMON_ARGS &&\
python -m trainer.glue_base --subset wnli $COMMON_ARGS