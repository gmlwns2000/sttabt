COMMON_ARGS="--not-wiki"
# python -m trainer.glue_base --subset cola $COMMON_ARGS &&\
# python -m trainer.glue_base --subset mnli $COMMON_ARGS &&\
# python -m trainer.glue_base --subset mrpc $COMMON_ARGS &&\
python -m trainer.glue_base --subset qnli $COMMON_ARGS &&\
python -m trainer.glue_base --subset qqp  $COMMON_ARGS &&\
python -m trainer.glue_base --subset rte  $COMMON_ARGS &&\
python -m trainer.glue_base --subset sst2 $COMMON_ARGS &&\
python -m trainer.glue_base --subset stsb $COMMON_ARGS &&\
python -m trainer.glue_base --subset wnli $COMMON_ARGS