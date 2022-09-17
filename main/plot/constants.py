def scale(lst, s): return list([it * s for it in lst])

GLUE_SUBSETS = "mnli qnli mrpc cola sst2 qqp wnli stsb rte".split()

METRIC_TO_NAME = {
    'acc': 'Accuracy (%)',
    'matthews_correlation': 'Matthews Correlation',
    'pearson': 'Pearson Correlation',
}

METRIC_TO_SCALER = {
    'acc': 100,
    'matthews_correlation': 100,
    'pearson': 1,
}

SUBSET_TO_NAME = {
    "cola": "CoLA",
    "mnli": "MNLI",
    "mrpc": "MRPC",
    "qnli": "QNLI",
    "qqp":  "QQP",
    "rte":  "RTE",
    "sst2": "SST-2",
    "stsb": "STSB",
    "wnli": "WNLI",
}
STR_IMAGENET_1K = 'ImageNet-1k'
STR_TOP1_ACCURACY = 'Top-1 Accuracy (%)'

PLT_STYLE = 'seaborn-bright'

STR_STTABT_APPROX = 'STTABT (Approx. Att.)'
STR_STTABT_ABSATT = 'STTABT (Actual Att.)'
STR_STTABT_APPROX_F4 = 'STTABT@f4 (Approx. Att.)'
STR_STTABT_ABSATT_F4 = 'STTABT@f4 (Actual Att.)'
STR_STTABT_APPROX_F8 = 'STTABT@f8 (Approx. Att.)'
STR_STTABT_ABSATT_F8 = 'STTABT@f8 (Actual Att.)'
STR_STTABT_CONCRETE_WITH_TRAIN = 'STTABT (Concrete, with train)'
STR_STTABT_CONCRETE_WITH_TRAIN_EMA = 'STTABT$_{ema}$ (Concrete, with train)'
STR_STTABT_CONCRETE = 'STTABT (Concrete)'
STR_STTABT_CONCRETE_F4 = 'STTABT@f4 (Concrete)'
STR_STTABT_CONCRETE_F8 = 'STTABT@f8 (Concrete)'
STR_STTABT_CONCRETE_WO_TRAIN = 'STTABT (Concrete, w/o train)'
STR_LTP_BEST_VALID = 'LTP (Best valid.)'
STR_DYNAMIC_VIT = 'DynamicViT'
STR_MANUAL_TOPK = 'Manual Top-k'
STR_BERT_BASE = 'BERT$_{BASE}$'
STR_DEIT_SMALL = 'DeiT$_{small}$'

STR_GFLOPS = 'GFLOPs'
STR_AVERAGE_KEEP_TOKEN_RATIO = 'Average Keep Token Ratio (%)'

COLOR_STTABT_APPROX = 'blue'
COLOR_STTABT_APPROX_F4 = 'blue'
COLOR_STTABT_ABSATT = 'lime'
COLOR_STTABT_ABSATT_F4 = 'lime'
COLOR_STTABT_CONCRETE_WITH_TRAIN = 'red'
COLOR_STTABT_CONCRETE_WO_TRAIN = '#ff9a3b'
COLOR_LTP_BEST_VALID = 'grey'
COLOR_MANUAL_TOPK = 'black'
COLOR_BERT_BASE = 'skyblue'