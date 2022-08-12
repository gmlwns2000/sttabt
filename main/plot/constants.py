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

PLT_STYLE = 'seaborn-bright'

STR_STTABT_APPROX = 'STTABT (Approx. Att.)'
STR_STTABT_ABSATT = 'STTABT (Actual Att.)'
STR_STTABT_CONCRETE_WITH_TRAIN = 'STTBTA (Concrete, with train)'
STR_STTABT_CONCRETE_WO_TRAIN = 'STTBTA (Concrete, w/o train)'
STR_LTP_BEST_VALID = 'LTP (Best valid.)'
STR_MANUAL_TOPK = 'Manual Top-k'
STR_BERT_BASE = 'BERT$_{BASE}$'

STR_GFLOPS = 'GFLOPs'
STR_AVERAGE_KEEP_TOKEN_RATIO = 'Average Keep Token Ratio (%)'

COLOR_STTABT_APPROX = 'blue'
COLOR_STTABT_ABSATT = 'lime'
COLOR_STTABT_CONCRETE_WITH_TRAIN = 'red'
COLOR_STTABT_CONCRETE_WO_TRAIN = '#ff9a3b'
COLOR_LTP_BEST_VALID = 'grey'
COLOR_MANUAL_TOPK = 'black'
COLOR_BERT_BASE = 'skyblue'