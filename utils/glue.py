def get_score(score):
    if 'accuracy' in score:
        return score['accuracy'], "acc"
    first_metric = list(score.keys())[0]
    return score[first_metric], first_metric