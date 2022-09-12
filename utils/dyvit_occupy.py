"""
== 는 직접 실행한 결과
.. 는 논문에서 제시한 결과
DyViT@0.1 avg occupy 0.25678525641025635
DyViT@0.2 avg occupy 0.2884006410256411
DyViT@0.3 avg occupy 0.327400641025641  == train failed at epoch 25, 64.706%:ema, 65.910%
DyViT@0.4 avg occupy 0.3751698717948718 == 74.854:ema, 74.948
DyViT@0.5 avg occupy 0.4330929487179487 .. 77.5
DyViT@0.6 avg occupy 0.5025544871794873 .. 78.5
DyViT@0.7 avg occupy 0.5849391025641026 .. 79.3
DyViT@0.8 avg occupy 0.6816314102564104 .. 79.6
DyViT@0.9 avg occupy 0.7940160256410257
DeiT-S .. 79.8
"""

def dyvit_occupy(k, num_layers=12, prune_loc = [3, 6, 9]):
    occupy = 1.0
    occupies = []
    for i in range(num_layers):
        if i in prune_loc:
            occupy *= k
        occupies.append(occupy)
    occupies.append(1/192)
    return sum(occupies) / len(occupies)

#init DYVIT_ACCURACY

DYVIT_BASE_RATE_TO_ACC = [
    (0.3, 65.910),
    (0.4, 74.948),
    (0.5, 77.5),
    (0.6, 78.5),
    (0.7, 79.4),
    (0.8, 79.6),
    (0.9, 79.7),
    (1.0, 79.8),
]

DYVIT_RESULTS = []

for base_rate, acc1 in DYVIT_BASE_RATE_TO_ACC:
    DYVIT_RESULTS.append({
        'base_rate': base_rate,
        'accuracy': acc1,
        'occupy': dyvit_occupy(base_rate)
    })

if __name__ == '__main__':
    for k in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        print(f'DyViT@{k} avg occupy', dyvit_occupy(k))
    
    for dic in DYVIT_RESULTS:
        print(f"DyViT@{dic['base_rate']},occupy:{dic['occupy']} = {dic['accuracy']}%")