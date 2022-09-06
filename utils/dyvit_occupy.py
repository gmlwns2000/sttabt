"""
DyViT@0.1 avg occupy 0.25678525641025635
DyViT@0.2 avg occupy 0.2884006410256411
DyViT@0.3 avg occupy 0.327400641025641
DyViT@0.4 avg occupy 0.3751698717948718
DyViT@0.5 avg occupy 0.4330929487179487
DyViT@0.6 avg occupy 0.5025544871794873
DyViT@0.7 avg occupy 0.5849391025641026
DyViT@0.8 avg occupy 0.6816314102564104
DyViT@0.9 avg occupy 0.7940160256410257
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

if __name__ == '__main__':
    for k in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        print(f'DyViT@{k} avg occupy', dyvit_occupy(k))