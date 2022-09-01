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
    for k in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        print(f'DyViT@{k} avg occupy', dyvit_occupy(k))