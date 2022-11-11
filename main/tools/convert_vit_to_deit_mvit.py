import torch

def run(from_path, to_path):
    from_state = torch.load(from_path, map_location='cpu')
    to_state = torch.load(to_path, map_location='cpu')
    
    from_model = {}
    for key in from_state['model'].keys():
        new_key = key
        if key.startswith('module.'):
            new_key = key[7:]
        from_model[new_key] = from_state['model'][key]
    
    for key in from_model:
        assert key in to_state['model']
        assert from_model[key].shape == to_state['model'][key].shape
        to_state['model'][key] = from_model[key]
    
    torch.save(to_state, to_path)
    print('done', from_path, '->', to_path)

if __name__ == '__main__':
    run('./saves/mvit-tiny-in1k.pth','./saves/mvit-tiny-deit/checkpoint.pth')
    run('./saves/mvit-tiny-in1k.pth','./saves/mvit-tiny-deit/best_checkpoint.pth')