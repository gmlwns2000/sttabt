import torch

def load_state_dict_interpolated(to_model, from_dict, ignores=['p_logit', 'ltp']):
    for name, to_param in to_model.named_parameters():
        if name in from_dict:
            from_data = from_dict[name].clone()
            to_shape = to_param.shape
            from_shape = from_data.shape
            if to_shape != from_shape:
                from_data = from_data.view(1, 1, -1)
                to_dim = 1
                for i in to_shape: to_dim *= i
                from_data = torch.nn.functional.interpolate(from_data, to_dim)
                from_data = from_data.view(*to_shape)
                to_param.data.copy_(from_data)
            else:
                to_param.data.copy_(from_data)
        else:
            ignored = False
            for ig in ignores:
                if ig in name:
                    ignored = True
                    break
            if not ignored: print(name, 'not found')