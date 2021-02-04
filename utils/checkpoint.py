import torch
def save_model(weights_filename, cp):
    dict = {}
    for k in cp:
        dict[k] = cp[k].state_dict()
    torch.save(dict, weights_filename)

def load_model(path, objs):
    checkpoint = torch.load(path)
    for k in objs:
        objs[k].load_state_dict(checkpoint[k])