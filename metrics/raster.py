import torch
def iou(y_, y, threshold=0.5):
    y_ = (y_ > threshold)
    y = (y > threshold)
    iou = (torch.sum(y_ & y) / torch.sum(y_ | y))
    assert (iou <= 1), f'{torch.sum(y_ & y)} and {torch.sum(y_ | y)}'
    return iou