import numpy as np
def iou(y_, y):
    y_ = y_.cpu().numpy()
    y = y.cpu().numpy()
    y_I = np.logical_and(y>0, y_>0)
    y_I_sum = np.sum(y_I, axis=(1, 2, 3))
    y_U = np.logical_or(y>0, y_>0)
    y_U_sum = np.sum(y_U, axis=(1, 2, 3))
    # print(y_I_sum, y_U_sum)
    nonzero_id = np.where(y_U_sum != 0)[0]
    if nonzero_id.shape[0] == 0:
        acc = 1.0
    else:
        acc = np.average(y_I_sum[nonzero_id] / y_U_sum[nonzero_id])
    return acc