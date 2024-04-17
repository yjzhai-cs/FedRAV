import copy

def personalize(id, w, weight_avg=None):
    """
    Federated personalization.

    Args:
        w: list of client model parameters
    
    return: updated server model parameters
    """

    w_id = w[id]

    backup_w = [w[i] for i in range(len(w)) if i != id]
    w = backup_w

    if weight_avg == None:
        weight_avg = [1/len(w) for i in range(len(w))]
        
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        w_avg[k] = w_avg[k].cuda() * weight_avg[0]
        
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] = w_avg[k].cuda() + w[i][k].cuda() * weight_avg[i]
        #w_avg[k] = torch.div(w_avg[k].cuda(), len(w))

    for k in w_avg.keys():
        w_avg[k] = w_id[k].cuda() * 0.5 + w_avg[k].cuda() * 0.5

    return w_avg