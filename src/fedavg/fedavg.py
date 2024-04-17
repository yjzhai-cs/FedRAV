import copy

def FedAvg(w, weight_avg=None):
    """
    Federated averaging.

    Args:
        w: list of client model parameters
    
    return: updated server model parameters
    """
    if weight_avg == None:
        weight_avg = [1/len(w) for i in range(len(w))]
        
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        w_avg[k] = w_avg[k].cuda() * weight_avg[0]
        
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] = w_avg[k].cuda() + w[i][k].cuda() * weight_avg[i]
        #w_avg[k] = torch.div(w_avg[k].cuda(), len(w)) 
    return w_avg

def aggregate(w, weight_avg=None):
    """
    Federated aggregation.

    Args:
        w: list of client model parameters
    
    return: updated server model parameters
    """
    if weight_avg == None:
        weight_avg = [1/len(w) for i in range(len(w))]
        
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        w_avg[k] = w_avg[k].cuda() * weight_avg[0]
        
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] = w_avg[k].cuda() + w[i][k].cuda() * weight_avg[i]
        #w_avg[k] = torch.div(w_avg[k].cuda(), len(w)) 
    return w_avg