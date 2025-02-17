import torch
import torch.nn.functional as F

from pdb import set_trace

def pairwise_loss(p1_out, p2_out, n1_out, n2_out, margin=100):
    pdist = torch.nn.PairwiseDistance(p=2)

    pos_dist = pdist(p1_out, p2_out)
    neg_dist = pdist(n1_out, n2_out)

    pair_loss =  torch.relu(pos_dist - neg_dist + margin).mean()

    return pair_loss


def orthogonal_loss(model_out, pair_matrix):
    model_out = F.normalize(model_out, 2, 1)

    pair_out = torch.mm(model_out, model_out.t())
    pair_loss = (pair_matrix - pair_out).pow(2).mean() 

    return pair_loss
