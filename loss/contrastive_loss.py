import torch.nn as nn
import torch
import math

class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature):
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.sim = nn.CosineSimilarity(dim=1)

    def compute_exp_sim(self, features_anchor, features_):
        """
        Compute sum(sim(anchor, pos)) or sum(sim(anchor, neg))
        """
        sim = self.sim(features_anchor, features_)
        exp_sim = torch.exp(torch.div(sim, self.temperature))
        return exp_sim

    def compute_exp_sim_with_rank(self, features_anchor, features_, rank1, rank2, max_rank_gap, epsilon=1e-6):
        """
        # RaCCon: A Rank-Consistent Contrastive Learning Framework for Mitigating Spurious Correlations
        # Written by Aiyang Han, PARNEC
        # https://github.com/hannaiiyanggit/RaCCon
        Compute sum(sim(anchor, pos)) with angle compensation of ranks
        rank1: image rank
        rank2: pos rank
        """
        sim = self.sim(features_anchor, features_)

        weight = torch.sin(math.pi * (rank1+rank2)/(2*max_rank_gap) - math.pi / 2)

        exp_sim = torch.exp(torch.div(sim, self.temperature))
        return exp_sim * (weight * 0.5 + 1)

    def forward(self, feature_ref, features_pos,  features_neg, ranks=None):
        # Compute negative similarities
        exp_neg = self.compute_exp_sim(feature_ref, features_neg)
        sum_exp_neg = exp_neg.sum(0, keepdim=True)

        if ranks is not None:
            image_rank, pos_rank, max_rank, min_rank = ranks
            exp_pos = self.compute_exp_sim_with_rank(feature_ref, features_pos, image_rank, pos_rank, max_rank - min_rank)

        else:
            exp_pos = self.compute_exp_sim(feature_ref, features_pos)

        log_probs = (torch.log(exp_pos) -
                     torch.log(sum_exp_neg + exp_pos.sum(0, keepdim=True)))
        loss = -1 * log_probs
        del exp_pos
        del exp_neg
        del log_probs
        return loss.mean()