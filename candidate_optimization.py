import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm


def optimize_linear(kp_features, cand_features, device='cpu'):
    """
    Optimize the linear assignment problem to find the best matching keypoint candidates for a set of keypoints.
    Cost is only based on features. This is a simple baseline for the optimization problem that uses the Hungarian
    algorithm to solve the linear assignment problem.

    :param kp_features: (K, F)
    :param cand_features: (N, F)
    :param device: device to use for computation
    :return: tuple: (K,), None
    """
    kp_features = torch.tensor(kp_features, dtype=torch.float32, device=device)
    cand_features = torch.tensor(cand_features, dtype=torch.float32, device=device)

    cost_matrix = torch.cdist(cand_features, kp_features, p=2).cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return row_ind, None


def optimize_keypoint_candidates(kp_relative_dists, cand_relative_dists, kp_features, cand_features,
                                 num_steps=2000, lr=0.001, dist_alpha=4., selection_beta=0, device='cpu'):
    """
    Custom optimization module that finds the set of keypoint candidates on an unseen shape that matches the seen
    keypoints on labeled shapes best. A selection matrix is optimized to fit both, features and geodesic distances to
    match the seen keypoints.

    :param kp_relative_dists: (K, K)
    :param cand_relative_dists: (N, N)
    :param kp_features: (K,F)
    :param cand_features: (N,F)
    :param num_steps: number of optimization steps
    :param lr: learning rate
    :param dist_alpha: weight for the distance loss
    :param selection_beta: weight for the selection reward
    :param device: device to use for computation
    :return: tuple: (K,), (N,K+1)
    """

    # This is neither the fastest nor the most accurate method to solve the quadratic assignment problem, but it is
    # sufficient for our purposes. Using proprietary solvers for unbalanced quadratic assignment should be much faster.

    kp_relative_dists = torch.tensor(kp_relative_dists, dtype=torch.float32, device=device)
    cand_relative_dists = torch.tensor(cand_relative_dists, dtype=torch.float32, device=device)

    selection_matrix = torch.randn(cand_features.shape[0], kp_features.shape[0] + 1, dtype=torch.float32, device=device)
    selection_matrix.requires_grad = True
    optimizer = torch.optim.Adam([selection_matrix], lr=lr)

    losses = []
    feature_losses = []
    distance_losses = []
    selection_rewards = []

    for _ in (pbar := tqdm(range(num_steps))):
        optimizer.zero_grad()
        sel_m = torch.softmax(selection_matrix, dim=1)

        feature_loss = torch.norm(torch.matmul(sel_m[:, :-1].t(), cand_features) - kp_features)
        distance_loss = dist_alpha * torch.norm(
            (sel_m[:, :-1].t() @ cand_relative_dists @ sel_m[:, :-1] - kp_relative_dists)[kp_relative_dists > 0])
        selection_reward = selection_beta * torch.sum(
            torch.max(sel_m[:, :-1], dim=0).values - torch.mean(sel_m[:, :-1], dim=0))
        loss = feature_loss + distance_loss - selection_reward
        losses.append(loss.item())
        feature_losses.append(feature_loss.item())
        distance_losses.append(distance_loss.item())
        selection_rewards.append(selection_reward.item())
        pbar.set_description(f'Loss: {loss.item():.3f}, Feature Loss: {feature_loss.item():.3f}, '
                             f'Distance Loss: {distance_loss.item():.3f}, Selection Reward: {selection_reward.item():.3f}')
        loss.backward()
        optimizer.step()

    sel_m = torch.softmax(selection_matrix, dim=1).detach().cpu().numpy()
    return np.argmax(sel_m[:, :-1], axis=0), sel_m
