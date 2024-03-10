import os

import numpy as np
import torch
from tqdm import tqdm

from candidate_optimization import optimize_keypoint_candidates
from feature_backprojection import compute_kp_dists_features, features_from_views, models
from utils import KeypointNetDataset, eval_iou, gen_geo_dists, setup_renderer, sample_view_points

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


def main(filename='experiment.txt', use_texture=False, use_model="dino", offset=0, gaussian_sigma=0.01):
    model = models[use_model](device=device)
    dist_weight = 4
    dist = 1.1
    views = sample_view_points(dist, 5)
    renderer = setup_renderer(device)
    few_shot = 3

    if not os.path.exists('results'):
        os.makedirs('results')
    f = open(f'results/{filename}', 'w')
    for cat_name in ['airplane', 'chair', 'table']:
        f.write(cat_name)
        f.write('\n')

        test_dataset = KeypointNetDataset(filter_classes=[cat_name], use_texture=use_texture)

        geo_dists = {}

        training_samples = test_dataset[offset:few_shot + offset]

        feature_function = lambda mesh, points, geo_dists: features_from_views(renderer, model, mesh,
                                                                               views, points, True,
                                                                               render_dist=dist,
                                                                               batch_size=None if not use_model == "sam" else 8,
                                                                               device=device, geo_dists=geo_dists,
                                                                               gaussian_sigma=gaussian_sigma)

        kp_features, kp_dists = compute_kp_dists_features(training_samples, feature_function)

        pred_all_iou = {
            cat_name: {}
        }
        gt_all = {
            cat_name: {}
        }

        for i in tqdm(range(offset + few_shot, len(test_dataset))):
            data = test_dataset[i]
            mesh, keypoints, class_title, mesh_id, pcd = data
            if mesh_id not in pred_all_iou[cat_name]:
                pred_all_iou[cat_name][mesh_id] = {}
                pred_all_iou[cat_name][mesh_id]["indices"] = []
                pred_all_iou[cat_name][mesh_id]["confidence"] = []
            if mesh_id not in gt_all[cat_name]:
                gt_all[cat_name][mesh_id] = []
            pcd = pcd.points_packed().numpy().astype(np.float32)

            geo_dists[mesh_id] = gen_geo_dists(pcd).astype(np.float32)

            geo_dists_mesh = geo_dists[mesh_id]
            geo_dists_mesh[np.isinf(geo_dists_mesh)] = geo_dists_mesh[~np.isinf(geo_dists_mesh)].max()
            normalized_geo_dists = geo_dists_mesh / np.max(geo_dists_mesh)
            features = feature_function(mesh, [{"xyz": pos} for pos in pcd], normalized_geo_dists)

            predictions_idx, selection_matrix = optimize_keypoint_candidates(kp_dists, normalized_geo_dists,
                                                                             kp_features, features,
                                                                             num_steps=5000, lr=0.1, device=device,
                                                                             dist_alpha=dist_weight,
                                                                             selection_beta=0)

            pred_all_iou[cat_name][mesh_id]["confidence"].extend(selection_matrix[:, :-1].max(axis=0).tolist())
            pred_all_iou[cat_name][mesh_id]["indices"].extend(predictions_idx.tolist())

            for kp in keypoints:
                gt_all[cat_name][mesh_id].append(kp["pcd_info"]["point_index"])

        for i in range(11):
            dist_thresh = 0.01 * i
            iou = eval_iou(pred_all_iou, gt_all, geo_dists, dist_thresh=dist_thresh)
            iou_l = list(iou.values())
            s = ""
            for x in iou_l:
                s += "{}\t".format(x)
            f.write('{}: {}\n'.format(dist_thresh, s))
            print('mIoU-{}: {}'.format(dist_thresh, s))

        f.write('\n')
    f.close()


if __name__ == '__main__':
    main(filename="keypointnet_experiment.txt", gaussian_sigma=0.01)
