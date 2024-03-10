import numpy as np
import torch
from pytorch3d.renderer import look_at_rotation, FoVPerspectiveCameras, PointLights
from tqdm import tqdm

from utils import check_visible_points, check_visible_vertices, VERBOSE


def compute_kp_dists_features(dataset, feature_function):
    """
    :param dataset: List of few-shot samples / small few-shot dataset as given from KeypointNetDataset
    :param feature_function: 'features_from_views' with only remaining arguments: 'mesh', 'keypoints', 'geo_dists'
    :return: Keypoint features, Pointwise normalized distances (np.array): (num_keypoints, emb_dim), (num_keypoints, num_keypoints)
    """

    kp_max_id = max(
        [kp["semantic_id"] for mesh, keypoints, class_title, mesh_id, pcd in dataset[:] for kp in keypoints])

    keypoint_dists = np.zeros((kp_max_id + 1, kp_max_id + 1))
    keypoint_features_dict = {kp_id: [] for kp_id in range(kp_max_id + 1)}

    for mesh, keypoints, class_title, mesh_id, pcd in tqdm(dataset):
        pcd_np = pcd.points_packed().cpu().numpy()

        kp_indices = [kp["semantic_id"] for kp in keypoints]
        kp_pcd_indices = [kp["pcd_info"]["point_index"] for kp in keypoints]
        from utils.evaluation import gen_geo_dists
        dists_between_points = gen_geo_dists(pcd_np).astype(np.float32)
        dists_between_points[np.isinf(dists_between_points)] = np.max(
            dists_between_points[~np.isinf(dists_between_points)])
        dists_normalized = dists_between_points / np.max(dists_between_points)
        kp_dists = dists_normalized[kp_pcd_indices, :][:, kp_pcd_indices]

        keypoint_features = feature_function(mesh, keypoints, kp_dists)

        keypoint_dists[np.ix_(kp_indices, kp_indices)] += kp_dists
        for kp_id, kp_features in zip(kp_indices, keypoint_features):
            keypoint_features_dict[kp_id].append(kp_features)

    empty_kp_ids = [kp_id for kp_id, kp_features in keypoint_features_dict.items() if len(kp_features) == 0]
    keypoint_features = torch.stack(
        [torch.mean(torch.stack(kp_features), dim=0) for kp_features in keypoint_features_dict.values() if
         len(kp_features) > 0])
    keypoint_dists = keypoint_dists[~np.isin(np.arange(kp_max_id + 1), empty_kp_ids)][:,
                     ~np.isin(np.arange(kp_max_id + 1), empty_kp_ids)]

    return keypoint_features, keypoint_dists


def features_from_views(renderer, model, mesh, views, keypoints=None, only_visible=True, render_dist=1.0,
                        batch_size=None, device="cpu", geo_dists=None, gaussian_sigma=0.1):
    """
    Compute the features extracted by 'model' from rendered images rendered with 'renderer (deprecated)' for the keypoints

    :param renderer: pytorch3d MeshRendererWithFragments
    :param model: 2D ViT pipeline (including pre- and post-processing) in: (N, H, W, C), out: (N, num_patches, emb_dim)
    :param mesh: pytorch3d Mesh
    :param views: viewpoints (e.g. from 'views_around_object')
    :param keypoints: list(dict) keypoints with 'xyz' (optional), if not given, features for vertices are returned
    :param only_visible: whether to only return the features if kp is in image
    :param render_dist: distance from which to render the images
    :param batch_size: optional batch size that is used in processing
    :param device: Device
    :param geo_dists: (N, N) matrix of pairwise distances between points / vertices
    :param gaussian_sigma: sigma for the gaussian geodesic re-weighting of the features
    :return: torch.Tensor point features (N, emb_dim)
    """
    mesh = mesh.to(device)
    points = torch.tensor(np.array([kp["xyz"] for kp in keypoints]), dtype=torch.float32).to(
        device) if keypoints is not None else mesh.verts_packed()
    res = renderer.rasterizer.raster_settings.image_size

    if geo_dists is not None:
        reweight = torch.tensor(np.exp(-geo_dists ** 2 / (2 * gaussian_sigma ** 2)), device=device)

    if batch_size is None:
        batch_size = len(views)

    num_views = len(views)

    # camera transform
    R = look_at_rotation(views, device=device)
    T = torch.tensor([0, 0, render_dist], device=device).repeat(len(views), 1)

    ret_array = None  # initialize when we know the embedding size
    point_values_counts = torch.zeros(len(points)).to(device)

    overall_visibility = torch.zeros(len(points))
    while len(views) > 0:
        batch_views = views[:batch_size]
        batch_R = R[:batch_size]
        batch_T = T[:batch_size]

        views = views[batch_size:]
        R = R[batch_size:]
        T = T[batch_size:]

        okay = False
        import time

        while not okay:
            try:
                # 1. Render the mesh
                camera = FoVPerspectiveCameras(R=batch_R, T=batch_T, device=device)
                light = PointLights(ambient_color=((0.5, 0.5, 0.5),), location=batch_views, device=device)

                start = time.time()
                with torch.no_grad():
                    images, fragments = renderer(mesh.extend(len(batch_views)), cameras=camera, lights=light)
                    images = images[..., :3]
                end = time.time()
                if VERBOSE:
                    print(end - start, "for rendering")

                pixel_coords_all_points = camera.transform_points_screen(points,
                                                                         image_size=(res, res)).cpu()  # (V, N, 3)

                # 2. Determine visibility
                start = time.time()
                if only_visible:
                    if keypoints is not None:
                        visible_points = check_visible_points(fragments.pix_to_face, mesh, points)
                    else:
                        visible_points = check_visible_vertices(fragments.pix_to_face, mesh)
                    overall_visibility += visible_points.cpu().sum(dim=0)
                end = time.time()
                if VERBOSE:
                    print(end - start, "for visibility")

                # 3. Extract features
                start = time.time()
                with torch.no_grad():
                    processed_images = model(images)  # (N, num_patches, emb_dim)
                end = time.time()
                if VERBOSE:
                    print(end - start, "for model")
                start = time.time()
                from feature_backprojection.model_wrappers import SAMWrapper
                features_per_view = get_feature_for_pixel_location(
                    processed_images, pixel_coords_all_points, image_size=res,
                    patch_size=int(res / np.sqrt(processed_images.shape[1])),
                    use_sam=isinstance(model, SAMWrapper))  # (V, N, emb_dim)
                end = time.time()
                if VERBOSE:
                    print(end - start, "for getting features for pixel location")

                # 4. Aggregate features
                if ret_array is None:
                    ret_array = torch.zeros(len(points), features_per_view.shape[-1]).to(device)

                start = time.time()
                if only_visible:
                    ret_array += torch.sum(features_per_view * visible_points[..., None], dim=0)
                    point_values_counts += visible_points.sum(dim=0)
                else:
                    ret_array += torch.sum(features_per_view, dim=0)
                    point_values_counts += features_per_view.size(0)  # Increment by the number of batch_views

                if only_visible and geo_dists is not None:
                    value_array = ret_array
                    point_counts = point_values_counts
                    point_values_counts = torch.zeros(len(points)).to(device)
                    ret_array = torch.zeros(len(points), features_per_view.shape[-1]).to(device)
                    for i in range(len(points)):
                        if point_counts[i] <= 0:
                            continue
                        reweighted_features = torch.outer(reweight[i],
                                                          value_array[i])
                        ret_array += reweighted_features
                        point_values_counts += reweight[i] * point_counts[i]
                end = time.time()
                if VERBOSE:
                    print(end - start, "for adding features")

                okay = True

            except AssertionError as e:
                print(e)
                batch_T = batch_T + 0.05

    if only_visible:
        if VERBOSE:
            print(f"Number of views: {num_views}",
                  f"Median visibility of points: {overall_visibility.median().item()}",
                  f"Mean visibility of points: {overall_visibility.mean().item()}")
        if torch.any(overall_visibility == 0):
            print(f"WARNING: {torch.sum(overall_visibility == 0)} points are not visible in any view! ")

    ret_array[point_values_counts > 0] /= point_values_counts[point_values_counts > 0][:, None]
    return ret_array


def get_feature_for_pixel_location(feature_map, pixel_locations, image_size=224, patch_size=14, use_sam=False):
    """
    :param feature_map: (V, (image_size / patch_size) ** 2, emb_dim)
    :param pixel_locations: (V, N, 3) from camera.transform_points_screen
    :param image_size: Size of the rendered image
    :param patch_size: Size of the patches in the feature map
    :param use_sam: Whether to use the SAM model
    :return: (V, N, emb_dim)
    """

    if use_sam:
        image_size = 1024
        patch_size = 16
        pixel_locations = pixel_locations * (image_size / 224)

    def transform_px_to_patch_id(pixel_locations, image_size=224, patch_size=14):
        """
        :param pixel_locations: (V, N, 3) from camera.transform_points_screen
        :param image_size: Size of the rendered image
        :param patch_size: Size of the patches in the feature map
        :return: (V, N) patch_id
        """
        if len(pixel_locations.shape) == 2:
            pixel_locations = pixel_locations.unsqueeze(0)
        assert pixel_locations.max() <= image_size, "Pixel locations must be in [0, image_size], but max is {}".format(
            pixel_locations.max())
        return (pixel_locations[:, :, 1] // patch_size * (image_size / patch_size) + pixel_locations[:, :,
                                                                                     0] // patch_size).long()

    patch_id = transform_px_to_patch_id(pixel_locations, image_size=image_size, patch_size=patch_size)  # (V, N)
    ret = torch.stack([feature_map[i, patch_id[i], :] for i in range(len(feature_map))])
    return ret
