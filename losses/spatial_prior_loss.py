from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt


def _process_single_mask(mask_np):
    if np.max(mask_np) == 0:
        return -np.ones_like(mask_np)

    pos_dist = distance_transform_edt(mask_np)
    neg_dist = distance_transform_edt(1 - mask_np)

    sdf = pos_dist - (neg_dist - 1)
    return np.tanh(sdf / 10.0)


def compute_sdf_multithread(mask_gt):
    mask_np = mask_gt.cpu().detach().numpy()

    if mask_np.ndim == 4:
        channel_idx = 1 if mask_np.shape[1] > 1 else 0
        mask_np = mask_np[:, channel_idx, :, :]

    batch_size = mask_np.shape[0]

    with ThreadPoolExecutor(max_workers=min(batch_size, 8)) as executor:
        results = list(executor.map(_process_single_mask, [mask_np[i] for i in range(batch_size)]))

    sdf_tensor = torch.tensor(np.array(results), dtype=torch.float32).unsqueeze(1)
    return sdf_tensor.to(mask_gt.device)


class SpatialPriorGeometryLoss(nn.Module):
    """Multi-scale SDF geometry supervision for evolved spatial priors."""

    def __init__(self):
        super(SpatialPriorGeometryLoss, self).__init__()
        self.geo_criterion = nn.MSELoss()
        self.weights = [0.2, 0.3, 0.4, 0.5]

    def forward(self, priors_list, target):
        device = target.device
        loss_geo_total = torch.tensor(0.0, device=device)

        with torch.no_grad():
            gt_sdf_full = compute_sdf_multithread(target)
            if gt_sdf_full.device != device:
                gt_sdf_full = gt_sdf_full.to(device)

        for i, pred_prior in enumerate(priors_list):
            if i >= len(self.weights):
                break

            current_size = pred_prior.shape[-2:]

            if gt_sdf_full.shape[-2:] != current_size:
                target_sdf = F.interpolate(
                    gt_sdf_full,
                    size=current_size,
                    mode="bilinear",
                    align_corners=True,
                )
            else:
                target_sdf = gt_sdf_full

            layer_geo_loss = self.geo_criterion(pred_prior, target_sdf)
            loss_geo_total += self.weights[i] * layer_geo_loss

        return loss_geo_total
