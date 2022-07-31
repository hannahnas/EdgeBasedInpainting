import torch
import numpy as np


def compute_eval_measures(gt, pred, mask):
    gt = gt[mask.bool()]
    pred = pred[mask.bool()]
    # Sometimes, pred = 0. Thus we need to increase it by a small bit to prevent zero division.
    pred[pred < 1e-6] = 1e-6

    # Compute the distances within 3 thresholds.
    thresh = torch.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).sum() / thresh.numel() #mean()
    a2 = (thresh < 1.25 ** 2).sum() / thresh.numel()
    a3 = (thresh < 1.25 ** 3).sum() / thresh.numel()

    # Compute RMSE.
    rmse = (gt - pred) ** 2
    rmse = torch.sqrt(rmse.mean())

    rmse_log = (torch.log(gt) - torch.log(pred)) ** 2
    rmse_log = torch.sqrt(rmse_log.mean())

    # Compute relative distances.
    abs_rel = torch.mean(torch.abs(gt - pred) / gt)
    sq_rel = torch.mean(((gt - pred) ** 2) / gt)

    # Compute the accuracy of values within one pixel (for CS) or one meter (for KITTI)
    # pixel_acc = torch.mean((torch.abs(gt - pred) <= 1.0).to(torch.float32))

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3



if __name__ == '__main__':
    pred = np.random.rand(4, 1, 128, 128)
    target = np.random.rand(4, 1, 128, 128)

    pred_torch = torch.from_numpy(pred)
    target_torch = torch.from_numpy(target)

    # errors = compute_errors(target, pred)
    torch_errors = compute_eval_measures(target_torch, pred_torch)
    print(torch_errors)
    # print(errors)