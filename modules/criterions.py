import torch.nn.functional as F
import torch


def L1_loss(pred, target, mask):
    masked_pred = pred * mask
    masked_target = target * mask
    l1 = F.l1_loss(masked_pred, masked_target, reduction='sum') / mask.sum()
    return l1

# Disparity smoothness loss

def gradient_x(img):
    # Pad input to keep output size consistent
    img = F.pad(img, (0, 1, 0, 0), mode="replicate")
    gx = img[:, :, :, :-1] - img[:, :, :, 1:]  # NCHW
    return gx

def gradient_y(img):
    # Pad input to keep output size consistent
    img = F.pad(img, (0, 0, 0, 1), mode="replicate")
    gy = img[:, :, :-1, :] - img[:, :, 1:, :]  # NCHW
    return gy

def single_disp_smoothness(disp, img):
    disp_gradients_x = gradient_x(disp)
    disp_gradients_y = gradient_y(disp)

    image_gradients_x = gradient_x(img)
    image_gradients_y = gradient_y(img)

    weights_x = torch.exp(-torch.mean(torch.abs(image_gradients_x), 1, keepdim=True))
    weights_y = torch.exp(-torch.mean(torch.abs(image_gradients_y), 1, keepdim=True))

    smoothness_x = disp_gradients_x * weights_x
    smoothness_y = disp_gradients_y * weights_y
    return (smoothness_x + smoothness_y).mean()