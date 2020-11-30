import torch
import time

def paste_mask_in_image(mask, box, im_h, im_w):
    torch.cuda.current_stream().synchronize()
    fun_st = time.time()
    TO_REMOVE = 1
    w = int(box[2] - box[0] + TO_REMOVE)
    h = int(box[3] - box[1] + TO_REMOVE)
    w = max(w, 1)
    h = max(h, 1)

    # Set shape to [batchxCxHxW]
    mask = mask.expand((1, 1, -1, -1))

    # Resize mask
    # mask = misc_nn_ops.interpolate(mask, size=(h, w), mode='bilinear', align_corners=False)

    mask = mask[0][0]

    im_mask = torch.zeros((im_h, im_w), dtype=mask.dtype, device=mask.device)

    x_0 = max(box[0], 0)
    x_1 = min(box[2] + 1, im_w)
    y_0 = max(box[1], 0)
    y_1 = min(box[3] + 1, im_h)

    im_mask[y_0:y_1, x_0:x_1] = mask[
        (y_0 - box[1]):(y_1 - box[1]), (x_0 - box[0]):(x_1 - box[0])
    ]
    torch.cuda.current_stream().synchronize()
    print("postprocess paste_masks_in_image paste_mask_in_image fun takes: ", time.time() - fun_st)
    return im_mask

device = torch.device('cuda')
# device = torch.device('cpu')
print(device)

mask = torch.rand(30, 30).float().to(device)
box = torch.rand(4).long().to(device).tolist()
im_h = 960
im_w = 540

paste_mask_in_image(mask, box, im_h, im_w)

torch.cuda.current_stream().synchronize()
st = time.time()
for _ in range(16):
    paste_mask_in_image(mask, box, im_h, im_w)
torch.cuda.current_stream().synchronize()
print("takes: ", time.time() - st)
