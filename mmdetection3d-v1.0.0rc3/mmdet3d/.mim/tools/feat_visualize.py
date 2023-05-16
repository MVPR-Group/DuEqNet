import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt


def plot_feat(feat, filepath):
    if not isinstance(feat, np.ndarray):
        feat = feat.data.cpu().numpy()

    feat_heatmap = np.maximum(feat, 0)
    feat_heatmap = np.mean(feat_heatmap, axis=0)
    feat_heatmap /= np.max(feat_heatmap)

    feat_heatmap = np.uint8(255 * feat_heatmap)
    feat_heatmap = cv2.applyColorMap(feat_heatmap, cv2.COLORMAP_JET)
    img = cv2.imwrite(filepath, feat_heatmap, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])


if __name__ == '__main__':
    # feat_1_path = '/home/leijiaming/painting/x.pth'
    # feat_1 = torch.load(feat_1_path).squeeze(dim=0).transpose(1, 2)
    #
    # plot_feat(feat_1, '/home/leijiaming/painting/x.png')
    #
    # lift_feat_path = '/home/leijiaming/painting/x_lift.pth'
    # lift_feat = torch.load(lift_feat_path).squeeze(dim=0).transpose(0, 1)
    # for i in range(4):
    #     plot_feat(lift_feat[i].transpose(1, 2), f'/home/leijiaming/painting/x_lift_{i}.png')

    # group_feat_path = '/home/leijiaming/painting/x_lift_group.pth'
    # group_feat = torch.load(group_feat_path).squeeze(dim=0).transpose(0, 1)
    # for i in range(4):
    #     plot_feat(group_feat[i].transpose(1, 2), f'/home/leijiaming/painting/x_lift_group_{i}.png')

    # 每个通道单独绘制
    feat_1_path = '/home/leijiaming/painting/x.pth'
    feat_1 = torch.load(feat_1_path).squeeze(dim=0).transpose(1, 2)
    for i in range(64):
        plot_feat(feat_1[i].unsqueeze(dim=0), f'/home/leijiaming/painting/x_painting/x_c{i}.png')

    x = 1
    y = 2
