import torch
from mmcv.cnn import build_norm_layer
from mmcv.runner import force_fp32
from torch import nn

from mmdet3d.ops import DynamicScatter
from ..builder import VOXEL_ENCODERS
from .utils import PFNLayer, get_paddings_indicator
from .utils_graph import GraphConvolutionLayer, get_norm_adjcency


@VOXEL_ENCODERS.register_module()
class LocalEqFeatureExtraction(nn.Module):
    """
        The implementation of Local Equivariance Feature Extraction.
    """

    def __init__(self,
                 in_channels,
                 feat_channels=(64,),
                 with_cluster_center=False,
                 voxel_size=(0.2, 0.2, 4),
                 point_cloud_range=(0, -40, -3, 70.4, 40, 1),
                 max_num_points=20,
                 norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
                 mode='max',
                 legacy=True):
        super(LocalEqFeatureExtraction, self).__init__()
        assert len(feat_channels) > 0
        self.legacy = legacy
        if with_cluster_center:
            in_channels += 3
        self._with_cluster_center = with_cluster_center
        in_channels += max_num_points
        self.fp16_enabled = False
        # Create gnn layers
        self.in_channels = in_channels
        feat_channels = [in_channels] + list(feat_channels)
        gnn_layers = []
        for i in range(len(feat_channels) - 1):
            in_filters = feat_channels[i]
            out_filters = feat_channels[i + 1]
            if i < len(feat_channels) - 2:
                last_layer = False
            else:
                last_layer = True
            gnn_layers.append(
                PFNLayer(
                    in_filters,
                    out_filters,
                    norm_cfg=norm_cfg,
                    last_layer=last_layer,
                    mode=mode))
        self.gnn_layers = nn.ModuleList(gnn_layers)

        # Need pillar (voxel) size and x/y offset in order to calculate offset
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.x_offset = self.vx / 2 + point_cloud_range[0]
        self.y_offset = self.vy / 2 + point_cloud_range[1]
        self.point_cloud_range = point_cloud_range

    @force_fp32(out_fp16=True)
    def forward(self, features, num_points, coors):
        """

        Args:
            features:
            num_points:
            coors: just for keeping the same style.
        Returns:

        """
        point_coords = features[:, :, :3]
        point_features = features[:, :, 3:]

        node_features = [point_coords, point_features]
        # Find distance of x, y, and z from cluster center
        if self._with_cluster_center:
            points_mean = point_coords.sum(
                dim=1, keepdim=True) / num_points.type_as(features).view(
                -1, 1, 1)  # xyz
            f_cluster = point_coords - points_mean
            node_features.append(f_cluster)  # c += 3
        node_features = torch.cat(node_features, dim=-1)

        #
        edge_features = []
        pairwise_distance = torch.cdist(point_coords, point_coords, p=2)  # (N, M, M)
        edge_features.append(pairwise_distance)
        edge_features = torch.cat(edge_features, dim=-1)

        # The feature decorations were calculated without regard to whether
        # pillar was empty. Need to ensure that
        # empty pillars remain set to zeros.
        voxel_count = features.shape[1]
        mask = get_paddings_indicator(num_points, voxel_count, axis=0)  # (N, M)
        node_mask = torch.unsqueeze(mask, -1).type_as(features)  # (N, M, 1)
        edge_mask = torch.unsqueeze(mask, dim=1).type_as(features)  # (N, 1, M)
        node_features *= node_mask  # 将填充部分的点的特征置为0
        edge_features *= edge_mask

        features = torch.cat((node_features, edge_features), dim=-1)

        for gnn in self.gnn_layers:
            features = gnn(features)

        return features.squeeze(1)


# deprecated
@VOXEL_ENCODERS.register_module()
class PillarFeatureNetGCN(nn.Module):
    """Pillar Feature Net with Graph convolution.

    The network prepares the pillar features and performs forward pass
    through GraphConvolutionLayer.

    Args:
        in_channels (int, optional): Number of input features,
            either x, y, z or x, y, z, r. Defaults to 4.
        feat_channels (tuple, optional): Number of features in each of the
            N PFNLayers. Defaults to (64, ).
        max_num_points (int): Max number of points in each pillar. Defaults to 32.
        voxel_size (tuple[float], optional): Size of voxels, only utilize x
            and y size. Defaults to (0.2, 0.2, 4).
        point_cloud_range (tuple[float], optional): Point cloud range, only
            utilizes x and y min. Defaults to (0, -40, -3, 70.4, 40, 1).
        norm_cfg ([type], optional): [description].
            Defaults to dict(type='BN1d', eps=1e-3, momentum=0.01).
        mode (str, optional): The mode to gather point features. Options are
            'max' or 'avg'. Defaults to 'max'.
        legacy (bool): Whether to use the new behavior or
            the original behavior. Defaults to True.
    """

    def __init__(self,
                 in_channels=4,
                 feat_channels=(64,),
                 with_cluster_center=True,
                 voxel_size=(0.2, 0.2, 4),
                 point_cloud_range=(0, -40, -3, 70.4, 40, 1),
                 norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
                 mode='max',
                 legacy=True):
        super(PillarFeatureNetGCN, self).__init__()
        assert len(feat_channels) > 0
        self.legacy = legacy
        if with_cluster_center:
            in_channels += 3
        self._with_cluster_center = with_cluster_center
        self.fp16_enabled = False
        # Create PillarFeatureNet layers
        self.in_channels = in_channels
        feat_channels = [in_channels] + list(feat_channels)
        gcn_layers = []
        for i in range(len(feat_channels) - 1):
            in_filters = feat_channels[i]
            out_filters = feat_channels[i + 1]
            if i < len(feat_channels) - 2:
                last_layer = False
            else:
                last_layer = True
            gcn_layers.append(
                GraphConvolutionLayer(
                    in_filters,
                    out_filters,
                    use_bias=True,
                    last_layer=last_layer))
        self.gcn_layers = nn.ModuleList(gcn_layers)

        # Need pillar (voxel) size and x/y offset in order to calculate offset
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.x_offset = self.vx / 2 + point_cloud_range[0]
        self.y_offset = self.vy / 2 + point_cloud_range[1]
        self.point_cloud_range = point_cloud_range

    @force_fp32(out_fp16=True)
    def forward(self, features, num_points, coors):
        """Forward function.

        Args:
            features (torch.Tensor): Point features or raw points in shape
                (N, M, C).
            num_points (torch.Tensor): Number of points in each pillar.
            coors (torch.Tensor): Coordinates of each voxel.

        Returns:
            torch.Tensor: Features of pillars.
        """
        point_coords = features[:, :, :3]
        point_features = features[:, :, 3:]

        features_ls = [point_coords, point_features]
        # Find distance of x, y, and z from cluster center
        if self._with_cluster_center:
            points_mean = point_coords.sum(
                dim=1, keepdim=True) / num_points.type_as(features).view(
                -1, 1, 1)  # xyz
            f_cluster = point_coords - points_mean
            features_ls.append(f_cluster)  # c += 3

        # Combine together feature decorations
        features = torch.cat(features_ls, dim=-1)
        # The feature decorations were calculated without regard to whether
        # pillar was empty. Need to ensure that
        # empty pillars remain set to zeros.
        voxel_count = features.shape[1]
        mask = get_paddings_indicator(num_points, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(features)
        features *= mask  # (N, M, C')

        norm_adjacency_matrix = get_norm_adjcency(num_points, voxel_count)
        for gcn in self.gcn_layers:
            features = gcn(features, norm_adjacency_matrix)

        return features.squeeze(1)
