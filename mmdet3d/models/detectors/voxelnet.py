import torch
from mmcv.runner import force_fp32
from torch.nn import functional as F

from mmdet3d.core import bbox3d2result, merge_aug_bboxes_3d
from mmdet3d.ops import Voxelization
from mmdet.models import DETECTORS
from .. import builder
from .single_stage import SingleStage3DDetector


@DETECTORS.register_module()
class VoxelNet(SingleStage3DDetector):
    r"""`VoxelNet <https://arxiv.org/abs/1711.06396>`_ for 3D detection."""

    def __init__(self,
                 voxel_layer,
                 voxel_encoder,
                 middle_encoder,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(VoxelNet, self).__init__( # 训练时，先调用父类init
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
        )
        self.voxel_layer = Voxelization(**voxel_layer)
        self.voxel_encoder = builder.build_voxel_encoder(voxel_encoder)
        self.middle_encoder = builder.build_middle_encoder(middle_encoder)

    def extract_feat(self, points, img_metas):
        """Extract features from points."""
        # 体素化 points[i].shape: [N,4] 
        # voxels:[V,32,4] V表示总体素个数 num_points：[V,1] 每个体素里面有几个点，后续会填充 
        # coors：voxel在原始场景的坐标 [N, 4] 4:(batch_idx, z, y, x)
        voxels, num_points, coors = self.voxelize(points) 
        # 将点云编码 voxel_features.shape:[V',64] 64表示已经编码
        # [N,M,C] -> [N, 64]
        voxel_features = self.voxel_encoder(voxels, num_points, coors)
        batch_size = coors[-1, 0].item() + 1
        # 中间编码层，根据具体方法实现不同功能。此处为将[N, 64]变为伪图像[N', H, W]
        x = self.middle_encoder(voxel_features, coors, batch_size) 
        x = self.backbone(x) # 使用backbone提取特征
        if self.with_neck:
            x = self.neck(x) # 使用neck融合和增强特征
        return x

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        """Apply hard voxelization to points."""
        voxels, coors, num_points = [], [], []
        for res in points:
            res_voxels, res_coors, res_num_points = self.voxel_layer(res)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return voxels, num_points, coors_batch

    def forward_train(self,
                      points,
                      img_metas,
                      gt_bboxes_3d,
                      gt_labels_3d,
                      gt_bboxes_ignore=None):
        """Training forward function.

        Args:
            points (list[torch.Tensor]): Point cloud of each sample.
            img_metas (list[dict]): Meta information of each sample
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.

        Returns:
            dict: Losses of each branch.
        """
        x = self.extract_feat(points, img_metas) # 提取特征
        outs = self.bbox_head(x) # 调用bbox_head得到tensor预测结果
        loss_inputs = outs + (gt_bboxes_3d, gt_labels_3d, img_metas) # 将预测结果与GT放在一起用于计算loss
        losses = self.bbox_head.loss( # 使用bbox_head计算loss并返回
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses

    def simple_test(self, points, img_metas, imgs=None, rescale=False):
        """Test function without augmentaiton."""
        x = self.extract_feat(points, img_metas) # 提取特征
        outs = self.bbox_head(x) # 使用bbox_head得到输出结果
        bbox_list = self.bbox_head.get_bboxes( # 得到bbox列表
            *outs, img_metas, rescale=rescale)
        bbox_results = [ # 转换得到预测结果
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results

    def aug_test(self, points, img_metas, imgs=None, rescale=False):
        """Test function with augmentaiton."""
        feats = self.extract_feats(points, img_metas)

        # only support aug_test for one sample
        aug_bboxes = []
        for x, img_meta in zip(feats, img_metas):
            outs = self.bbox_head(x)
            bbox_list = self.bbox_head.get_bboxes(
                *outs, img_meta, rescale=rescale)
            bbox_list = [
                dict(boxes_3d=bboxes, scores_3d=scores, labels_3d=labels)
                for bboxes, scores, labels in bbox_list
            ]
            aug_bboxes.append(bbox_list[0])

        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes = merge_aug_bboxes_3d(aug_bboxes, img_metas,
                                            self.bbox_head.test_cfg)

        return [merged_bboxes]
