# Copyright (c) OpenMMLab. All rights reserved.
from locale import normalize
import tempfile
import json
from os import path as osp


import mmcv
import numpy as np
import pandas as pd
import pyquaternion
from pyquaternion import Quaternion

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import cv2

from PIL import Image
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.data_classes import Box as NuScenesBox
from nuscenes.utils.geometry_utils import transform_matrix, view_points, BoxVisibility, box_in_image

from ..core import show_result
from ..core.bbox import Box3DMode, Coord3DMode, LiDARInstance3DBoxes
from .builder import DATASETS
from .custom_3d import Custom3DDataset
from .pipelines import Compose
from ..core.visualizer.image_vis import draw_lidar_bbox3d_on_img


@DATASETS.register_module()
class NuScenesDataset(Custom3DDataset):
    r"""NuScenes Dataset.

    This class serves as the API for experiments on the NuScenes Dataset.

    Please refer to `NuScenes Dataset <https://www.nuscenes.org/download>`_
    for data downloading.

    Args:
        ann_file (str): Path of annotation file.
        pipeline (list[dict], optional): Pipeline used for data processing.
            Defaults to None.
        data_root (str): Path of dataset root.
        classes (tuple[str], optional): Classes used in the dataset.
            Defaults to None.
        load_interval (int, optional): Interval of loading the dataset. It is
            used to uniformly sample the dataset. Defaults to 1.
        with_velocity (bool, optional): Whether include velocity prediction
            into the experiments. Defaults to True.
        modality (dict, optional): Modality to specify the sensor data used
            as input. Defaults to None.
        box_type_3d (str, optional): Type of 3D box of this dataset.
            Based on the `box_type_3d`, the dataset will encapsulate the box
            to its original format then converted them to `box_type_3d`.
            Defaults to 'LiDAR' in this dataset. Available options includes.
            - 'LiDAR': Box in LiDAR coordinates.
            - 'Depth': Box in depth coordinates, usually for indoor dataset.
            - 'Camera': Box in camera coordinates.
        filter_empty_gt (bool, optional): Whether to filter empty GT.
            Defaults to True.
        test_mode (bool, optional): Whether the dataset is in test mode.
            Defaults to False.
        eval_version (bool, optional): Configuration version of evaluation.
            Defaults to  'detection_cvpr_2019'.
        use_valid_flag (bool, optional): Whether to use `use_valid_flag` key
            in the info file as mask to filter gt_boxes and gt_names.
            Defaults to False.
    """
    NameMapping = {
        'movable_object.barrier': 'barrier',
        'vehicle.bicycle': 'bicycle',
        'vehicle.bus.bendy': 'bus',
        'vehicle.bus.rigid': 'bus',
        'vehicle.car': 'car',
        'vehicle.construction': 'construction_vehicle',
        'vehicle.motorcycle': 'motorcycle',
        'human.pedestrian.adult': 'pedestrian',
        'human.pedestrian.child': 'pedestrian',
        'human.pedestrian.construction_worker': 'pedestrian',
        'human.pedestrian.police_officer': 'pedestrian',
        'movable_object.trafficcone': 'traffic_cone',
        'vehicle.trailer': 'trailer',
        'vehicle.truck': 'truck'
    }
    DefaultAttribute = {
        'car': 'vehicle.parked',
        'pedestrian': 'pedestrian.moving',
        'trailer': 'vehicle.parked',
        'truck': 'vehicle.parked',
        'bus': 'vehicle.moving',
        'motorcycle': 'cycle.without_rider',
        'construction_vehicle': 'vehicle.parked',
        'bicycle': 'cycle.without_rider',
        'barrier': '',
        'traffic_cone': '',
    }
    AttrMapping = {
        'cycle.with_rider': 0,
        'cycle.without_rider': 1,
        'pedestrian.moving': 2,
        'pedestrian.standing': 3,
        'pedestrian.sitting_lying_down': 4,
        'vehicle.moving': 5,
        'vehicle.parked': 6,
        'vehicle.stopped': 7,
    }
    AttrMapping_rev = [
        'cycle.with_rider',
        'cycle.without_rider',
        'pedestrian.moving',
        'pedestrian.standing',
        'pedestrian.sitting_lying_down',
        'vehicle.moving',
        'vehicle.parked',
        'vehicle.stopped',
    ]
    # https://github.com/nutonomy/nuscenes-devkit/blob/57889ff20678577025326cfc24e57424a829be0a/python-sdk/nuscenes/eval/detection/evaluate.py#L222 # noqa
    ErrNameMapping = {
        'trans_err': 'mATE',
        'scale_err': 'mASE',
        'orient_err': 'mAOE',
        'vel_err': 'mAVE',
        'attr_err': 'mAAE'
    }
    CLASSES = ('car', 'truck', 'trailer', 'bus', 'construction_vehicle',
               'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
               'barrier')

    def __init__(self,
                 ann_file,
                 pipeline=None,
                 data_root=None,
                 classes=None,
                 load_interval=1,
                 with_velocity=True,
                 modality=None,
                 box_type_3d='LiDAR',
                 filter_empty_gt=True,
                 test_mode=False,
                 eval_version='detection_cvpr_2019',
                 use_valid_flag=False):
        self.load_interval = load_interval
        self.use_valid_flag = use_valid_flag
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            classes=classes,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode)

        self.with_velocity = with_velocity
        self.eval_version = eval_version
        from nuscenes.eval.detection.config import config_factory
        self.eval_detection_configs = config_factory(self.eval_version)
        if self.modality is None:
            self.modality = dict(
                use_camera=False,
                use_lidar=True,
                use_radar=False,
                use_map=False,
                use_external=False,
            )

    def get_cat_ids(self, idx):
        """Get category distribution of single scene.

        Args:
            idx (int): Index of the data_info.

        Returns:
            dict[list]: for each category, if the current scene
                contains such boxes, store a list containing idx,
                otherwise, store empty list.
        """
        info = self.data_infos[idx]
        if self.use_valid_flag:
            mask = info['valid_flag']
            gt_names = set(info['gt_names'][mask])
        else:
            gt_names = set(info['gt_names'])

        cat_ids = []
        for name in gt_names:
            if name in self.CLASSES:
                cat_ids.append(self.cat2id[name])
        return cat_ids

    def load_annotations(self, ann_file):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations sorted by timestamps.
        """
        data = mmcv.load(ann_file, file_format='pkl')
        data_infos = list(sorted(data['infos'], key=lambda e: e['timestamp']))
        data_infos = data_infos[::self.load_interval]
        self.metadata = data['metadata']
        self.version = self.metadata['version']
        return data_infos

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]
        # standard protocol modified from SECOND.Pytorch
        input_dict = dict(
            sample_idx=info['token'],
            pts_filename=info['lidar_path'],
            sweeps=info['sweeps'],
            timestamp=info['timestamp'] / 1e6,
        )

        if self.modality['use_camera']:  # todo 20220920
            image_paths = []
            lidar2img_rts = []
            for cam_type, cam_info in info['cams'].items():
                image_paths.append(cam_info['data_path'])
                # obtain lidar to image transformation matrix
                lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
                lidar2cam_t = cam_info[
                    'sensor2lidar_translation'] @ lidar2cam_r.T
                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[3, :3] = -lidar2cam_t
                intrinsic = cam_info['cam_intrinsic']
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = (viewpad @ lidar2cam_rt.T)
                lidar2img_rts.append(lidar2img_rt)

            input_dict.update(
                dict(
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                ))

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos

        return input_dict

    def get_ann_info(self, index):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: Annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`):
                    3D ground truth bboxes
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - gt_names (list[str]): Class names of ground truths.
        """
        info = self.data_infos[index]
        # filter out bbox containing no points
        if self.use_valid_flag:
            mask = info['valid_flag']
        else:
            mask = info['num_lidar_pts'] > 0
        gt_bboxes_3d = info['gt_boxes'][mask]
        gt_names_3d = info['gt_names'][mask]
        gt_labels_3d = []
        for cat in gt_names_3d:
            if cat in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(cat))
            else:
                gt_labels_3d.append(-1)
        gt_labels_3d = np.array(gt_labels_3d)

        if self.with_velocity:
            gt_velocity = info['gt_velocity'][mask]
            nan_mask = np.isnan(gt_velocity[:, 0])
            gt_velocity[nan_mask] = [0.0, 0.0]
            gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)

        # the nuscenes box center is [0.5, 0.5, 0.5], we change it to be
        # the same as KITTI (0.5, 0.5, 0)
        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_names=gt_names_3d)
        return anns_results

    def _format_bbox(self, results, jsonfile_prefix=None):
        """Convert the results to the standard format.

        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str): The prefix of the output jsonfile.
                You can specify the output directory/filename by
                modifying the jsonfile_prefix. Default: None.

        Returns:
            str: Path of the output json file.
        """
        nusc_annos = {}
        mapped_class_names = self.CLASSES

        print('Start to convert detection format...')
        for sample_id, det in enumerate(mmcv.track_iter_progress(results)):
            annos = []
            boxes = output_to_nusc_box(det)  # 将模型预测输出转换为nusc box的格式
            sample_token = self.data_infos[sample_id]['token']
            boxes = lidar_nusc_box_to_global(self.data_infos[sample_id], boxes,
                                             mapped_class_names,
                                             self.eval_detection_configs,
                                             self.eval_version)
            # 上述语句 将模型预测输出框由lidar坐标系转换为global坐标系 （因为模型的输入和输出均是在lidar坐标系）
            # 具体过程：先从lidar坐标系转换到ego坐标系，然后再转换到global坐标系
            for i, box in enumerate(boxes):
                name = mapped_class_names[box.label]
                if np.sqrt(box.velocity[0]**2 + box.velocity[1]**2) > 0.2:
                    if name in [
                            'car',
                            'construction_vehicle',
                            'bus',
                            'truck',
                            'trailer',
                    ]:
                        attr = 'vehicle.moving'
                    elif name in ['bicycle', 'motorcycle']:
                        attr = 'cycle.with_rider'
                    else:
                        attr = NuScenesDataset.DefaultAttribute[name]
                else:
                    if name in ['pedestrian']:
                        attr = 'pedestrian.standing'
                    elif name in ['bus']:
                        attr = 'vehicle.stopped'
                    else:
                        attr = NuScenesDataset.DefaultAttribute[name]

                nusc_anno = dict(
                    sample_token=sample_token,
                    translation=box.center.tolist(),
                    size=box.wlh.tolist(),
                    rotation=box.orientation.elements.tolist(),
                    velocity=box.velocity[:2].tolist(),
                    detection_name=name,
                    detection_score=box.score,
                    attribute_name=attr)  # 将模型输出框转换成nusc anno的格式
                annos.append(nusc_anno)
            nusc_annos[sample_token] = annos
        nusc_submissions = {
            'meta': self.modality,
            'results': nusc_annos,
        }

        mmcv.mkdir_or_exist(jsonfile_prefix)
        res_path = osp.join(jsonfile_prefix, 'results_nusc.json')
        print('Results writes to', res_path)
        mmcv.dump(nusc_submissions, res_path)
        return res_path

    def _evaluate_single(self,
                         result_path,
                         logger=None,
                         metric='bbox',
                         result_name='pts_bbox'):
        """Evaluation for a single model in nuScenes protocol.

        Args:
            result_path (str): Path of the result file.
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Default: None.
            metric (str, optional): Metric name used for evaluation.
                Default: 'bbox'.
            result_name (str, optional): Result name in the metric prefix.
                Default: 'pts_bbox'.

        Returns:
            dict: Dictionary of evaluation details.
        """
        from nuscenes import NuScenes
        from nuscenes.eval.detection.evaluate import NuScenesEval

        output_dir = osp.join(*osp.split(result_path)[:-1])
        nusc = NuScenes(
            version=self.version, dataroot=self.data_root, verbose=True)
        eval_set_map = {
            'v1.0-mini': 'mini_val',
            'v1.0-trainval': 'val',
        }
        nusc_eval = NuScenesEval(
            nusc,
            config=self.eval_detection_configs,
            result_path=result_path,
            eval_set=eval_set_map[self.version],
            output_dir=output_dir,
            verbose=True)
        nusc_eval.main(plot_examples=0,  # 可视化数量，绘制gt和预测框，俯视视角
                       render_curves=False,  # 是否绘制相关评价曲线，包括每一类和总体的
                       )

        # record metrics
        metrics = mmcv.load(osp.join(output_dir, 'metrics_summary.json'))
        detail = dict()
        metric_prefix = f'{result_name}_NuScenes'
        for name in self.CLASSES:
            for k, v in metrics['label_aps'][name].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}_AP_dist_{}'.format(metric_prefix, name, k)] = val
            for k, v in metrics['label_tp_errors'][name].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}_{}'.format(metric_prefix, name, k)] = val
            for k, v in metrics['tp_errors'].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}'.format(metric_prefix,
                                      self.ErrNameMapping[k])] = val

        detail['{}/NDS'.format(metric_prefix)] = metrics['nd_score']
        detail['{}/mAP'.format(metric_prefix)] = metrics['mean_ap']
        return detail

    def format_results(self, results, jsonfile_prefix=None):
        """Format the results to json (standard format for COCO evaluation).

        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: Returns (result_files, tmp_dir), where `result_files` is a
                dict containing the json filepaths, `tmp_dir` is the temporal
                directory created for saving json files when
                `jsonfile_prefix` is not specified.
        """
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
            format(len(results), len(self)))

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
            # tmp_dir = '/home/leijiaming/Code'
            # jsonfile_prefix = osp.join(tmp_dir, 'results')
            # 上述是用于测试的时候指定文件路径
        else:
            tmp_dir = None

        # currently the output prediction results could be in two formats
        # 1. list of dict('boxes_3d': ..., 'scores_3d': ..., 'labels_3d': ...)
        # 2. list of dict('pts_bbox' or 'img_bbox':
        #     dict('boxes_3d': ..., 'scores_3d': ..., 'labels_3d': ...))
        # this is a workaround to enable evaluation of both formats on nuScenes
        # refer to https://github.com/open-mmlab/mmdetection3d/issues/449
        if not ('pts_bbox' in results[0] or 'img_bbox' in results[0]):
            result_files = self._format_bbox(results, jsonfile_prefix)
        else:
            # should take the inner dict out of 'pts_bbox' or 'img_bbox' dict
            result_files = dict()
            for name in results[0]:
                print(f'\nFormating bboxes of {name}')
                results_ = [out[name] for out in results]
                tmp_file_ = osp.join(jsonfile_prefix, name)
                result_files.update(
                    {name: self._format_bbox(results_, tmp_file_)})
        return result_files, tmp_dir

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,  # 模型预测结果转成coco格式后的保存路径
                 result_names=['pts_bbox'],
                 show=False,
                 out_dir=None,
                 pipeline=None):
        """Evaluation in nuScenes protocol.

        Args:
            results (list[dict]): Testing results of the dataset.
            metric (str | list[str], optional): Metrics to be evaluated.
                Default: 'bbox'.
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str, optional): The prefix of json files including
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            show (bool, optional): Whether to visualize.
                Default: False.
            out_dir (str, optional): Path to save the visualization results.
                Default: None.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.

        Returns:
            dict[str, float]: Results of each evaluation metric.
        """
        result_files, tmp_dir = self.format_results(results, jsonfile_prefix)  # 将模型预测结果转成coco格式，准备评价
        """上述，会将coco格式的预测结果保存到jsonfile_prefix/pts_bbox/results_nusc.json中
                如果json_prefix非空的话，则会创建临时路径，即返回的tmp_dir非空
                """
        # result_files = dict(
        #     pts_bbox=osp.join(jsonfile_prefix, 'pts_bbox/results_nusc.json'),
        # )
        # tmp_dir = None
        '''
        if evaluate_each_sample:
            evaluate_each_sample_out_dir = osp.join(jsonfile_prefix, 'pts_bbox/')
        else:
            evaluate_each_sample_out_dir = None
        '''

        if isinstance(result_files, dict):
            results_dict = dict()
            for name in result_names:
                print('Evaluating bboxes of {}'.format(name))
                ret_dict = self._evaluate_single(result_files[name])
            results_dict.update(ret_dict)
        elif isinstance(result_files, str):
            results_dict = self._evaluate_single(result_files)

        if tmp_dir is not None:
            tmp_dir.cleanup()

        if show or out_dir:  # 这个可视化需要图形界面才可以运行
            self.show(results, out_dir, show=show, pipeline=pipeline)
        return results_dict

    def _build_default_pipeline(self):
        """Build the default pipeline for this dataset."""
        pipeline = [
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=5,
                file_client_args=dict(backend='disk')),
            dict(
                type='LoadPointsFromMultiSweeps',
                sweeps_num=10,
                file_client_args=dict(backend='disk')),
            dict(
                type='DefaultFormatBundle3D',
                class_names=self.CLASSES,
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ]
        return Compose(pipeline)

    def show(self, results, out_dir, show=False, pipeline=None):
        """Results visualization.

        Args:
            results (list[dict]): List of bounding boxes results.
            out_dir (str): Output directory of visualization result.
            show (bool): Whether to visualize the results online.
                Default: False.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.
        """
        assert out_dir is not None, 'Expect out_dir, got none.'
        pipeline = self._get_pipeline(pipeline)
        for i, result in enumerate(results):
            if 'pts_bbox' in result.keys():
                result = result['pts_bbox']  # 只取预测的相关信息
                # result=dict('boxes_3d': LiDARInstance3DBoxes, 'scores_3d': Tensor, 'labels_3d': Tensor)
            data_info = self.data_infos[i]  # 第i个sample的信息
            # 里面有token，可视化的时候可以用token
            # sample_token = data_info['token']
            pts_path = data_info['lidar_path']  # lidar points
            file_name = osp.split(pts_path)[-1].split('.')[0]
            points = self._extract_data(i, pipeline, 'points').numpy()  # 取sample中的points
            # for now we convert points into depth mode
            points = Coord3DMode.convert_point(points, Coord3DMode.LIDAR,
                                               Coord3DMode.DEPTH)  # 将点从lidar视角转换到depth视角
            inds = result['scores_3d'] > 0.1  # 只取预测框里得分超过0.1的框
            gt_bboxes = self.get_ann_info(i)['gt_bboxes_3d'].tensor.numpy()
            # ndarray, 只取gt框的信息，9维，xyzwlh+r+v_x+v_y
            show_gt_bboxes = Box3DMode.convert(gt_bboxes, Box3DMode.LIDAR,
                                               Box3DMode.DEPTH)
            pred_bboxes = result['boxes_3d'][inds].tensor.numpy()
            # 取预测框的信息，同样是9维。而且得分要超过0.1。
            show_pred_bboxes = Box3DMode.convert(pred_bboxes, Box3DMode.LIDAR,
                                                 Box3DMode.DEPTH)
            show_result(points, show_gt_bboxes, show_pred_bboxes, out_dir,
                        file_name, show)

    def show_2d(self, results, out_dir, pipeline=None, underlay_map=True,
                box_vis_level: BoxVisibility = BoxVisibility.ANY,
                axes_limit: float = 40,
                verbose=False,
                each_sample_summary=False):
        """Results visualization in 2d image

        Args:
            results (list[dict]): List of bounding boxes results.
            out_dir (str): Output directory of visualization result.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.
            underlay_map (bool): When set to true, lidar data is plotted
                onto the map. This can be slow.
            box_vis_level (BoxVisibility): If sample_data is an image,
                this sets required visibility for boxes.
            axes_limit (float): Axes limit for lidar and radar (measured in meters).
            verbose (bool): Whether to display the image after it is rendered.
            each_sample_summary:

        """
        if each_sample_summary:
            json_file = osp.split(out_dir)[0] + "/pts_bbox/mertrics_summary_each_sample.json"
            with open(json_file, 'r') as f:
                metric_summary_each_sample = json.load(f)
            csv_data = []
        else:
            csv_data = None

        assert out_dir is not None, 'Expect out_dir, got none.'
        pipeline = self._get_pipeline(pipeline)
        nusc = NuScenes(version='v1.0-trainval', dataroot='/dataset/nuscenes', verbose=True)
        nsweeps = 1
        for transform in getattr(pipeline, 'transforms'):
            if transform.__class__.__name__ == 'LoadPointsFromMultiSweeps':
                nsweeps += getattr(transform, 'sweeps_num')
                break

        label_to_name_abb = {
            1: 'car', 2: 'truck', 3: 'cons.', 4: 'bus', 5: 'trail.',
            6: 'barr.', 7: 'motor.', 8: 'bicy.', 9: 'peds.', 10: 'traf.'
        }

        # lidar坐标系下的可视化
        # 模型预测是在lidar坐标系，可视化也是在lidar坐标系，所以不需换转换坐标系
        for i, result in enumerate(mmcv.track_iter_progress(results)):
            if 'pts_bbox' in result.keys():
                result = result['pts_bbox']  # 只取预测的相关信息
                # result=dict('boxes_3d': LiDARInstance3DBoxes, 'scores_3d': Tensor, 'labels_3d': Tensor)
            data_info = self.data_infos[i]  # 第i个sample的信息
            if csv_data:
                metric_summary = metric_summary_each_sample[data_info['token']]
                csv_data.append({'token': data_info['token'],
                                 'orient_err': metric_summary['tp_errors']['orient_err']})
            # 里面有token，可视化的时候可以用token
            sample_record = nusc.get('sample', data_info['token'])
            # copy from nuscenes
            chan = 'LIDAR_TOP'
            lidar_sample_data_token = sample_record['data'][chan]
            lidar_sample_data = nusc.get('sample_data', lidar_sample_data_token)
            ref_chan = 'LIDAR_TOP'
            ref_sample_data_token = sample_record['data'][ref_chan]
            ref_sample_data_record = nusc.get('sample_data', ref_sample_data_token)

            # Get aggregated lidar point cloud in lidar frame.
            pc, times = LidarPointCloud.from_file_multisweep(nusc, sample_record, chan, ref_chan,
                                                             nsweeps=nsweeps)
            velocities = None

            # By default we render the sample_data top down in the sensor frame.
            # This is slightly inaccurate when rendering the map as the sensor frame may not be perfectly upright.
            # Using use_flat_vehicle_coordinates we can render the map in the ego frame instead.

            # Retrieve transformation matrices for reference point cloud.
            cs_record = nusc.get('calibrated_sensor', ref_sample_data_record['calibrated_sensor_token'])
            pose_record = nusc.get('ego_pose', ref_sample_data_record['ego_pose_token'])
            ref_to_ego = transform_matrix(translation=cs_record['translation'],
                                          rotation=Quaternion(cs_record["rotation"]))

            # Compute rotation between 3D vehicle pose and "flat" vehicle pose (parallel to global z plane).
            ego_yaw = Quaternion(pose_record['rotation']).yaw_pitch_roll[0]
            rotation_vehicle_flat_from_vehicle = np.dot(
                Quaternion(scalar=np.cos(ego_yaw / 2), vector=[0, 0, np.sin(ego_yaw / 2)]).rotation_matrix,
                Quaternion(pose_record['rotation']).inverse.rotation_matrix)
            vehicle_flat_from_vehicle = np.eye(4)
            vehicle_flat_from_vehicle[:3, :3] = rotation_vehicle_flat_from_vehicle
            viewpoint = np.dot(vehicle_flat_from_vehicle, ref_to_ego)

            # Init axes.
            _, ax = plt.subplots(1, 1, figsize=(9, 9))

            # Render map if requested.
            if underlay_map:
                nusc.explorer.render_ego_centric_map(sample_data_token=lidar_sample_data_token,
                                                     axes_limit=axes_limit, ax=ax)

            # Show point cloud.
            points = view_points(pc.points[:3, :], viewpoint, normalize=False)  # map 3d points to a 2d plane
            dists = np.sqrt(np.sum(pc.points[:2, :] ** 2, axis=0))
            # colors = np.minimum(1, dists / axes_limit / np.sqrt(2))
            colors = (np.array((31, 36, 33)) / 255.0).reshape(1, -1)  # 象牙黑

            point_scale = 0.2  # 0.2 for lidar
            scatter = ax.scatter(points[0, :], points[1, :], c=colors, s=point_scale)  # 将点绘制在图上

            # Show ego vehicle.
            ax.plot(0, 0, 'x', color='red')

            # Get boxes in lidar frame.
            _, gt_boxes, _ = nusc.get_sample_data(ref_sample_data_token, box_vis_level=box_vis_level,
                                                  use_flat_vehicle_coordinates=True)

            # Show gt_boxes.
            for gt_box in gt_boxes:
                # c = np.array(nusc.explorer.get_color(gt_box.name)) / 255.0
                c = np.array((0, 0, 255)) / 255.0  # blue
                gt_box.render(ax, view=np.eye(4), colors=(c, c, c), linewidth=0.8)
                # gt_box执行了vehicle_flat_from_vehicle旋转，所以这里的view是单位阵

            # Show pred_boxes.
            inds = result['scores_3d'] > 0.5  # 只取预测框里得分超过0.5的框
            # as well as gt boxes and pred boxes, stay in LIDAR mode
            pred_bboxes = result['boxes_3d'][inds]
            pred_bboxes_labels = result['labels_3d'][inds]
            pred_bboxes_scores = result['scores_3d'][inds]
            pred_bboxes_center = pred_bboxes.gravity_center.numpy()
            pred_bboxes_dims = pred_bboxes.dims.numpy()[:, [1, 0, 2]]
            pred_bboxes_yaw = pred_bboxes.yaw.numpy()
            # pred_bboxes_yaw = -pred_bboxes_yaw - np.pi / 2

            pred_bbox_list = []
            for i in range(len(pred_bboxes)):
                quat = pyquaternion.Quaternion(axis=[0, 0, 1], radians=pred_bboxes_yaw[i])
                velocity = (*pred_bboxes.tensor[i, 7:9], 0.0)
                pred_bbox = NuScenesBox(
                    pred_bboxes_center[i],  # xyz center
                    pred_bboxes_dims[i],  # size
                    quat,
                    label=pred_bboxes_labels[i],
                    score=pred_bboxes_scores[i],
                    velocity=velocity)
                pred_bbox_list.append(pred_bbox)  # lidar坐标系

            for pred_bbox in pred_bbox_list:
                color = np.array((0, 255, 0)) / 255.0  # green
                pred_bbox.render(ax, view=viewpoint, colors=(color, color, color), linewidth=0.8)
                # 因为pred_bbox是lidar坐标系下，所以需要转到vehicle_flat_from_vehicle
                # 因此这里的view是vehicle_flat_from_vehicle的viewpoint

            # Limit visible range.
            ax.set_xlim(-axes_limit, axes_limit)
            ax.set_ylim(-axes_limit, axes_limit)

            ax.axis('off')
            # ax.set_title('{} {labels_type}'.format(
            #     sd_record['channel'], labels_type='(predictions)' if lidarseg_preds_bin_path else ''))
            ax.set_aspect('equal')

            # Save the figure.
            pts_path = data_info['lidar_path']  # lidar points
            file_name = osp.split(pts_path)[-1].split('.')[0]
            out_dir_new = out_dir + "_LIDAR_TOP"
            mmcv.mkdir_or_exist(out_dir_new)
            if out_dir is not None:
                plt.savefig(osp.join(out_dir_new, f'{file_name}_LIDAR_TOP.png'),
                            bbox_inches='tight', pad_inches=0, dpi=200)

            plt.close()

            if verbose:
                plt.show()

        if csv_data:
            pdd = pd.DataFrame(data=csv_data, columns=csv_data[0].keys())
            pdd.to_csv(osp.join(osp.split(out_dir)[0], 'pts_bbox/aoe_sample.csv'))



    # 2023.2.8 编写show_2d 2.0版本，添加要可视化的传感器参数以及可视化时要添加的属性
    # TODO gt框和预测框的颜色是个问题，目前预测框会根据预测类别画不同的颜色，那gt怎么办呢？
    def show_2d_v2(self, results, out_dir, show_gt=False,
                   pipeline=None, underlay_map=True,
                   box_vis_level: BoxVisibility = BoxVisibility.ANY,
                   axes_limit: float = 40,
                   verbose=False,
                   visualization_chan=['LIDAR_TOP'],
                   pred_attributes=None) -> None:
        """
        
        :param results:
        :param out_dir:
        :param show_gt:
        :param pipeline:
        :param underlay_map:
        :param box_vis_level:
        :param axes_limit:
        :param verbose:
        :param visualization_chan:
        :param pred_attributes:
        """
        assert out_dir is not None, 'Expect out_dir, got none.'
        pipeline = self._get_pipeline(pipeline)
        nusc = NuScenes(version='v1.0-trainval', dataroot='/dataset/nuscenes', verbose=True)
        nsweeps = 1
        for transform in getattr(pipeline, 'transforms'):
            if transform.__class__.__name__ == 'LoadPointsFromMultiSweeps':
                nsweeps += getattr(transform, 'sweeps_num')
                break

        label_to_name = ('vehicle.car', 'vehicle.truck', 'vehicle.construction',
                         'vehicle.bus.rigid', 'vehicle.trailer', 'movable_object.barrier',
                         'vehicle.motorcycle', 'vehicle.bicycle', 'human.pedestrian.adult',
                         'movable_object.trafficcone')
        label_to_name_abb = {
            0: 'car', 1: 'truck', 2: 'cons.', 3: 'bus', 4: 'trail.',
            5: 'barr.', 6: 'motor.', 7: 'bicy.', 8: 'peds.', 9: 'traf.'
        }

        # 对每个sample进行可视化
        for i, result in enumerate(mmcv.track_iter_progress(results)):
            if 'pts_bbox' in result.keys():
                result = result['pts_bbox']  # 只取预测的相关信息
                # result=dict('boxes_3d': LiDARInstance3DBoxes, 'scores_3d': Tensor, 'labels_3d': Tensor)
            data_info = self.data_infos[i]  # 第i个sample的信息

            # 里面有token，可视化的时候可以用token
            sample_record = nusc.get('sample', data_info['token'])

            # get prediction boxes
            inds = result['scores_3d'] > 0.5  # 只取预测框里得分超过0.5的框
            # as well as gt boxes and pred boxes, stay in LIDAR mode
            pred_bboxes = result['boxes_3d'][inds]
            pred_bboxes_labels = result['labels_3d'][inds]
            pred_bboxes_scores = result['scores_3d'][inds]
            pred_bboxes_center = pred_bboxes.gravity_center.numpy()
            pred_bboxes_dims = pred_bboxes.dims.numpy()
            pred_bboxes_yaw = pred_bboxes.yaw.numpy()
            # 2023.2.8 修复因为新版本的 mmdet3d 重新定义坐标系以造成可视化错误的问题
            pred_bboxes_dims = pred_bboxes_dims[:, [1, 0, 2]]

            pred_bbox_list = []
            for i in range(len(pred_bboxes)):
                quat = pyquaternion.Quaternion(axis=[0, 0, 1], radians=pred_bboxes_yaw[i])
                velocity = (*pred_bboxes.tensor[i, 7:9], 0.0)
                pred_bbox = NuScenesBox(
                    pred_bboxes_center[i],  # xyz center
                    pred_bboxes_dims[i],  # size
                    quat,
                    label=pred_bboxes_labels[i],
                    score=pred_bboxes_scores[i],
                    name=label_to_name[pred_bboxes_labels[i]],
                    velocity=velocity)
                pred_bbox_list.append(pred_bbox)  # lidar坐标系

            for chan in visualization_chan:
                # 1. LIDAR可视化
                if chan == 'LIDAR_TOP':
                    # copy from nuscenes
                    lidar_sample_data_token = sample_record['data'][chan]
                    lidar_sample_data = nusc.get('sample_data', lidar_sample_data_token)
                    ref_chan = 'LIDAR_TOP'
                    ref_sample_data_token = sample_record['data'][ref_chan]
                    ref_sample_data_record = nusc.get('sample_data', ref_sample_data_token)

                    # Get aggregated lidar point cloud in lidar frame.
                    pc, times = LidarPointCloud.from_file_multisweep(nusc, sample_record, chan, ref_chan,
                                                                    nsweeps=nsweeps)
                    velocities = None

                    # By default we render the sample_data top down in the sensor frame.
                    # This is slightly inaccurate when rendering the map as the sensor frame may not be perfectly upright.
                    # Using use_flat_vehicle_coordinates we can render the map in the ego frame instead.

                    # Retrieve transformation matrices for reference point cloud.
                    cs_record = nusc.get('calibrated_sensor', ref_sample_data_record['calibrated_sensor_token'])
                    pose_record = nusc.get('ego_pose', ref_sample_data_record['ego_pose_token'])
                    ref_to_ego = transform_matrix(translation=cs_record['translation'],
                                                rotation=Quaternion(cs_record["rotation"]))

                    # Compute rotation between 3D vehicle pose and "flat" vehicle pose (parallel to global z plane).
                    ego_yaw = Quaternion(pose_record['rotation']).yaw_pitch_roll[0]
                    rotation_vehicle_flat_from_vehicle = np.dot(
                        Quaternion(scalar=np.cos(ego_yaw / 2), vector=[0, 0, np.sin(ego_yaw / 2)]).rotation_matrix,
                        Quaternion(pose_record['rotation']).inverse.rotation_matrix)
                    vehicle_flat_from_vehicle = np.eye(4)
                    vehicle_flat_from_vehicle[:3, :3] = rotation_vehicle_flat_from_vehicle
                    viewpoint = np.dot(vehicle_flat_from_vehicle, ref_to_ego)

                    # Init axes.
                    _, ax = plt.subplots(1, 1, figsize=(9, 9))

                    # Render map if requested.
                    if underlay_map:
                        nusc.explorer.render_ego_centric_map(sample_data_token=lidar_sample_data_token,
                                                            axes_limit=axes_limit, ax=ax)

                    # Show point cloud.
                    points = view_points(pc.points[:3, :], viewpoint, normalize=False)  # map 3d points to a 2d plane
                    dists = np.sqrt(np.sum(pc.points[:2, :] ** 2, axis=0))
                    # colors = np.minimum(1, dists / axes_limit / np.sqrt(2))
                    colors = (np.array((31, 36, 33)) / 255.0).reshape(1, -1)  # 象牙黑

                    point_scale = 0.15  # 0.2 for lidar
                    scatter = ax.scatter(points[0, :], points[1, :], c=colors, s=point_scale)  # 将点绘制在图上

                    # Show ego vehicle.
                    ax.plot(0, 0, 'x', color='red')

                    if show_gt:  # gt框的可视化
                        # Get boxes in lidar frame.
                        _, gt_boxes, _ = nusc.get_sample_data(ref_sample_data_token, box_vis_level=box_vis_level,
                                                              use_flat_vehicle_coordinates=True)

                        # Show gt_boxes.
                        for gt_box in gt_boxes:
                            # c = np.array(nusc.explorer.get_color(gt_box.name)) / 255.0
                            c = np.array((0, 0, 255)) / 255.0  # blue
                            gt_box.render(ax, view=np.eye(4), colors=(c, c, c), linewidth=0.8)
                            # gt_box执行了vehicle_flat_from_vehicle旋转，所以这里的view是单位阵

                    # Show pred_boxes.
                    for pred_bbox in pred_bbox_list:
                        bbox = pred_bbox.copy()  #
                        color = np.array((nusc.explorer.get_color(bbox.name))) / 255.0  # color depends on box'name
                        # color = np.array((0, 255, 0)) / 255.0  # green
                        bbox.render(ax, view=viewpoint, colors=(color, color, color), linewidth=0.8)
                        # 因为pred_bbox是lidar坐标系下，所以需要转到vehicle_flat_from_vehicle
                        # 因此这里的view是vehicle_flat_from_vehicle的viewpoint

                    # Limit visible range.
                    ax.set_xlim(-axes_limit, axes_limit)
                    ax.set_ylim(-axes_limit, axes_limit)

                    ax.axis('off')
                    # ax.set_title('{} {labels_type}'.format(
                    #     sd_record['channel'], labels_type='(predictions)' if lidarseg_preds_bin_path else ''))
                    ax.set_aspect('equal')

                    # Save the figure.
                    pts_path = data_info['lidar_path']  # lidar points
                    file_name = osp.split(pts_path)[-1].split('.')[0]
                    out_dir_new = out_dir + "_" + chan + '_v2'
                    mmcv.mkdir_or_exist(out_dir_new)
                    if out_dir_new is not None:
                        plt.savefig(osp.join(out_dir_new, f'{file_name}_{chan}.png'),
                                    bbox_inches='tight', pad_inches=0, dpi=200)

                    plt.close()

                    if verbose:
                        plt.show()
                elif 'RADAR' in chan:
                    pass
                elif 'CAM' in chan:
                    cam_sample_data_token = sample_record['data'][chan]
                    cam_sample_data = nusc.get('sample_data', cam_sample_data_token)
                    cam_data_path, gt_boxes, cam_intrinsic = nusc.get_sample_data(cam_sample_data_token)

                    imsize = (cam_sample_data['width'], cam_sample_data['height'])

                    lidar_sensor_token = sample_record['data']['LIDAR_TOP']
                    lidar_sensor = nusc.get('sample_data', lidar_sensor_token)

                    data = Image.open(cam_data_path)

                    # First step: transform the pointcloud to the ego vehicle frame for the timestamp of the sweep.
                    cs_record_first = nusc.get('calibrated_sensor', lidar_sensor['calibrated_sensor_token'])
                    # Second step: transform from ego to the global frame.
                    pose_record_second = nusc.get('ego_pose', lidar_sensor['ego_pose_token'])
                    # Third step: transform from global into the ego vehicle frame for the timestamp of the image.
                    pose_record_third = nusc.get('ego_pose', cam_sample_data['ego_pose_token'])
                    # Fourth step: transform from ego into the camera.
                    cs_record_fourth = nusc.get('calibrated_sensor', cam_sample_data['calibrated_sensor_token'])

                    # 坐标系转换
                    pred_bbox_list_list = []
                    for pred_bbox in pred_bbox_list:
                        bbox = pred_bbox.copy()

                        # Points live in the point sensor frame. So they need to be transformed via global to the image plane.
                        # First step: transform the pointcloud to the ego vehicle frame for the timestamp of the sweep.
                        bbox.rotate(Quaternion(cs_record_first['rotation']))
                        bbox.translate(np.array(cs_record_first['translation']))

                        # Second step: transform from ego to the global frame.
                        bbox.rotate(Quaternion(pose_record_second['rotation']))
                        bbox.translate(np.array(pose_record_second['translation']))

                        # Third step: transform from global into the ego vehicle frame for the timestamp of the image.
                        bbox.translate(-np.array(pose_record_third['translation']))
                        bbox.rotate(Quaternion(pose_record_third['rotation']).inverse)

                        # Fourth step: transform from ego into the camera.
                        bbox.translate(-np.array(cs_record_fourth['translation']))
                        bbox.rotate(Quaternion(cs_record_fourth['rotation']).inverse)

                        if box_in_image(bbox, cam_intrinsic, imsize, vis_level=box_vis_level):
                            pred_bbox_list_list.append(bbox)

                    _, ax1 = plt.subplots(1, 1, figsize=(9, 16))

                    # show image
                    ax1.imshow(data)

                    # show pred boxes
                    for pred_bbox in pred_bbox_list_list:
                        color = np.array((nusc.explorer.get_color(pred_bbox.name))) / 255.0  # color depends on box'name
                        # color = np.array((0, 255, 0)) / 255.0  # green
                        pred_bbox.render(ax1, view=np.array(cs_record_fourth['camera_intrinsic']),
                                         colors=(color, color, color), linewidth=0.8, normalize=True)
                        # 因为pred_bbox是lidar坐标系下，所以需要转到vehicle_flat_from_vehicle
                        # 因此这里的view是vehicle_flat_from_vehicle的viewpoint
                        corners = view_points(pred_bbox.corners(), np.array(cs_record_fourth['camera_intrinsic']),
                                              normalize=True)
                        corner_1 = corners[:, 0]
                        corner_2 = corners[:, 1]
                        start_x = (corner_1[0] + corner_2[0]) / 2
                        start_y = min(corner_1[1], corner_2[1]) - 5

                        # 标注预测目标的属性
                        if pred_attributes is not None:
                            assert isinstance(pred_attributes, list)
                            attribute_count = 0
                            for attribute in pred_attributes:
                                attribute_count += 1
                                pos_x = start_x
                                pos_y = start_y - (attribute_count-1) * 10
                                # 2023.2.10 对标注的坐标进行范围限制 -> 修复属性标注会超出图像范围的问题
                                if 25 >= pos_x or pos_x >= data.size[0] - 25 or 25 >= pos_y or pos_y >= data.size[1] - 25:
                                    continue
                                if attribute == 'category':
                                    plt.text(pos_x, pos_y, s=label_to_name_abb[pred_bbox.label],
                                             size=7.0, ha='center', va='center',
                                             bbox=dict(boxstyle='round',
                                                       edgecolor=color,
                                                       facecolor=color
                                                       ),
                                             fontsize=6.0, color='white')
                                elif attribute == 'velocity':
                                    # 速度是3个方向的数值，暂时跳过
                                    pass
                                    # plt.text(pos_x, pos_y, s='{:.2f}'.format(pred_bbox.velocity), fontsize=10.0, color=color)
                                elif attribute == 'orientation':
                                    pass
                                else:
                                    attribute_count -= 1
                                    raise ValueError('invalid attribute')

                    if show_gt:
                        for gt_box in gt_boxes:
                            color = np.array(nusc.explorer.get_color(gt_box.name)) / 255.0
                            gt_box.render(ax1, view=cam_intrinsic, normalize=True, colors=(color, color, color))

                    # Limit visible range.
                    ax1.set_xlim(0, data.size[0])
                    ax1.set_ylim(data.size[1], 0)

                    ax1.axis('off')
                    # ax.set_title('{} {labels_type}'.format(
                    #     sd_record['channel'], labels_type='(predictions)' if lidarseg_preds_bin_path else ''))
                    ax1.set_aspect('equal')

                    # Save the figure.
                    file_name = osp.split(cam_data_path)[-1].split('.')[0]
                    out_dir_new = out_dir + "_" + chan + '_v2'
                    mmcv.mkdir_or_exist(out_dir_new)
                    if out_dir_new is not None:
                        plt.savefig(osp.join(out_dir_new, f'{file_name}_{chan}.png'),
                                    bbox_inches='tight', pad_inches=0, dpi=200)

                    plt.close()

                    if verbose:
                        plt.show()


def output_to_nusc_box(detection):
    """Convert the output to the box class in the nuScenes.

    Args:
        detection (dict): Detection results.

            - boxes_3d (:obj:`BaseInstance3DBoxes`): Detection bbox.
            - scores_3d (torch.Tensor): Detection scores.
            - labels_3d (torch.Tensor): Predicted box labels.

    Returns:
        list[:obj:`NuScenesBox`]: List of standard NuScenesBoxes.
    """
    box3d = detection['boxes_3d']
    scores = detection['scores_3d'].numpy()
    labels = detection['labels_3d'].numpy()

    box_gravity_center = box3d.gravity_center.numpy()
    box_dims = box3d.dims.numpy()
    box_yaw = box3d.yaw.numpy()

    # our LiDAR coordinate system -> nuScenes box coordinate system
    nus_box_dims = box_dims[:, [1, 0, 2]]

    box_list = []
    for i in range(len(box3d)):
        quat = pyquaternion.Quaternion(axis=[0, 0, 1], radians=box_yaw[i])
        velocity = (*box3d.tensor[i, 7:9], 0.0)
        # velo_val = np.linalg.norm(box3d[i, 7:9])
        # velo_ori = box3d[i, 6]
        # velocity = (
        # velo_val * np.cos(velo_ori), velo_val * np.sin(velo_ori), 0.0)
        box = NuScenesBox(
            box_gravity_center[i],
            nus_box_dims[i],
            quat,
            label=labels[i],
            score=scores[i],
            velocity=velocity)
        box_list.append(box)
    return box_list


def lidar_nusc_box_to_global(info,
                             boxes,
                             classes,
                             eval_configs,
                             eval_version='detection_cvpr_2019'):
    """Convert the box from ego to global coordinate.

    Args:
        info (dict): Info for a specific sample data, including the
            calibration information.
        boxes (list[:obj:`NuScenesBox`]): List of predicted NuScenesBoxes.
        classes (list[str]): Mapped classes in the evaluation.
        eval_configs (object): Evaluation configuration object.
        eval_version (str, optional): Evaluation version.
            Default: 'detection_cvpr_2019'

    Returns:
        list: List of standard NuScenesBoxes in the global
            coordinate.
    """
    box_list = []
    for box in boxes:
        # Move box to ego vehicle coord system
        box.rotate(pyquaternion.Quaternion(info['lidar2ego_rotation']))
        box.translate(np.array(info['lidar2ego_translation']))
        # filter det in ego.
        cls_range_map = eval_configs.class_range
        radius = np.linalg.norm(box.center[:2], 2)
        det_range = cls_range_map[classes[box.label]]
        if radius > det_range:
            continue
        # Move box to global coord system
        box.rotate(pyquaternion.Quaternion(info['ego2global_rotation']))
        box.translate(np.array(info['ego2global_translation']))
        box_list.append(box)
    return box_list
