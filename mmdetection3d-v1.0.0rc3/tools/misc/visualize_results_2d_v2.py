# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import mmcv
from mmcv import Config

from mmdet3d.datasets import build_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet3D visualize the results')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('--result', help='results file in pickle format')
    parser.add_argument(
        '--show-dir', help='directory where visualize results will be saved')
    parser.add_argument('-chan', nargs='*', help='sensors to be visualized', default=['CAM_FRONT', 'LIDAR_TOP'])
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    if args.result is not None and \
            not args.result.endswith(('.pkl', '.pickle')):
        raise ValueError('The results file must be a pkl file.')

    cfg = Config.fromfile(args.config)
    cfg.data.test.test_mode = True

    # build the dataset
    dataset = build_dataset(cfg.data.test)
    results = mmcv.load(args.result)

    if getattr(dataset, 'show_2d', None) is not None:
        # data loading pipeline for showing
        eval_pipeline = cfg.get('eval_pipeline', {})
        valid_chan = ('LIDAR_TOP', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT',
                      'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
                      'RADAR_BACK_LEFT', 'RADAR_BACK_RIGHT',
                      'RADAR_FRONT', 'RADAR_FRONT_LEFT', 'RADAR_FRONT_RIGHT')
        # LIDAR_TOP, CAM_BACK, CAM_BACK_LEFT, CAM_BACK_RIGHT, CAM_FRONT, CAM_FRONT_LEFT
        # CAM_FRONT_RIGHT (RADAR_BACK_LEFT, RADAR_BACK_RIGHT, RADAR_FRONT, RADAR_FRONT_LEFT, RADAR_FRONT_RIGHT)
        # 目前不支持radar可视化
        # visualization_chan = ['LIDAR_TOP', 'CAM_FRONT']
        visualization_chan = args.chan
        for chan in visualization_chan:
            if chan not in valid_chan:
                assert KeyError('invalid sensor channel input')
        # visualization_chan = ['CAM_FRONT']
        pred_attributes = ['category']
        if eval_pipeline:
            dataset.show_2d_v2(results, args.show_dir, show_gt=False,
                               pipeline=eval_pipeline, underlay_map=False,
                               visualization_chan=visualization_chan,
                               pred_attributes=pred_attributes)
        else:
            dataset.show_2d_v2(results, args.show_dir, show_gt=False,
                               underlay_map=False,
                               visualization_chan=visualization_chan,
                               pred_attributes=pred_attributes)  # use default pipeline
    else:
        raise NotImplementedError(
            'Show is not implemented for dataset {}!'.format(
                type(dataset).__name__))


if __name__ == '__main__':
    main()

"""
python visualize_results_2d_v2.py ${config_file} --result ${result_file} --show-dir ${show_path}
# "${config_file}" 必需，配置
# "--result ${result_file}" 必需，测试时模型的预测结果，.pkl文件
# "--show-dir ${show_path}" 必需，可视化结果保存路径。就在对应的路径下一个visualization文件夹
# 例子：python visualize_results_2d.py /home/leijiaming/mmdetection3d/configs/centerpoint/centerpoint_02pillar_second_secfpn_4x8_cyclic_20e_nus.py \
            --result /home/leijiaming/mmdetection3d/work_dirs/centerpoint_02pillar_second_secfpn_4x8_cyclic_20e_nus/results_24_epoch.pkl \
            --show-dir /home/leijiaming/mmdetection3d/work_dirs/centerpoint_02pillar_second_secfpn_4x8_cyclic_20e_nus/visualization
"""
