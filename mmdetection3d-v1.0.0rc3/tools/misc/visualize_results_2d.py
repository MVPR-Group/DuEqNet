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
        if eval_pipeline:
            dataset.show_2d(results, args.show_dir, pipeline=eval_pipeline, underlay_map=False)
        else:
            dataset.show_2d(results, args.show_dir)  # use default pipeline
    else:
        raise NotImplementedError(
            'Show is not implemented for dataset {}!'.format(
                type(dataset).__name__))


if __name__ == '__main__':
    main()

"""
用于根据测试推理生成的结果进行点云的俯瞰视角可视化，带gt框。TODO，修改为根据传入的传感器通道进行可视化输出
python visualize_results_2d.py ${config_file} --result ${result_file} --show-dir ${show_path}
# "${config_file}" 必需，配置
# "--result ${result_file}" 必需，测试时模型的预测结果，.pkl文件
# "--show-dir ${show_path}" 必需，可视化结果保存路径。就在对应的路径下一个visualization文件夹
# 例子：python visualize_results_2d.py /home/leijiaming/mmdetection3d/configs/centerpoint/centerpoint_02pillar_second_secfpn_4x8_cyclic_20e_nus.py \
            --result /home/leijiaming/mmdetection3d/work_dirs/centerpoint_02pillar_second_secfpn_4x8_cyclic_20e_nus/results_24_epoch.pkl \
            --show-dir /home/leijiaming/mmdetection3d/work_dirs/centerpoint_02pillar_second_secfpn_4x8_cyclic_20e_nus/visualization
"""
