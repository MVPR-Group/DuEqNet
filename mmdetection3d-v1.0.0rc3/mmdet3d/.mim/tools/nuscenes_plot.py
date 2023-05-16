from nuscenes.nuscenes import NuScenes


def nuscenes_scene_plot_lidartop(nusc,
                                 scene_idx: int,
                                 out_path: str,
                                 with_anns: bool = True,
                                 nsweeps: int = 1,
                                 underlay_map: bool = True,
                                 verbose: bool = False) -> None:
    my_scene = nusc.scene[scene_idx]
    last_sample_token = my_scene['last_sample_token']
    last_sample = nusc.get('sample', last_sample_token)
    sample_data_token = last_sample['data']['LIDAR_TOP']
    nusc.render_sample_data(sample_data_token, with_anns=with_anns, nsweeps=nsweeps, verbose=verbose, out_path=out_path,
                            underlay_map=underlay_map)


if __name__ == '__main__':
    nusc = NuScenes(version='v1.0-trainval', dataroot='/dataset/nuscenes', verbose=True)
    # scene_idx = 522
    # nuscenes_scene_plot_lidartop(nusc, scene_idx, nsweeps=1,
    #                              out_path=f'/home/leijiaming/painting/lidar-1sweep-{scene_idx}-notitle.png')
    # nuscenes_scene_plot_lidartop(nusc, scene_idx, nsweeps=10,
    #                              out_path=f'/home/leijiaming/painting/lidar-10sweep-{scene_idx}-notitle.png')
    scene_idx = 400
    # nuscenes_scene_plot_lidartop(nusc, scene_idx, nsweeps=1,
    #                              out_path=f'/home/leijiaming/painting/lidar-1sweep-{scene_idx}-notitle.png')
    # nuscenes_scene_plot_lidartop(nusc, scene_idx, nsweeps=10,
    #                              out_path=f'/home/leijiaming/painting/lidar-10sweep-{scene_idx}-notitle.png')
    # nuscenes_scene_plot_lidartop(nusc, scene_idx, nsweeps=10, underlay_map=False,
    #                              out_path=f'/home/leijiaming/painting/lidar-10sweep-{scene_idx}-notitle-nomap.png')

    # scene_idx = 400
    nuscenes_scene_plot_lidartop(nusc, scene_idx, with_anns=False, nsweeps=3, underlay_map=False,
                                 out_path=f'/home/leijiaming/painting/lidar-3sweep-{scene_idx}-notitle-nomap-noanns.png')
    nuscenes_scene_plot_lidartop(nusc, scene_idx, with_anns=True, nsweeps=3, underlay_map=False,
                                 out_path=f'/home/leijiaming/painting/lidar-3sweep-{scene_idx}-notitle-nomap-anns.png')
