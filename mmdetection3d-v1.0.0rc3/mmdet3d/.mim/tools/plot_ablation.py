import matplotlib.pyplot as plt

# gnn_c4 版本的消融
if __name__ == '__main__':

    plt.figure(figsize=(4, 3))
    fig, ax = plt.subplots(dpi=200)

    models = ['PointPillars', 'SSN', 'Free-anchor3d', 'CenterPoint']
    map_c4_w_DA = [0.3429, 0.4162, 0.4474, 0.5017]
    map_0_w_DA = [0.3409, 0.4117, 0.4396, 0.4890]
    map_c4_wo_DA = [0.3336, 0.4077, 0.4436, 0.4821]
    map_0_wo_DA = [0.3343, 0.3982, 0.3703, 0.4648]
    for i in range(4):
        map_c4_w_DA[i] *= 100
        map_0_w_DA[i] *= 100
        map_c4_wo_DA[i] *= 100
        map_0_wo_DA[i] *= 100

    ax.plot(models, map_c4_w_DA, color='steelblue', marker='o', linewidth=2.0, label='ours')
    ax.plot(models, map_0_w_DA, ls='--', color='r', marker='s', linewidth=2.0, label='baseline')
    plt.tick_params(labelsize=12)

    ax.set_ylabel('mAP(%)', fontsize=14)

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
              fancybox=False, shadow=False, ncol=5, frameon=False)  # 将图例放在x轴下方

    plt.tight_layout()
    plt.savefig('/home/leijiaming/painting/comparsion_map.png')
    plt.close()
    ########################################

    plt.figure(figsize=(4, 3))
    fig, ax = plt.subplots(dpi=200)

    models_aoe = ['PointPillars', 'SSN', 'Free-anchor3d', 'CenterPoint']
    maoe_c4_w_DA = [0.4762, 0.3959, 0.5104, 0.3598]
    maoe_0_w_DA = [0.5231, 0.4355, 0.5298,  0.3850]
    maoe_c4_wo_DA = [0.5009, 0.4523, 0.5212, 0.3694]
    maoe_0_wo_DA = [0.5471, 0.4549, 0.5339, 0.4033]

    ax.plot(models_aoe, maoe_c4_w_DA, color='steelblue', marker='o', linewidth=2.0, label='ours')
    ax.plot(models_aoe, maoe_0_w_DA, ls='--', color='r', marker='s', linewidth=2.0, label='baseline')
    plt.tick_params(labelsize=12)

    ax.set_ylabel('mAOE', fontsize=14)

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
              fancybox=False, shadow=False, ncol=5, frameon=False)  # 将图例放在x轴下方

    plt.tight_layout()
    plt.savefig('/home/leijiaming/painting/comparsion_maoe.png')
    plt.close()

    x = 1
    y = 2
