import matplotlib.pyplot as plt

# gnn_c4
if __name__ == '__main__':
    x = ['PointPillars', 'SSN', 'Free-anchor3d', 'CenterPoint', 'Ours']
    x_ = ['PPs', 'SSN', 'F-a3d', 'CP', 'Ours']
    car_ap = [0.7868, 0.8101, 0.8120, 0.8389, 0.8416]
    ped_ap = [0.6123, 0.6623, 0.7472, 0.7726, 0.7890]
    for i in range(5):
        car_ap[i] *= 100
        ped_ap[i] *= 100

    car_aoe = [0.1569, 0.1531, 0.1618, 0.1628, 0.1494]
    ped_aoe = [0.4265, 0.4103, 0.3632, 0.4108, 0.4013]

    plt.figure(figsize=(8, 6))
    fig, ax = plt.subplots(dpi=200)

    ax.plot(x, car_ap, color='steelblue', marker='o', linewidth=2.0, label='Car')
    ax.plot(x, ped_ap, color='red', marker='s', linewidth=2.0, label='Ped.')
    plt.tick_params(labelsize=13)
    ax.set_ylabel('AP(%)', fontsize=14)

    # ax2 = ax.twinx()
    # ax2.plot(x, car_aoe)
    # ax2.set_ylabel('AOE')

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
              fancybox=False, shadow=False, ncol=5, frameon=False,
              fontsize=14)  # 将图例放在x轴下方

    plt.tight_layout()
    plt.savefig('/home/leijiaming/painting/gnn_c4_car&ped_ap.png', bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(8, 6))
    fig, ax = plt.subplots(dpi=200)
    ax.plot(x, car_aoe, color='steelblue', marker='o', linewidth=2.0, label='Car')
    ax.plot(x, ped_aoe, color='red', marker='s', linewidth=2.0, label='Ped.')
    plt.tick_params(labelsize=13)
    ax.set_ylabel('AOE', fontsize=14)

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
              fancybox=False, shadow=False, ncol=5, frameon=False,
              fontsize=14)  # 将图例放在x轴下方

    plt.tight_layout()
    plt.savefig('/home/leijiaming/painting/gnn_c4_car&ped_aoe.png', bbox_inches='tight')
    plt.close()

    models = ['PointPillars', 'SSN', 'Free-anchor3d', 'Ours']
    map_1_with_DA = []
    map_0_with_DA = []
    map_1_wo_DA = []
    map_0_wo_DA = []

    x = 1
    y = 2
