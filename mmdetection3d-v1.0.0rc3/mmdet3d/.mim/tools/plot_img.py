import numpy as np
from PIL import Image, ImageDraw
import torch
from mmdet3d.core.bbox import Box3DMode, CameraInstance3DBoxes, points_cam2img

if __name__ == '__main__':

    car1_tensor = torch.tensor([[-2.70, 1.74, 3.68, 3.23, 1.57, 1.60, -1.29]], dtype=torch.float32)

    car1_box = CameraInstance3DBoxes(car1_tensor)

    P2_08 = np.array([[7.215377000000e+02, 0.000000000000e+00, 6.095593000000e+02, 4.485728000000e+01],
                      [0.000000000000e+00, 7.215377000000e+02, 1.728540000000e+02, 2.163791000000e-01],
                      [0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00, 2.745884000000e-03]],
                     dtype=np.float32)
    P2_08 = torch.tensor(P2_08)
    R0_rect_08 = np.array([[9.999239000000e-01, 9.837760000000e-03, -7.445048000000e-03, 0],
                           [-9.869795000000e-03, 9.999421000000e-01, -4.278459000000e-03, 0],
                           [7.402527000000e-03, 4.351614000000e-03, 9.999631000000e-01, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

    car1_box.convert_to(Box3DMode.CAM, R0_rect_08)
    car1_box_corners = car1_box.corners
    car1_box_corners_in_images = points_cam2img(car1_box_corners, P2_08)

    minxy = torch.min(car1_box_corners_in_images, dim=1)[0].squeeze()
    maxxy = torch.max(car1_box_corners_in_images, dim=1)[0].squeeze()

    file_name = '/dataset/kitti/training/image_2/000008.png'

    kitti_08 = Image.open(file_name)
    draw = ImageDraw.Draw(kitti_08)
    draw.line((minxy[0], minxy[1], maxxy[0], maxxy[1]), 'cyan')

    kitti_08.save('/home/leijiaming/kitti_08.jpeg', 'JPEG')


    pass