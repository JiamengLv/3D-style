import os
from OpencvAndOpen3d import point_cloud_generator


# 1.利用KITTI预训练模型，对单张图像进行深度估计
demo_path = "./LapDepth-release-master/demo.py"
model_dir = "./LapDepth-release-master/pretrain_model/LDRN_KITTI_ResNext101_pretrained_data_grad.pkl"
dataset_pretrained = "KITTI"

data_image = "./show_image/"

try:
    os.system("python {} --model_dir {} --img_folder_dir {}  --pretrained {} ".format(demo_path, model_dir, data_image,dataset_pretrained))
except Exception as e:
    print('MyErrorAtStart:  %s' % e)


# 2.进行3D重建
camera_intrinsics = [1084, 910,2217, 2297]            # 相机角度以及位置 u0,v0,dx,dy  # 关键是相机的角度以及位置 如何设置  ？？？？？

depth_image = "./out_show_image/"
save_ply_path = "./ply/"

if not os.path.exists(save_ply_path):
    os.makedirs(save_ply_path)

for index in range(len(os.listdir(data_image))):
    rgb_file = data_image + sorted(os.listdir(data_image))[index]
    depth_file = depth_image + sorted(os.listdir(depth_image))[index]
    save_ply = save_ply_path + rgb_file.split("/")[-1].split(".")[-2] + ".ply"
    a = point_cloud_generator(rgb_file=rgb_file,
                              depth_file=depth_file,
                              save_ply=save_ply,
                              camera_intrinsics=camera_intrinsics
                              )
    a.compute()
    a.write_ply()
    a.show_point_cloud()