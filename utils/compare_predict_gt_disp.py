from losses_and_merics import EndPointError, Bad3
import cv2
import tensorflow as tf
import numpy as np
import os
from file_utils import MkdirSimple
import matplotlib.pyplot as plt

def val_cv2_tf_io(disp_file):

    disp_bytes = tf.io.read_file(disp_file)
    # Using uint16 for higher precision
    disp_map = tf.io.decode_png(disp_bytes, dtype=tf.uint16)
    print("uint16 max: ,min: ",np.max(disp_map), np.min(disp_map))
    disp_map = tf.cast(disp_map, dtype=tf.float32)
    print("float32 max: ,min: ",np.max(disp_map), np.min(disp_map))

    disp_map = disp_map / 256.0

    print("float32/256 max: ,min: ",np.max(disp_map), np.min(disp_map))

    cv_image = cv2.imread(disp_file, -1) /256.0
    print("cv_image max: ,min: ",np.max(cv_image), np.min(cv_image))

    print("sum: {}".format(np.sum(cv_image - disp_map[:,:,0])))

    print(disp_map.shape,cv_image.shape)

def getAbsdiff(depth_map, disparity_map, path, name):
    name = os.path.splitext(name)[0]
    output_depth = os.path.join(path, "diff", name)
    MkdirSimple(output_depth)

    depth_map = np.squeeze(depth_map)
    print("depth map : ", depth_map)

    disparity_map =np.squeeze(disparity_map)# * 256.0
    print("disparity map: ", disparity_map)

    mask = disparity_map > 0

    # 归一化数据范围
    # depth_map_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    depth_map_norm = depth_map
    depth_map_norm_mask = depth_map * mask
    # disparity_map_norm = cv2.normalize(disparity_map, None, 0, 255, cv2.NORM_MINMAX)
    disparity_map_norm = disparity_map * mask
    # print("depth_map_norm: ", depth_map_norm)

    # 计算深度图和视差图的差异
    # diff_map = cv2.absdiff(depth_map_norm_mask.astype("float32"), disparity_map_norm.astype("float32"))
    diff_map = np.abs(depth_map_norm_mask.astype("float32") - disparity_map_norm.astype("float32"))
    # diff_map = cv2.bitwise_and(diff_map, mask)

    # 计算深度图和视差图的差异
    fig, axes = plt.subplots(4, 1, figsize=(6, 12))
    # axes[0].subplot(1, 3, 1)
    axes[0].imshow(depth_map_norm, cmap='gray')
    axes[0].set_title('Depth Map')
    axes[0].axis('off')
    cv2.imwrite(os.path.join(output_depth, "depth_map_norm.png"), depth_map_norm)

    axes[1].imshow(depth_map_norm_mask, cmap='gray')
    axes[1].set_title('Depth Map Mask')
    axes[1].axis('off')
    cv2.imwrite(os.path.join(output_depth, "depth_map_norm_mask.png"), depth_map_norm_mask)

    # axes[1].subplot(1, 3, 2)
    axes[2].imshow(disparity_map_norm, cmap='gray')
    axes[2].set_title('Disparity Map')
    axes[2].axis('off')
    cv2.imwrite(os.path.join(output_depth, "disparity_map_norm.png"), disparity_map_norm)

    # axes[2].subplot(1, 3, 3)
    cax = axes[3].imshow(diff_map, cmap='jet')
    axes[3].set_title('Difference Map')
    # axes[2].colorbar()
    axes[3].axis('off')
    diff_map = cv2.applyColorMap(np.asarray(diff_map).astype("uint8"), cv2.COLORMAP_JET)
    cv2.imwrite(os.path.join(output_depth, "diff_map.png"), diff_map)

    concat = np.vstack([cv2.cvtColor(depth_map_norm, cv2.COLOR_GRAY2BGR), cv2.cvtColor(depth_map_norm_mask, cv2.COLOR_GRAY2BGR)
                           ,cv2.cvtColor(disparity_map_norm, cv2.COLOR_GRAY2BGR), diff_map])
    cv2.imwrite(os.path.join(output_depth, "concat_compare_map.png"), concat)

    fig.subplots_adjust(hspace=0.05, wspace=0.05, top=0.95, bottom=0.05)
    plt.colorbar(cax, ax=axes[3])

    # 调整图像尺寸
    fig.set_size_inches(10, 15)

    plt.tight_layout()
    plt.show()
    plt.savefig("myplot.png")


def compare_depth_disp(output_dir, op, depth_file, disp_file, bf=None):
    print(depth_file, disp_file)
    gt = cv2.imread(disp_file, -1) /256.0
    if bf is not None:
        gt = cv2.imread(disp_file, -1)
        gt[gt > 0] = bf / gt[gt > 0]
    disparities = cv2.imread(depth_file, -1)
    if bf is not None:
        disparities[disparities > 0] = bf / disparities[disparities > 0]

    epe = EndPointError()
    gt_flaot = gt.astype(np.float32)
    disparities_float = disparities.astype(np.float32)
    print(gt_flaot.shape, disparities_float.shape)
    epe.update_state(gt_flaot, disparities_float)
    print("EPE: ", epe.result())

    bad3 = Bad3()
    bad3.update_state(gt_flaot, disparities_float)
    print("Bad3: ", bad3.result())

    # getAbsdiff(gt_flaot, disparities_float, output_dir, op)

if __name__ == '__main__':
    # compare_depth_disp("test_c","/home/indemind/Code/PycharmProjects/Depth_Estimation/Stereo/madnet/result_D10.0.7_2000_test/gray/000000_10.png","/home/indemind/Code/PycharmProjects/Depth_Estimation/Stereo/madnet/result_D10.0.7_2000_test/gray/000000_10.png","/home/indemind/Code/PycharmProjects/Depth_Estimation/Stereo/madnet/test_images/kitti/disp/000000_10.png")
    compare_depth_disp("test_c", "/home/indemind/Code/PycharmProjects/Depth_Estimation/Stereo/madnet/result_D10.0.7_2000_test/gray/000000_10.png","/home/indemind/Code/PycharmProjects/Depth_Estimation/Stereo/madnet/result_D10.0.7_2000_test/gray/000000_10.png", "/home/indemind/Code/PycharmProjects/Depth_Estimation/Stereo/madnet/test_images/kitti/disp/000000_10.png")