import os
import re
import cv2
import numpy as np
import sys

CURRENT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(CURRENT_DIR, '../../'))

def MkdirSimple(path):
    path_current = path
    suffix = os.path.splitext(os.path.split(path)[1])[1]

    if suffix != "":
        path_current = os.path.dirname(path)
        if path_current in ["", "./", ".\\"]:
            return
    if not os.path.exists(path_current):
        os.makedirs(path_current)

def Walk(path, suffix:list):
    file_list = []
    suffix = [s.lower() for s in suffix]
    if not os.path.exists(path):
        print("not exist path {}".format(path))
        return []

    if os.path.isfile(path):
        return [path,]

    for root, dirs, files in os.walk(path):
        for file in files:
            if os.path.splitext(file)[1].lower()[1:] in suffix:
                file_list.append(os.path.join(root, file))

    try:
        file_list.sort(key=lambda x:int(re.findall('\d+', os.path.splitext(os.path.basename(x))[0])[0]))
    except:
        pass
    return file_list

def get_left_right_files(data_dir):
    left_files = []
    right_files = []
    if os.path.isdir(data_dir):
        paths = Walk(data_dir, ['jpg', 'png', 'jpeg'])
        print(paths)
        for image_name in paths:
            if "left" in image_name or "cam0" in image_name:
                left_files.append(image_name)
            elif "right" in image_name or "cam1" in image_name:
                right_files.append(image_name)
    else:
        print("need --images for input images' dir")
        assert 0
    assert len(left_files) == len(right_files), "left(cam0) images' number != right(cam1) images' number! "
    return left_files, right_files

def get_files(data_dir):
    if os.path.isdir(data_dir):
        paths = Walk(data_dir, ['jpg', 'png', 'jpeg'])
    else:
        print("need --images for input images' dir")
        assert 0
    return paths

def GetDepthImg(img):
    depth_img_rest = img.copy()
    depth_img_R = depth_img_rest.copy()
    depth_img_R[depth_img_rest > 255] = 255
    depth_img_rest[depth_img_rest < 255] = 255
    depth_img_rest -= 255
    depth_img_G = depth_img_rest.copy()
    depth_img_G[depth_img_rest > 255] = 255
    depth_img_rest[depth_img_rest < 255] = 255
    depth_img_rest -= 255
    depth_img_B = depth_img_rest.copy()
    depth_img_B[depth_img_rest > 255] = 255
    depth_img_rgb = np.stack([depth_img_R, depth_img_G, depth_img_B], axis=2)
    return depth_img_rgb.astype(np.uint8)

def GetDepthImgPSL(img):
    depth_img_rest = img.copy()
    depth_img_R = depth_img_rest.copy()
    depth_img_R[depth_img_rest > 255] = 0
    depth_img_rest[depth_img_rest < 255] = 255
    depth_img_rest -= 255
    depth_img_G = depth_img_rest.copy()
    depth_img_G[depth_img_rest > 255] = 0
    depth_img_rest[depth_img_rest < 255] = 255
    depth_img_rest -= 255
    depth_img_B = depth_img_rest.copy()
    depth_img_B[depth_img_rest > 255] = 255
    depth_img_rgb = np.stack([depth_img_R, depth_img_G, depth_img_B], axis=2)
    return depth_img_rgb.astype(np.uint8)

def WriteDepth(depth, limg, path, name, bf=None, scale=100):
    if bf is not None:
        bf = float(bf)

    output_depth = os.path.join(path, "depth", name)
    predict_np = depth.squeeze()
    print(predict_np.shape)
    MkdirSimple(output_depth)
    print(np.min(predict_np), np.max(predict_np))
    # get depth
    if bf is not None:
        depth_img_depth = bf / predict_np * scale
        print(np.min(depth_img_depth), np.max(depth_img_depth))

        depth_img_depth[depth_img_depth < 0] = 0
        depth_img_depth[depth_img_depth > 65535] = 65535
        print(depth_img_depth.shape, output_depth)
        cv2.imwrite(output_depth, depth_img_depth.astype(np.uint16))
    else:
        assert 0, "please confirm bf parameter"
def get_last_name(file_name):
    if file_name is not None:
        return file_name.split("/")[-1].split(".")[0]
    else:
        return None
