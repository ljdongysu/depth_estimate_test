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

def WriteDepth(depth, limg, path, name, bf=None):
    if bf is not None:
        bf = float(bf)
    output_concat_color = os.path.join(path, "concat_color", name)
    output_resize = os.path.join(path, "resize", name)
    output_concat_disp = os.path.join(path, "concat_disp", name)
    output_disp = os.path.join(path, "disp", name)
    output_disp_scale = os.path.join(path, "disp_scaleX256_uint16", name)
    output_depth = os.path.join(path, "depth", name)
    output_depth_psl = os.path.join(path, "depth_psl", name)
    output_color = os.path.join(path, "color", name)
    output_concat_depth = os.path.join(path, "concat_depth", name)
    output_concat = os.path.join(path, "concat", name)
    MkdirSimple(output_concat_color)
    MkdirSimple(output_concat_disp)
    MkdirSimple(output_concat_depth)
    MkdirSimple(output_disp)
    MkdirSimple(output_depth)
    MkdirSimple(output_depth_psl)
    MkdirSimple(output_color)
    MkdirSimple(output_concat)
    MkdirSimple(output_disp_scale)
    MkdirSimple(output_resize)

    predict_np = depth.squeeze()
    print(predict_np.max(), " ", predict_np.min())
    predict_scale = (predict_np - np.min(predict_np))* 255 / (np.max(predict_np) - np.min(predict_np))

    predict_scale = predict_scale.astype(np.uint8)
    predict_np_int = predict_scale
    color_img = cv2.applyColorMap(predict_np_int, cv2.COLORMAP_HOT)
    limg_cv = limg
    concat_img_color = np.vstack([limg_cv, color_img])
    predict_np_rgb = np.stack([predict_np, predict_np, predict_np], axis=2)
    concat_img_gray = np.vstack([limg_cv, predict_np_rgb])

    # get depth
    depth_img_rgb = GetDepthImg(predict_np)
    if bf is not None:
        depth_img_rgb_psl = bf / predict_np
        depth_img_rgb_psl = GetDepthImgPSL(depth_img_rgb_psl)
        cv2.imwrite(output_depth_psl, depth_img_rgb_psl)
    else:
        print("Warning bf is None, please confirm bf is right!")
        depth_img_rgb_psl = GetDepthImgPSL(predict_np)
        cv2.imwrite(output_depth_psl, depth_img_rgb_psl)
    concat_img_depth = np.vstack([limg_cv, depth_img_rgb])
    concat = np.hstack([np.vstack([limg_cv, color_img]), np.vstack([predict_np_rgb, depth_img_rgb])])

    cv2.imwrite(output_concat_color, concat_img_color)
    cv2.imwrite(output_concat_disp, concat_img_gray)
    cv2.imwrite(output_color, color_img)

    predict_np_gray_scale = predict_np * 256
    predict_np_gray_scale[predict_np_gray_scale < 0] = 0
    predict_np_gray_scale[predict_np_gray_scale > 65535] = 65535
    cv2.imwrite(output_disp_scale, predict_np_gray_scale.astype(np.uint16))
    cv2.imwrite(output_disp, np.squeeze(predict_np))
    print(predict_np.shape, np.squeeze(predict_np).shape)
    cv2.imwrite(output_depth, depth_img_rgb)

    cv2.imwrite(output_concat_depth, concat_img_depth)
    cv2.imwrite(output_concat, concat)
    cv2.imwrite(output_resize, limg)

def get_last_name(file_name):
    if file_name is not None:
        return file_name.split("/")[-1].split(".")[0]
    else:
        return None
