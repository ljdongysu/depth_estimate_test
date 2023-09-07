import argparse
import sys, os
import cv2
import numpy as np

CURRENT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(CURRENT_DIR, '../'))

from onnx_utils.onnxmodel import ONNXModel
from utils.file_utils import get_left_right_files

from utils.preprocess_postprocess import preprocess_hit, preprocess_madnet
from utils.file_utils import WriteDepth
from utils.file_utils import get_files, get_last_name
from utils.compare_tof import compare_depth_tof
from utils.compare_predict_gt_disp import compare_depth_disp
from utils.compare_tof import get_boundary, get_boundary_wh
def get_parameter():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', default=None, required=False, type=str, help='Data directory for prediction')

    parser.add_argument('--model_type', default="madnet", type=str, help='model type of onnx')

    parser.add_argument('--height', default=None, type=int, help='Image height for inference')
    parser.add_argument('--width', default=None, type=int, help='Image width for inference')

    parser.add_argument('--output_dir', default=None, type=str,
                        help='Directory to save inference results and test results')

    parser.add_argument('--onnx_file', default=None, type=str,
                        help='File to save onnx inference model')

    parser.add_argument('--scale', type=str, default=100, help="depth image real cm * scale for test depth image")

    parser.add_argument('--bf', type=str, default=3424, help="bf")

    return parser.parse_args()
def main():
    args = get_parameter()

    print("current dataset's bf is {}".format(args.bf))

    if args.data_dir is not None and os.path.isdir(args.data_dir):
        left_files, right_files = get_left_right_files(args.data_dir)
        disp_true = [None] * len(left_files)

        # load onnx file
        model = ONNXModel(args.onnx_file)

        root_len = len(args.data_dir)

        for left_file, right_file, disp_file in zip(left_files, right_files, disp_true):
            if left_file[root_len:][0] == '/':
                op = os.path.join(args.output_dir, left_file[root_len + 1:])
            else:
                op = os.path.join(args.output_dir, left_file[root_len:])
            left_image = cv2.imread(left_file)
            right_image = cv2.imread(right_file)

            if args.width is not None and args.height is not None:
                left, right, top, bottom = get_boundary_wh(left_image, width=int(args.width), height=int(args.height))
                left_image = left_image[top: bottom, left: right]
                right_image = right_image[top: bottom, left: right]
            left_copy = left_image.copy()
            if args.model_type == "madnet":
                left_image = preprocess_madnet(left_image)
                right_image = preprocess_madnet(right_image)
            elif args.model_type == "hitnet":
                left_image = preprocess_hit(left_image)
                right_image = preprocess_hit(right_image)

            output = model.forward2((left_image, right_image))
            print(output[0].shape)
            disp = output[0]

            op = op.replace(".jpg", ".png")

            WriteDepth(disp, left_copy, args.output_dir, op, args.bf, args.scale)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()