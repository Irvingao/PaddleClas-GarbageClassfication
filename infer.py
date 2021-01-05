import argparse
import numpy as np
import paddle.fluid as fluid
from utils import utils
import cv2
import os

'''
（1）环境配置：
    !git clone https://github.com/yeyupiaoling/PaddlePaddle-Classification.git

    !pip install ujson opencv-python pillow tqdm PyYAML visualdl -i https://mirrors.aliyun.com/pypi/simple/
    !pip install paddleslim==1.1.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
    !pip install PaddlePaddle-Classification/utils/ppcls-0.0.3-py3-none-any.whl
（2）提供图片预测和视频流预测两种模式。
    视频流预测：默认
    图片预测：根据输入图片路径自行修改后缀参数，或在parse中的default中修改即可。

'''
# 获取后缀(为了方便，就全部默认即可，修改default修改参数)
def parse_args():
    def str2bool(v):
        return v.lower() in ("true", "t", "1")

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str,      default="./trained_model/ResNet50_vd_ssld")
    parser.add_argument("--use_gpu",    type=str2bool, default=False)
    parser.add_argument("--img_size",   type=int,      default=224)
    return parser.parse_args()

# 加载模型
def create_predictor(args):
    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)

    [program, feed_names, fetch_lists] = fluid.io.load_inference_model(dirname=args.model_path,
                                                                       executor=exe,
                                                                       model_filename='__model__',
                                                                       params_filename='__params__')
    compiled_program = fluid.compiler.CompiledProgram(program)

    return exe, compiled_program, feed_names, fetch_lists

# 获取预处理op
def create_operators(args):
    img_mean = [0.485, 0.456, 0.406]
    img_std = [0.229, 0.224, 0.225]
    img_scale = 1.0 / 255.0

    decode_op = utils.DecodeImage()
    resize_op = utils.ResizeImage(resize_short=256)
    crop_op = utils.CropImage(size=(args.img_size, args.img_size))
    normalize_op = utils.NormalizeImage(scale=img_scale, mean=img_mean, std=img_std)
    totensor_op = utils.ToTensor()

    return [decode_op, resize_op, crop_op, normalize_op, totensor_op]


# 执行预处理
def preprocess(fname, ops):
    data = open(fname, 'rb').read()
    for op in ops:
        data = op(data)
    return data


# 提取预测结果
def postprocess(outputs, topk=5):
    output = outputs[0]
    prob = np.array(output).flatten()
    index = prob.argsort(axis=0)[-topk:][::-1].astype('int32')
    return zip(index, prob[index])

# 视频流
def video(image_path):
    cap = cv2.VideoCapture(1)
    ret, frame = cap.read()
    cv2.imwrite(image_path, frame)
    cv2.imshow("frame", frame)
    
if __name__ == '__main__':
    label_dict = {0:"metal", 1:"paper", 2:"plastic", 3:"glass"}
    image_path = "./tem.jpg"

    args = parse_args()
    # 初始化
    operators = create_operators(args)
    exe, program, feed_names, fetch_lists = create_predictor(args)
    
    while True:
        video(image_path)
        data = preprocess(image_path, operators)
        data = np.expand_dims(data, axis=0)
        
        outputs = exe.run(program,
                        feed={feed_names[0]: data},
                        fetch_list=fetch_lists,
                        return_numpy=False)
        lab, porb = postprocess(outputs).__next__()
        print("结果为：%s, 概率为：%f" % (label_dict[lab], porb)) #lab
        os.remove("./tem.jpg")
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break