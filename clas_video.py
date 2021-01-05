from paddlelite.lite import *
import cv2
import numpy as np
import sys
import time
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

# 加载模型
def create_predictor(model_dir):
    config = MobileConfig()
    config.set_model_from_file(model_dir)
    predictor = create_paddle_predictor(config)
    return predictor    

#图像归一化处理
def process_img(image, input_image_size):
    origin = image
    img = origin.resize(input_image_size, Image.BILINEAR)
    resized_img = img.copy()
    if img.mode != 'RGB':
      	img = img.convert('RGB')
    img = np.array(img).astype('float32').transpose((2, 0, 1))  # HWC to CHW
    img -= 127.5
    img *= 0.007843
    img = img[np.newaxis, :]
    return origin,img

# 预测
def predict(image, predictor, input_image_size):
    #输入数据处理
    input_tensor = predictor.get_input(0)
    input_tensor.resize([1, 3, input_image_size[0], input_image_size[1]])
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA))
    origin, img = process_img(image, input_image_size)
    image_data = np.array(img).flatten().tolist()
    input_tensor.set_float_data(image_data)
    #执行预测
    predictor.run()
    #获取输出
    output_tensor = predictor.get_output(0)
    print("output_tensor.float_data()[:] : ", output_tensor.float_data()[:])
    res = output_tensor.float_data()[:]
    return res

# 展示结果
def post_res(label_dict, res):
    # print(max(res))
    target_index = res.index(max(res))
    print("结果是:" + "   " + label_dict[target_index], "准确率为：",  max(res))
    
if __name__ == '__main__':
    # 初始定义
    label_dict = {0:"metal", 1:"paper", 2:"plastic", 3:"glass"}
    image = "./test_pic/images_orginal/plastic/plastic300.jpg"
    model_dir = "./trained_model/ResNet50_trash_x86_model.nb"
    image_size = (224, 224)
    # 初始化
    predictor = create_predictor(model_dir)
    
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        # 预测
        print('Predict Start')
        time_start=time.time()
        res = predict(frame, predictor, image_size)
        # 显示结果
        post_res(label_dict, res)
        print('Time Cost：{}'.format(time.time()-time_start) , "s")
        print('Predict End')
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break