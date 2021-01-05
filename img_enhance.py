import os
import shutil
import cv2
import numpy as np
'''opencv数据增强
    对图片进行色彩增强、高斯噪声、水平镜像、放大、旋转、剪切
'''

def contrast_brightness_image(src1, a, g, path_out):
    '''
        色彩增强（通过调节对比度和亮度）
    '''
    h, w, ch = src1.shape  # 获取shape的数值，height和width、通道
    # 新建全零图片数组src2,将height和width，类型设置为原图片的通道类型(色素全为零，输出为全黑图片)
    src2 = np.zeros([h, w, ch], src1.dtype)
    # addWeighted函数说明:计算两个图像阵列的加权和
    dst = cv2.addWeighted(src1, a, src2, 1 - a, g)
    cv2.imwrite(path_out, dst)


def gasuss_noise(image, path_out_gasuss, mean=0, var=0.001):
    '''
        添加高斯噪声
        mean : 均值
        var : 方差
    '''
    image = np.array(image / 255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    out = image + noise
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out * 255)
    cv2.imwrite(path_out_gasuss, out)


def mirror(image, path_out_mirror):
    '''
        水平镜像
    '''
    h_flip = cv2.flip(image, 1)
    cv2.imwrite(path_out_mirror, h_flip)


def resize(image, path_out_large):
    '''
        放大两倍
    '''
    height, width = image.shape[:2]
    large = cv2.resize(image, (2 * width, 2 * height))
    cv2.imwrite(path_out_large, large)


def rotate(image, path_out_rotate):
    '''
        旋转
    '''
    rows, cols = image.shape[:2]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 10, 1)
    dst = cv2.warpAffine(image, M, (cols, rows))
    cv2.imwrite(path_out_rotate, dst)


def shear(image, path_out_shear):
    '''
        剪切
    '''
    height, width = image.shape[:2]
    cropped = image[int(height / 9):height, int(width / 9):width]
    cv2.imwrite(path_out_shear, cropped)


if __name__ == '__main__':
    
    image_path = './test_pic/images_orginal/plastic/'
    image_out_path = './test_picimages_orginal/plastic/'
    
    if not os.path.exists(image_out_path):
        os.mkdir(image_out_path)
    list = os.listdir(image_path)
    print(list)
    print("----------------------------------------")
    print("The original data path:" + image_path)
    print("The original data set size:" + str(len(list)))
    print("----------------------------------------")

    class_list = ["glass", "metal", "paper", "plastic"]
    for i in range(0, len(list)):
        
        for j in range(0, len(imageNameList)):
            path_out = os.path.join(
                image_out_path, out_image_name + imageNameList[j])
            image = cv2.imread(path)
            if j == 0:
                contrast_brightness_image(image, 1.2, 10, path_out)
            elif j == 1:
                gasuss_noise(image, path_out)
            elif j == 2:
                mirror(image, path_out)
            elif j == 3:
                resize(image, path_out)
            elif j == 4:
                rotate(image, path_out)
            elif j == 5:
                shear(image, path_out)
            else:
                shutil.copy(path, path_out)
        print(out_image_name + "success！", end='\t')
    print("----------------------------------------")
    print("The data augmention path:" + image_out_path)
    outlist = os.listdir(image_out_path)
    print("The data augmention sizes:" + str(len(outlist)))
    print("----------------------------------------")
    print("Rich sample for:" + str(len(outlist) - len(list)))

