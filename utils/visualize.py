import cv2
import numpy as np


def generate_mask_image(image_path, output_path):
    # 读取输入图像
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print(f"Failed to load image {image_path}")
        return

    # 计算图像的均值
    mean_value = np.mean(image)

    # 创建空的彩色图像
    mask_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

    # 根据均值生成蒙版图像
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i, j] > mean_value:
                # 高于均值部分为蓝色
                mask_image[i, j] = [255, 0, 0]
            else:
                # 低于均值部分为红色，颜色越深的部分越红
                mask_image[i, j] = [0, 0, 255 - int(image[i, j] / mean_value * 255)]

    # 保存输出蒙版图像
    cv2.imwrite(output_path, mask_image)
    print(f"Mask image saved as {output_path}")


# 输入和输出图像路径
input_image_path = 'test_images/1111.png'
output_image_path = 'test_images/27mask.png'

# 生成蒙版图像
generate_mask_image(input_image_path, output_image_path)
