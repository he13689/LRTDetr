from utils.common import generate_colors

model_yaml = 'configs/rtdetr/rtdetr_custom17.yml'  # 模型yaml配置文件
resume = 'weights/best_model71.pth'  # 继续训练
use_amp = False  # amp加速
tuning = 'weights/rtdetr_r50vd_6x_coco_from_paddle.pth'  # 预训练模型位置
iou_type = 'GIOU'  # MPDIOU, InnerIOU, GIOU

test_only = False
chs = 3
device = 'cuda'
half = False
v8_weights = 'weights/yolol.pt'
# v7_weights = 'weights/yolov5l.pt'  # 暂时代替一下
v7_weights = 'weights/yolov7.pth'
save_counter = 0
colors = generate_colors(55)
best_ap = 0.300

conf_thres = .25
nms_thres = .7
