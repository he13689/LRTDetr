import torch
from loguru import logger
import config
from src.core import YAMLConfig
from utils.trainer import DetTrainer

logger.add('output/test5/test_log.txt')
config.model_yaml = 'configs/rtdetr/rtdetr_r101vd_6x_coco.yml'
config.resume = 'output/rtdetr_r101vd_6x_coco/latest_model.pth'
config.tuning = ''
config.test_only = True

if __name__ == '__main__':
    # 初始化
    cfg = YAMLConfig(
        config.model_yaml,
        resume=config.resume,
        use_amp=config.use_amp,
        tuning=config.tuning
    )

    trainer = DetTrainer(cfg)

    ckpt = torch.load(config.resume, map_location='cpu')
    msg = cfg.model.load_state_dict(ckpt)
    print(msg)

    trainer.testify()

