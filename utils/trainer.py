# 定义了detection训练器，用于训练目标检测方法，其中定义了许多方法，比如训练，验证
import datetime
import json
import time

import loguru
import torch

import config
from src.data import get_coco_api_from_dataset
from src.solver import BaseSolver
from src.solver.det_engine import evaluate, train_one_epoch


class DetTrainer(BaseSolver):

    def testify(self, ):
        self.train()
        self.model.eval()
        n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        loguru.logger.warning(f'number of params: {n_parameters}')

        base_ds = get_coco_api_from_dataset(self.val_dataloader.dataset)

        module = self.ema.module if self.ema else self.model
        test_stats, coco_evaluator = evaluate(
            module, self.criterion, self.postprocessor, self.val_dataloader, base_ds, self.device, self.output_dir
        )

        # 将结果打印到文件log.txt中
        log_stats = {**{f'test_{k}': v for k, v in test_stats.items()}, 'n_parameters': n_parameters}
        with (self.output_dir / "log.txt").open("a") as f:
            f.write(json.dumps(log_stats) + "\n")

    def training(self, ):
        # 真正的执行函数入口
        loguru.logger.error("Start training......")
        self.train()  # 设定训练用的loader optimizer等

        args = self.cfg

        n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        loguru.logger.warning(f'number of params:{n_parameters}')

        base_ds = get_coco_api_from_dataset(self.val_dataloader.dataset)
        # best_stat = {'coco_eval_bbox': 0, 'coco_eval_masks': 0, 'epoch': -1, }
        best_stat = {'epoch': -1, }

        start_time = time.time()
        for epoch in range(self.last_epoch + 1, args.epoches):

            train_stats = train_one_epoch(self.model, self.criterion, self.train_dataloader, self.optimizer, self.device, epoch, args.clip_max_norm, print_freq=args.log_step, ema=self.ema, scaler=self.scaler)

            self.lr_scheduler.step()

            if self.output_dir:
                checkpoint_paths = [self.output_dir / 'checkpoint.pth']
                # extra checkpoint before LR drop and every 100 epochs
                if (epoch + 1) % args.checkpoint_step == 0:
                    checkpoint_paths.append(self.output_dir / f'checkpoint{epoch:04}.pth')

            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module, self.criterion, self.postprocessor, self.val_dataloader, base_ds, self.device, self.output_dir
            )

            # TODO
            for k in test_stats.keys():
                if k in best_stat:
                    best_stat['epoch'] = epoch if test_stats[k][0] > best_stat[k] else best_stat['epoch']
                    best_stat[k] = max(best_stat[k], test_stats[k][0])
                else:
                    best_stat['epoch'] = epoch
                    best_stat[k] = test_stats[k][0]
            loguru.logger.error(f'best_stat: {best_stat}')  # 当前最好的训练结果

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}

            if self.output_dir:
                with (self.output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                map5095 = log_stats['test_coco_eval_bbox'][0]
                if map5095 > config.best_ap:
                    config.best_ap = map5095
                    torch.save(self.model.state_dict(), self.output_dir / f'best_model{epoch}.pth')

                # for evaluation logs
                if coco_evaluator is not None:
                    (self.output_dir / 'eval').mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        filenames = ['latest.pth']
                        if epoch % 50 == 0:
                            filenames.append(f'{epoch:03}.pth')
                            torch.save(self.model.state_dict(), self.output_dir / f'{epoch:03}_model.pth')
                        for name in filenames:
                            torch.save(coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval" / name)
                        torch.save(self.model.state_dict(), self.output_dir / 'latest_model.pth')

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))
        torch.cuda.empty_cache()

    def val(self, ):
        self.eval()

        base_ds = get_coco_api_from_dataset(self.val_dataloader.dataset)

        module = self.ema.module if self.ema else self.model
        test_stats, coco_evaluator = evaluate(module, self.criterion, self.postprocessor,
                                              self.val_dataloader, base_ds, self.device, self.output_dir)

        # if self.output_dir:
        #     dist.save_on_master(coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth")

        return

    def test(self, ):
        self.eval()

        base_ds = get_coco_api_from_dataset(self.val_dataloader.dataset)

        module = self.ema.module if self.ema else self.model
        test_stats, coco_evaluator = evaluate(module, self.criterion, self.postprocessor,
                                              self.val_dataloader, base_ds, self.device, self.output_dir)

        # if self.output_dir:
        #     dist.save_on_master(coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth")

        return
