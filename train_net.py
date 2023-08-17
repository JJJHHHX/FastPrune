#!/usr/bin/env python
# encoding: utf-8
"""
@author:  L1aoXingyu, guan'an wang
@contact: sherlockliao01@gmail.com, guan.wang0706@gmail.com
"""

import sys
sys.path.append('.')
from fastreid.config import get_cfg
from fastreid.engine import default_argument_parser, default_setup, DefaultTrainer, launch, SimpleTrainer
from fastreid.utils.params import ContiguousParams
from fastreid.utils.checkpoint import Checkpointer

from fastprune import *
import time

class PruneTrainer(SimpleTrainer):
    """
    Trainer class with support for pruning.
    """
    def __init__(self, model, data_loader, optimizer, param_wrapper):
        super().__init__(model, data_loader, optimizer, param_wrapper)

    def run_step(self):
        """
        Implement the standard training logic described above.
        """
        assert self.model.training, "[PruneTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        """
        If your want to do something with the data, you can wrap the dataloader.
        """
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        """
        If your want to do something with the heads, you can wrap the model.
        """

        loss_dict = self.model(data)
        losses = sum(loss_dict.values())

        """
        If you need accumulate gradients or something similar, you can
        wrap the optimizer with your custom `zero_grad()` method.
        """
        # import pdb; pdb.set_trace()
        self.optimizer.zero_grad()

        losses.backward()

        self._write_metrics(loss_dict, data_time)

        self.model.backbone.updateBN()

        """
        If you need gradient clipping/scaling or other processing, you can
        wrap the optimizer with your custom `step()` method.
        """
        self.optimizer.step()
        if isinstance(self.param_wrapper, ContiguousParams):
            self.param_wrapper.assert_buffer_is_valid()

class Trainer(DefaultTrainer):
    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        """
        # import pdb; pdb.set_trace()
        data_loader = self.build_train_loader(cfg)
        cfg = self.auto_scale_hyperparams(cfg, data_loader.dataset.num_classes)
        model = self.build_model(cfg)
        optimizer, param_wrapper = self.build_optimizer(cfg, model)

        defined_trainer = PruneTrainer(
            model, data_loader, optimizer, param_wrapper
        )
        super().__init__(cfg, trainer=defined_trainer, model=model, data_loader=data_loader, optimizer=optimizer, param_wrapper=param_wrapper)
    

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg




def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = DefaultTrainer.build_model(cfg)
        Checkpointer(model, save_dir=cfg.OUTPUT_DIR).load(cfg.MODEL.WEIGHTS)
        res = DefaultTrainer.test(cfg, model)
        return res

    trainer = Trainer(cfg)

    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    parser = default_argument_parser()
    args = parser.parse_args()

    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
