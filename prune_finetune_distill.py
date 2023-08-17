import sys
sys.path.append('.')
from fastreid.config import get_cfg
from fastreid.engine import default_argument_parser, default_setup, DefaultTrainer, launch, SimpleTrainer
from fastreid.utils.params import ContiguousParams
from fastreid.utils.checkpoint import Checkpointer
from fastreid.modeling.meta_arch import build_model
import torch
import torch.nn as nn

from fastprune import *
import time

sys.path.append('./projects/FastDistill')
from fastdistill import *


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

def get_pruned_model(model,cfg_fastreid, percent):
    ### record all bn weight and get threshold based on prune percentage
    total = 0
    for name, m in model.backbone.named_modules():
        # if isinstance(m, nn.BatchNorm2d):
        #     if name.endswith("BN"): 
        #         continue ### do not regularize IBN module
        #     # import pdb; pdb.set_trace()
        if isinstance(m, nn.BatchNorm2d) and (name.endswith("bn1") or name.endswith("bn2")) and name!='bn1':
            total += m.weight.data.shape[0]

    # import pdb; pdb.set_trace()
    bn = torch.zeros(total)
    index = 0
    for name, m in model.backbone.named_modules():
        # if isinstance(m, nn.BatchNorm2d):
        #     if name.endswith("BN"): 
        #         continue ### do not regularize IBN module
        if isinstance(m, nn.BatchNorm2d) and (name.endswith("bn1") or name.endswith("bn2")) and name!='bn1':
            size = m.weight.data.shape[0]
            bn[index:(index+size)] = m.weight.data.abs().clone()
            index += size
    # import pdb; pdb.set_trace()
    y, i = torch.sort(bn) ### sort all bn weights
    thre_index = int(total * percent)
    thre = y[thre_index]
    print("bn pruned threshold: ", thre)
    ########
    pruned = 0
    cfg = []
    cfg_mask = {}
    pruned_bn_index = []

    old_modules = list(model.backbone.named_modules())
    for idx, (name, m) in enumerate(old_modules):
        if isinstance(m, nn.BatchNorm2d) and (name.endswith("bn1") or name.endswith("bn2")) and name!='bn1':
            weight_copy = m.weight.data.abs().clone()
            mask = weight_copy.gt(thre).cuda()
            pruned = pruned + mask.shape[0] - torch.sum(mask)
            # import pdb; pdb.set_trace()
            # m.weight.data.mul_(mask)
            m.weight.data = m.weight.data[mask]
            # m.bias.data.mul_(mask)
            m.bias.data = m.bias.data[mask]
            m.running_mean.data = m.running_mean.data[mask]
            m.running_var.data = m.running_var.data[mask]
            cfg.append(int(torch.sum(mask)))
            cfg_mask[idx] = mask.clone()
            print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
                format(idx, mask.shape[0], int(torch.sum(mask))))
            pruned_bn_index.append(idx)

    # import pdb; pdb.set_trace()
    pruned_ratio = pruned/total
    print('pruned_ratio ', pruned_ratio)

    ### get pruned model (cut conv1 and conv2 channel , change the weight shape)
    for layer_id , (name, m) in enumerate(model.backbone.named_modules()):
        if isinstance(m,  nn.Conv2d) and layer_id+1 in pruned_bn_index:
            m.weight.data =  m.weight.data[cfg_mask[layer_id+1]]
        if isinstance(m,  nn.Conv2d) and layer_id-1 in pruned_bn_index:
            m.weight.data = m.weight.data[:, cfg_mask[layer_id-1]]
            

    #### save pruned model weight
    # Checkpointer(model, save_dir="./pruned_model").save(f"pruned_percent_{args.percent}")
    ### pruned_model_test
    new_model_res = DefaultTrainer.test(cfg_fastreid, model)
    return model

class Trainer(DefaultTrainer):
    def __init__(self, cfg, pruned_model):
        """
        Args:
            cfg (CfgNode):
        """
        # import pdb; pdb.set_trace()
        data_loader = self.build_train_loader(cfg)
        cfg = self.auto_scale_hyperparams(cfg, data_loader.dataset.num_classes)
        optimizer, param_wrapper = self.build_optimizer(cfg, pruned_model)

        defined_trainer = SimpleTrainer(
            pruned_model, data_loader, optimizer, param_wrapper
        )
        super().__init__(cfg, trainer=defined_trainer, model=pruned_model, data_loader=data_loader, optimizer=optimizer, param_wrapper=param_wrapper)


def main(args):
    cfg_fastreid = setup(args)
    
    model = build_model(cfg_fastreid)
    Checkpointer(model).load(cfg_fastreid.MODEL.WEIGHTS)
    # org_res = DefaultTrainer.test(cfg_fastreid, model)
    model = get_pruned_model(model, cfg_fastreid, args.percent)

    if args.eval_only:
        return 
    ### start finetune
    trainer = Trainer(cfg_fastreid, model)
    return trainer.train()



if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument("--percent", type=float, default=0.5, help="pruned bn percent")
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