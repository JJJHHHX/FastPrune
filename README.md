# FastDistill in FastReID

This project provides a strong distillation method for both embedding and classification training.
The feature distillation comes from [overhaul-distillation](https://github.com/clovaai/overhaul-distillation/tree/master/ImageNet).


## Datasets Prepration
- DukeMTMC-reID


## Train and Evaluation
```shell
# teacher model training
python3 projects/FastDistill/train_net.py \
--config-file projects/FastDistill/configs/sbs_r101ibn.yml \
--num-gpus 4

# loss distillation
python3 projects/FastDistill/train_net.py \
--config-file projects/FastDistill/configs/kd-sbs_r101ibn-sbs_r34.yaml \
--num-gpus 4 \
MODEL.META_ARCHITECTURE Distiller
KD.MODEL_CONFIG '("projects/FastDistill/logs/dukemtmc/r101_ibn/config.yaml",)' \
KD.MODEL_WEIGHTS '("projects/FastDistill/logs/dukemtmc/r101_ibn/model_best.pth",)'

# loss+overhaul distillation
python3 projects/FastDistill/train_net.py \
--config-file projects/FastDistill/configs/kd-sbs_r101ibn-sbs_r34.yaml \
--num-gpus 4 \
MODEL.META_ARCHITECTURE DistillerOverhaul
KD.MODEL_CONFIG '("projects/FastDistill/logs/dukemtmc/r101_ibn/config.yaml",)' \
KD.MODEL_WEIGHTS '("projects/FastDistill/logs/dukemtmc/r101_ibn/model_best.pth",)'


CUDA_VISIBLE_DEVICES=4,5,6,7 nohup python projects/FastDistill/train_net.py \
--config-file projects/FastDistill/configs/kd-sbs_r50ibn-sbs_r34ibn.yml --num-gpus 4 \
SOLVER.IMS_PER_BATCH 1024 MODEL.META_ARCHITECTURE DistillerOverhaul  > distill.out & 
```

## Experimental Results

### Settings

All the experiments are conducted with 4 V100 GPUs.


### DukeMTMC-reID

| Model | Rank@1 | mAP |
| --- | --- | --- |
| R101_ibn (teacher) | 90.66 | 81.14 |
| R34 (student) | 86.31 | 73.28 |
| JS Div | 88.60 | 77.80 |
| JS Div + Overhaul | 88.73 | 78.25 |

## Contact
This project is conducted by [Xingyu Liao](https://github.com/L1aoXingyu) and [Guan'an Wang](https://wangguanan.github.io/) (guan.wang0706@gmail).


MAX_EPOCH: 60  SCHED: CosineAnnealingLR  DELAY_EPOCHS: 30
CUDA_VISIBLE_DEVICES=4,5,6,7 nohup python projects/FastDistill/train_net.py \
--config-file projects/FastDistill/configs/kd-sbs_r50ibn-sbs_r34ibn.yml --num-gpus 4 \
SOLVER.IMS_PER_BATCH 1024 MODEL.META_ARCHITECTURE DistillerOverhaul  > distill.out & 


MAX_EPOCH: 120  SCHED: CosineAnnealingLR  DELAY_EPOCHS: 60
CUDA_VISIBLE_DEVICES=4,5,6,7 nohup python projects/FastDistill/train_net.py \
--config-file projects/FastDistill/configs/kd-sbs_r50ibn-sbs_r34ibn.yml --num-gpus 4 \
SOLVER.MAX_EPOCH 120   SOLVER.DELAY_EPOCHS 60  \
SOLVER.IMS_PER_BATCH 1024 MODEL.META_ARCHITECTURE DistillerOverhaul  > distill_120.out & 

#### 主要降低loss_overhual (需要较大 lr) 从/FastDistill/logs/Market1501/kd-r34ibn-r50ibn/model_0059.pth 开始
CUDA_VISIBLE_DEVICES=4,5,6,7 nohup python projects/FastDistill/train_net.py \
--config-file projects/FastDistill/configs/kd-sbs_r50ibn-sbs_r34ibn.yml --num-gpus 4 \
SOLVER.MAX_EPOCH 100  SOLVER.DELAY_EPOCHS 40 SOLVER.FREEZE_ITERS 0 SOLVER.WARMUP_ITERS 0 \
MODEL.WEIGHTS projects/FastDistill/logs/Market1501/kd-r34ibn-r50ibn/model_0059.pth \
OUTPUT_DIR projects/FastDistill/logs/Market1501/kd-r34ibn-r50ibn-160epoch \
SOLVER.IMS_PER_BATCH 1024 MODEL.META_ARCHITECTURE DistillerOverhaul  > distill_160.out &


############ test ibn False
MAX_EPOCH: 180 SCHED: CosineAnnealingLR  DELAY_EPOCHS: 100
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python projects/FastDistill/train_net.py \
--config-file projects/FastDistill/configs/kd-sbs_r50ibn-sbs_r34ibn.yml --dist-url tcp://127.0.0.1:1234 --num-gpus 4 \
SOLVER.MAX_EPOCH 180  SOLVER.DELAY_EPOCHS 120  MODEL.BACKBONE.WITH_IBN False \
OUTPUT_DIR projects/FastDistill/logs/Market1501/kd-r34-r50ibn-160epoch \
SOLVER.IMS_PER_BATCH 1024 MODEL.META_ARCHITECTURE DistillerOverhaul > distill_noIBN.out & 
### 主要降低loss_overhual (需要较大 lr) 从/FastDistill/logs/Market1501/kd-r34ibn-r50ibn/model_0059.pth 开始 consineAnnealingLR 下降
MAX_EPOCH: 220 SCHED: CosineAnnealingLR  DELAY_EPOCHS: 160

CUDA_VISIBLE_DEVICES=4,5,6,7 nohup python projects/FastDistill/train_net.py \
--config-file projects/FastDistill/configs/kd-sbs_r50ibn-sbs_r34ibn.yml --num-gpus 4 \
SOLVER.MAX_EPOCH 100  SOLVER.DELAY_EPOCHS 40 SOLVER.FREEZE_ITERS 0 SOLVER.WARMUP_ITERS 0 \
MODEL.BACKBONE.WITH_IBN False \
MODEL.WEIGHTS projects/FastDistill/logs/Market1501/kd-r34-r50ibn-160epoch/model_0119.pth \
OUTPUT_DIR projects/FastDistill/logs/Market1501/kd-r34-r50ibn-220epoch \
SOLVER.IMS_PER_BATCH 1024 MODEL.META_ARCHITECTURE DistillerOverhaul  > distill_noIBN_220epoch.out &

###

MAX_EPOCH: 320 SCHED: CosineAnnealingLR  DELAY_EPOCHS: 200

CUDA_VISIBLE_DEVICES=4,5,6,7 nohup python projects/FastDistill/train_net.py \
--config-file projects/FastDistill/configs/kd-sbs_r50ibn-sbs_r34ibn.yml --num-gpus 4 \
SOLVER.MAX_EPOCH 200 SOLVER.DELAY_EPOCHS 80 SOLVER.FREEZE_ITERS 0 SOLVER.WARMUP_ITERS 0 \
MODEL.BACKBONE.WITH_IBN False \
MODEL.WEIGHTS projects/FastDistill/logs/Market1501/kd-r34-r50ibn-160epoch/model_0119.pth \
OUTPUT_DIR projects/FastDistill/logs/Market1501/kd-r34-r50ibn-320epoch \
SOLVER.IMS_PER_BATCH 1024 MODEL.META_ARCHITECTURE DistillerOverhaul  > distill_noIBN_320epoch.out &


MAX_EPOCH: 260 SCHED: CosineAnnealingLR  DELAY_EPOCHS: 160

CUDA_VISIBLE_DEVICES=4,5,6,7 nohup python projects/FastDistill/train_net.py \
--config-file projects/FastDistill/configs/kd-sbs_r50ibn-sbs_r34ibn.yml --num-gpus 4 \
SOLVER.MAX_EPOCH 100  SOLVER.DELAY_EPOCHS 40 SOLVER.FREEZE_ITERS 0 SOLVER.WARMUP_ITERS 0 \
MODEL.BACKBONE.WITH_IBN False \
MODEL.WEIGHTS projects/FastDistill/logs/Market1501/kd-r34-r50ibn-160epoch/model_0119.pth \
OUTPUT_DIR projects/FastDistill/logs/Market1501/kd-r34-r50ibn-220epoch \
SOLVER.IMS_PER_BATCH 1024 MODEL.META_ARCHITECTURE DistillerOverhaul  > distill_noIBN_220epoch.out &


############ test ibn True
MAX_EPOCH: 160 SCHED: CosineAnnealingLR  DELAY_EPOCHS: 100

CUDA_VISIBLE_DEVICES=4,5,6,7 nohup python projects/FastDistill/train_net.py \
--config-file projects/FastDistill/configs/kd-sbs_r50ibn-sbs_r34ibn.yml --num-gpus 4 \
SOLVER.MAX_EPOCH 180  SOLVER.DELAY_EPOCHS 120  MODEL.BACKBONE.WITH_IBN True \
OUTPUT_DIR projects/FastDistill/logs/Market1501/kd-r34ibn-r50ibn-180epoch \
SOLVER.IMS_PER_BATCH 1024 MODEL.META_ARCHITECTURE DistillerOverhaul > distill_180.out & 

### test teacher
CUDA_VISIBLE_DEVICES=4,5,6,7  python projects/FastDistill/train_net.py --eval-only Ture \
--config-file configs/Market1501/bagtricks_R50-ibn.yml --num-gpus 4 \
MODEL.WEIGHTS logs/market1501/bagtricks_R50/market_bot_R50-ibn.pth  TEST.IMS_PER_BATCH 1024


