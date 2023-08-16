# DLP Lab4 - CVAE Video Predictor
## Environments
```
conda create --name DLP_Lab4 python=3.9
conda activate DLP_Lab4
pip install -r requirements.txt
```

## Highest PSNR
#### training
```
python Trainer.py --DR LAB4_Dataset/ --save_root checkpoints/max_psnr/ --num_epoch 100 --fast_train --fast_train_epoch 9 --kl_anneal_type NoKL_Annealing --tfr 0 
```

![Alt text](graph/loss_curve_max_psnr.png)

![Alt text](graph/tfr_max_psnr.png)

![Alt text](graph/VAL_PSNR_max_psnr.png)

#### validation 
```
python Trainer.py --DR LAB4_Dataset/ --save_root checkpoints/max_psnr/ --test --ckpt checkpoints/max_psnr/epoch=99.ckpt
```

![Alt text](graph/per_frame_quality_max_psnr.png)

#### testing
```
python Tester.py --DR LAB4_Dataset/ --save_root outputs/max_psnr/ --test --ckpt checkpoints/max_psnr/epoch=99.ckpt
```

![Alt text](screenshots/max_psnr.png)