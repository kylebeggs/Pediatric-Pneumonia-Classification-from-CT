# Pediatric-Pneumonia-Classification-from-CT

The [train.py](train.py) file in the main directory of the repo is the only file of importance. There is a helper file with some plotting functions,
etc named [helper.py](helper.py) but everything is contained in the train file.

We make use of argparse here, so simply type ``` python train.py --help ``` to see the potential training settings as such:

```shell
> python train.py --help

optional arguments:
  -h, --help            show this help message and exit
  --name NAME           Name the model you are training.
  --pretrain            Initliaze the weights found from pretraining with ImageNet.
  --balance             Weight the classes according to their imbalance.
  --augment             Augment the training data.
  --epochs EPOCHS       Set max number of epochs.
  --lr LR, --learning-rate LR
                        Set the initial learning rate to be used with the reduced step scheduler.
```

The file outputs a TensorBoard file in the [runs/](runs) directory with the name which you gave as input as seen above. Launch the TensorBoard session with
```tensorboard --logdir=runs``` from the main directory of this repo. A log file is output containing the training settings and test set performance metrics such
as accuracy and F1 score in the [logs/](logs) directory. The model weights are saved on the last epoch into the [trained_models/](trained_models) directory named according to the 
name you provide when launching the training. The train file automatically detects if you have a CUDA enabled device and selects it for training. 
If not, it uses the CPU.
