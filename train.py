#%%
# training script for pneumonia classification task
from torch.utils.tensorboard import SummaryWriter
import PIL
import os
import logging
from tqdm import tqdm
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import torch
import torchvision.transforms as T
from torchvision import datasets
from torch.utils.data import DataLoader
import torchvision.models as models
from monai.metrics import ROCAUCMetric
from monai.transforms import RandRotate, RandZoom, RandFlip, ScaleIntensity

# my own functions
from helper import load_images, visualize_raw_data, show_batch, log_metrics, show_misclassified

import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', type=str, required=True, help='Name the model you are training.')
    parser.add_argument('--pretrain', action='store_true', help='Initliaze the weights found from pretraining with ImageNet.')
    parser.add_argument('--balance', action='store_true' , help='Weight the classes according to their imbalance.')
    parser.add_argument('--augment', action='store_true' , help='Augment the training data.')
    parser.add_argument('--epochs', default=100, type=int , help='Set max number of epochs.')
    parser.add_argument('--lr', '--learning-rate', default=5e-5, type=float, help='Set the initial learning rate to be used with the reduced step scheduler.')
    
    args = parser.parse_args()

    return args


#%% ---------------------------------------------------------------------------
# user input parameters

args = parse_args()

run_name = args.name

if torch.cuda.is_available():
    device = torch.device("cuda")
    workers = 1  # im not sure why, but it yelled at me when tryingn to use more with cuda
else:
    device = torch.device("cpu")
    workers = 8

print("\nUsing ", device)

# model
use_pretrained = args.pretrain
balance = args.balance

# dataloader
augment = args.augment
target_img_size = 224  # start with this as resnet18 originally used this
batch_size = 8

# data stores
data_dir_train = "data/train"
data_dir_val = "data/val"
data_dir_test = "data/test"

# training
decay_learning_rate = True
learning_rate = args.lr
max_epochs = args.epochs
val_interval = 1  # after how many epochs to run validation set

logging.basicConfig(filename="logs/"+run_name+".log",
                            filemode='a',
                            format='%(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG)

# print info to screen
logging.info("Running "+run_name)
logging.info(f"  pretrained: {use_pretrained}")
logging.info(f"  balance: {balance}")
logging.info(f"  augment: {augment}")
logging.info(f"  learning rate: {learning_rate}")
logging.info(f"  batch size: {batch_size}")

#%% ---------------------------------------------------------------------------
# load images from data directories

root_dir = os.getcwd()

train_input, train_label, train_class_info = load_images(data_dir_train)
val_input, val_label, val_class_info = load_images(data_dir_val)
test_input, test_label, test_class_info = load_images(data_dir_test)

# make sure number of classes are correct as sanity check, set global num_classes if all good
if len(train_class_info) == len(val_class_info) == len(test_class_info):
    num_class = len(train_class_info)
else:
    raise ValueError(
        f"Number of classes in train, test, val is not the same:\n train: {len(train_class_info)}\n val: {len(val_class_info)}\n test: {len(test_class_info)}"
    )


#%% ---------------------------------------------------------------------------
# define transforms

class SquarePad:
    def __call__(self, image):
        max_wh = max(image.size)
        p_left, p_top = [(max_wh - s) // 2 for s in image.size]
        p_right, p_bottom = [
            max_wh - (s + pad) for s, pad in zip(image.size, [p_left, p_top])
        ]
        padding = (p_left, p_top, p_right, p_bottom)
        return T.functional.pad(image, padding, 0, 'constant')

if augment:
    train_transforms = T.Compose([
        SquarePad(),  # pad the image to be square so all samples are the same size
        T.Resize(target_img_size),  # resize for computational efficiency
        T.Grayscale(),  # ensure only 1 channel
        T.ToTensor(),
        ScaleIntensity(),  # normalize the data
        RandRotate(range_x=np.pi / 12, prob=0.5, keep_size=True),
        RandFlip(spatial_axis=0, prob=0.5),
        RandFlip(spatial_axis=1, prob=0.5),
        RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5),
    ])
else:
    train_transforms = T.Compose([
        SquarePad(),  # pad the image to be square so all samples are the same size
        T.Resize(target_img_size),  # resize for computational efficiency
        T.Grayscale(),  # ensure only 1 channel
        T.ToTensor(),
        ScaleIntensity()  # normalize the data
    ])

val_transforms = T.Compose([
    SquarePad(),  # pad the image to be square so all samples are the same size
    T.Resize(target_img_size),  # resize for computational efficiency
    T.Grayscale(),  # ensure only 1 channel
    T.ToTensor(),
    ScaleIntensity()  # normalize the data
])

test_transforms = val_transforms  # test should be same procedure as validation


#%% ---------------------------------------------------------------------------
# view transformed images

# get normalized image
img = PIL.Image.open(train_input[np.random.randint(0, len(train_input))])
img_normalized = train_transforms(img)

# convert normalized image to numpy
# array
img_np = np.array(img)
img_np_norm = np.array(img_normalized)
img_np_hist = np.ma.masked_where(img_np <= 0, img_np)
img_np_norm_hist = np.ma.masked_where(img_np_norm <= 0, img_np_norm)

# plot the pixel values histogram
plt.subplots(2, 2, figsize=(8, 7))
plt.subplot(2, 2, 1)
plt.hist(img_np_hist.ravel(), bins=50, density=True)
plt.xlabel("pixel values")
plt.ylabel("relative frequency")
plt.title("Regular Image")
plt.subplot(2, 2, 2)
plt.hist(img_np_norm_hist.ravel(), bins=50, density=True)
plt.xlabel("pixel values")
plt.ylabel("relative frequency")
plt.title("Transformed Image")
# show image
plt.subplot(2, 2, 3)
plt.imshow(img_np, cmap="gray")
plt.subplot(2, 2, 4)
plt.imshow(img_np_norm[0, :, :], cmap="gray")

#%% ---------------------------------------------------------------------------
# create dataloaders

train_ds = datasets.ImageFolder(data_dir_train, transform=train_transforms)
train_loader = DataLoader(
    train_ds,
    batch_size=batch_size,
    shuffle=True,
    num_workers=workers,
    pin_memory=torch.cuda.is_available(),
)

val_ds = datasets.ImageFolder(data_dir_val, transform=val_transforms)
val_loader = DataLoader(
    val_ds,
    batch_size=batch_size,
    num_workers=workers,
    pin_memory=torch.cuda.is_available(),
)

test_ds = datasets.ImageFolder(data_dir_test, transform=test_transforms)
test_loader = DataLoader(
    test_ds,
    batch_size=batch_size,
    num_workers=workers,
    pin_memory=torch.cuda.is_available(),
)

#%% ---------------------------------------------------------------------------
# define model and parameters

if use_pretrained:
    model = models.resnet18(pretrained=True)
else:
    model = models.resnet18()

# make changes to model as it is greyscale and binary classification
model.conv1 = torch.nn.Conv2d(1,
                              64,
                              kernel_size=(7, 7),
                              stride=(2, 2),
                              padding=(3, 3),
                              bias=False)
model.fc = torch.nn.Linear(in_features=512, out_features=2, bias=True)
model.to(device)

auc_metric = ROCAUCMetric()

optimizer = torch.optim.Adam(model.parameters(), learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)

# weighting loss function for class imbalance
if balance:
    total = train_class_info["NORMAL"] + train_class_info["PNEUMONIA"]
    weight0 = train_class_info['PNEUMONIA'] / total
    weight1 = train_class_info['NORMAL'] / total
    class_weights = torch.tensor([weight0, weight1], dtype=torch.float).cuda()
    loss_function = torch.nn.CrossEntropyLoss(weight=class_weights)
else:
    loss_function = torch.nn.CrossEntropyLoss()

#%% ---------------------------------------------------------------------------
# train the model

# start a typical PyTorch training
writer_train = SummaryWriter('runs/' + run_name + '/training')
writer_val = SummaryWriter('runs/' + run_name + '/val')

epoch_loop = tqdm(range(max_epochs), total=max_epochs, leave=False)
for epoch in epoch_loop:
    model.train()
    current_learning_rate = optimizer.param_groups[0]["lr"]

    # perform epoch training step
    train_loss = 0
    num_correct = 0
    acc_count = 0
    step = 0
    for batch_data in train_loader:
        step += 1
        train_inputs, train_labels = batch_data[0].to(
            device), batch_data[1].to(device)
        optimizer.zero_grad()

        train_outputs = model(train_inputs)
        loss = loss_function(train_outputs, train_labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        value = torch.eq(train_outputs.argmax(dim=1), train_labels)
        acc_count += len(value)
        num_correct += value.sum().item()

    # calculate metrics
    train_loss /= step
    train_acc = num_correct / acc_count

    log_metrics(writer_train, epoch, train_loss, train_acc, current_learning_rate)
    
    if decay_learning_rate:
        scheduler.step()

    if (epoch + 1) % val_interval == 0:
        model.eval()
        with torch.no_grad():
            val_loss = 0
            num_correct = 0
            acc_count = 0
            step = 0
            for val_data in val_loader:
                step += 1
                val_inputs, val_labels = val_data[0].to(
                    device), val_data[1].to(device)

                val_outputs = model(val_inputs)
                loss = loss_function(val_outputs, val_labels)
                val_loss += loss.item()

                value = torch.eq(val_outputs.argmax(dim=1), val_labels)
                acc_count += len(value)
                num_correct += value.sum().item()

            # calculate metrics
            val_loss /= step
            val_acc = num_correct / acc_count

        log_metrics(writer_val, epoch, val_loss, val_acc, current_learning_rate)
        epoch_loop.set_description(f"Epoch ")
        epoch_loop.set_postfix(train_acc=train_acc, val_acc=val_acc)


torch.save(model.state_dict(), "trained_models/" + run_name + ".pth")

writer_train.flush()
writer_val.close()
writer_train.flush()
writer_val.close()

#%% ---------------------------------------------------------------------------
# evaluate model

model.load_state_dict(torch.load("trained_models/" + run_name + ".pth"))
model.eval()
y_true = []
y_pred = []
with torch.no_grad():
    for test_data in test_loader:
        test_images, test_labels = (
            test_data[0].to(device),
            test_data[1].to(device),
        )
        pred = model(test_images).argmax(dim=1)
        for i in range(len(pred)):
            y_true.append(test_labels[i].item())
            y_pred.append(pred[i].item())

class_names = [class_name for class_name in test_class_info.keys()]            
logging.info(classification_report(y_true, y_pred, target_names=class_names, digits=4))
logging.shutdown()


#%% ---------------------------------------------------------------------------
# display cases of mis-classification

#show_misclassified(model, test_loader, device, class_names)

incorrect_examples = []

model.eval()
for data, label in test_loader:
    data , label = data.to(device), label.to(device)
    output = model(data)
    _, predictions = torch.max(output,dim=1)
    for sample in range(data.shape[0]):
        if(label[sample]!=predictions[sample]):
            incorrect_examples.append(data[sample].cpu())

 
incorrect_examples = random.sample(incorrect_examples, 4)

fig = plt.figure(figsize=(8, 8))

for idx in range(len(incorrect_examples)):
    img = np.array(incorrect_examples[idx])
    ax = fig.add_subplot(2, 2, idx+1, xticks=[], yticks=[])
    #img = img/2 + 0.5
    #img = np.clip(img, 0, 1)
    plt.imshow(img[0,:,:], cmap="gray")
    #ax.set_title(f"{class_names[predictions[idx]]}: x%\n (label: {class_names[label[idx]]})")
    #color=(“green” if pred[idx]==target[idx].item() else “red”))


#%% ---------------------------------------------------------------------------
