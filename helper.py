# helper functions

import os
import matplotlib.pyplot as plt
import PIL
import numpy as np
from torchvision.utils import make_grid
import torch


def load_images(data_dir):
    print("\n"+data_dir)
    class_names = sorted(
        x for x in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, x))
    )
    num_class = len(class_names)
    image_files = [
        [
            os.path.join(data_dir, class_names[i], x)
            for x in os.listdir(os.path.join(data_dir, class_names[i]))
        ]
        for i in range(num_class)
    ]
    num_each = [len(image_files[i]) for i in range(num_class)]
    image_files_list = []
    image_class = []
    for i in range(num_class):
        image_files_list.extend(image_files[i])
        image_class.extend([i] * num_each[i])
    num_total = len(image_class)

    print(f"Total image count: {num_total}")
    print(
        f"Label names: {class_names} corresponding to {[i for i in range(num_class)]}"
    )
    print(f"Label counts: {num_each}")

    class_info = dict(zip(class_names, num_each))

    return image_files_list, image_class, class_info


def visualize_raw_data(input, label):
    assert len(input) == len(label)
    plt.subplots(3, 3, figsize=(8, 8))
    for i, k in enumerate(np.random.randint(len(input), size=9)):
        im = PIL.Image.open(input[k])
        arr = np.array(im)
        plt.subplot(3, 3, i + 1)
        plt.xlabel("Sample %i, Label: %i" % (k, label[k]))
        plt.imshow(arr, cmap="gray", vmin=0, vmax=255)
    plt.tight_layout()
    plt.show()

def show_misclassified(model, loader, device, class_names):
    incorrect_examples = []

    model.eval()
    for data,target in loader:
        data , target = data.to(device), target.to(device)
        output = model(data)
        _, pred = torch.max(output,1)
        idxs_mask = ((pred == target) == False).nonzero()
        incorrect_examples.append(data[idxs_mask].cpu().numpy())
    
    fig = plt.figure(figsize=(8, 8))

    for idx in np.arange(9):
        ax = fig.add_subplot(3, 3, idx+1, xticks=[], yticks=[])
        img = incorrect_examples[idx][idx]
        img = img/2 + 0.5
        img = np.clip(img, 0, 1)
        plt.imshow(img[0,0,:,:])
        ax.set_title(f"{class_names[pred[idx]]}: x%\n (label: {class_names[target[idx]]})")
        #color=(“green” if pred[idx]==target[idx].item() else “red”))


def show_images(images, nmax=64):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(make_grid((images.detach()[:nmax]), nrow=8).permute(1, 2, 0))

def show_batch(dl, nmax=25):
    for images in dl:
        show_images(images, nmax)
        break

def log_metrics(writer, epoch, loss, acc, lr): #, auc):
    writer.add_scalar("loss", loss, epoch + 1)
    writer.add_scalar("accuracy", acc, epoch + 1)
    writer.add_scalar("learning rate", lr, epoch + 1)
    #writer.add_scalar("auc-roc", auc, epoch + 1)
