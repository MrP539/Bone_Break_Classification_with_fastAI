import pandas as pd
import matplotlib.pyplot as plt
import fastai.vision.all
import os
import shutil
import torch

fields = fastai.vision.all.DataBlock(
    blocks=(fastai.vision.all.ImageBlock,fastai.vision.all.CategoryBlock),
    get_items = fastai.vision.all.get_image_files,
    get_y = fastai.vision.all.parent_label,
    splitter = fastai.vision.all.RandomSplitter(valid_pct=0.2,seed=42),
    item_tfms=fastai.vision.all.RandomResizedCrop(224,min_scale = 0.5),
    batch_tfms=fastai.vision.all.aug_transforms()
)

loot_dir = r"data\Bone_Break_Classification_train"



dls = fields.dataloaders(loot_dir,device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),num_workers = 0,bs=64,shuffle = True)

csv_logger = fastai.vision.all.CSVLogger("result.csv")
learner = fastai.vision.all.vision_learner(dls=dls,arch=fastai.vision.all.resnet50,metrics=[fastai.vision.all.error_rate,fastai.vision.all.accuracy],cbs=csv_logger)
# lr = learner.lr_find()
# print(lr)
learner.fine_tune(epochs=100,freeze_epochs = 1,base_lr = 3e-3)
learner.remove_cb(csv_logger)
learner.export("model.pkl")

csv_path = learner.path/"result.csv"
df = pd.read_csv(csv_path)
df = pd.read_csv("result.csv")
df.columns = df.columns.str.strip()

plt.figure(figsize=(6,6))
plt.plot(df.epoch,df.train_loss,label = "train_loss",color = "blue",marker="x")
plt.plot(df.epoch,df.valid_loss,label = "valid_loss",color = "red",marker="x")
plt.xlabel("epochs")
plt.ylabel("loss")
plt.legend()
plt.title("Train and Validation Loss by Epoch (resnet34)")
plt.show()

plt.figure(figsize=(6,6))
plt.plot(df.epoch,df.error_rate,label = "error_rate", color = "blue",marker="x")
plt.xlabel("Epoch")
plt.ylabel("error_rate")
plt.title('Validation error_rate by Epoch (resnet34)')
plt.show()

plt.figure(figsize=(6,6))
plt.plot(df.epoch,df.accuracy,label = "accuracy", color = "blue",marker="x")
plt.xlabel("Epoch")
plt.ylabel("accuracy")
plt.title('Validation accuracy by Epoch (resnet34)')
plt.show()
