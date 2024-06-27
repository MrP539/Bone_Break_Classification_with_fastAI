import torch
import fastai.vision.all
import pandas as pd
import matplotlib.pyplot as plt

fields = fastai.vision.all.DataBlock(
    blocks=(fastai.vision.all.ImageBlock,fastai.vision.all.CategoryBlock),
    get_items = fastai.vision.all.get_image_files,
    get_y = fastai.vision.all.parent_label,
    item_tfms=fastai.vision.all.Resize(224)
)

test_path = r"data\Bone Break Classification_test"

test_dls = fields.dataloaders(test_path,bs=64,shuffle = False,device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),num_workers = 0)

model_path = r"D:\machine_learning_AI_Builders\บท4\Classification\Bone_Break_Classification_with_fastAI\models\resnet34(100epochs)\model.pkl"
model = fastai.vision.all.load_learner(model_path)
preds,targs = model.get_preds(dl=test_dls)

accuracy = (preds.argmax(dim=1) == targs).float().mean()
print(f"Test Accuracy: {accuracy:.4f}")

# แสดงผลภาพที่มีการทำนายพร้อมกับ label
interp = fastai.vision.all.ClassificationInterpretation.from_learner(model, dl=test_dls)
interp.plot_top_losses(9, figsize=(15, 10))
plt.show() 
#คำนวณและแสดง confusion matrix
interp.plot_confusion_matrix(figsize=(10, 10))
plt.show() 


