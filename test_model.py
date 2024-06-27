import fastai
import torch
import fastai.vision.all
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os

model_path = r"D:\machine_learning_AI_Builders\บท4\Classification\Bone_Break_Classification_with_fastAI\models\resnet34(100epochs)\model.pkl"
model = fastai.vision.all.load_learner(model_path)
img_test_path = "test.jpg"
perd = model.predict(os.path.join(img_test_path))
class_name = list(perd)[0]
img_cv2 = cv2.imread(img_test_path)
img_cv2 = cv2.resize(img_cv2,(500,500))
cv2.putText(img_cv2,f"{class_name}",(20,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),4)

cv2.imshow("test",img_cv2)
cv2.waitKey(0)
cv2.destroyAllWindows()
