import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import numpy as np
import cv2
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
import argparse
from ssd import build_ssd
import torch_mlu
import torch_mlu.core.mlu_model as ct
import warnings 
warnings.filterwarnings("ignore")
print("Use MLU Deivce ......")
parser = argparse.ArgumentParser()
parser.add_argument('--image_path', default='demo/example.jpg',type=str)
parser.add_argument('--model_path', default='./mlu_weights/ssd300_VOC_8000.pth',type=str)
parser.add_argument('--output_path', default='../data/output/infer',type=str)
args = parser.parse_args()
net = build_ssd('test', 300, 21)    # initialize SSD
net.load_weights(args.model_path)
from data import VOCDetection, VOCAnnotationTransform
# here we specify year (07 or 12) and dataset ('test', 'val', 'train') 
#VOC_ROOT="../"
#testset = VOCDetection(VOC_ROOT, [('2007', 'val')], None, VOCAnnotationTransform())
#img_id = 0
#image = testset.pull_image(img_id)
if not os.path.exists(args.output_path):
    os.makedirs(args.output_path, mode=0o777, exist_ok=True)
image=cv2.imread(args.image_path)
# rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# rgb_image = cv2.cvtColor(image)
x = cv2.resize(image, (300, 300)).astype(np.float32)
x -= (104.0, 117.0, 123.0)
x = x.astype(np.float32)
x = x[:, :, ::-1].copy()
x = torch.from_numpy(x).permute(2, 0, 1)
xx = Variable(x.unsqueeze(0))     # wrap tensor in Variable
if torch.cuda.is_available():
    xx = xx.cuda()
y = net(xx)
from data import VOC_CLASSES as labels
# top_k=10
detections = y.data
# scale each detection back up to the image
scale = torch.Tensor(image.shape[1::-1]).repeat(2)
colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
for i in range(detections.size(1)):
    j = 0
    while detections[0,i,j,0] >= 0.6:
        score = detections[0,i,j,0]
        label_name = labels[i-1]
        display_txt = '%s: %.2f'%(label_name, score)
        pt = (detections[0,i,j,1:]*scale).cpu().numpy()
        color = colors[i]
        a=(round(pt[0]), round(pt[1]))
        b=(round(pt[2]), round(pt[3]))
        c=(round(pt[0]), round(pt[1])-2)
        cv2.rectangle(image, a, b, color, 2)
        font=cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, display_txt, c, font, 0.5, color, 1)
        cv2.imwrite(args.output_path+"/result.jpg",image)
        print(display_txt)
        j+=1

