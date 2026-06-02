import argparse
import os
import time
from os import listdir

import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision.transforms import ToTensor
from tqdm import tqdm

from data_utils import is_image_file
from model import Net

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test Super Resolution')
    parser.add_argument('--upscale_factor', default=3, type=int, help='super resolution upscale factor')
    parser.add_argument('--model_name', default='epoch_3_100.pt', type=str, help='super resolution model name')
    opt = parser.parse_args()

    UPSCALE_FACTOR = opt.upscale_factor
    MODEL_NAME = opt.model_name

    path = 'data/images/'
    images_name = [x for x in listdir(path) if is_image_file(x)]
    model = Net(upscale_factor=UPSCALE_FACTOR)
    if torch.cuda.is_available():
        model = model.cuda()
    model_path = 'epochs/' + MODEL_NAME
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, weights_only=True))
        print(f'Loaded weights from {model_path}')
    else:
        print(f'WARNING: {model_path} not found, using random weights for throughput benchmark')

    out_path = 'results/' + str(UPSCALE_FACTOR) + '/'
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    total_inference_time = 0.0
    total_samples = 0
    for image_name in tqdm(images_name, desc='convert LR images to HR images'):

        img = Image.open(path + image_name).convert('YCbCr')
        y, cb, cr = img.split()
        image = Variable(ToTensor()(y)).view(1, -1, y.size[1], y.size[0])
        if torch.cuda.is_available():
            image = image.cuda()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        infer_start = time.time()
        out = model(image)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        infer_end = time.time()
        total_inference_time += (infer_end - infer_start)
        total_samples += 1

        out = out.cpu()
        out_img_y = out.data[0].numpy()
        out_img_y *= 255.0
        out_img_y = out_img_y.clip(0, 255)
        out_img_y = Image.fromarray(np.uint8(out_img_y[0]), mode='L')
        out_img_cb = cb.resize(out_img_y.size, Image.BICUBIC)
        out_img_cr = cr.resize(out_img_y.size, Image.BICUBIC)
        out_img = Image.merge('YCbCr', [out_img_y, out_img_cb, out_img_cr]).convert('RGB')
        out_img.save(out_path + image_name)

    # Print inference throughput and latency
    if total_samples > 0:
        avg_latency_ms = (total_inference_time / total_samples) * 1000
        throughput = total_samples / total_inference_time
        print(f'\nInference throughput: {throughput:.2f} images/s')
        print(f'Average inference latency: {avg_latency_ms:.2f} ms/image')
        print(f'Total inference time: {total_inference_time:.2f} s')
    if torch.cuda.is_available():
        print(f'GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB')
        print(f'GPU memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB')
