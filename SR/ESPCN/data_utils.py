import argparse
import os
from os import listdir
from os.path import join

from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import Compose, CenterCrop, Resize
import torchvision.transforms as transforms
from tqdm import tqdm


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.JPG', '.JPEG', '.PNG'])


def is_video_file(filename):
    return any(filename.endswith(extension) for extension in ['.mp4', '.avi', '.mpg', '.mkv', '.wmv', '.flv'])


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def input_transform(crop_size, upscale_factor):
    valid_size = calculate_valid_crop_size(crop_size, upscale_factor)
    return transforms.Compose([
        transforms.CenterCrop(valid_size),
        transforms. Resize(valid_size // upscale_factor, interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor()
    ])


def target_transform(crop_size, upscale_factor):
    valid_size = calculate_valid_crop_size(crop_size, upscale_factor)
    return transforms.Compose([
        transforms.CenterCrop(valid_size),
        transforms.ToTensor()
    ])


class DatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, upscale_factor, input_transform=None, target_transform=None):
        super(DatasetFromFolder, self).__init__()
        self.upscale_factor = upscale_factor
        self.image_dir = dataset_dir
        self.input_transform = input_transform
        self.target_transform = target_transform

        self.image_filenames = [join(self.image_dir, x) for x in listdir(self.image_dir) if is_image_file(x)]
    
    def __getitem__(self, index):
        image = Image.open(self.image_filenames[index]).convert('YCbCr')
        y, cb, cr = image.split()
        target = y.copy()

        if self.input_transform:
            y = self.input_transform(y) 
        if self.target_transform:
            target = self.target_transform(target)

        assert y.dim() == 3 and target.dim() == 3, "Output must be [C, H, W]"
        return y, target

    def __len__(self):
        return len(self.image_filenames)


def generate_dataset(data_type, upscale_factor):
    images_name = [x for x in listdir('/dataset/VOC2012-ESPCN/' + data_type) if is_image_file(x)]
    crop_size = calculate_valid_crop_size(256, upscale_factor)
    lr_transform = input_transform(crop_size, upscale_factor)
    hr_transform = target_transform(crop_size, upscale_factor)

    root = 'data/' + data_type
    if not os.path.exists(root):
        os.makedirs(root)
    path = root + '/SRF_' + str(upscale_factor)
    if not os.path.exists(path):
        os.makedirs(path)
    image_path = path + '/data'
    if not os.path.exists(image_path):
        os.makedirs(image_path)
    target_path = path + '/target'
    if not os.path.exists(target_path):
        os.makedirs(target_path)

    for image_name in tqdm(images_name, desc='generate ' + data_type + ' dataset with upscale factor = '
            + str(upscale_factor) + ' from VOC2012'):
        image = Image.open('/dataset/VOC2012-ESPCN/' + data_type + '/' + image_name)
        target = image.copy()
        image = lr_transform(image)
        target = hr_transform(target)

        image_pil = transforms.ToPILImage()(image)
        target_pil = transforms.ToPILImage()(target)

        image_pil.save(os.path.join(image_path, image_name))
        target_pil.save(os.path.join(target_path, image_name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate Super Resolution Dataset')
    parser.add_argument('--upscale_factor', default=3, type=int, help='super resolution upscale factor')
    opt = parser.parse_args()
    UPSCALE_FACTOR = opt.upscale_factor

    generate_dataset(data_type='train', upscale_factor=UPSCALE_FACTOR)
    generate_dataset(data_type='val', upscale_factor=UPSCALE_FACTOR)

