from __future__ import  absolute_import

import torch
import time
from tqdm import tqdm

from utils.config import opt
from data.dataset import Dataset, TestDataset, inverse_normalize
from model import FasterRCNNVGG16
from torch.utils import data as data_
from trainer import FasterRCNNTrainer
from utils import array_tool as at
from utils.eval_tool import eval_detection_voc

# fix for ulimit
# https://github.com/pytorch/pytorch/issues/973#issuecomment-346405667
import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (20480, rlimit[1]))


def eval(dataloader, faster_rcnn, test_num=10000):
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    gt_bboxes, gt_labels, gt_difficults = list(), list(), list()
    total_inference_time = 0.0
    total_samples = 0
    for ii, (imgs, sizes, gt_bboxes_, gt_labels_, gt_difficults_) in tqdm(enumerate(dataloader)):
        sizes = [sizes[0][0].item(), sizes[1][0].item()]
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        infer_start = time.time()
        pred_bboxes_, pred_labels_, pred_scores_ = faster_rcnn.predict(imgs, [sizes])
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        infer_end = time.time()
        total_inference_time += (infer_end - infer_start)
        total_samples += 1
        gt_bboxes += list(gt_bboxes_.numpy())
        gt_labels += list(gt_labels_.numpy())
        gt_difficults += list(gt_difficults_.numpy())
        pred_bboxes += pred_bboxes_
        pred_labels += pred_labels_
        pred_scores += pred_scores_
        if ii == test_num: break

    result = eval_detection_voc(
        pred_bboxes, pred_labels, pred_scores,
        gt_bboxes, gt_labels, gt_difficults,
        use_07_metric=True)

    # Print inference throughput and latency
    if total_samples > 0:
        avg_latency_ms = (total_inference_time / total_samples) * 1000
        throughput = total_samples / total_inference_time
        print(f'\nInference throughput: {throughput:.2f} samples/s')
        print(f'Average inference latency: {avg_latency_ms:.2f} ms/sample')
        print(f'Total inference time: {total_inference_time:.2f} s')
    if torch.cuda.is_available():
        print(f'GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB')
        print(f'GPU memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB')

    return result


def main(**kwargs):
    opt._parse(kwargs)

    testset = TestDataset(opt)
    test_dataloader = data_.DataLoader(testset,
                                       batch_size=1,
                                       num_workers=opt.test_num_workers,
                                       shuffle=False, \
                                       pin_memory=True
                                       )
    faster_rcnn = FasterRCNNVGG16()
    print('model construct completed')
    trainer = FasterRCNNTrainer(faster_rcnn).cuda()
    if opt.load_path:
        trainer.load(opt.load_path)
        print('load pretrained model from %s' % opt.load_path)
    else:
        print("ckpt path not found")
        return 

    eval_result = eval(test_dataloader, faster_rcnn, test_num=opt.test_num)

    print(f"mAP: {eval_result['map']}")



if __name__ == '__main__':
    import sys
    argv = sys.argv[1:]
    # Skip command name if present (e.g. 'main')
    if argv and not argv[0].startswith('--'):
        argv = argv[1:]
    kwargs = {}
    for arg in argv:
        if arg.startswith('--'):
            key, _, val = arg[2:].partition('=')
            kwargs[key.replace('-', '_')] = val
    main(**kwargs)