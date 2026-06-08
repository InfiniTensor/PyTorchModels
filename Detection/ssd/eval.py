import argparse
import time
import torch
from utils import *
from datasets import PascalVOCDataset
from tqdm import tqdm
from pprint import PrettyPrinter

# Good formatting when printing the APs for each class and mAP
pp = PrettyPrinter()

# Command-line argument parsing
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SSD model")
    
    # Adding arguments
    parser.add_argument('--data', type=str, default='./data', help='Path to dataset folder')
    parser.add_argument('--keep_difficult', type=bool, default=True, help='Whether to keep difficult objects in mAP calculation')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for evaluation')
    parser.add_argument('--workers', type=int, default=4, help='Number of workers for DataLoader')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cuda', help='Device to run the model on (cpu or cuda)')
    parser.add_argument('--checkpoint', type=str, default='./checkpoint_ssd300.pth.tar', help='Path to model checkpoint')
    parser.add_argument('--max_batches', type=int, default=0, help='Max batches for eval (0=all)')

    return parser.parse_args()

# Main evaluation function
def evaluate(test_loader, model, max_batches=0):
    """
    Evaluate.

    :param test_loader: DataLoader for test data
    :param model: model
    """

    # Make sure it's in eval mode
    model.eval()

    # Lists to store detected and true boxes, labels, scores
    det_boxes = list()
    det_labels = list()
    det_scores = list()
    true_boxes = list()
    true_labels = list()
    true_difficulties = list()  # it is necessary to know which objects are 'difficult', see 'calculate_mAP' in utils.py

    total_inference_time = 0.0
    total_samples = 0
    with torch.no_grad():
        # Batches
        for i, (images, boxes, labels, difficulties) in enumerate(tqdm(test_loader, desc='Evaluating')):
            if max_batches > 0 and i >= max_batches:
                break
            images = images.to(device)  # (N, 3, 300, 300)
            batch_size_i = images.size(0)

            # Forward prop.
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            inference_start = time.time()
            predicted_locs, predicted_scores = model(images)

            # Detect objects in SSD output
            det_boxes_batch, det_labels_batch, det_scores_batch = model.detect_objects(predicted_locs, predicted_scores,
                                                                                       min_score=0.01, max_overlap=0.45,
                                                                                       top_k=200)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            inference_end = time.time()
            total_inference_time += (inference_end - inference_start)
            total_samples += batch_size_i

            # Store this batch's results for mAP calculation
            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]
            difficulties = [d.to(device) for d in difficulties]

            det_boxes.extend(det_boxes_batch)
            det_labels.extend(det_labels_batch)
            det_scores.extend(det_scores_batch)
            true_boxes.extend(boxes)
            true_labels.extend(labels)
            true_difficulties.extend(difficulties)

            # Print cumulative throughput after each batch
            if i > 0:
                tput = total_samples / total_inference_time if total_inference_time > 0 else 0
                avg_lat = (total_inference_time / total_samples) * 1000 if total_samples > 0 else 0
                print(f'Inference throughput: {tput:.2f} samples/s', flush=True)
                print(f'Average inference latency: {avg_lat:.2f} ms/sample', flush=True)

    # Print inference throughput and latency BEFORE mAP calculation
    if total_samples > 0:
        avg_latency_ms = (total_inference_time / total_samples) * 1000
        throughput = total_samples / total_inference_time
        print(f'\nInference throughput: {throughput:.2f} samples/s', flush=True)
        print(f'Average inference latency: {avg_latency_ms:.2f} ms/sample', flush=True)
        print(f'Total inference time: {total_inference_time:.2f} s', flush=True)

        # Calculate mAP
        APs, mAP = calculate_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels, true_difficulties)

        # Print AP for each class
        pp.pprint(APs)

        print('\nMean Average Precision (mAP): %.3f' % mAP, flush=True)
    if torch.cuda.is_available():
        print(f'GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB')
        print(f'GPU memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB')

if __name__ == '__main__':
    # Parse command-line arguments
    args = parse_args()

    # Set device (CPU or GPU)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Load model checkpoint
    checkpoint = torch.load(args.checkpoint, weights_only=False)
    model = checkpoint['model']
    model = model.to(device)

    # Switch to eval mode
    model.eval()

    # Load test data
    test_dataset = PascalVOCDataset(args.data,
                                    split='test',
                                    keep_difficult=args.keep_difficult)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                              collate_fn=test_dataset.collate_fn, num_workers=args.workers, pin_memory=True)

    # Evaluate the model
    evaluate(test_loader, model, max_batches=args.max_batches)

