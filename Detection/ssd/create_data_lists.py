import argparse
from utils import create_data_lists

def main():
    parser = argparse.ArgumentParser(description="Generate data lists for VOC datasets.")
    parser.add_argument('--voc07_path', type=str, default='../data/VOCdevkit/VOC2007',
                        help="Path to the VOC 2007 dataset")
    parser.add_argument('--voc12_path', type=str, default='../data/VOCdevkit/VOC2012',
                        help="Path to the VOC 2012 dataset")
    parser.add_argument('--output_folder', type=str, default='./data',
                        help="Directory to save output data lists")

    args = parser.parse_args()

    create_data_lists(voc07_path=args.voc07_path,
                      voc12_path=args.voc12_path,
                      output_folder=args.output_folder)

if __name__ == '__main__':
    main()
