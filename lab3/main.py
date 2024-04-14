from ultralytics import YOLO
import argparse


def main(args):
    model = YOLO('yolov8l-seg.yaml')
    model.train(data='dataset/data.yaml', epochs=1000)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--val', type=str, default='runs/detect/train6/weights/best.pt', help='validate')
    parser.add_argument('--train', type=str, default=None, help='train')
    parser.add_argument('--test', type=str, default='runs/detect/train8/weights/best.pt', help='test')
    args = parser.parse_args()
    main(args)
