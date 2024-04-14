from ultralytics import YOLO
import argparse


def main(args):
    if args.train:
        model = YOLO('yolov8l.yaml')
        model.train(data='dataset/data.yaml', epochs=1000, imgsz=416)
    if args.val:
        model = YOLO(args.val)
        model.val()
    if args.test:
        model = YOLO(args.test)
        model.predict('dataset/test/images', save=True, imgsz=416, conf=0.5)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--val', type=str, default='runs/detect/train8/weights/best.pt', help='validate')
    parser.add_argument('--train', type=str, default=None, help='train')
    parser.add_argument('--test', type=str, default='runs/detect/train8/weights/best.pt', help='test')
    args = parser.parse_args()
    main(args)
