from ultralytics import YOLO
import argparse


def main(args):
    if args.train:
        model = YOLO('yolov8n.yaml')
        model.train(data='mushrooms2/data.yaml', epochs=1000, imgsz=416, batch=32)
    if args.val:
        if not args.train:
            model = YOLO(args.val)
        model.val(data='mushrooms2/data.yaml')
    if args.test:
        if not args.train:
            model = YOLO(args.test)
        model.predict('mushrooms2/test/images', save=True, imgsz=640, conf=0.5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--val', type=str, default='runs/detect/train14/weights/best.pt', help='validate')
    parser.add_argument('--train', type=str, default=None, help='train')
    parser.add_argument('--test', type=str, default='runs/detect/train14/weights/best.pt', help='test')
    args = parser.parse_args()
    main(args)
