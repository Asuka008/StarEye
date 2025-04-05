import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

# Official Parameters: https://docs.ultralytics.com/modes/train/#resuming-interrupted-trainings:~:text=a%20training%20run.-,Train%20Settings,-The%20training%20settings

if __name__ == '__main__':
    model = YOLO('./StarEye/models/StarEye.yaml')
    # model.load('yolov8n.pt') # loading pretrain weights
    model.train(data='./StarEye/dataset/data.yaml',
                cache=False,
                imgsz=640,
                epochs=350,
                batch=16,
                close_mosaic=0,
                workers=4,
                # device='0',
                optimizer='SGD', # using SGD
                # patience=0, # close earlystop
                # resume=True, # Universal checkpointing
                # amp=False, # close amp
                # fraction=0.2,
                project='runs/train',
                name='realhumansssstar', # Output file name
                )