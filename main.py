from ultralytics import YOLO
# View your own model
if __name__=='__main__':
    model=YOLO('./StarEye.yaml')
    model.info()