import time
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from ultralytics import YOLO





def StartRepeatTrainYOLOV8():

    data = "./dnf.yaml"
    device= "0"
    cache = True
    workers = 8
    batch = 16
    epochs = 500
    print("train start first train ")
    model = 'yolov8n.pt'
    args = dict(model=model,
                data=data,
                device=device,
                cache=cache,
                batch=batch,
                workers=workers,
                epochs=epochs
                )
    YOLO(model).train(**args)
    print(" first train success")




if __name__ == '__main__':


    print("start yolo Repeat train")
    StartRepeatTrainYOLOV8()
    print("yolo train success")