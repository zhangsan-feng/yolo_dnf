import time

import win32gui
import win32print
import win32ui
import win32con
import pygetwindow
import numpy
import cv2
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from ultralytics import YOLO







def TestYOLOModel(model):

    print("start load yolo model")
    # model = YOLO('../train_model/yolov8n.pt')
    model = YOLO('../train_model/runs/detect/train' + str(model) + '/weights/best.pt')
    print("load model success ")

    window_process = pygetwindow.getActiveWindow()

    title = window_process.title
    left = window_process.left
    top = window_process.top
    width = window_process.width
    height = window_process.height

    d = {
        "title :": title,
        "left  :": left,
        "top   :": top,
        "width :": width,
        "height:": height,
    }
    print("get windows app process info :", d)


    while True:
        hdesktop = win32gui.GetDesktopWindow()
        desktop_dc = win32gui.GetWindowDC(hdesktop)
        img_dc = win32ui.CreateDCFromHandle(desktop_dc)
        mem_dc = img_dc.CreateCompatibleDC()
        screenshot = win32ui.CreateBitmap()
        screenshot.CreateCompatibleBitmap(img_dc, width, height)
        mem_dc.SelectObject(screenshot)
        mem_dc.BitBlt((0, 0), (width, height), img_dc, (left, top), win32con.SRCCOPY)

        # 展示图片 使用 numpy 转换图片的字节流 --> BGR 图片格式
        signedIntsArray = screenshot.GetBitmapBits(True)
        img = numpy.frombuffer(signedIntsArray, dtype='uint8')
        img.shape = (height, width, 4)

        # opencv 转换图片格式  BGR --> RGB
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        res = model.predict(image_rgb,device='0')
        # res = model.predict(image_rgb,device='cpu')

        # opencv 转换图片格式 RGB --> BGR
        image_bgr = cv2.cvtColor(res[0].plot(), cv2.COLOR_RGB2BGR)

        # 展示图片
        cv2.imshow("img",image_bgr)
        # cv2.imshow("img", img)
        cv2.waitKey(1)

        # 释放临时内存
        mem_dc.DeleteDC()
        win32gui.DeleteObject(screenshot.GetHandle())





if __name__ == '__main__':

    model = "10"
    # model = ""
    # model = ""
    # model = ""
    # model = ""

    TestYOLOModel(model)
    # print(get_real_resolution())
