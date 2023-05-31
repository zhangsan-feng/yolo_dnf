import win32gui
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

        # 展示图片 使用 numpy 转换
        signedIntsArray = screenshot.GetBitmapBits(True)
        img = numpy.frombuffer(signedIntsArray, dtype='uint8')
        img.shape = (height, width, 4)
        res = model.predict('',device='0',)

        # 展示图片
        # cv2.imshow("img", img)  # 显示
        cv2.imshow("img", res[0].plot())  # 显示
        cv2.waitKey(1)

        # 释放临时内存
        mem_dc.DeleteDC()
        win32gui.DeleteObject(screenshot.GetHandle())





if __name__ == '__main__':

    model = "4"
    # model = ""
    # model = ""
    # model = ""
    # model = ""

    TestYOLOModel(model)

