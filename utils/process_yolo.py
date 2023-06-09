import win32gui
import win32ui
import win32con
import pygetwindow
import numpy
import cv2

def StartYOLOV8():
    count = 0
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


        # cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        # cv2.imwrite("img.jpg",img,[int(cv2.IMWRITE_JPEG_QUALITY), 100])
        # cv2.namedWindow('img') #命名窗口
        cv2.imshow("img", img)  # 显示
        cv2.waitKey(1)

        # 保存图片
        # screenshot.SaveBitmapFile(mem_dc, 'screenshot.png')

        # 释放临时内存
        mem_dc.DeleteDC()
        win32gui.DeleteObject(screenshot.GetHandle())

        count += 1
        print("截图了", count, "次")

        print("start yolov8")