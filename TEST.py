import cv2
import numpy as np

ix,iy = -1,-1
cor = []
stack = 0
x_upl = 0
y_upl = 0
x_upr = 0
y_upr = 0
x_dol = 0
y_dol = 0
x_dor = 0
y_dor = 0

def draw_circle(event, x,y, flags, param):
        global ix, iy, stack

        if event == cv2.EVENT_LBUTTONDOWN:
                ix, iy = x,y
        elif event == cv2.EVENT_LBUTTONUP:
                if stack == 0:
                        x_upl = np.abs(ix + x) / 2
                        y_upl = np.abs(iy + y) / 2
                        if x_upl != 0 and y_upl != 0 :
                                print("x_upl y_upl print")
                                print(x_upl, y_upl)
                                cor.insert(0, [x_upl, y_upl])
                                x_upl = 0
                                y_upl = 0
                                stack = 1
                                print(stack)
                elif stack == 1:
                        x_upr = np.abs(ix + x) / 2
                        y_upr = np.abs(iy + y) / 2
                        if x_upr != 0 and y_upr != 0 :
                                print("x_upr y_upr print")
                                print(x_upr, y_upr)
                                cor.insert(1, [x_upr, y_upr])
                                x_upr = 0
                                y_upr = 0
                                stack = 2
                                print(stack)
                elif stack == 2:
                        x_dor = np.abs(ix + x) / 2
                        y_dor = np.abs(iy + y) / 2
                        if x_dor != 0 and y_dor != 0:
                                print("x_dor y_dor print")
                                print(x_dor, y_dor)
                                cor.insert(2, [x_dor, y_dor])
                                x_dor = 0
                                y_dor = 0
                                x = 0
                                y = 0
                                stack = 3
                                print(stack)
                elif stack == 3:
                        x_dol = np.abs(ix + x) / 2
                        y_dol = np.abs(iy + y) / 2
                        if x_dol != 0 and y_dol != 0:
                                print("x_dol y_dol print")
                                print(x_dol, y_dol)
                                cor.insert(3, [x_dol, y_dol])
                                isObjectinLane(cor)
                                cor.clear()
                                x_dol = 0
                                y_dol = 0
                                stack = 0
                                print(stack)
                else:
                        print("Error!!!!")


def isObjectinLane(cor):
        #cv2.rectangle(img, (line1xmin, line1ymin), (line1xmax, line1ymax), (255, 0, 0), 3)
        # cv2.rectangle(img, (line2xmin, line2ymin), (line2xmax, line2ymax), (0, 255, 0), 3)
        # cv2.rectangle(img, (line3xmin, line3ymin), (line3xmax, line3ymax), (0, 0, 255), 3)
        line1poly = np.array(cor)
        line1poly = line1poly.reshape(-1, 1, 2)
        cv2.polylines(img, np.int32([line1poly]), True, (255,0,0))

img = cv2.imread("test_data/test1.jpg")
cv2.namedWindow('image')
cv2.setMouseCallback('image',draw_circle)

while True:
        cv2.imshow('image', img)
        k = cv2.waitKey(1) & 0xFF

cv2.destroyAllWindows()