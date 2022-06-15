import cv2
import numpy as np
import os
pts = []


def draw_roi(event, x, y, flags, param):
    img2 = img.copy()
    if event == cv2.EVENT_LBUTTONDOWN:
        pts.append((x, y))
    if event == cv2.EVENT_RBUTTONDOWN:
        pts.pop()
    if event == cv2.EVENT_MBUTTONDOWN:
        mask = np.zeros(img.shape, np.uint8)
        points = np.array(pts, np.int32)
        points = points.reshape((-1, 1, 2))
        mask = cv2.polylines(mask, [points], True, (255, 255, 255), 2)
        mask2 = cv2.fillPoly(mask.copy(), [points], (255, 255, 255))
        mask3 = cv2.fillPoly(mask.copy(), [points], (0, 255, 0))
        show_image = cv2.addWeighted(
            src1=img, alpha=0.8, src2=mask3, beta=0.2, gamma=0)
        cv2.imshow("mask", mask2)
        cv2.imshow("show_img", show_image)
        ROI = cv2.bitwise_and(mask2, img)
        cv2.imshow("ROI", ROI)
        cv2.waitKey(0)
    if len(pts) > 0:
        cv2.circle(img2, pts[-1], 1, (0, 0, 255), -1)
    if len(pts) > 1:
        for i in range(len(pts) - 1):
            cv2.circle(img2, pts[i], 1, (0, 0, 255), -1)
            cv2.line(img=img2, pt1=pts[i], pt2=pts[i + 1],
                     color=(255, 0, 0), thickness=1)
    cv2.imshow('image', img2)


img = cv2.imread("./img/Beach.jpg")
cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_roi)

mask = np.zeros_like(img)
while True:
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    if key == ord("s"):
        cv2.fillPoly(mask, [np.array(pts)], (255, 255, 255))
        cv2.imshow("mask", mask)
        pts = []
cv2.imshow("mask", mask)
files = os.listdir("./mask")
name = len(files) +1
print(files)
cv2.imwrite(f"./mask/mask{name}.jpg", mask)
cv2.waitKey()
# cv2.destroyAllWindows()
