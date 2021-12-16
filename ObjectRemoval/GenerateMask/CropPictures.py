import cv2
import numpy as np


def getPoints(path):
    img = cv2.imread(path)
    print(img.shape)
    a = []
    b = []

    def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            xy = "%d,%d" % (x, y)
            a.append(x)
            b.append(y)
            cv2.circle(img, (x, y), 1, (255, 0, 0), thickness=-1)
            cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                        1.0, (0, 0, 0), thickness=1)
            cv2.imshow("image", img)

    cv2.namedWindow("image")
    cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
    cv2.imshow("image", img)
    cv2.waitKey(0)
    if len(a) < 2 or len(b) < 2:
        raise Exception("Please choose 2 points!")
    return [a[0], b[0]], [a[1], b[1]]


if __name__ == '__main__':
    fig_path = "test_demo/prairie.png"
    point1, point2 = getPoints(fig_path)
    file = open("points.txt", "w")
    print(point1, point2)
    file.write(str(point1[0]) + '\t' + str(point1[1]) + '\t' +
               str(point2[0]) + '\t' + str(point2[1]) + '\n')
    file.write(fig_path + '\n')
    file.close()
