import numpy as np
import math
import cv2

width = 0
height = 0
eggCount = 0
distance_tresh = 200
radius_min, radius_max = 18, 45 #raio maximo e minimo para deteccao do ovo: Ajustar para obter melhor resultado. Valor em pixels
area_min, area_max = 370, 1300 #Area maxima e minima para deteccao dos ovos: Ajustar para obter melhor resultado.


def getDistance(coordYEgg1, coordYEgg2):
    dist = abs(coordYEgg1 - coordYEgg2)
    return dist

def do_segment(frame):
    height = frame.shape[0]
    width = frame.shape[1]
    polygons = np.array([
                            [(0, height), (width - 50, height), (width - 50, 0),(0, 0)]
                        ])
    mask = np.zeros_like(frame)
    cv2.fillPoly(mask, polygons, 255)
    segment = cv2.bitwise_and(frame, mask)
    return segment

im = cv2.imread("01.jpg", 1)
frame40 = im
height = np.size(frame40, 0)
width = np.size(frame40, 1)
hsv = cv2.cvtColor(frame40, cv2.COLOR_BGR2HSV)
threshold, bitwise = cv2.threshold(hsv[:, :, 2], 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
morph = cv2.morphologyEx(bitwise, cv2.MORPH_CLOSE, kernel)
dist = cv2.distanceTransform(morph, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)

borderSize = 40
distborder = cv2.copyMakeBorder(dist, borderSize, borderSize, borderSize, borderSize,
                                cv2.BORDER_CONSTANT | cv2.BORDER_ISOLATED, 0)
gap = 10
kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * (borderSize - gap) + 1, 2 * (borderSize - gap) + 1))
kernel2 = cv2.copyMakeBorder(kernel2, gap, gap, gap, gap, cv2.BORDER_CONSTANT | cv2.BORDER_ISOLATED, 0)

distTempl = cv2.distanceTransform(kernel2, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)

nxcor = cv2.matchTemplate(distborder, distTempl, cv2.TM_CCOEFF_NORMED)

mn, mx, _, _ = cv2.minMaxLoc(nxcor)
threshold, bitwise = cv2.threshold(nxcor, mx * 0.5, 255, cv2.THRESH_BINARY)
peaks8u = cv2.convertScaleAbs(bitwise)
peaks8u = do_segment(peaks8u)
_, contours, hierarchy = cv2.findContours(peaks8u, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

flag = False
egg_list = []
egg_index = 0
eggCount = 0
for i in range(len(contours)):
    contour = contours[i]
    (x, y), radius = cv2.minEnclosingCircle(contour)
    radius = int(radius)
    (x, y, w, h) = cv2.boundingRect(contour)
    egg_index = i
    egg_list.append([x, y, flag])

    if len(contour) >= 5:

        if (radius <= int(radius_max) and radius >= int(radius_min)):
            ellipse = cv2.fitEllipse(contour)
            (center, axis, angle) = ellipse
            coordXContour, coordYContour = int(center[0]), int(center[1])
            ax1, ax2 = int(axis[0]) - 2, int(axis[1]) - 2
            orientation = int(angle)
            area = cv2.contourArea(contour)

            if area >= int(area_min) and area <= int(area_max):
                cv2.ellipse(frame40, (coordXContour, coordYContour), (ax1, ax2), orientation, 0, 360, (255, 0, 255), 2)  # roxo
                cv2.putText(frame40, "{}".format(str(coordXContour)), (coordXContour, coordYContour), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(60, 50, 40), 1)
                for k in range(len(egg_list)):                    
                    egg_new_X = x
                    egg_new_Y = y
                    dist = getDistance(egg_new_Y, egg_list[k][1])
                    if dist > distance_tresh:                     
                        egg_list.append([egg_new_X, egg_new_Y, flag])
                eggCount+=1

# number_eggs = 48
# precision = int((eggCount/number_eggs) * 100)
cv2.putText(frame40, "Ovos: {}".format(str(eggCount)), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 2,(250, 255, 1), 2)
# cv2.putText(frame40, "Precisao(%): {}".format(str(precision)), (20, 400), cv2.FONT_HERSHEY_SIMPLEX, 1,(250, 0, 1), 2)
cv2.imshow("Frame", frame40)
while True:
    cv2.imshow("Frame", frame40)
    key = cv2.waitKey(1)
    if key == 27:
        break
cv2.destroyAllWindows()