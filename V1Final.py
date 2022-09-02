import cv2
import numpy as np
import scipy
import scipy.spatial.distance
import math

widthImg = 3840
heightImg = 2160

cap = cv2.VideoCapture(1)
cap.set(3,widthImg)
cap.set(4,heightImg)

def auto_canny(image, sigma=0.33):
    v = np.median(image)+150
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    return edged

def preProcessiong(img):

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 0)
    ret, binary = cv2.threshold(imgBlur, 50, 255, cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE)
    imgCanny = cv2.Canny(binary, 100, 200)
    kernel = np.ones((5, 5))
    imgDial = cv2.dilate(imgCanny, kernel, iterations=3)
    imgThres = cv2.erode(imgDial, kernel, iterations=1)

    return imgThres

def getContours(img):
    biggest = np.array([])
    maxArea = 0
    contours,hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area>5000:
            peri = cv2.arcLength(cnt,True)
            approx = cv2.approxPolyDP(cnt,0.01*peri,True)
            if area >maxArea and len(approx) == 4:
                biggest = approx
                maxArea = area
    cv2.drawContours(imgContour, biggest, -1, (255, 0, 0), 60)
    return biggest


def reorder(myPoints):
    myPoints = myPoints.reshape((4, 2))

    myPointsNew = np.zeros((4, 1, 2), np.int32)

    px = 9999999999999999999
    py = 9999999999999999999

    for x in range(4):
        if myPoints[x][0]<px:
            if myPoints[x][1]<py:
                px = myPoints[x][0]
                py = myPoints[x][1]

    myPointsNew[0] = [px,py]

    px = 0
    py = 0

    for x in range(4):
        if myPoints[x][0] > px:
            if myPoints[x][1] > py:
                px = myPoints[x][0]
                py = myPoints[x][1]

    myPointsNew[3] = [px, py]

    myPoints = np.delete(myPoints, (np.where(myPoints == myPointsNew[0])[0][0]), axis=0)
    myPoints = np.delete(myPoints, (np.where(myPoints == myPointsNew[3])[0][0]), axis=0)

    px = 9999999999999999999

    for x in range(2):
        if myPoints[x][0] < px:
                px = myPoints[x][0]
                py = myPoints[x][1]

    myPointsNew[2] = [px, py]

    px = 0

    for x in range(2):
        if myPoints[x][0] > px:
                px = myPoints[x][0]
                py = myPoints[x][1]

    myPointsNew[1] = [px, py]

    dist1 = np.linalg.norm(myPointsNew[0] - myPointsNew[1])
    dist2 = np.linalg.norm(myPointsNew[0] - myPointsNew[2])
    myPointsNew2 = np.copy(myPointsNew)

    if dist1>dist2:
        myPointsNew[0] = myPointsNew2[2]
        myPointsNew[1] = myPointsNew2[0]
        myPointsNew[2] = myPointsNew2[3]
        myPointsNew[3] = myPointsNew2[1]

    return myPointsNew


def getWarp(img, biggest):

    biggest = reorder(biggest)
    p=[]
    '''p = np.float32(biggest)'''

    for x in range(0,len(biggest)):
        p.append((biggest[x][0][0],biggest[x][0][1]))

    (rows, cols, _) = img.shape
    u0 = (cols) / 2.0
    v0 = (rows) / 2.0

    w1 = scipy.spatial.distance.euclidean(p[0], p[1])
    w2 = scipy.spatial.distance.euclidean(p[2], p[3])

    h1 = scipy.spatial.distance.euclidean(p[0], p[2])
    h2 = scipy.spatial.distance.euclidean(p[1], p[3])

    w = max(w1, w2)
    h = max(h1, h2)

    # visible aspect ratio
    ar_vis = float(w) / float(h)

    # make numpy arrays and append 1 for linear algebra
    m1 = np.array((p[0][0], p[0][1], 1)).astype('float32')
    m2 = np.array((p[1][0], p[1][1], 1)).astype('float32')
    m3 = np.array((p[2][0], p[2][1], 1)).astype('float32')
    m4 = np.array((p[3][0], p[3][1], 1)).astype('float32')

    # calculate the focal disrance
    k2 = np.dot(np.cross(m1, m4), m3) / np.dot(np.cross(m2, m4), m3)
    k3 = np.dot(np.cross(m1, m4), m2) / np.dot(np.cross(m3, m4), m2)

    n2 = k2 * m2 - m1
    n3 = k3 * m3 - m1

    n21 = n2[0]
    n22 = n2[1]
    n23 = n2[2]

    n31 = n3[0]
    n32 = n3[1]
    n33 = n3[2]

    f = math.sqrt(np.abs((1.0 / (n23 * n33)) * ((n21 * n31 - (n21 * n33 + n23 * n31) * u0 + n23 * n33 * u0 * u0) + (
            n22 * n32 - (n22 * n33 + n23 * n32) * v0 + n23 * n33 * v0 * v0))))

    A = np.array([[f, 0, u0], [0, f, v0], [0, 0, 1]]).astype('float32')

    At = np.transpose(A)
    Ati = np.linalg.inv(At)
    Ai = np.linalg.inv(A)

    # calculate the real aspect ratio
    ar_real = math.sqrt(np.dot(np.dot(np.dot(n2, Ati), Ai), n2) / np.dot(np.dot(np.dot(n3, Ati), Ai), n3))

    if ar_real < ar_vis:
        W = int(w)
        H = int(W / ar_real)
    else:
        H = int(h)
        W = int(ar_real * H)

    pts1 = np.array(p).astype('float32')
    pts2 = np.float32([[0, 0], [W, 0], [0, H], [W, H]])

    # project the image with the new w/h
    M = cv2.getPerspectiveTransform(pts1, pts2)

    dst = cv2.warpPerspective(img, M, (W, H))

    return dst

def stackImages(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]),
                                                None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        hor_con = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver

while True:
    success, img = cap.read()
    img = cv2.resize(img,(widthImg,heightImg))
    imgContour = img.copy()

    imgThres = preProcessiong(img)
    biggest = getContours(imgThres)

    if biggest.size != 0:
        imgWarped = getWarp(img, biggest)
    else:
        imgWarped = img.copy()
    imgWarped = img.copy()

    imageArray = ([img,imgThres],[imgContour,imgWarped])

    stackedImages = stackImages(0.2, imageArray)
    cv2.imshow("WorkFlow", stackedImages)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


'''img= cv2.imread("resources/e4.jpg")
imgContour = img.copy()

imgThres = preProcessiong(img)
biggest = getContours(imgThres)
imageArray = ([imgContour,imgThres])
stackedImages = stackImages(0.15, imageArray)

cv2.imshow("Image", stackedImages)

if biggest.size != 0:
    imgWarped = getWarp(img, biggest)
else:
    imgWarped = img.copy()'''

def leEscale(img):

    scale_percent = 17

    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)

    dsize = (width, height)

    output = cv2.resize(img, dsize)
    return output

image1 = leEscale(imgWarped)

cv2.imshow("Result", image1)

cv2.waitKey(0)

