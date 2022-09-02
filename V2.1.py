from tkinter import *
from PIL import Image
from PIL import ImageTk
import numpy as np
import tkinter as tk
from tkinter import filedialog
import cv2
import imutils
import scipy.spatial.distance
import math
import os
from pathlib import Path
import shutil


def resize(event):
    new_width = event.width
    new_height = event.height
    main.minsize(width=1625, height=int(new_width*0.505))

    lblVideo.place(x=10, y=10, width=(new_width*0.7), height=((new_width*0.7)*0.5625))
    lblResultado.place(x=((new_width*0.7)+20), y=10, width=(new_width-(new_width*0.7)-30), height=((new_width-(new_width*0.7)-30)*1.77))

    btnGrabar.place(x=10, y=(((new_width*0.7)*0.5625)+20), width=(((new_width*0.7)-10)/2), height=40)
    btnFinalizar.place(x=((((new_width*0.7)-10)/2)+20), y=(((new_width*0.7)*0.5625)+20), width=(((new_width*0.7)-10)/2), height=40)
    btnCapturar1.place(x=10, y=(((new_width * 0.7) * 0.5625) + 80), width=(((new_width * 0.7) - 10) / 2), height=40)
    btnCapturar2.place(x=((((new_width * 0.7) - 10) / 2) + 20), y=(((new_width * 0.7) * 0.5625) + 80),width=(((new_width * 0.7) - 10) / 2), height=40)

def auto_canny(image, sigma=0.33):
    v = np.median(image)+150
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    return edged

def preProcessiong(img):

    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray,(5,5),0)
    ret, binary = cv2.threshold(imgBlur, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE)
    imgCanny = auto_canny(imgBlur)
    kernel = np.ones((5, 5))
    imgDial = cv2.dilate(imgCanny, kernel, iterations=3)
    imgThres = cv2.erode(imgDial, kernel, iterations=1)

    return imgThres

def getContours(img):
    global imgContour
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
    cv2.drawContours(imgContour, biggest, -1, (255, 0, 0), 20)
    return biggest


def reorder(myPoints):
    myPoints = myPoints.reshape((4, 2))

    myPointsNew = np.zeros((4, 1, 2), np.int32)
    add = myPoints.sum(1)

    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]

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
    p = []
    '''p = np.float32(biggest)'''

    for x in range(0, len(biggest)):
        p.append((biggest[x][0][0], biggest[x][0][1]))

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

def visualizar():
    global cap, imgContour, imgWarped, band

    btnFinalizar['state'] = NORMAL
    btnGrabar['state'] = DISABLED

    try:
        success, image = cap.read()
        imgContour = image.copy()
        imgThres = preProcessiong(image)
        biggest = getContours(imgThres)
        imgContour0 = imgContour.copy()

        imgContour0 = imutils.resize(imgContour0, height=lblVideo.winfo_height(),width=lblVideo.winfo_width())
        imgContour0 = cv2.cvtColor(imgContour0, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(imgContour0)
        img = ImageTk.PhotoImage(image=im)
        lblVideo.configure(image=img)
        lblVideo.image = img

        try:
            if biggest.size != 0:
                imgWarped = getWarp(image, biggest)
                imgWarped = cv2.cvtColor(imgWarped, cv2.COLOR_BGR2RGB)

                imgWarped0 = imutils.resize(imgWarped, height=lblResultado.winfo_height(), width=lblResultado.winfo_width())

                im2 = Image.fromarray(imgWarped0)
                img2 = ImageTk.PhotoImage(image=im2)
                lblResultado.configure(image=img2)
                lblResultado.image = img2
                band=1
                btnCapturar1['state'] = NORMAL
                btnCapturar2['state'] = NORMAL
        except:
            pass
        if band == 1:
            imgWarped0 = imutils.resize(imgWarped, height=lblResultado.winfo_height(), width=lblResultado.winfo_width())
            im2 = Image.fromarray(imgWarped0)
            img2 = ImageTk.PhotoImage(image=im2)
            lblResultado.configure(image=img2)
            lblResultado.image = img2
        else:
            lblResultado.configure(image='')

        lblVideo.after(5, visualizar)
    except:
        btnGrabar['state'] = NORMAL
        btnFinalizar['state'] = DISABLED
        lblVideo.configure(image='')
        lblResultado.configure(image='')

def iniciar():
    global cap
    cap = cv2.VideoCapture(0)
    cap.set(3, 3840)
    cap.set(4, 2160)
    visualizar()

def finalizar():
    cap.release()
    global  imgWarped
    imgWarped0 = imutils.resize(imgWarped, height=lblResultado.winfo_height(), width=lblResultado.winfo_width())
    im2 = Image.fromarray(imgWarped0)
    img2 = ImageTk.PhotoImage(image=im2)
    lblResultado.configure(image=img2)
    lblResultado.image = img2

def PathC():
    archi1 = open("db/path.txt", "r")
    lineas = archi1.readlines()
    source = lineas[0]
    archi1.close()
    destination = tk.filedialog.askdirectory()
    files = os.listdir(source)

    for file in files:
        new_path = shutil.move(f"{source}/{file}", destination)
        print(new_path)

    archi1 = open("db/path.txt", "w")
    archi1.write(destination)
    archi1.close()

filetypes = (
    ('PNG files', '*.png'),
    ('JPEG files', '*.jpeg'),
    ('JPG files', '*.jpg'),
)

def capturar():
    global imgWarped

    filename = tk.filedialog.asksaveasfilename(
        title='Save as...',
        filetypes=filetypes,
        defaultextension='.png'
    )
    try:
        filename.grab_set()
    except:
        pass

    imgWarped = cv2.cvtColor(imgWarped, cv2.COLOR_BGR2RGB)
    cv2.imwrite(filename, imgWarped)

def capturar2():
    pass

band = 0
cap = None
imgContour=np.zeros_like((100,100,3),np.uint8)
imgWarped=np.zeros_like((100,100,3),np.uint8)

main = Tk()
main.title("Prueba Graficos")
main.state('zoomed')
main.minsize(width=1625, height=820)

menubarra = Menu(main, relief=RAISED, bd=10)
menuarchivo = Menu(menubarra, tearoff=0, relief=SOLID, bd=0)
menuPath = Menu(menuarchivo, tearoff=0)

menuarchivo.add_cascade(label="Configuracion", menu=menuPath)
menuPath.add_command(label="Cambiar ruta de almacenamiento", command=PathC)
menuarchivo.add_separator()
menuarchivo.add_command(label="Salir", command=lambda: main.destroy())
menubarra.add_cascade(label="Menu", menu=menuarchivo)
main.config(menu=menubarra)

frame = Frame(main)
frame.pack(fill=BOTH, expand=YES)
frame.bind("<Configure>", resize)

btnGrabar = Button(frame, text="Grabar", width=45, command=iniciar)

btnFinalizar = Button(frame, text="Finalizar", width=45, command=finalizar)
btnFinalizar['state'] = DISABLED

btnCapturar1 = Button(frame, text="Guardar como", width=45, command=capturar)
btnCapturar1['state'] = DISABLED

btnCapturar2 = Button(frame, text="Guardar", width=45, command=capturar2)
btnCapturar2['state'] = DISABLED

lblVideo = Label(frame)
lblVideo.configure(bg="#808080")

lblResultado = Label(frame)
lblResultado.configure(bg="#808080")

cpt = sum([len(files) for r, d, files in os.walk('db')])

if cpt == 0 :
    downloads_dir = str(Path.home() / "Downloads")
    directorio = downloads_dir+'\Capturas Documentos'
    try:
        os.mkdir(directorio)
    except OSError:
        print("La creación del directorio %s falló" % directorio)
    else:
        print("Se ha creado el directorio: %s " % directorio)

    archi1 = open("db/path.txt", "w")
    archi1.write(directorio)
    archi1.close()

main.mainloop()