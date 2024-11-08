import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import random

# Deklarasi Path untuk mengambil data foto untuk di compare dari path FOTO
path = 'Foto'
images = []
classNames = []

# Meload gambar dari path
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

# Fungsi untuk Encoding gambar
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)
        if encode:  # Check if encoding was found
            encodeList.append(encode[0])
    return encodeList

# Me list encode yang sudah di encode dari File Basic
encodeListKnown = findEncodings(images)
print('Encoding Complete')

# Membuka webCam
cap = cv2.VideoCapture(0)

# Membuat warna random untuk setiap orang
colors = [tuple(np.random.randint(0, 255, 3).tolist()) for _ in range(len(classNames))]

#Perulangan
while True:
    success, img = cap.read()
    #Jika Gagal dalam membuka webcam maka akan Selsai
    if not success:
        break

    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    # Mencari wajah dan mencocokannya dengan hasil encoding tadi
    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            # Menggunakan warna untuk Setiap Individu yang terdeteksi
            color = colors[matchIndex]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), color, cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            
    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) & 0xFF == ord('e'):  # Pencet e untuk Keluar (exit)
        break

cap.release()
cv2.destroyAllWindows()
