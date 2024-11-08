import cv2
import face_recognition

# Memuat gambar dari file dan mengubahnya menjadi format RGB
imgDani = face_recognition.load_image_file('Foto/Dani.jpg')
imgDani = cv2.cvtColor(imgDani, cv2.COLOR_BGR2RGB)
imgEcha = face_recognition.load_image_file('Foto/Echa.jpg')
imgEcha = cv2.cvtColor(imgEcha, cv2.COLOR_BGR2RGB)

# Mendeteksi lokasi wajah dan mengkodekan wajah dari gambar Dani
faceLoc = face_recognition.face_locations(imgDani)[0]  # Mengambil lokasi wajah pertama
encodeDani = face_recognition.face_encodings(imgDani)[0]  # Mengambil encoding wajah pertama
# Menggambar kotak di sekitar wajah Dani
cv2.rectangle(imgDani, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 255), 2)

# Mendeteksi lokasi wajah dan mengkodekan wajah dari gambar Echa
faceLocTest = face_recognition.face_locations(imgEcha)[0]  # Mengambil lokasi wajah pertama
encodeTest = face_recognition.face_encodings(imgEcha)[0]  # Mengambil encoding wajah pertama
# Menggambar kotak di sekitar wajah Echa
cv2.rectangle(imgEcha, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (255, 0, 255), 2)

# Membandingkan wajah Dani dengan gambar uji
results = face_recognition.compare_faces([encodeDani], encodeTest)  # Membandingkan encoding
faceDis = face_recognition.face_distance([encodeDani], encodeTest)  # Menghitung jarak wajah
# Menampilkan hasil perbandingan dan jarak di gambar uji
print(results, faceDis)  # Menampilkan hasil perbandingan dan jarak ke konsol
cv2.putText(imgEcha, f'{results} {round(faceDis[0], 2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

# Menampilkan gambar dengan kotak wajah dan hasil
cv2.imshow('Dani', imgDani)
cv2.imshow('Echa', imgEcha)
cv2.waitKey(0)
