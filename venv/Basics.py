import cv2
import face_recognition

# Memuat gambar dari file dan mengubahnya menjadi format RGB
imgDani = face_recognition.load_image_file('Foto/Dani.jpg')
imgDani = cv2.cvtColor(imgDani, cv2.COLOR_BGR2RGB)
imgEcha = face_recognition.load_image_file('Foto/Echa.jpg')
imgEcha = cv2.cvtColor(imgEcha, cv2.COLOR_BGR2RGB)
imgArpid = face_recognition.load_image_file('Foto/Arpid.jpg')
imgArpid = cv2.cvtColor(imgArpid, cv2.COLOR_BGR2RGB)

# Mendeteksi lokasi wajah dan mengkodekan wajah dari gambar Dani
faceLocDani = face_recognition.face_locations(imgDani)[0]  # Mengambil lokasi wajah pertama
encodeDani = face_recognition.face_encodings(imgDani)[0]  # Mengambil encoding wajah pertama
cv2.rectangle(imgDani, (faceLocDani[3], faceLocDani[0]), (faceLocDani[1], faceLocDani[2]), (255, 0, 255), 2)

# Mendeteksi lokasi wajah dan mengkodekan wajah dari gambar Echa
faceLocEcha = face_recognition.face_locations(imgEcha)[0]
encodeEcha = face_recognition.face_encodings(imgEcha)[0]
cv2.rectangle(imgEcha, (faceLocEcha[3], faceLocEcha[0]), (faceLocEcha[1], faceLocEcha[2]), (255, 0, 255), 2)

# Mendeteksi lokasi wajah dan mengkodekan wajah dari gambar Arpid
faceLocArpid = face_recognition.face_locations(imgArpid)[0]
encodeArpid = face_recognition.face_encodings(imgArpid)[0]
cv2.rectangle(imgArpid, (faceLocArpid[3], faceLocArpid[0]), (faceLocArpid[1], faceLocArpid[2]), (255, 0, 255), 2)

# Membandingkan wajah Dani dengan wajah Echa
resultsEcha = face_recognition.compare_faces([encodeDani], encodeEcha)
faceDisEcha = face_recognition.face_distance([encodeDani], encodeEcha)
print(f"Comparison with Echa: {resultsEcha}, Distance: {faceDisEcha}")
cv2.putText(imgEcha, f'{resultsEcha} {round(faceDisEcha[0], 2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

# Membandingkan wajah Dani dengan wajah Arpid
resultsArpid = face_recognition.compare_faces([encodeDani], encodeArpid)
faceDisArpid = face_recognition.face_distance([encodeDani], encodeArpid)
print(f"Comparison with Arpid: {resultsArpid}, Distance: {faceDisArpid}")
cv2.putText(imgArpid, f'{resultsArpid} {round(faceDisArpid[0], 2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

# Menampilkan gambar dengan kotak wajah dan hasil
cv2.imshow('Dani', imgDani)
cv2.imshow('Echa', imgEcha)
cv2.imshow('Arpid', imgArpid)

cv2.waitKey(0)
cv2.destroyAllWindows()
