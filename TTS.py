import cv2
import face_recognition

# Coba load gambar referensi
try:
    known_image = face_recognition.load_image_file("Gweh.jpg")
    known_face_encoding = face_recognition.face_encodings(known_image)[0]
except FileNotFoundError:
    print("Gambar referensi tidak ditemukan. Pastikan path file sudah benar.")
    exit()
except IndexError:
    print("Tidak ada wajah yang terdeteksi dalam gambar referensi.")
    exit()

# Inisialisasi kamera
video_capture = cv2.VideoCapture(0)

# Set kamera ke resolusi lebih kecil untuk meningkatkan kecepatan
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Untuk menghitung frame agar deteksi tidak dilakukan di setiap frame
frame_counter = 0

try:
    while True:
        # Tangkap frame dari video
        ret, frame = video_capture.read()
        if not ret:
            print("Gagal mendapatkan frame dari kamera.")
            break

        # Hanya lakukan face recognition di setiap 5 frame untuk menghemat kinerja
        frame_counter += 1
        if frame_counter % 5 == 0:
            # Konversi frame dari BGR (OpenCV default) ke RGB (face_recognition default)
            rgb_frame = frame[:, :, ::-1]

            # Temukan semua wajah dalam frame saat ini
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            # Periksa apakah wajah yang dikenali sesuai dengan wajah yang sudah disimpan
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                matches = face_recognition.compare_faces([known_face_encoding], face_encoding)
                name = "Unknown"

                if True in matches:
                    name = "Known Person"

                # Gambar persegi panjang di sekitar wajah
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Tampilkan hasil
        cv2.imshow('Video', frame)

        # Tekan 'q' untuk keluar dari loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except Exception as e:
    print("Terjadi error:", e)
finally:
    # Bersihkan segala resource yang digunakan
    video_capture.release()
    cv2.destroyAllWindows()
