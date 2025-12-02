import cv2
import numpy as np

# Memuat model dan konfigurasi YOLO
weights_path = "weights/yolov3-tiny_training_last.weights"
config_path = "cfg/yolov3-tiny.cfg"
net = cv2.dnn.readNet(weights_path, config_path)

# Memuat nama-nama objek
with open('data/obj.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Mengatur kamera
cap = cv2.VideoCapture(0)  # Gunakan 0 untuk kamera default

# Cek apakah kamera berhasil dibuka
if not cap.isOpened():
    print("Error: Kamera tidak bisa diakses.")
    exit()

while True:
    # Baca frame dari kamera
    ret, frame = cap.read()
    if not ret:
        print("Error: Gagal membaca frame dari kamera.")
        break

    # Mengubah ukuran frame untuk input model
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)

    # Mengambil layer output
    layer_names = net.getLayerNames()
    output_layers_indices = net.getUnconnectedOutLayers()

    # Jika output_layers_indices adalah array satu dimensi
    if output_layers_indices.ndim == 1:
        output_layers = [layer_names[i - 1] for i in output_layers_indices]
    else:  # Jika output_layers_indices adalah array dua dimensi
        output_layers = [layer_names[i[0] - 1] for i in output_layers_indices]

    # Melakukan deteksi
    outputs = net.forward(output_layers)

    # Mengolah hasil deteksi
    boxes = []
    confidences = []
    class_ids = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]  # Ambil nilai confidence
            class_id = np.argmax(scores)  # Dapatkan id kelas dengan probabilitas tertinggi
            confidence = scores[class_id]  # Dapatkan confidence

            if confidence > 0.5:  # Ambil hanya deteksi yang melebihi threshold
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Koordinat persegi panjang untuk NMS
                boxes.append([center_x - w // 2, center_y - h // 2, w, h])  # Persegi panjang
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Menghapus kotak yang tumpang tindih
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Menggambar kotak di frame
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = (0, 255, 0)  # Hijau
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)  # Kotak
            cv2.putText(
                frame,
                label,
                (x, y - 10),  # Label di atas kotak
                cv2.FONT_HERSHEY_PLAIN,
                2,
                color,
                2
            )

    # Menampilkan frame dengan deteksi
    cv2.imshow("Image", frame)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Menutup kamera dan jendela
cap.release()
cv2.destroyAllWindows()
