import cv2
from ultralytics import YOLO
import os

def detect_keypoints(image_path):
    # Inisialisasi model
    model = YOLO("yolov8n-pose.pt")
    
    # Proses deteksi
    results = model(image_path)
    
    # Tampilkan hasil
    for result in results:
        plotted_img = result.plot()
        cv2.imshow("Hasil Deteksi", plotted_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # Simpan gambar ke C:\OUTPUT
        output_dir = r" " # isi dengan direktori yang diinginkan
        os.makedirs(output_dir, exist_ok=True)
        
        # Ekstrak nama file asli
        filename = os.path.basename(image_path)
        output_path = os.path.join(output_dir, f"result_{filename}")
        
        # Handle karakter khusus dalam path
        try:
            cv2.imwrite(output_path, plotted_img)
            print(f"✅ Hasil disimpan di: {output_path}")
        except Exception as e:
            print(f"❌ Gagal menyimpan hasil: {str(e)}")

if __name__ == "__main__":
    # Hardcode path atau gunakan input
    image_path = r" " # Anda bisa memasukkan path gambar di sini 
    
    # Validasi path
    if os.path.exists(image_path):
        detect_keypoints(image_path)
    else:
        print("❌ File tidak ditemukan. Pastikan:")
        print(f"1. Path benar: {image_path}")
        print("2. File ada di lokasi tersebut")
        print("3. Tidak ada typo dalam path")