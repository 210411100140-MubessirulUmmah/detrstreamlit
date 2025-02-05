import os
import cv2
import torch
import shutil
import zipfile
import numpy as np
import streamlit as st
import supervision as sv
from PIL import Image
from transformers import DetrForObjectDetection, DetrImageProcessor


# =========================== SETUP STREAMLIT =========================== #
st.set_page_config(page_title="Deteksi Cacat Pengelasan", layout="wide")

st.title("🔥 Deteksi Cacat Pengelasan dengan DETR 🔥")
st.write("Upload gambar pengelasan dan model AI akan mendeteksi cacat seperti **slag inclusion, spatter,** dan **undercut**.")

# =========================== LOAD MODEL =========================== #
@st.cache_resource
def load_model():
    HOME = "./"  # Sesuaikan lokasi model
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = DetrForObjectDetection.from_pretrained(HOME).to(DEVICE).eval()
    image_processor = DetrImageProcessor.from_pretrained(HOME)
    
    return model, image_processor, DEVICE

model, image_processor, DEVICE = load_model()
st.success("✅ Model berhasil dimuat!")

# =========================== KONFIGURASI =========================== #
CLASS_NAMES = {
    0: "weld-defect-dete",
    1: "slag inclusion",
    2: "spatter",
    3: "undercut"
}

UPLOAD_DIR = "uploaded_images"
RESULTS_DIR = "detected_results"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# =========================== UNGGAH GAMBAR =========================== #
uploaded_files = st.file_uploader("📤 Upload gambar pengelasan", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

# Threshold Slider
threshold = st.slider("🎚️ Atur Threshold Deteksi", 0.1, 1.0, 0.5, step=0.05)

if uploaded_files:
    st.write(f"🔍 **{len(uploaded_files)} gambar telah diunggah!**")

    for uploaded_file in uploaded_files:
        # Simpan gambar yang diunggah
        image_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

    st.success("✅ Semua gambar berhasil diunggah!")

# =========================== DETEKSI OBJEK =========================== #
def detect_objects(image_path, threshold):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with torch.no_grad():
        inputs = image_processor(images=image_rgb, return_tensors="pt").to(DEVICE)
        outputs = model(**inputs)

        target_sizes = torch.tensor([image.shape[:2]]).to(DEVICE)
        results = image_processor.post_process_object_detection(outputs, threshold=threshold, target_sizes=target_sizes)[0]

        bboxes = results["boxes"].cpu().numpy()
        scores = results["scores"].cpu().numpy()
        class_ids = results["labels"].cpu().numpy()

    # Buat anotasi
    box_annotator = sv.BoundingBoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    labels = [f"{CLASS_NAMES[class_id]} ({score:.2f})" for class_id, score in zip(class_ids, scores)]
    detections = sv.Detections(xyxy=bboxes, confidence=scores, class_id=class_ids)

    # Anotasi gambar
    annotated_image = box_annotator.annotate(scene=image_rgb.copy(), detections=detections)
    annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

    return annotated_image

# =========================== TAMPILKAN HASIL DETEKSI =========================== #
if st.button("🔍 Jalankan Deteksi"):
    if not uploaded_files:
        st.warning("⚠️ Silakan unggah gambar terlebih dahulu.")
    else:
        detected_images = []
        st.write("🚀 **Memproses deteksi...**")

        for file in uploaded_files:
            image_path = os.path.join(UPLOAD_DIR, file.name)
            result_image = detect_objects(image_path, threshold)

            result_path = os.path.join(RESULTS_DIR, file.name)
            Image.fromarray(result_image).save(result_path)

            detected_images.append(result_path)

            # Tampilkan hasil
            st.image(result_image, caption=f"Hasil Deteksi - {file.name}", use_column_width=True)

        st.success("✅ Deteksi selesai!")

# =========================== DOWNLOAD HASIL =========================== #
if os.listdir(RESULTS_DIR):
    st.write("📥 **Unduh Hasil Deteksi:**")

    # **Download satu per satu**
    for result_file in os.listdir(RESULTS_DIR):
        result_path = os.path.join(RESULTS_DIR, result_file)
        with open(result_path, "rb") as f:
            st.download_button(f"📥 Download {result_file}", f, file_name=result_file)

    # **Download semua sebagai ZIP**
    zip_path = "detected_results.zip"
    with zipfile.ZipFile(zip_path, "w") as zipf:
        for result_file in os.listdir(RESULTS_DIR):
            zipf.write(os.path.join(RESULTS_DIR, result_file), result_file)

    with open(zip_path, "rb") as f:
        st.download_button("📥 Download Semua Hasil sebagai ZIP", f, file_name="detected_results.zip")

# =========================== HAPUS GAMBAR =========================== #
st.write("🗑 **Hapus Gambar yang Telah Diunggah:**")

# **Hapus satu per satu**
if os.listdir(UPLOAD_DIR):
    selected_file = st.selectbox("Pilih gambar untuk dihapus:", os.listdir(UPLOAD_DIR))
    if st.button("🗑 Hapus Gambar"):
        os.remove(os.path.join(UPLOAD_DIR, selected_file))
        st.success(f"✅ {selected_file} telah dihapus!")

# **Hapus semua gambar**
if st.button("🗑 Hapus Semua Gambar"):
    shutil.rmtree(UPLOAD_DIR)
    shutil.rmtree(RESULTS_DIR)
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    st.success("✅ Semua gambar berhasil dihapus!")

# =========================== FOOTER =========================== #
st.markdown("---")
st.markdown("🎯 **Dibuat oleh Mubessirul Ummah | Skripsi Deteksi Cacat Pengelasan**")

