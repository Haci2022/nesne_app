import os
import cv2
import numpy as np
from PIL import Image
import streamlit as st
from ultralytics import YOLO

# SSL sertifikalari klasorde yoksa hemen basitce olustur
if not os.path.exists("cert.pem") or not os.path.exists("key.pem"):
    os.system(
        'openssl req -x509 -newkey rsa:2048 -keyout key.pem -out cert.pem -days 365 -nodes -subj "/CN=localhost"'
    )


st.title("Boru Sayma Programi")

# Modeli yukle
try:
    model = YOLO("best.pt")
except:
    model = YOLO("yolov8n.pt")

# Stok listesi
if "stok" not in st.session_state:
    st.session_state.stok = {}

# Kamera kismi
kamera_resmi = st.camera_input("Fotograf Cek")
anlik_sayim = 0

if kamera_resmi is not None:
    resim = Image.open(kamera_resmi)
    resim_np = np.array(resim)

    # Yapay zeka ile tespit
    sonuclar = model(resim_np)

    for sonuc in sonuclar:
        for kutu in sonuc.boxes:
            anlik_sayim = anlik_sayim + 1
            koordinat = kutu.xyxy[0]
            x1 = int(koordinat[0])
            y1 = int(koordinat[1])
            x2 = int(koordinat[2])
            y2 = int(koordinat[3])

            cv2.rectangle(resim_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                resim_np,
                f"Boru {anlik_sayim}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

    st.image(resim_np, caption="Bulunan Borular")
    st.write("Bulunan Boru Sayisi:", anlik_sayim)

    # Stoga ekleme
    tur = st.text_input("Boru Boyutu / Turu", "Standart Boru")
    if st.button("Stoga Kaydet"):
        if tur in st.session_state.stok:
            st.session_state.stok[tur] = (
                st.session_state.stok[tur] + anlik_sayim
            )
        else:
            st.session_state.stok[tur] = anlik_sayim
        st.success("Stoga eklendi!")

# Stok Goruntuleme
st.write("---")
st.write("### Stok Durumu")

toplam = 0
for k, v in st.session_state.stok.items():
    st.write(k, ":", v, "tane")
    toplam = toplam + v

st.write("**Genel Toplam:**", toplam)