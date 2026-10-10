import cv2
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.textinput import TextInput
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.scrollview import ScrollView
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from ultralytics import YOLO

class KameraSayimPaneli(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(orientation='horizontal', padding=10, spacing=10, **kwargs)
        self.stok_verisi = {}
        self.frame_sayaci = 0
        self.son_tespit_metni = "Henuz tespit yok"
        self.model = YOLO('yolov8n.pt')
        self.capture = cv2.VideoCapture(0)
        kamera_layout = BoxLayout(orientation='vertical', size_hint_x=0.6, spacing=5)
        kamera_layout.add_widget(Label(
            text="Canli Nesne Algilama ve Sayim Kamerasi",
            size_hint_y=0.08,
            font_size='16sp',
            bold=True
        ))
        self.kamera_goruntu = Image(size_hint_y=0.82, allow_stretch=True)
        kamera_layout.add_widget(self.kamera_goruntu)
        self.lbl_canli_tespit = Label(
            text="Algilanan: Henuz tespit yok",
            size_hint_y=0.1,
            color=(0.3, 0.9, 0.4, 1),
            bold=True
        )
        kamera_layout.add_widget(self.lbl_canli_tespit)
        self.add_widget(kamera_layout)
        sag_panel = BoxLayout(orientation='vertical', size_hint_x=0.4, spacing=10)
        sag_panel.add_widget(Label(
            text="Malzeme Sayim Paneli",
            size_hint_y=0.08,
            font_size='18sp',
            bold=True
        ))
        giris_layout = BoxLayout(orientation='horizontal', size_hint_y=0.1, spacing=5)
        self.input_tur = TextInput(hint_text="Nesne Tipi", multiline=False, size_hint_x=0.5)
        self.input_adet = TextInput(hint_text="Adet", multiline=False, input_filter='int', size_hint_x=0.25)
        btn_ekle = Button(text="Ekle", size_hint_x=0.25, background_color=(0.2, 0.7, 0.3, 1))
        btn_ekle.bind(on_press=self.manuel_malzeme_ekle)
        giris_layout.add_widget(self.input_tur)
        giris_layout.add_widget(self.input_adet)
        giris_layout.add_widget(btn_ekle)
        sag_panel.add_widget(giris_layout)
        self.lbl_toplam = Label(
            size_hint_y=0.06,
            font_size='15sp',
            color=(0.3, 0.8, 1, 1),
            bold=True
        )
        sag_panel.add_widget(self.lbl_toplam)
        self.scroll = ScrollView(size_hint_y=0.76)
        self.listele_layout = GridLayout(
            cols=1, 
            spacing=5, 
            size_hint_y=None,
            row_default_height=40,
            row_force_default=True
        )
        self.listele_layout.bind(minimum_height=self.listele_layout.setter('height'))
        self.scroll.add_widget(self.listele_layout)
        sag_panel.add_widget(self.scroll)
        self.add_widget(sag_panel)
        self.arayuzu_guncelle()
        if self.capture.isOpened():
            Clock.schedule_interval(self.kare_guncelle, 1.0 / 30.0)
        else:
            self.lbl_canli_tespit.text = "Hata: Kamera baslatilamadi!"
            self.lbl_canli_tespit.color = (0.9, 0.2, 0.2, 1)
    def kare_guncelle(self, dt):
        if not self.capture.isOpened():
            return
        ret, frame = self.capture.read()
        if not ret:
            return
        self.frame_sayaci += 1
        if self.frame_sayaci % 3 == 0:
            sonuclar = self.model(frame, verbose=False)
            anlik_sayimlar = {}
            for sonuc in sonuclar:
                for kutu in sonuc.boxes:
                    sinif_id = int(kutu.cls[0])
                    etiket = self.model.names[sinif_id]
                    anlik_sayimlar[etiket] = anlik_sayimlar.get(etiket, 0) + 1
                    x1, y1, x2, y2 = map(int, kutu.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, etiket, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            if anlik_sayimlar:
                tespit_metni = ", ".join([f"{k}: {v}" for k, v in anlik_sayimlar.items()])
                self.son_tespit_metni = f"Algilanan: {tespit_metni}"
            else:
                self.son_tespit_metni = "Algilanan: Nesne bulunamadi"
            self.lbl_canli_tespit.text = self.son_tespit_metni
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        buffer = cv2.flip(frame_rgb, 0).tobytes()
        dokusu = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='rgb')
        dokusu.blit_buffer(buffer, colorfmt='rgb', bufferfmt='ubyte')
        self.kamera_goruntu.texture = dokusu
    def manuel_malzeme_ekle(self, instance):
        tur = self.input_tur.text.strip()
        adet_str = self.input_adet.text.strip()
        if tur and adet_str:
            try:
                adet = int(adet_str)
                if adet > 0:
                    self.stok_verisi[tur] = self.stok_verisi.get(tur, 0) + adet
                    self.input_tur.text = ""
                    self.input_adet.text = ""
                    self.arayuzu_guncelle()
            except ValueError:
                pass
    def adet_degistir(self, tur, miktar):
        if tur in self.stok_verisi:
            self.stok_verisi[tur] += miktar
            if self.stok_verisi[tur] <= 0:
                del self.stok_verisi[tur]
            self.arayuzu_guncelle()
    def arayuzu_guncelle(self):
        self.listele_layout.clear_widgets()
        toplam_sayi = 0
        for tur, adet in self.stok_verisi.items():
            toplam_sayi += adet
            satir = BoxLayout(orientation='horizontal', size_hint_y=None, height=40, spacing=5)
            satir.add_widget(Label(text=str(tur), size_hint_x=0.5))
            satir.add_widget(Label(text=f"{adet} adet", size_hint_x=0.2, bold=True))
            btn_artir = Button(text="+1", size_hint_x=0.15, background_color=(0.3, 0.6, 0.9, 1))
            btn_artir.bind(on_press=lambda inst, t=tur: self.adet_degistir(t, 1))
            btn_azalt = Button(text="-1", size_hint_x=0.15, background_color=(0.9, 0.3, 0.3, 1))
            btn_azalt.bind(on_press=lambda inst, t=tur: self.adet_degistir(t, -1))
            satir.add_widget(btn_artir)
            satir.add_widget(btn_azalt)
            self.listele_layout.add_widget(satir)
        self.lbl_toplam.text = f"Toplam Kayit: {toplam_sayi}"
    def on_stop(self):
        if hasattr(self, 'capture') and self.capture.isOpened():
            self.capture.release()
class KameraSayimApp(App):
    def build(self):
        self.title = "Nesne Algilama ve Sayim Paneli"
        return KameraSayimPaneli()
    def on_stop(self):
        if self.root:
            self.root.on_stop()
            
if __name__ == '__main__':
    KameraSayimApp().run()