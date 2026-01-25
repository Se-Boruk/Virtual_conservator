import sys
import os
from PyQt6 import uic
from PyQt6.QtWidgets import QApplication, QFileDialog
from PyQt6.QtGui import QPixmap, QCursor
from PyQt6.QtCore import Qt, QCoreApplication
import time
import numpy as np
import pyarrow as pa
from PIL import Image

class Encoder:
    def __init__(self):
        self.image_path = None

    def convert_to_arrow(self, path):
        """Konwertuje plik obrazu do formatu tabelarycznego Apache Arrow"""
        try:
            # 1. Otwarcie obrazu i konwersja na tablicę NumPy (RGB)
            img = Image.open(path).convert('RGB')
            img_data = np.array(img)
            
            # Pobieramy wymiary
            height, width, channels = img_data.shape
            
            # 2. Spłaszczamy dane do formy wektora (dla tabeli Arrow)
            # Każdy wiersz w tabeli może reprezentować jeden piksel lub cały obraz jako blob
            flat_data = img_data.tobytes()
            
            # 3. Tworzymy strukturę danych Arrow
            # Tutaj zapisujemy metadane (wymiary) oraz same dane binarnie
            data = [
                pa.array([width]),
                pa.array([height]),
                pa.array([flat_data])
            ]
            
            table = pa.Table.from_arrays(data, names=['width', 'height', 'image_bytes'])
            self.image_as_table = table
            print(f"Sukces: Obraz {path} skonwertowany do Arrow Table.")
            return table
        
        except Exception as e:
            print(f"Błąd podczas konwersji obrazu: {e}")
            return None
        
    def arrow_to_jpg(self, output_path="result.jpg"):
        """Konwertuje tabelę Apache Arrow z powrotem na plik JPG"""
        try:
            # 1. Ekstrakcja danych z tabeli Arrow
            # Pobieramy pierwszy element z każdej kolumny
            width = self.image_as_table.column('width')[0].as_py()
            height = self.image_as_table.column('height')[0].as_py()
            image_bytes = self.image_as_table.column('image_bytes')[0].as_py()

            # 2. Konwersja bajtów z powrotem na tablicę NumPy
            # Musimy wiedzieć, że obraz był w formacie RGB (3 kanały)
            flat_data = np.frombuffer(image_bytes, dtype=np.uint8)
            img_data = flat_data.reshape((height, width, 3))

            # 3. Tworzenie obiektu obrazu PIL i zapis do JPG
            img = Image.fromarray(img_data, 'RGB')
            img.save(output_path, "JPEG")

            print(f"Sukces: Obraz został odtworzony i zapisany w {output_path}")
            return output_path

        except Exception as e:
            print(f"Błąd konwersji z Arrow do JPG: {e}")
            return None

    def crash_image(self):
        print("Crashing image...")
        crashed_img = self.convert_to_arrow(self.image_path)
        self.crashed_img_arrow = crashed_img
        return self.arrow_to_jpg("crashed_image.jpg")

    def Auto_encode(self):
        print("Auto encoding image...")
        self.fixed_image = self.crashed_img_arrow
        return self.arrow_to_jpg("fixed_image.jpg")

    def upscaler(self):
        print("Upscaling image...")
        self.upscaled_image = self.fixed_image
        return self.arrow_to_jpg("upscaled_image.jpg")
        
class GUI():
    def __init__(self):
        self.window = None
        self.second_window = None
        self.third_window = None 
        self.image = None 

    def klikniecie_w_obraz(self, event):
        """
        Ta funkcja zastąpi standardowe zachowanie labela przy kliknięciu.
        Argument 'event' jest przekazywany automatycznie przez PyQt.
        """
        # Sprawdzamy, czy kliknięto Lewym Przyciskiem Myszy
        if event.button() == Qt.MouseButton.LeftButton:
            # Otwieramy okno wyboru pliku
            plik, _ = QFileDialog.getOpenFileName(
                self.window, 
                "Wybierz zdjęcie", 
                "", 
                "Obrazy (*.png *.jpg *.jpeg *.bmp)"
            )

            if plik:
                # Tworzymy obraz
                self.image = plik
                pixmap = QPixmap(plik)
                
                # Skalujemy do rozmiaru ramki
                wymiar_ramki = self.window.imageLabel.size()
                skalowane_foto = pixmap.scaled(
                    wymiar_ramki,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
                
                # Ustawiamy obraz
                self.window.imageLabel.setPixmap(skalowane_foto)
                self.window.imageLabel.setText("")  # Czyścimy tekst "Brak zdjęcia"
                self.window.btnStart.setEnabled(True)
                self.aktualizuj_status("Click start to process the image")
    
    def drugie_okno(self):
        """Funkcja ładująca i wyświetlająca Second_window.ui"""
        # 1. Zmieniamy status na przetwarzanie
        self.aktualizuj_status("Processing...")
        QCoreApplication.processEvents() # Odświeżamy UI, by napis się pojawił
        
        # Symulacja czasu pracy (np. 1 sekunda)
        time.sleep(1)

        # 2. Ładujemy drugie okno, jeśli jeszcze nie istnieje
        if self.second_window is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            second_ui_path = os.path.join(current_dir, "Second_window.ui")
            self.second_window = uic.loadUi(second_ui_path)

        # 3. Wyświetlamy drugie okno
        self.second_window.show()
        
        # 4. (Opcjonalnie) Możemy ukryć pierwsze okno:
        # self.window.hide()
        
        self.aktualizuj_status("Done! Window opened.")

    def trzecie_okno(self):
        """Funkcja ładująca i wyświetlająca Third_window.ui"""
        # 1. Zmieniamy status na przetwarzanie
        self.aktualizuj_status("Processing...")
        QCoreApplication.processEvents() # Odświeżamy UI, by napis się pojawił
        
        # Symulacja czasu pracy (np. 1 sekunda)
        time.sleep(1)

        # 2. Ładujemy drugie okno, jeśli jeszcze nie istnieje
        if self.third_window is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            third_ui_path = os.path.join(current_dir, "Third_window.ui")
            self.third_window = uic.loadUi(third_ui_path)

        # 3. Wyświetlamy drugie okno
        self.third_window.show()
        
        # 4. (Opcjonalnie) Możemy ukryć pierwsze okno:
        # self.window.hide()
        
        self.aktualizuj_status("Done! Window opened.")

    def wyczysc_wszystko(self):
        """Funkcja do czyszczenia obrazu i resetowania statusu"""
        self.window.imageLabel.clear()
        self.window.imageLabel.setText("Click to add image")
        self.aktualizuj_status("Image cleared. Click to add a new image.")
        self.window.btnStart.setEnabled(False)

    def start_process(self):
        """Funkcja wywoływana po kliknięciu Start - sprawdza DropDown i wybiera okno"""
        wybor = self.window.DropDown.currentText()

        E = Encoder()  # Inicjalizacja klasy Encoder
        E.image_path = self.image
        path1 = E.crash_image()
        path2 = E.Auto_encode()
        path3 = E.upscaler()
        
        if wybor == "Fix image":
            self.drugie_okno()
            self.ustaw_obraz_w_labelu(self.second_window.Fixed_img, path2)
            self.ustaw_obraz_w_labelu(self.second_window.Damaged_img, path1)

        elif wybor == "Fix & Upscale image":
            self.trzecie_okno()
            self.ustaw_obraz_w_labelu(self.third_window.Fixed_img, path2)
            self.ustaw_obraz_w_labelu(self.third_window.Damaged_img, path1)
            self.ustaw_obraz_w_labelu(self.third_window.Upscaled_img, path3)
            

        else:
            self.aktualizuj_status("Select an option first!")

    def ustaw_obraz_w_labelu(self, label_obj, sciezka_obrazu):
        """Pomocnicza funkcja do ładowania obrazu do konkretnego labela"""
        pixmap = QPixmap(sciezka_obrazu)
        if not pixmap.isNull():
            label_obj.setPixmap(pixmap.scaled(
                label_obj.size(), 
                Qt.AspectRatioMode.KeepAspectRatio, 
                Qt.TransformationMode.SmoothTransformation
            ))
            label_obj.setText("")

    def aktualizuj_status(self, tekst):
        """Pomocnicza funkcja do zmiany tekstu w dolnej etykiecie"""
        self.window.label.setText(tekst)

    # --- GŁÓWNA CZĘŚĆ PROGRAMU ---

    def run(self):
        app = QApplication(sys.argv)

        # Ustalenie ścieżki do pliku .ui
        current_dir = os.path.dirname(os.path.abspath(__file__))
        ui_file_path = os.path.join(current_dir, "Main_window.ui")

        try:
            self.window = uic.loadUi(ui_file_path)
            self.window.btnStart.setEnabled(False)

            # Nadpisujemy metodę mousePressEvent dla elementu imageLabel.
            self.window.imageLabel.mousePressEvent = self.klikniecie_w_obraz

            # Zmieniamy kursor na "łapkę" po najechaniu na obszar zdjęcia,
            self.window.imageLabel.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))

            # Obsługa pozostałych przycisków 
            if hasattr(self.window, 'btnExit'):
                self.window.btnExit.clicked.connect(self.window.close)
            
            if hasattr(self.window, 'btnClear'):
                self.window.btnClear.clicked.connect(self.wyczysc_wszystko)

            if hasattr(self.window, 'btnStart'):
                self.window.btnStart.clicked.connect(self.start_process)

            self.window.show()
            sys.exit(app.exec())

        except FileNotFoundError:
            print(f"BŁĄD: Nie znaleziono pliku: {ui_file_path}")
        except Exception as e:
            print(f"Wystąpił błąd: {e}")

G = GUI()
G.run()