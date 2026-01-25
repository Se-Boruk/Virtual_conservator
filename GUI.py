###################################################################
# ( 1 ) Libs
###################################################################
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
from DataBase.DataBase_Functions import Random_Damage_Generator
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

###################################################################
# ( 2 ) Class encoder, contains all functions for image processing
###################################################################

class Encoder:
    def __init__(self):
        self.image_path = None

    def convert_to_arrow(self, path):
        """Converts a JPG image to an Apache Arrow Table"""
        try:
            # Open image using PIL
            img = Image.open(path).convert('RGB')
            img_data = np.array(img)
            
            # Download image dimensions
            height, width, channels = img_data.shape
            
            flat_data = img_data.tobytes()
            
            # Create Arrow Table
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
        """Converts an Apache Arrow Table back to a JPG image"""

        output_path = "GUI/" + output_path
        try:
            # Extract data from the Arrow table
            width = self.image_as_table.column('width')[0].as_py()
            height = self.image_as_table.column('height')[0].as_py()
            image_bytes = self.image_as_table.column('image_bytes')[0].as_py()

            # Convert bytes back to a NumPy array (RGB - 3 channels)
            flat_data = np.frombuffer(image_bytes, dtype=np.uint8)
            img_data = flat_data.reshape((height, width, 3))

            # 3. Create a PIL image object and save as JPG
            img = Image.fromarray(img_data, 'RGB')
            img.save(output_path, "JPEG")

            print(f"Sukces: Obraz został odtworzony i zapisany w {output_path}")
            return output_path

        except Exception as e:
            print(f"Błąd konwersji z Arrow do JPG: {e}")
            return None
        
    def save_tensor_as_jpg(self, image_tensor, output_path="damaged_output.jpg"):
        """
        image_tensor: Tensor of shape (1, C, H, W) or (C, H, W) in range [0, 1]
        output_path: File save path
        """
        # Remove the Batch dimension (if it exists) and move to CPU
        if image_tensor.dim() == 4:
            image_tensor = image_tensor.squeeze(0)
        
        # Conversion: (C, H, W) -> (H, W, C)
        img_np = image_tensor.permute(1, 2, 0).cpu().numpy()
        
        # Scale to [0, 255] and convert to integer type (uint8)
        img_np = (img_np * 255.0).clip(0, 255).astype(np.uint8)
        
        # Create PIL Image object and save
        final_image = Image.fromarray(img_np)
        final_image.save(output_path, quality=95)
        return output_path

    def crash_image(self):
        print("Crashing image...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        try:
            # Load, convert to RGB and resize (class default uses shape 256x256)
            original_pil = Image.open(self.image_path).convert("RGB")
            original_pil = original_pil.resize((256, 256)) 
            
            # Convert to numpy array and normalize to [0, 1] range
            img_np = np.array(original_pil, dtype=np.float32) / 255.0
            
            # Convert to PyTorch tensor: (H, W, C) -> (C, H, W) -> (B, C, H, W)
            # B (Batch) = 1
            img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device)
        except FileNotFoundError:
            # If the file is not found, generate noise so the code remains functional
            print("Nie znaleziono pliku, używam losowego szumu.")
            img_tensor = torch.rand((1, 3, 256, 256)).to(device)

        # Initialize the damage generator
        dmg_gen = Random_Damage_Generator(device=device)

        # Generate mask with expected shape (B, H, W)
        B, C, H, W = img_tensor.shape
        mask, metadata = dmg_gen.generate(shape=(B, H, W))

        # Apply damage (following logic in Async_DataLoader)
        # Add channel dimension to the mask: (B, 1, H, W)
        damaged_img_tensor = img_tensor * (1.0 - mask.unsqueeze(1))

        damaged_image_path = self.save_tensor_as_jpg(damaged_img_tensor, "crashed_image.jpg")
        
        crashed_img = self.convert_to_arrow(damaged_image_path)
        self.crashed_img_arrow = crashed_img
        return damaged_image_path

    def Auto_encode(self):
        print("Auto encoding image...")
        self.fixed_image = self.crashed_img_arrow
        return self.arrow_to_jpg("fixed_image.jpg")

    def upscaler(self):
        print("Upscaling image...")
        self.upscaled_image = self.fixed_image
        return self.arrow_to_jpg("upscaled_image.jpg")
    
###################################################################
# ( 3 ) Class GUI, responsible for the graphical interface
###################################################################
        
class GUI():
    def __init__(self):
        self.window = None
        self.second_window = None
        self.third_window = None 
        self.image = None 

    def klikniecie_w_obraz(self, event):
        """
        This function replaces the standard label behavior on click.
        The 'event' argument is passed automatically by PyQt.
        """
        # Check if Left Mouse Button was clicked
        if event.button() == Qt.MouseButton.LeftButton:
            # Open file dialog
            plik, _ = QFileDialog.getOpenFileName(
                self.window, 
                "Wybierz zdjęcie", 
                "", 
                "Obrazy (*.png *.jpg *.jpeg *.bmp)"
            )

            if plik:
                # Create the image
                self.image = plik
                pixmap = QPixmap(plik)
                
                # Scale to frame size
                wymiar_ramki = self.window.imageLabel.size()
                skalowane_foto = pixmap.scaled(
                    wymiar_ramki,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
                
                # Set the image
                self.window.imageLabel.setPixmap(skalowane_foto)
                self.window.imageLabel.setText("")  # Clear the "Brak zdjęcia" (No image) text
                self.window.btnStart.setEnabled(True)
                self.aktualizuj_status("Click start to process the image")
    
    def drugie_okno(self):
        """Function loading and displaying Second_window.ui"""
        # Change status to processing
        self.aktualizuj_status("Processing...")
        QCoreApplication.processEvents() # Refresh UI so the label updates
        
        # Simulate processing time (e.g., 1 second)
        time.sleep(1)

        # Load the second window if it doesn't exist yet
        if self.second_window is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            second_ui_path = os.path.join(current_dir, "GUI/Second_window.ui")
            self.second_window = uic.loadUi(second_ui_path)

        # Display the second window
        self.second_window.show()
        
        self.aktualizuj_status("Done! Window opened.")

    def trzecie_okno(self):
        """Function loading and displaying Third_window.ui"""
        # Change status to processing
        self.aktualizuj_status("Processing...")
        QCoreApplication.processEvents() # Refresh UI so the label updates
        
        # Simulate processing time (e.g., 1 second)
        time.sleep(1)

        # Load the third window if it doesn't exist yet
        if self.third_window is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            third_ui_path = os.path.join(current_dir, "GUI/Third_window.ui")
            self.third_window = uic.loadUi(third_ui_path)

        # Display the third window
        self.third_window.show()
        
        self.aktualizuj_status("Done! Window opened.")

    def wyczysc_wszystko(self):
        """Function to clear image and reset status"""
        self.window.imageLabel.clear()
        self.window.imageLabel.setText("Click to add image")
        self.aktualizuj_status("Image cleared. Click to add a new image.")
        self.window.btnStart.setEnabled(False)

    def start_process(self):
        """Function called on Start click - checks DropDown and selects window"""
        wybor = self.window.DropDown.currentText()

        E = Encoder()  # Initialize Encoder class
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
        """Helper function to load an image into a specific label"""
        pixmap = QPixmap(sciezka_obrazu)
        if not pixmap.isNull():
            label_obj.setPixmap(pixmap.scaled(
                label_obj.size(), 
                Qt.AspectRatioMode.KeepAspectRatio, 
                Qt.TransformationMode.SmoothTransformation
            ))
            label_obj.setText("")

    def aktualizuj_status(self, tekst):
        """Helper function to change the text in the bottom label"""
        self.window.label.setText(tekst)

###################################################################
# ( 4 ) RUN GUI
###################################################################

    def run(self):
        app = QApplication(sys.argv)

        # Determine path to .ui file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        ui_file_path = os.path.join(current_dir, "GUI/Main_window.ui")

        try:
            self.window = uic.loadUi(ui_file_path)
            self.window.btnStart.setEnabled(False)

            # Override the mousePressEvent method for the imageLabel element
            self.window.imageLabel.mousePressEvent = self.klikniecie_w_obraz

            # Change cursor to "pointing hand" when hovering over the image area
            self.window.imageLabel.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))

            # Handle remaining buttons 
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
