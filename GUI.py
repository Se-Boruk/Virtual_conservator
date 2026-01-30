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
from Upscaling.Upscaling_GUI import upscale_x4_progressive as up
from InPainter.Inpainter_GUI import InPainteR_GUI
import random
from Clustering.Clustering_GUI import Clustering


###################################################################
# ( 2 ) Class encoder, contains all functions for image processing
###################################################################

class Encoder:
    def __init__(self):
        self.image_path = None
        self.class_number = None

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
        
    def arrow_to_jpg(self, output_path="result.png"):
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
            img.save(output_path, "PNG")

            print(f"Sukces: Obraz został odtworzony i zapisany w {output_path}")
            return output_path

        except Exception as e:
            print(f"Błąd konwersji z Arrow do JPG: {e}")
            return None
        
    def save_tensor_as_png(self, image_tensor, output_path="damaged_output.png"):
        """
        image_tensor: Tensor of shape (1, C, H, W) or (C, H, W) in range [0, 1]
        output_path: File save path
        """
        output_path = "GUI/" + output_path
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
    
    def jpg_to_tensor(self, image_path, device):
            # Load, convert to RGB and resize (class default uses shape 256x256)
            original_pil = Image.open(image_path).convert("RGB")
            original_pil = original_pil.resize((256, 256)) 
            
            # Convert to numpy array and normalize to [0, 1] range
            img_np = np.array(original_pil, dtype=np.float32) / 255.0
            
            # Convert to PyTorch tensor: (H, W, C) -> (C, H, W) -> (B, C, H, W)
            # B (Batch) = 1
            img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device)

            return img_tensor

    def crash_image(self):
        print("Crashing image...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        try:
            img_tensor = self.jpg_to_tensor(self.image_path, device=device)
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

        self.damaged_image_path = self.save_tensor_as_png(damaged_img_tensor, "crashed_image.png")

        return self.damaged_image_path

    def Auto_encode(self):
        print("Fixing image...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        img_tensor = self.jpg_to_tensor(self.damaged_image_path, device=device)
        self.fixed_img_path, _ = InPainteR_GUI(img_tensor, "fixed_image.png")
        self.class_number = Clustering(self.image_path)
        return self.fixed_img_path

    def upscaler(self):
        print("Upscaling image...")
        return up(image_path= self.fixed_img_path, output_name="upscaled_image.png")

    
###################################################################
# ( 3 ) Class GUI, responsible for the graphical interface
###################################################################
        
class GUI():
    def __init__(self):
        self.window = None
        self.second_window = None
        self.third_window = None 
        self.fourth_window = None 
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
                self.status_update("Click start to process the image")
    
    def open_second_window(self):
        """Function loading and displaying Fix_Image_window.ui"""
        # Change status to processing
        self.status_update("Processing...")
        QCoreApplication.processEvents() # Refresh UI so the label updates
        
        # Simulate processing time (e.g., 1 second)
        time.sleep(1)

        # Load the second window if it doesn't exist yet
        if self.second_window is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            second_ui_path = os.path.join(current_dir, "GUI/Fix_Image_window.ui")
            self.second_window = uic.loadUi(second_ui_path)

        # Display the second window
        self.second_window.show()
        
        self.status_update("Done! Window opened.")

    def open_third_window(self):
        """Function loading and displaying Fix_&_Upscale_Image.ui"""
        # Change status to processing
        self.status_update("Processing...")
        QCoreApplication.processEvents() # Refresh UI so the label updates
        
        # Simulate processing time (e.g., 1 second)
        time.sleep(1)

        # Load the third window if it doesn't exist yet
        if self.third_window is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            third_ui_path = os.path.join(current_dir, "GUI/Fix_&_Upscale_Image.ui")
            self.third_window = uic.loadUi(third_ui_path)

        # Display the third window
        self.third_window.show()
        
        self.status_update("Done! Window opened.")

    def open_fourth_window(self):
        """Function loading and displaying Third_window.ui"""
        # Change status to processing
        self.status_update("Processing...")
        QCoreApplication.processEvents() # Refresh UI so the label updates
        
        # Simulate processing time (e.g., 1 second)
        time.sleep(1)

        # Load the third window if it doesn't exist yet
        if self.fourth_window is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            fourth_ui_path = os.path.join(current_dir, "GUI/Compare_Image.ui")
            self.fourth_window = uic.loadUi(fourth_ui_path)

        # Display the third window
        self.fourth_window.show()
        
        self.status_update("Done! Window opened.")

    def wyczysc_wszystko(self):
        """Function to clear image and reset status"""
        self.window.imageLabel.clear()
        self.window.imageLabel.setText("Click to add image")
        self.status_update("Image cleared. Click to add a new image.")
        self.window.btnStart.setEnabled(False)
    
    def load_image_from_class(self, class_no):
        folder_path = os.path.join("GUI", "zapisane_klasy", str(class_no))
        if not os.path.exists(folder_path):
            print(f"BŁĄD: Folder klasy nie istnieje: {folder_path}")
            return []
        
        try:
            all_files = os.listdir(folder_path)
            valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
            images = [f for f in all_files if f.lower().endswith(valid_extensions)]
        except Exception as e:
            print(f"Błąd odczytu folderu: {e}")
            return []
        
        if len(images) < 4:
            selected_files = images
            print(f"Uwaga: W klasie {class_no} znaleziono tylko {len(images)} zdjęć.")
        else:
            selected_files = random.sample(images, 4)

        # 5. Tworzenie pełnych ścieżek do plików
        full_paths = [os.path.join(folder_path, f) for f in selected_files]
        
        return full_paths


    def start_process(self):
        """Function called on Start click - checks DropDown and selects window"""
        wybor = self.window.DropDown.currentText()
        # tick_box = self.window.CompareImageCheckBox.isChecked()

        E = Encoder()  # Initialize Encoder class
        E.image_path = self.image
        path1 = E.crash_image()
        path2 = E.Auto_encode()
        path3 = E.upscaler()
        class_type = E.class_number
        
        if wybor == "Fix image":
            self.open_second_window()
            self.Load_Image_To_Label(self.second_window.Fixed_img, path2)
            self.Load_Image_To_Label(self.second_window.Damaged_img, path1)
            self.second_window.Class_no.setText(f"Numer klasy: {class_type}")


        elif wybor == "Fix & Upscale image":
            self.open_third_window()
            self.Load_Image_To_Label(self.third_window.Fixed_img, path2)
            self.Load_Image_To_Label(self.third_window.Damaged_img, path1)
            self.Load_Image_To_Label(self.third_window.Upscaled_img, path3)
            self.third_window.Class_no.setText(f"Numer klasy: {class_type}")

        # elif wybor == "Fix image" and tick_box == True:
        #     self.open_second_window()
        #     self.Load_Image_To_Label(self.second_window.Fixed_img, path2)
        #     self.Load_Image_To_Label(self.second_window.Damaged_img, path1)

        #     # self.open_fourth_window()
        #     # self.fourth_window.label.setText(f"Numer klasy: {class_type}")
        #     # Image_labels = [self.fourth_window.Image1, self.fourth_window.Image2, self.fourth_window.Image3, self.fourth_window.Image4]
        #     # examples = self.load_image_from_class(class_type)
        #     # for label, ex in zip(Image_labels, examples):
        #     #     self.Load_Image_To_Label(label, ex)

        # elif wybor == "Fix & Upscale image" and tick_box == True:
        #     self.open_third_window()
        #     self.Load_Image_To_Label(self.third_window.Fixed_img, path2)
        #     self.Load_Image_To_Label(self.third_window.Damaged_img, path1)
        #     self.Load_Image_To_Label(self.third_window.Upscaled_img, path3)
            
        #     # self.open_fourth_window()
        #     # self.fourth_window.label.setText(f"Numer klasy: {class_type}")
        #     # Image_labels = [self.fourth_window.Image1, self.fourth_window.Image2, self.fourth_window.Image3, self.fourth_window.Image4]
        #     # examples = self.load_image_from_class(class_type)
        #     # for label, ex in zip(Image_labels, examples):
        #     #     self.Load_Image_To_Label(label, ex)
            
        else:
            self.status_update("Select an option first!")

    def Load_Image_To_Label(self, label_obj, sciezka_obrazu):
        """Helper function to load an image into a specific label"""
        pixmap = QPixmap(sciezka_obrazu)
        if not pixmap.isNull():
            label_obj.setPixmap(pixmap.scaled(
                label_obj.size(), 
                Qt.AspectRatioMode.KeepAspectRatio, 
                Qt.TransformationMode.SmoothTransformation
            ))
            label_obj.setText("")

    def status_update(self, tekst):
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
