import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
photo_path = os.path.join(current_dir, "042222-256x256.png")
model_path = os.path.join(current_dir, "srcnn_epoch_31.pth")

# --- KLASA SRCNN ---
class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        # Etap 0: Powiększenie wstępne (Bicubic) do rozmiaru docelowego
        self.upsample = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=False)

        # Etap 1: Ekstrakcja cech (Patch extraction and representation)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.relu1 = nn.ReLU(inplace=True)

        # Etap 2: Mapowanie nieliniowe (Non-linear mapping)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU(inplace=True)

        # Etap 3: Rekonstrukcja (Reconstruction)
        self.conv3 = nn.Conv2d(32, 3, kernel_size=5, padding=2)
        
    def forward(self, x):
        x = self.upsample(x) 
        out = self.relu1(self.conv1(x))
        out = self.relu2(self.conv2(out))
        out = self.conv3(out)
        return out

# --- FUNKCJA UPSCALINGU x4 W DWÓCH KROKACH ---
def upscale_x4_progressive(image_path, model_path= model_path, output_name="result_1024x1024.png"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Ładowanie modelu
    print(f"Ładowanie modelu z: {model_path}")
    model = SRCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 2. Wczytanie obrazu (256x256)
    img = Image.open(image_path).convert('RGB')
    transform = T.ToTensor()
    # input_tensor: [1, 3, 256, 256]
    input_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        stage1_output = model(input_tensor)
        stage1_output = stage1_output.clamp(0, 1)
        final_output = model(stage1_output)
        final_output = final_output.clamp(0, 1)

    # 3. Zapis i wizualizacja
    output_path = "GUI/" + output_name
    result_img = T.ToPILImage()(final_output.squeeze(0).cpu())
    result_img.save(output_path)

    return output_path
