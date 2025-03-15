import os
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import random
import glob  # Adăugat importul lipsă


# Configurare
input_file = r"D:\Python\OCR\NewProject\CreateDataset\romanian.xlsx"  # Schimbă dacă numele fișierului este diferit
output_dir = "word_dataset"
font_dir = r"/Font"  # Schimbă la fonturile tale
img_size = (256, 64)  # Ajustează dimensiunea dacă este necesar
background_color = (255, 255, 255)
text_color = (0, 0, 0)

# Creează directorul de output
os.makedirs(output_dir, exist_ok=True)

# Încarcă lista de cuvinte
df = pd.read_excel(input_file, usecols=[0], header=None)  # Citește coloana A
words = df[0].dropna().astype(str).tolist()

# Listează fonturile disponibile
font_paths = glob.glob(font_dir + "/*.ttf") + glob.glob(font_dir + "/*.otf")



def generate_word_images():
    for font_path in font_paths:
        try:
            font = ImageFont.truetype(font_path, 40)
            font_name = os.path.basename(font_path).split('.')[0]

            for word in words:
                img = Image.new('RGB', img_size, background_color)
                draw = ImageDraw.Draw(img)

                bbox = draw.textbbox((0, 0), word, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]

                x = (img_size[0] - text_width) / 2 - bbox[0]
                y = (img_size[1] - text_height) / 2 - bbox[1]

                draw.text((x, y), word, fill=text_color, font=font)
                img.save(f"{output_dir}/{font_name}_{word}.png")

        except Exception as e:
            print(f"Font {font_path} incompatibil: {str(e)}")
            continue


if __name__ == "__main__":
    generate_word_images()
    print("Generare completă!")
