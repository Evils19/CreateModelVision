from PIL import Image, ImageDraw, ImageFont
import os
import glob

# Configurare
output_dir = "synthetic_dataset"
font_dir = r"D:\Python\CreateModelVision\Font"  # Schimbă la fonturile tale (ex: C:/Windows/Fonts pe Windows)
img_size = (64, 64)
background_color = (255, 255, 255)  # Alb
text_color = (0, 0, 0)  # Negru

# Caractere de generat (A-Z, diacritice, 0-9)
chars = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
    'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
    'Ă', 'Â', 'Î', 'Ș', 'Ț', 'ă', 'â', 'î', 'ș', 'ț',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'
]



# Creează directoare pentru fiecare caracter
for char in chars:
    os.makedirs(f"{output_dir}/{char}", exist_ok=True)

# Listează toate fonturile disponibile
font_paths = glob.glob(font_dir + "/*.ttf") + glob.glob(font_dir + "/*.otf")


def generate_char_images():
    for font_path in font_paths:
        try:
            font = ImageFont.truetype(font_path, 40)
            font_name = os.path.basename(font_path).split('.')[0]

            for char in chars:
                # Creează imagine nouă
                img = Image.new('RGB', img_size, background_color)
                draw = ImageDraw.Draw(img)

                # Calculează dimensiunea textului
                bbox = draw.textbbox((0, 0), char, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]

                # Centrare text
                x = (img_size[0] - text_width) / 2 - bbox[0]
                y = (img_size[1] - text_height) / 2 - bbox[1]

                # Desenează caracterul
                draw.text((x, y), char, fill=text_color, font=font)

                # Salvează imaginea
                img.save(f"{output_dir}/{char}/{font_name}_{char}.png")

        except Exception as e:
            print(f"Font {font_path} incompatibil: {str(e)}")
            continue


if __name__ == "__main__":
    generate_char_images()
    print("Generare completă!")