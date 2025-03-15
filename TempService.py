import os
import zipfile
import glob
from pathlib import Path


def extract_fonts_from_zips(source_dir: str, target_dir: str) -> int:
    """
    Extrage fișierele .ttf și .otf din toate arhivele .zip dintr-un folder sursă.

    Args:
        source_dir (str): Către folderul cu arhive .zip
        target_dir (str): Către folderul destinație pentru fonturi

    Returns:
        int: Numărul de fișiere extrase
    """
    # Creează folderul destinație dacă nu există
    Path(target_dir).mkdir(parents=True, exist_ok=True)

    # Caută toate fișierele .zip în folderul sursă
    zip_files = glob.glob(os.path.join(source_dir, "*.zip"))
    extracted_count = 0


    for zip_path in zip_files:
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                for file_info in zip_ref.infolist():
                    # Ignoră directoarele
                    if file_info.is_dir():
                        continue

                    # Verifică extensia fișierului
                    file_ext = os.path.splitext(file_info.filename)[1].lower()
                    if file_ext in ('.ttf', '.otf'):
                        # Extrage fișierul
                        zip_ref.extract(file_info, target_dir)
                        extracted_count += 1

                        # Redenumește fișierul extras pentru a evita conflictele
                        original_path = os.path.join(target_dir, file_info.filename)
                        new_path = os.path.join(target_dir, f"{Path(zip_path).stem}_{Path(file_info.filename).name}")
                        os.rename(original_path, new_path)

        except zipfile.BadZipFile:
            print(f"⚠ Avertisment: {zip_path} este corupt sau nu este o arhivă validă.")
        except Exception as e:
            print(f"⛔ Eroare la procesarea {zip_path}: {str(e)}")

    return extracted_count


if __name__ == "__main__":
    source = r"C:\Users\jitar\Downloads\Temp"  # Folderul cu fișiere .zip
    destination = r"C:\Users\jitar\Downloads\Fonts"  # Folderul destinație

    count = extract_fonts_from_zips(source, destination)
    print(f"✅ Extrase {count} fișiere font în {destination}")