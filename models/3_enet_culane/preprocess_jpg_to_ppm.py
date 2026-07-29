import sys
from pathlib import Path
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_DIR = SCRIPT_DIR / "input"
OUTPUT_DIR = SCRIPT_DIR / "output"

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    jpgs = list(INPUT_DIR.glob("*.jpg")) + list(INPUT_DIR.glob("*.jpeg")) + \
           list(INPUT_DIR.glob("*.JPG")) + list(INPUT_DIR.glob("*.JPEG"))

    if not jpgs:
        print(f"[jpg_to_ppm] No jpg files found in {INPUT_DIR}")
        return

    for jpg_path in jpgs:
        img = Image.open(jpg_path).convert("RGB")
        out_path = OUTPUT_DIR / (jpg_path.stem + ".ppm")
        img.save(out_path, format="PPM")
        print(f"[jpg_to_ppm]  {jpg_path.name} -> {out_path.name}  ({img.width}x{img.height})")

    print(f"[jpg_to_ppm] Converted {len(jpgs)} image(s).")

if __name__ == "__main__":
    main()