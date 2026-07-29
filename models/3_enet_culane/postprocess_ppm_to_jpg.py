import time
from pathlib import Path
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "output"

def save_with_retry(img, out_path, attempts=5, delay=0.4):
    last_err = None
    for i in range(attempts):
        try:
            img.save(out_path, format="JPEG", quality=95)
            return True
        except OSError as e:
            last_err = e
            time.sleep(delay)
    print(f"[ppm_to_jpg]  FAILED after {attempts} attempts: {out_path.name} -- {last_err}")
    return False

def main():
    ppms = list(OUTPUT_DIR.glob("*_overlay.ppm"))
    if not ppms:
        print(f"[ppm_to_jpg] No *_overlay.ppm files found in {OUTPUT_DIR}")
        return

    converted = 0
    for ppm_path in ppms:
        img = Image.open(ppm_path).convert("RGB")
        out_path = ppm_path.with_suffix(".jpg")
        if save_with_retry(img, out_path):
            print(f"[ppm_to_jpg]  {ppm_path.name} -> {out_path.name}")
            img.close()
            ppm_path.unlink()
            converted += 1
        else:
            img.close()

    for txt_path in OUTPUT_DIR.glob("*_exist.txt"):
        txt_path.unlink()
        print(f"[ppm_to_jpg]  removed {txt_path.name}")

    print(f"[ppm_to_jpg] Converted {converted}/{len(ppms)} image(s). Output folder now contains jpg only.")

if __name__ == "__main__":
    main()