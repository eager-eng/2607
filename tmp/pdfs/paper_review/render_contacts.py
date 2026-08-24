from pathlib import Path
import sys

import pypdfium2 as pdfium
from PIL import Image, ImageDraw


pdf_path = Path(sys.argv[1])
output_dir = Path(sys.argv[2])
output_dir.mkdir(parents=True, exist_ok=True)
document = pdfium.PdfDocument(pdf_path)
pages = []

for index in range(min(48, len(document))):
    image = document[index].render(scale=1.0).to_pil().convert("RGB")
    image.thumbnail((620, 880))
    canvas = Image.new("RGB", (640, 920), "white")
    canvas.paste(image, ((640 - image.width) // 2, 30))
    ImageDraw.Draw(canvas).text((16, 8), f"Page {index + 1}", fill="black")
    pages.append(canvas)

for start in range(0, len(pages), 6):
    batch = pages[start : start + 6]
    sheet = Image.new("RGB", (1920, 1840), "#DDDDDD")
    for offset, page in enumerate(batch):
        x = (offset % 3) * 640
        y = (offset // 3) * 920
        sheet.paste(page, (x, y))
    first = start + 1
    last = start + len(batch)
    sheet.save(output_dir / f"contact_{first:02d}_{last:02d}.png")
