from pathlib import Path
import sys
from docx import Document

SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(r"<third_party_sources>\融合多源遥感与城市语义的条件层级概率融合香港土地利用土地覆盖制图.docx")
OUT = Path(__file__).with_name("manuscript_structure.txt")

doc = Document(SRC)
lines = []
for i, p in enumerate(doc.paragraphs):
    txt = p.text.replace("\t", " ")
    if txt.strip():
        lines.append(f"P{i:03d} [{p.style.name}] {txt}")
lines.append("\n=== TABLES ===")
for ti, table in enumerate(doc.tables):
    lines.append(f"TABLE {ti}: {len(table.rows)} rows x {len(table.columns)} cols")
    for ri, row in enumerate(table.rows):
        cells = [c.text.replace("\n", " | ").replace("\t", " ") for c in row.cells]
        lines.append(f"  R{ri}: " + " || ".join(cells))
OUT.write_text("\n".join(lines), encoding="utf-8")
print(OUT)
