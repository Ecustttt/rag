import os
import json
from dataclasses import dataclass, field
from docx import Document
import argparse


# ======================
# 配置
# ======================
@dataclass
class ParseConfig:
    image_ext_default: str = "png"
    image_name_pattern: str = "img_{index:04d}.{ext}"
    body_start_level: int = 1
    split_on_image_boundary: bool = True


# ======================
# 解析器
# ======================
class DocxParser:
    def __init__(self, docx_path, image_dir, config: ParseConfig = None):
        self.config = config or ParseConfig()
        self.docx_path = docx_path
        self.image_dir = image_dir
        os.makedirs(image_dir, exist_ok=True)
        self.doc = Document(docx_path)

        self.chunks = []
        self.current_titles = {}
        self.buffer = []
        self.images = []
        self.order = 0
        self.image_index = 0
        self.in_body = False
        self.seg_id = 0
        self.last_title_path = None

    def save_image(self, image_part):
        ext = image_part.content_type.split("/")[-1] or self.config.image_ext_default
        self.image_index += 1
        filename = self.config.image_name_pattern.format(
            index=self.image_index, ext=ext
        )
        path = os.path.join(self.image_dir, filename)
        with open(path, "wb") as f:
            f.write(image_part.blob)
        return path

    def extract_image_from_paragraph(self, paragraph):
        images = []
        for run in paragraph.runs:
            drawings = run._element.xpath(".//a:blip")
            for drawing in drawings:
                rId = drawing.get(
                    "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed"
                )
                if rId and rId in self.doc.part.related_parts:
                    image_part = self.doc.part.related_parts[rId]
                    img_path = self.save_image(image_part)
                    images.append(img_path)
        return images

    def get_title_path(self):
        return [
            self.current_titles[i]
            for i in sorted(self.current_titles)
            if self.current_titles.get(i)
        ]

    def _reset_titles_from(self, level, text):
        self.current_titles[level] = text
        for k in list(self.current_titles.keys()):
            if k > level:
                self.current_titles.pop(k, None)

    def flush_chunk(self):
        if not self.buffer:
            return

        current_path = self.get_title_path()
        if current_path != self.last_title_path:
            self.seg_id = 1
            self.last_title_path = current_path
        else:
            self.seg_id += 1

        self.order += 1

        title_str = current_path[-1] if current_path else ""
        content = "\n".join(self.buffer)
        if title_str and not content.startswith(title_str):
            content = f"{title_str}\n{content}"

        chunk = {
            "id": self.order,
            "seg_id": self.seg_id,
            "title_path": current_path,
            "content": content,
            "images": self.images.copy(),
        }
        self.chunks.append(chunk)
        self.buffer.clear()
        self.images.clear()

    def parse(self):
        for p in self.doc.paragraphs:
            text = p.text.strip()
            style = p.style.name

            if style.startswith("Heading"):
                level_str = style.replace("Heading ", "")
                if not level_str.isdigit():
                    continue
                level = int(level_str)

                if self.buffer or self.images:
                    self.flush_chunk()

                if level == self.config.body_start_level:
                    self.in_body = True

                if self.in_body:
                    self._reset_titles_from(level, text)
                    self.buffer.append(text)
                continue

            if not self.in_body:
                continue

            if (
                self.config.split_on_image_boundary
                and text
                and self.images
                and self.buffer
            ):
                self.flush_chunk()

            imgs = self.extract_image_from_paragraph(p)
            if imgs:
                self.images.extend(imgs)

            if text:
                self.buffer.append(text)

        self.flush_chunk()
        return self.chunks


# ======================
# 批处理
# ======================
def parse_docx_file(docx_path, image_dir, output_json, config: ParseConfig = None):
    parser = DocxParser(docx_path, image_dir, config)
    chunks = parser.parse()
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    return chunks


def parse_dir(doc_dir, image_base, config: ParseConfig = None,
              output_dir=None, exts=(".docx",)):
    out_root = output_dir or doc_dir
    os.makedirs(out_root, exist_ok=True)
    os.makedirs(image_base, exist_ok=True)

    results = {}
    for filename in os.listdir(doc_dir):
        if not filename.endswith(exts):
            continue
        doc_name = os.path.splitext(filename)[0]
        docx_path = os.path.join(doc_dir, filename)
        image_dir = os.path.join(image_base, doc_name)
        out_json = os.path.join(out_root, doc_name + ".json")

        chunks = parse_docx_file(docx_path, image_dir, out_json, config)
        results[filename] = len(chunks)
        print(f"Done: {filename} -> {len(chunks)} chunks")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="GWM", help="数据集名，如 GWM / MDM")
    args = ap.parse_args()
    ds = args.dataset

    cfg = ParseConfig()
    parse_dir(
        f"./data/docs/{ds}",
        f"./data/images/{ds}/",
        cfg,
        f"./data/docs/{ds}_json/",
    )