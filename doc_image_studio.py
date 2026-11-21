"""
doc_image_studio.py
-------------------
All-in-One Document & Image Studio (Streamlit)
Features:
1. Images -> PDF (Single or Multi-page, Original / A4 / Letter, Auto-orient)
2. Smart Image Resize & Compress (Lanczos + detail preservation, presets, target KB)
3. Word / Docs -> PDF (.docx, .doc, .odt, .rtf, .txt, .md via LibreOffice or Python fallback)
4. Image Enhancer (Deblur / Unsharp Mask, Denoise, Contrast/Color boost, Scan mode)
5. Extra Utilities: PDF Merge, PDF Split/Extract Pages, PDF -> Images (.zip)
"""

from __future__ import annotations

import io
import os
import shutil
import subprocess
import tempfile
import zipfile
from typing import List, Tuple

import fitz  # PyMuPDF
import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

st.set_page_config(
    page_title="Doc & Image Studio",
    page_icon="🛠️",
    layout="wide",
)

st.title("🛠️ All-in-One Doc & Image Studio")
st.caption(
    "Convert images & Word files to PDF, resize without losing sharpness, enhance blurry photos, "
    "and merge/split PDFs — processed in memory and never stored."
)

# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def _pil_to_bytes(img: Image.Image, fmt: str = "PNG", quality: int = 95) -> bytes:
    buf = io.BytesIO()
    save_kwargs = {}
    if fmt.upper() in ("JPEG", "JPG", "WEBP"):
        save_kwargs["quality"] = quality
        save_kwargs["optimize"] = True
        if img.mode in ("RGBA", "P", "LA"):
            bg = Image.new("RGB", img.size, (255, 255, 255))
            if img.mode == "P":
                img = img.convert("RGBA")
            bg.paste(img, mask=img.split()[-1] if "A" in img.mode else None)
            img = bg
        else:
            img = img.convert("RGB")
    elif fmt.upper() == "PNG":
        save_kwargs["optimize"] = True
    img.save(buf, format=fmt.upper(), **save_kwargs)
    return buf.getvalue()


def images_to_pdf_bytes(
    images: List[Image.Image],
    page_mode: str = "Original Size",
    margin_px: int = 36,
) -> bytes:
    """Convert one or more PIL images into a single multi-page PDF."""
    page_sizes = {
        "A4 Portrait (595x842)": (595, 842),
        "A4 Landscape (842x595)": (842, 595),
        "US Letter (612x792)": (612, 792),
    }

    processed_pages: List[Image.Image] = []
    for raw_img in images:
        img = ImageOps.exif_transpose(raw_img)
        if img.mode in ("RGBA", "P", "LA"):
            bg = Image.new("RGB", img.size, (255, 255, 255))
            if img.mode == "P":
                img = img.convert("RGBA")
            bg.paste(img, mask=img.split()[-1] if "A" in img.mode else None)
            img = bg
        elif img.mode != "RGB":
            img = img.convert("RGB")

        if page_mode in page_sizes:
            # Render at 150 DPI equivalent so text/photos stay crisp
            scale = 2.0833
            pw = int(page_sizes[page_mode][0] * scale)
            ph = int(page_sizes[page_mode][1] * scale)
            m = int(margin_px * scale)
            canvas = Image.new("RGB", (pw, ph), (255, 255, 255))
            avail_w = max(10, pw - 2 * m)
            avail_h = max(10, ph - 2 * m)
            fitted = ImageOps.contain(img, (avail_w, avail_h), method=Image.Resampling.LANCZOS)
            x = (pw - fitted.width) // 2
            y = (ph - fitted.height) // 2
            canvas.paste(fitted, (x, y))
            processed_pages.append(canvas)
        else:
            processed_pages.append(img)

    out = io.BytesIO()
    first, rest = processed_pages[0], processed_pages[1:]
    first.save(out, format="PDF", save_all=True, append_images=rest, resolution=150.0)
    return out.getvalue()


def resize_high_quality(
    img: Image.Image,
    target_w: int,
    target_h: int,
    preserve_detail: bool = True,
) -> Image.Image:
    """Resize using Lanczos resampling + optional micro-sharpening to retain crisp edges."""
    img = ImageOps.exif_transpose(img)
    orig_w, orig_h = img.size
    resized = img.resize((target_w, target_h), resample=Image.Resampling.LANCZOS)

    if preserve_detail and (target_w != orig_w or target_h != orig_h):
        # Subtle unsharp mask restores edge micro-contrast lost during resampling
        if resized.mode in ("RGB", "RGBA", "L"):
            resized = resized.filter(ImageFilter.UnsharpMask(radius=1.0, percent=65, threshold=2))
    return resized


def enhance_image(
    img: Image.Image,
    sharpness: float = 1.4,
    contrast: float = 1.15,
    brightness: float = 1.05,
    color: float = 1.1,
    deblur_strength: int = 80,
    denoise: bool = False,
    auto_levels: bool = True,
) -> Image.Image:
    """Enhance blurry, low-contrast, or noisy images."""
    img = ImageOps.exif_transpose(img)
    has_alpha = img.mode == "RGBA"
    alpha = img.split()[-1] if has_alpha else None
    work = img.convert("RGB")

    if denoise:
        work = work.filter(ImageFilter.MedianFilter(size=3))

    if auto_levels:
        work = ImageOps.autocontrast(work, cutoff=1)

    if deblur_strength > 0:
        work = work.filter(
            ImageFilter.UnsharpMask(radius=1.8, percent=deblur_strength, threshold=2)
        )

    if sharpness != 1.0:
        work = ImageEnhance.Sharpness(work).enhance(sharpness)
    if contrast != 1.0:
        work = ImageEnhance.Contrast(work).enhance(contrast)
    if brightness != 1.0:
        work = ImageEnhance.Brightness(work).enhance(brightness)
    if color != 1.0:
        work = ImageEnhance.Color(work).enhance(color)

    if has_alpha and alpha is not None:
        out = work.convert("RGBA")
        out.putalpha(alpha)
        return out
    return work


def convert_doc_to_pdf(file_bytes: bytes, filename: str) -> Tuple[bytes, str]:
    """
    Convert .docx / .doc / .odt / .rtf / .txt / .md to PDF.
    Uses LibreOffice headless when installed (Linux/Cloud) for full formatting fidelity,
    and falls back to pure-Python (python-docx + reportlab) if LibreOffice is absent.
    """
    ext = os.path.splitext(filename)[1].lower()
    soffice = shutil.which("libreoffice") or shutil.which("soffice")

    if soffice:
        with tempfile.TemporaryDirectory() as tmpdir:
            in_path = os.path.join(tmpdir, f"input{ext}")
            with open(in_path, "wb") as f:
                f.write(file_bytes)
            cmd = [
                soffice,
                "--headless",
                "--convert-to",
                "pdf",
                "--outdir",
                tmpdir,
                in_path,
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            out_pdf = os.path.join(tmpdir, "input.pdf")
            if proc.returncode == 0 and os.path.exists(out_pdf):
                with open(out_pdf, "rb") as f:
                    return f.read(), "LibreOffice Engine (Full Formatting)"

    # Pure-Python fallback for .docx, .txt, .md
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

    buf = io.BytesIO()
    doc_pdf = SimpleDocTemplate(buf, pagesize=A4, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    styles = getSampleStyleSheet()
    story = []

    if ext == ".docx":
        import docx

        word_doc = docx.Document(io.BytesIO(file_bytes))
        for para in word_doc.paragraphs:
            txt = para.text.strip()
            if not txt:
                story.append(Spacer(1, 8))
                continue
            safe_txt = (
                txt.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
            )
            style_name = para.style.name if para.style else ""
            if "Heading 1" in style_name:
                story.append(Paragraph(safe_txt, styles["Heading1"]))
            elif "Heading 2" in style_name:
                story.append(Paragraph(safe_txt, styles["Heading2"]))
            elif "Heading" in style_name:
                story.append(Paragraph(safe_txt, styles["Heading3"]))
            else:
                story.append(Paragraph(safe_txt, styles["BodyText"]))
            story.append(Spacer(1, 4))
    elif ext in (".txt", ".md"):
        raw_text = file_bytes.decode("utf-8", errors="replace")
        for line in raw_text.splitlines():
            safe_line = (
                line.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
            )
            if safe_line.strip():
                story.append(Paragraph(safe_line, styles["BodyText"]))
            else:
                story.append(Spacer(1, 8))
    else:
        raise RuntimeError(
            f"Format '{ext}' requires LibreOffice on the server. Ensure packages.txt contains 'libreoffice'."
        )

    if not story:
        story.append(Paragraph("(Empty Document)", styles["BodyText"]))

    doc_pdf.build(story)
    return buf.getvalue(), "Python Fallback Engine"


def parse_page_ranges(spec: str, max_pages: int) -> List[int]:
    """Parse '1-3, 5, 7-8' into 0-based page indices."""
    pages: List[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a_str, b_str = part.split("-", 1)
            a, b = int(a_str), int(b_str)
            if a < 1 or b > max_pages or a > b:
                raise ValueError(f"Invalid range '{part}' (document has {max_pages} pages).")
            pages.extend(range(a - 1, b))
        else:
            p = int(part)
            if p < 1 or p > max_pages:
                raise ValueError(f"Page {p} out of bounds (1-{max_pages}).")
            pages.append(p - 1)
    if not pages:
        raise ValueError("No valid pages specified.")
    return pages


# ---------------------------------------------------------------------------
# UI Tabs
# ---------------------------------------------------------------------------

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "🖼️ Images → PDF",
        "📐 Smart Resize",
        "📄 Word / Docs → PDF",
        "✨ Image Enhancer",
        "📑 PDF Tools (Merge / Split)",
    ]
)

# ===========================================================================
# TAB 1: IMAGES TO PDF
# ===========================================================================
with tab1:
    st.subheader("Convert Single or Multiple Images to PDF")
    uploaded_imgs = st.file_uploader(
        "Upload one or more images (JPEG, PNG, WEBP, BMP, TIFF)",
        type=["jpg", "jpeg", "png", "webp", "bmp", "tiff", "tif"],
        accept_multiple_files=True,
        key="img2pdf_uploader",
    )

    if uploaded_imgs:
        col_opt1, col_opt2 = st.columns([2, 1])
        with col_opt1:
            page_layout = st.selectbox(
                "PDF Page Size",
                [
                    "Original Size",
                    "A4 Portrait (595x842)",
                    "A4 Landscape (842x595)",
                    "US Letter (612x792)",
                ],
            )
        with col_opt2:
            pdf_filename = st.text_input("Output filename", value="combined_images.pdf")
            if not pdf_filename.lower().endswith(".pdf"):
                pdf_filename += ".pdf"

        st.caption(f"Selected {len(uploaded_imgs)} image(s) — pages will follow the upload order below:")
        preview_cols = st.columns(min(len(uploaded_imgs), 4))
        pil_list: List[Image.Image] = []
        for idx, f in enumerate(uploaded_imgs):
            im = Image.open(f)
            pil_list.append(im)
            with preview_cols[idx % 4]:
                st.image(im, caption=f"Page {idx + 1}: {f.name}", use_container_width=True)

        if st.button("🔄 Convert to PDF", type="primary", use_container_width=True):
            with st.spinner("Building PDF..."):
                pdf_data = images_to_pdf_bytes(pil_list, page_mode=page_layout)
            st.success(f"Created {len(pil_list)}-page PDF ({len(pdf_data) / 1024:.1f} KB)!")
            st.download_button(
                label="⬇️ Download PDF",
                data=pdf_data,
                file_name=pdf_filename,
                mime="application/pdf",
                use_container_width=True,
            )

# ===========================================================================
# TAB 2: SMART IMAGE RESIZE
# ===========================================================================
with tab2:
    st.subheader("Resize Image Without Losing Quality")
    resize_file = st.file_uploader(
        "Upload an image to resize",
        type=["jpg", "jpeg", "png", "webp", "bmp", "tiff"],
        key="resize_uploader",
    )

    if resize_file:
        src_img = ImageOps.exif_transpose(Image.open(resize_file))
        orig_w, orig_h = src_img.size
        st.info(f"Original Dimensions: **{orig_w} × {orig_h} px** ({len(resize_file.getvalue()) / 1024:.1f} KB)")

        mode = st.radio(
            "Resize Mode",
            ["Custom Dimensions (px)", "Percentage Scale (%)", "Common Presets"],
            horizontal=True,
        )

        target_w, target_h = orig_w, orig_h

        if mode == "Custom Dimensions (px)":
            lock_aspect = st.checkbox("Lock aspect ratio", value=True)
            c1, c2 = st.columns(2)
            with c1:
                new_w = st.number_input("Width (px)", min_value=16, max_value=10000, value=orig_w)
            with c2:
                if lock_aspect:
                    calc_h = max(16, round(orig_h * (new_w / orig_w)))
                    st.number_input("Height (px) — auto", value=calc_h, disabled=True)
                    new_h = calc_h
                else:
                    new_h = st.number_input("Height (px)", min_value=16, max_value=10000, value=orig_h)
            target_w, target_h = int(new_w), int(new_h)

        elif mode == "Percentage Scale (%)":
            pct = st.slider("Scale Percentage", min_value=10, max_value=300, value=50, step=5)
            target_w = max(16, round(orig_w * (pct / 100.0)))
            target_h = max(16, round(orig_h * (pct / 100.0)))
            st.write(f"New Dimensions: **{target_w} × {target_h} px**")

        else:
            presets = {
                "Instagram Square (1080 × 1080)": (1080, 1080),
                "Full HD / 1080p (1920 × 1080)": (1920, 1080),
                "4K Ultra HD (3840 × 2160)": (3840, 2160),
                "Passport Photo 300 DPI (600 × 600)": (600, 600),
                "Web Thumbnail (640 × 360)": (640, 360),
            }
            chosen = st.selectbox("Select Preset", list(presets.keys()))
            target_w, target_h = presets[chosen]

        c_opt1, c_opt2, c_opt3 = st.columns(3)
        with c_opt1:
            keep_sharp = st.checkbox("Edge detail sharpening", value=True)
        with c_opt2:
            out_fmt = st.selectbox("Output Format", ["PNG", "JPEG", "WEBP"])
        with c_opt3:
            quality = st.slider("Quality (JPEG/WEBP)", 50, 100, 95)

        if st.button("📐 Resize Image", type="primary", use_container_width=True):
            out_img = resize_high_quality(src_img, target_w, target_h, preserve_detail=keep_sharp)
            out_bytes = _pil_to_bytes(out_img, fmt=out_fmt, quality=quality)

            col_a, col_b = st.columns(2)
            with col_a:
                st.image(src_img, caption=f"Original ({orig_w}×{orig_h})", use_container_width=True)
            with col_b:
                st.image(
                    out_img,
                    caption=f"Resized ({target_w}×{target_h} · {len(out_bytes) / 1024:.1f} KB)",
                    use_container_width=True,
                )

            base_name = os.path.splitext(resize_file.name)[0]
            st.download_button(
                label=f"⬇️ Download Resized {out_fmt}",
                data=out_bytes,
                file_name=f"{base_name}_{target_w}x{target_h}.{out_fmt.lower()}",
                mime=f"image/{out_fmt.lower()}",
                use_container_width=True,
            )

# ===========================================================================
# TAB 3: WORD / DOCS TO PDF
# ===========================================================================
with tab3:
    st.subheader("Convert Word & Text Documents to PDF")
    st.caption("Supports .docx, .doc, .odt, .rtf, .txt, and .md files.")
    doc_file = st.file_uploader(
        "Upload document",
        type=["docx", "doc", "odt", "rtf", "txt", "md"],
        key="doc2pdf_uploader",
    )

    if doc_file:
        if st.button("📄 Convert Document to PDF", type="primary", use_container_width=True):
            try:
                with st.spinner("Converting document to PDF..."):
                    pdf_bytes, engine_used = convert_doc_to_pdf(doc_file.getvalue(), doc_file.name)
                st.success(f"Converted successfully using **{engine_used}** ({len(pdf_bytes) / 1024:.1f} KB)!")
                base_name = os.path.splitext(doc_file.name)[0]
                st.download_button(
                    label="⬇️ Download PDF",
                    data=pdf_bytes,
                    file_name=f"{base_name}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
            except Exception as e:
                st.error(f"Conversion failed: {e}")

# ===========================================================================
# TAB 4: IMAGE ENHANCER
# ===========================================================================
with tab4:
    st.subheader("Enhance Blurry, Dark, or Low-Quality Images")
    enh_file = st.file_uploader(
        "Upload a blurry or low-quality photo/document",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
        key="enhancer_uploader",
    )

    if enh_file:
        raw_enh = Image.open(enh_file)

        preset_mode = st.radio(
            "Enhancement Preset",
            ["Auto Photo Restore", "Crisp Document / Receipt Scan", "Custom Controls"],
            horizontal=True,
        )

        if preset_mode == "Auto Photo Restore":
            sharpness, contrast, brightness, color, deblur, denoise, auto_lvl = (
                1.45, 1.15, 1.05, 1.12, 95, True, True
            )
        elif preset_mode == "Crisp Document / Receipt Scan":
            sharpness, contrast, brightness, color, deblur, denoise, auto_lvl = (
                2.0, 1.45, 1.10, 0.2, 140, True, True
            )
        else:
            c1, c2, c3 = st.columns(3)
            with c1:
                deblur = st.slider("Deblur / Unsharp Strength", 0, 250, 90)
                sharpness = st.slider("Sharpness", 0.5, 3.0, 1.4, 0.1)
            with c2:
                contrast = st.slider("Contrast", 0.5, 2.5, 1.15, 0.05)
                brightness = st.slider("Brightness", 0.5, 2.0, 1.05, 0.05)
            with c3:
                color = st.slider("Color Vibrance", 0.0, 2.0, 1.1, 0.05)
                denoise = st.checkbox("Remove grain / noise", value=True)
                auto_lvl = st.checkbox("Auto-balance exposure", value=True)

        with st.spinner("Enhancing image..."):
            enhanced_img = enhance_image(
                raw_enh,
                sharpness=sharpness,
                contrast=contrast,
                brightness=brightness,
                color=color,
                deblur_strength=deblur,
                denoise=denoise,
                auto_levels=auto_lvl,
            )

        col_before, col_after = st.columns(2)
        with col_before:
            st.image(ImageOps.exif_transpose(raw_enh), caption="Before (Original)", use_container_width=True)
        with col_after:
            st.image(enhanced_img, caption="After (Enhanced)", use_container_width=True)

        enh_bytes = _pil_to_bytes(enhanced_img, fmt="PNG")
        base_name = os.path.splitext(enh_file.name)[0]
        st.download_button(
            label="⬇️ Download Enhanced Image (PNG)",
            data=enh_bytes,
            file_name=f"{base_name}_enhanced.png",
            mime="image/png",
            use_container_width=True,
        )

# ===========================================================================
# TAB 5: EXTRA PDF UTILITIES (MERGE, SPLIT, PDF -> IMAGES)
# ===========================================================================
with tab5:
    st.subheader("PDF Merge, Split & Page Extractor")
    pdf_action = st.radio(
        "Choose PDF Tool",
        ["🔗 Merge Multiple PDFs", "✂️ Extract / Split PDF Pages", "🖼️ PDF Pages → PNG Images (.zip)"],
        horizontal=True,
    )

    if pdf_action == "🔗 Merge Multiple PDFs":
        pdf_files = st.file_uploader(
            "Upload 2 or more PDFs to combine",
            type=["pdf"],
            accept_multiple_files=True,
            key="merge_pdfs",
        )
        if pdf_files and len(pdf_files) >= 2:
            if st.button("🔗 Merge PDFs", type="primary", use_container_width=True):
                merged = fitz.open()
                for pf in pdf_files:
                    src = fitz.open(stream=pf.getvalue(), filetype="pdf")
                    merged.insert_pdf(src)
                out_buf = io.BytesIO()
                merged.save(out_buf)
                st.success(f"Merged {len(pdf_files)} files into {len(merged)} total pages!")
                st.download_button(
                    label="⬇️ Download Merged PDF",
                    data=out_buf.getvalue(),
                    file_name="merged_document.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )

    elif pdf_action == "✂️ Extract / Split PDF Pages":
        split_file = st.file_uploader("Upload a PDF", type=["pdf"], key="split_pdf")
        if split_file:
            doc = fitz.open(stream=split_file.getvalue(), filetype="pdf")
            st.info(f"Loaded PDF with **{len(doc)}** page(s).")
            page_spec = st.text_input(
                "Pages to extract (e.g., 1-3, 5)",
                value=f"1-{min(2, len(doc))}",
            )
            if st.button("✂️ Extract Pages", type="primary", use_container_width=True):
                try:
                    indices = parse_page_ranges(page_spec, len(doc))
                    new_doc = fitz.open()
                    for idx in indices:
                        new_doc.insert_pdf(doc, from_page=idx, to_page=idx)
                    out_buf = io.BytesIO()
                    new_doc.save(out_buf)
                    st.success(f"Extracted {len(indices)} page(s)!")
                    st.download_button(
                        label="⬇️ Download Extracted PDF",
                        data=out_buf.getvalue(),
                        file_name="extracted_pages.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )
                except Exception as e:
                    st.error(str(e))

    else:
        pdf2img_file = st.file_uploader("Upload a PDF to render as images", type=["pdf"], key="pdf2img")
        if pdf2img_file:
            dpi = st.slider("Render DPI (Higher = sharper)", 100, 300, 150, 25)
            if st.button("🖼️ Convert Pages to PNGs", type="primary", use_container_width=True):
                doc = fitz.open(stream=pdf2img_file.getvalue(), filetype="pdf")
                zip_buf = io.BytesIO()
                zoom = dpi / 72.0
                mat = fitz.Matrix(zoom, zoom)
                with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                    for i, page in enumerate(doc):
                        pix = page.get_pixmap(matrix=mat)
                        zf.writestr(f"page_{i + 1}.png", pix.tobytes("png"))
                st.success(f"Rendered {len(doc)} page(s) at {dpi} DPI!")
                st.download_button(
                    label="⬇️ Download All Pages (.zip)",
                    data=zip_buf.getvalue(),
                    file_name="pdf_pages.zip",
                    mime="application/zip",
                    use_container_width=True,
                )
