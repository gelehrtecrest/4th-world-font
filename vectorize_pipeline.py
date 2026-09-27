import os
import re
import sys
import argparse
import cv2
import numpy as np
from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen
from PIL import Image, ImageDraw, ImageFont

# -------------------------------------------------------------
# Configuration & Typography Metrics (UPM = 1000)
# -------------------------------------------------------------
UPM = 1000
ASCENT = 850
DESCENT = -200
CAP_HEIGHT = 720
X_HEIGHT = 520
BASELINE_SVG = 800  # in 1000x1000 SVG canvas

DATA_DIR = "data"
OUT_DIR = "output"
CLEAN_PNG_DIR = os.path.join(OUT_DIR, "cleaned_png")
SVG_DIR = os.path.join(OUT_DIR, "svg")
FONT_DIR = os.path.join(OUT_DIR, "fonts")

for d in [OUT_DIR, CLEAN_PNG_DIR, SVG_DIR, FONT_DIR]:
    os.makedirs(d, exist_ok=True)

# -------------------------------------------------------------
# File name parser: large_x.jpg / small_x.jpg / large-x.jpg
# -------------------------------------------------------------
def parse_char_info(filename):
    m = re.match(r"(large|small)[_-]([a-zA-Z0-9])\.(jpg|jpeg|png)", filename, re.IGNORECASE)
    if not m:
        return None
    case_type = m.group(1).lower()
    char_raw = m.group(2)
    char_str = char_raw.upper() if case_type == "large" else char_raw.lower()
    return char_str, (case_type == "large"), char_str

# -------------------------------------------------------------
# Image Cleaning with smooth_level (0..100)
# -------------------------------------------------------------
def clean_and_binarize(img_bgr, filename="", smooth_level=50):
    t = max(0.0, min(100.0, float(smooth_level))) / 100.0
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    scale = 4
    h_orig, w_orig = gray.shape
    gray_up = cv2.resize(gray, (w_orig * scale, h_orig * scale), interpolation=cv2.INTER_CUBIC)
    
    # Adaptive filter based on smooth_level
    if t < 0.1:
        filtered = cv2.GaussianBlur(gray_up, (3, 3), 0)
    elif t < 0.5:
        filtered = cv2.bilateralFilter(gray_up, 5, 40, 40)
    else:
        filtered = cv2.bilateralFilter(gray_up, 7, 60, 60)
        
    _, thresh = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh)
    if num_labels <= 1:
        return thresh
        
    areas = [stats[i, cv2.CC_STAT_AREA] for i in range(1, num_labels)]
    max_area_idx = 1 + int(np.argmax(areas))
    max_area = stats[max_area_idx, cv2.CC_STAT_AREA]
    
    clean_mask = np.zeros_like(thresh)
    h_up, w_up = thresh.shape
    
    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        
        touches_left = (x == 0)
        touches_right = (x + w >= w_up)
        
        # Remove small edge artifacts from neighboring cropped letters
        if (touches_left or touches_right) and i != max_area_idx:
            if area < max_area * 0.35 or w < w_up * 0.15:
                print(f"[{filename}] Cleaned neighboring fragment #{i}: area={area}, bbox=({x},{y},{w},{h})")
                continue
                
        if area < 60:
            continue
            
        clean_mask[labels == i] = 255

    return clean_mask

# -------------------------------------------------------------
# Blended Contour Smoothing (0: Raw, 50: Target, 100: Max Smooth)
# -------------------------------------------------------------
def get_smoothed_contours(cleaned_mask, smooth_level=50):
    t = max(0.0, min(100.0, float(smooth_level))) / 100.0
    
    # Epsilon scales continuously from 0.7 (raw) to 1.35 (max smooth)
    epsilon = 0.7 + (1.35 - 0.7) * t
    
    contours, hierarchy = cv2.findContours(cleaned_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hierarchy is None or len(contours) == 0:
        return []
        
    hier = hierarchy[0]
    result = []
    
    for idx, cnt in enumerate(contours):
        approx = cv2.approxPolyDP(cnt, epsilon, closed=True)
        if len(approx) < 3:
            continue
        pts_raw = approx.reshape(-1, 2).astype(np.float64)
        n = len(pts_raw)
        
        if t <= 0.01:
            is_hole = (hier[idx][3] != -1)
            result.append((pts_raw, is_hole, np.zeros(n, dtype=bool)))
            continue
            
        # Corner detection (turning angle threshold scales with t)
        # Lower t retains more subtle angles as corners
        corner_angle_deg = 42.0 + 16.0 * t
        cos_thresh = np.cos(np.radians(corner_angle_deg))
        is_corner = np.zeros(n, dtype=bool)
        for i in range(n):
            v1 = pts_raw[i] - pts_raw[(i - 1) % n]
            v2 = pts_raw[(i + 1) % n] - pts_raw[i]
            n1 = np.linalg.norm(v1)
            n2 = np.linalg.norm(v2)
            if n1 > 1e-4 and n2 > 1e-4:
                dot = np.dot(v1, v2) / (n1 * n2)
                if dot < cos_thresh:
                    is_corner[i] = True
                    
        # Laplacian smoothing on non-corner points
        iters = int(np.round(1 + 2 * t))
        weight = 0.15 + 0.35 * t
        
        pts_smooth = pts_raw.copy()
        for _ in range(iters):
            temp = pts_smooth.copy()
            for i in range(n):
                if not is_corner[i]:
                    neighbor_avg = 0.5 * (pts_smooth[(i - 1) % n] + pts_smooth[(i + 1) % n])
                    temp[i] = (1.0 - weight) * pts_smooth[i] + weight * neighbor_avg
            pts_smooth = temp
            
        # Linear blend between raw and smoothed
        final_pts = (1.0 - t) * pts_raw + t * pts_smooth
        is_hole = (hier[idx][3] != -1)
        result.append((final_pts, is_hole, is_corner))
        
    return result

# -------------------------------------------------------------
# Smooth SVG Generation
# -------------------------------------------------------------
def build_svg(contour_data, mask_shape, is_large, out_svg_path):
    h_mask, w_mask = mask_shape
    target_h = CAP_HEIGHT if is_large else X_HEIGHT
    scale = target_h / h_mask
    
    all_pts = []
    for pts, _, _ in contour_data:
        all_pts.extend(pts)
    if not all_pts:
        return 0, 0
    all_pts = np.array(all_pts)
    min_x, max_x = all_pts[:, 0].min(), all_pts[:, 0].max()
    min_y, max_y = all_pts[:, 1].min(), all_pts[:, 1].max()
    
    glyph_w = (max_x - min_x) * scale
    glyph_h = (max_y - min_y) * scale
    
    svg_off_x = (1000.0 - glyph_w) / 2.0 - min_x * scale
    svg_off_y = BASELINE_SVG - max_y * scale
    
    svg_paths = []
    for pts, _, is_corner in contour_data:
        n = len(pts)
        if n < 3:
            continue
        trans = pts * scale + [svg_off_x, svg_off_y]
        
        d = []
        start_pt = ((trans[-1] + trans[0]) / 2.0)
        d.append(f"M {start_pt[0]:.2f} {start_pt[1]:.2f}")
        
        for i in range(n):
            curr = trans[i]
            nxt = trans[(i + 1) % n]
            mid = (curr + nxt) / 2.0
            if is_corner[i]:
                d.append(f"L {curr[0]:.2f} {curr[1]:.2f}")
                d.append(f"L {mid[0]:.2f} {mid[1]:.2f}")
            else:
                d.append(f"Q {curr[0]:.2f} {curr[1]:.2f}, {mid[0]:.2f} {mid[1]:.2f}")
        d.append("Z")
        svg_paths.append(" ".join(d))
        
    svg_str = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 1000" width="1000" height="1000">
  <path d="{' '.join(svg_paths)}" fill="#1a1a1a" fill-rule="evenodd" />
</svg>'''
    with open(out_svg_path, "w", encoding="utf-8") as f:
        f.write(svg_str)
        
    return glyph_w, glyph_h

# -------------------------------------------------------------
# Smooth TTF Glyph Generation
# -------------------------------------------------------------
def build_ttf_glyph(contour_data, mask_shape, is_large):
    h_mask, w_mask = mask_shape
    target_h = CAP_HEIGHT if is_large else X_HEIGHT
    scale = target_h / h_mask
    
    all_pts = []
    for pts, _, _ in contour_data:
        all_pts.extend(pts)
    if not all_pts:
        pen = TTGlyphPen(None)
        return pen.glyph(), 500, 0
        
    all_pts = np.array(all_pts)
    min_x, max_x = all_pts[:, 0].min(), all_pts[:, 0].max()
    min_y, max_y = all_pts[:, 1].min(), all_pts[:, 1].max()
    
    glyph_w = (max_x - min_x) * scale
    lsb = 50
    advance_width = int(glyph_w + lsb * 2)
    
    pen = TTGlyphPen(None)
    
    for pts, is_hole, is_corner in contour_data:
        n = len(pts)
        if n < 3:
            continue
            
        font_pts = []
        corner_flags = []
        for i in range(n):
            fx = (pts[i][0] - min_x) * scale + lsb
            fy = (max_y - pts[i][1]) * scale
            font_pts.append((fx, fy))
            corner_flags.append(is_corner[i])
            
        # Orientation check
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += font_pts[i][0] * font_pts[j][1] - font_pts[j][0] * font_pts[i][1]
            
        if not is_hole and area > 0:
            font_pts.reverse()
            corner_flags.reverse()
        elif is_hole and area < 0:
            font_pts.reverse()
            corner_flags.reverse()
            
        # Draw with TrueType quadratic curves
        start_pt = ((font_pts[-1][0] + font_pts[0][0]) / 2.0, (font_pts[-1][1] + font_pts[0][1]) / 2.0)
        pen.moveTo(start_pt)
        for i in range(n):
            curr = font_pts[i]
            nxt = font_pts[(i + 1) % n]
            mid = ((curr[0] + nxt[0]) / 2.0, (curr[1] + nxt[1]) / 2.0)
            if corner_flags[i]:
                pen.lineTo(curr)
                pen.lineTo(mid)
            else:
                pen.qCurveTo(curr, mid)
        pen.closePath()
        
    return pen.glyph(), advance_width, lsb

# -------------------------------------------------------------
# Generate Levels Comparison Image (0, 25, 50, 75, 100)
# -------------------------------------------------------------
def generate_levels_comparison_image(out_path):
    test_chars = ["large_i.jpg", "large_u.jpg", "small_e.jpg", "small_n.jpg", "small_r.jpg"]
    test_chars = [f for f in test_chars if os.path.exists(os.path.join(DATA_DIR, f))]
    if not test_chars:
        return
        
    levels = [0, 25, 50, 75, 100]
    cell_w, cell_h = 160, 180
    rows = len(levels)
    cols = len(test_chars)
    comp_img = np.full((cell_h * rows + 60, cell_w * cols + 150, 3), 255, dtype=np.uint8)

    for col_idx, f in enumerate(test_chars):
        x_s = 150 + col_idx * cell_w
        cv2.putText(comp_img, f.split('.')[0], (x_s + 35, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 40, 40), 1, cv2.LINE_AA)

    for row_idx, lvl in enumerate(levels):
        y_s = 60 + row_idx * cell_h
        label_text = f"Level {lvl}"
        if lvl == 0:
            label_text += " (Raw)"
        elif lvl == 50:
            label_text += " (Default)"
        elif lvl == 100:
            label_text += " (Smooth)"
        col_txt = (0, 80, 200) if lvl == 50 else (80, 80, 80)
        cv2.putText(comp_img, label_text, (10, y_s + 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col_txt, 1, cv2.LINE_AA)
        
        for col_idx, f in enumerate(test_chars):
            raw_img = cv2.imread(os.path.join(DATA_DIR, f))
            mask = clean_and_binarize(raw_img, f, smooth_level=lvl)
            cnts = get_smoothed_contours(mask, smooth_level=lvl)
            
            sub = np.full((cell_h, cell_w, 3), 255, dtype=np.uint8)
            s = 130.0 / mask.shape[0]
            off_x = (cell_w - mask.shape[1] * s) / 2.0
            
            for pts, is_h, _ in cnts:
                pts_s = (pts * s + [off_x, 20]).astype(np.int32)
                col = (255, 255, 255) if is_h else ((25, 60, 130) if lvl == 50 else (45, 45, 45))
                cv2.drawContours(sub, [pts_s], -1, col, -1, lineType=cv2.LINE_AA)
                cv2.polylines(sub, [pts_s], True, (0, 120, 220) if lvl == 50 else (120, 120, 120), 1, lineType=cv2.LINE_AA)
                
            x_start = 150 + col_idx * cell_w
            comp_img[y_s:y_s+cell_h, x_start:x_start+cell_w] = sub

    cv2.imwrite(out_path, comp_img)

# -------------------------------------------------------------
# Main Pipeline Execution
# -------------------------------------------------------------
def run_pipeline(smooth_level=50):
    print("=" * 60)
    print(f"FF14 Fourth World Font Generation Pipeline (Smooth Level: {smooth_level})")
    print("=" * 60)
    
    files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith((".jpg", ".jpeg", ".png"))])
    print(f"Found {len(files)} files in '{DATA_DIR}'.")
    
    char_records = []
    ttf_glyphs = {}
    ttf_metrics = {}
    cmap = {}
    
    # Default notdef & space glyphs
    notdef_pen = TTGlyphPen(None)
    notdef_pen.moveTo((100, 0))
    notdef_pen.lineTo((100, CAP_HEIGHT))
    notdef_pen.lineTo((400, CAP_HEIGHT))
    notdef_pen.lineTo((400, 0))
    notdef_pen.closePath()
    ttf_glyphs[".notdef"] = notdef_pen.glyph()
    ttf_metrics[".notdef"] = (500, 100)
    
    space_pen = TTGlyphPen(None)
    ttf_glyphs["space"] = space_pen.glyph()
    ttf_metrics["space"] = (300, 0)
    cmap[ord(' ')] = "space"
    
    for f in files:
        parsed = parse_char_info(f)
        if not parsed:
            continue
            
        char_str, is_large, glyph_name = parsed
        img_path = os.path.join(DATA_DIR, f)
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        cleaned = clean_and_binarize(img, f, smooth_level=smooth_level)
        base_name = os.path.splitext(f)[0]
        
        clean_png_path = os.path.join(CLEAN_PNG_DIR, f"{base_name}.png")
        cv2.imwrite(clean_png_path, cleaned)
        
        # Extract blended contours
        contours = get_smoothed_contours(cleaned, smooth_level=smooth_level)
        
        # Generate SVG
        svg_filename = f"{base_name}.svg"
        svg_path = os.path.join(SVG_DIR, svg_filename)
        gw, gh = build_svg(contours, cleaned.shape, is_large, svg_path)
        
        # Generate TTF glyph
        glyph, adv_w, lsb = build_ttf_glyph(contours, cleaned.shape, is_large)
        ttf_glyphs[glyph_name] = glyph
        ttf_metrics[glyph_name] = (adv_w, lsb)
        cmap[ord(char_str)] = glyph_name
        
        char_records.append({
            "char": char_str,
            "is_large": is_large,
            "orig_file": f,
            "clean_file": f"{base_name}.png",
            "svg_file": svg_filename,
            "width": int(gw),
            "height": int(gh),
            "adv_width": adv_w,
        })
        print(f"  Processed '{char_str}' ({'Upper' if is_large else 'Lower'}): size={gw:.1f}x{gh:.1f}, advance={adv_w}")

    # Build TTF Font
    font_path = os.path.join(FONT_DIR, "FF14FourthWorld.ttf")
    fb = FontBuilder(UPM, isTTF=True)
    glyph_order = [".notdef", "space"] + [k for k in ttf_glyphs.keys() if k not in [".notdef", "space"]]
    fb.setupGlyphOrder(glyph_order)
    fb.setupCharacterMap(cmap)
    fb.setupGlyf(ttf_glyphs)
    fb.setupHorizontalMetrics(ttf_metrics)
    fb.setupHorizontalHeader(ascent=ASCENT, descent=DESCENT)
    fb.setupNameTable({
        "familyName": "FF14 Fourth World",
        "styleName": "Regular",
        "uniqueFontIdentifier": f"FF14FourthWorld-Regular:S{smooth_level}:2026",
        "fullName": "FF14 Fourth World Regular",
        "version": f"Version 1.3 (Smooth {smooth_level}); 2026",
        "psName": "FF14FourthWorld-Regular",
    })
    fb.setupOS2(
        sTypoAscender=ASCENT,
        sTypoDescender=DESCENT,
        usWinAscent=ASCENT,
        usWinDescent=abs(DESCENT),
        sxHeight=X_HEIGHT,
        sCapHeight=CAP_HEIGHT,
    )
    fb.setupPost()
    fb.save(font_path)
    print(f"\nSuccessfully compiled TrueType font: {font_path}")

    # Generate levels comparison chart image
    chart_img_path = os.path.join(OUT_DIR, "compare_levels.png")
    generate_levels_comparison_image(chart_img_path)
    print(f"Successfully generated smooth levels chart: {chart_img_path}")

    # Build Interactive HTML Preview
    html_path = os.path.join(OUT_DIR, "preview.html")
    build_html_preview(char_records, html_path, smooth_level)
    print(f"Successfully generated preview dashboard: {html_path}")

    # Generate visual sample image
    try:
        font_large = ImageFont.truetype(font_path, 72)
        font_small = ImageFont.truetype(font_path, 40)
        font_label = ImageFont.load_default()

        canvas = Image.new("RGB", (960, 480), color=(252, 252, 254))
        draw = ImageDraw.Draw(canvas)

        draw.text((32, 24), f"FF14 Fourth World Font (Smooth Level: {smooth_level}/100)", fill=(40, 40, 50), font=font_label)
        draw.text((32, 58), f"Available Glyphs ({len(char_records)}): {' '.join([r['char'] for r in char_records])}", fill=(100, 100, 110), font=font_label)
        draw.text((32, 85), " ".join([r['char'] for r in char_records]), fill=(25, 30, 45), font=font_large)

        draw.text((32, 195), "Sample Text 1: 'Unmoored Isle'", fill=(100, 100, 110), font=font_label)
        draw.text((32, 222), "Unmoored Isle", fill=(25, 30, 45), font=font_large)

        draw.text((32, 335), "Sample Text 2: 'red rose / iron soul / moon order'", fill=(100, 100, 110), font=font_label)
        draw.text((32, 362), "red rose  iron soul  moon order", fill=(25, 30, 45), font=font_small)

        sample_img_path = os.path.join(OUT_DIR, "sample_render.png")
        canvas.save(sample_img_path)
        print(f"Successfully generated sample image: {sample_img_path}")
    except Exception as e:
        print(f"Sample image generation skipped: {e}")

# -------------------------------------------------------------
# HTML Preview Dashboard Generation
# -------------------------------------------------------------
def build_html_preview(records, html_path, smooth_level=50):
    rows_html = []
    for r in records:
        rows_html.append(f"""
        <tr>
            <td style="font-weight:bold; font-size:1.4rem; text-align:center;">{r['char']}</td>
            <td style="text-align:center;"><span class="badge {'badge-upper' if r['is_large'] else 'badge-lower'}">{'大文字' if r['is_large'] else '小文字'}</span></td>
            <td style="text-align:center;"><img src="../data/{r['orig_file']}" class="preview-img" alt="{r['char']} original" /></td>
            <td style="text-align:center;"><img src="cleaned_png/{r['clean_file']}" class="preview-img" alt="{r['char']} clean" /></td>
            <td style="text-align:center;"><img src="svg/{r['svg_file']}" class="preview-svg" alt="{r['char']} svg" /></td>
            <td style="font-family:'FF14FourthWorld'; font-size:2.4rem; text-align:center; color:#2c3e50;">{r['char']}</td>
            <td style="font-size:0.85rem; color:#666; text-align:center;">{r['width']} x {r['height']} (Adv: {r['adv_width']})</td>
        </tr>
        """)
        
    html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<title>FF14 第四世界文字 フォントプレビュー (Smooth Level: {smooth_level})</title>
<style>
    @font-face {{
        font-family: 'FF14FourthWorld';
        src: url('fonts/FF14FourthWorld.ttf?t={os.urandom(4).hex()}') format('truetype');
    }}
    body {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        background-color: #f5f7fa;
        color: #333;
        margin: 0;
        padding: 24px;
    }}
    .container {{
        max-width: 1100px;
        margin: 0 auto;
        background: #fff;
        padding: 32px;
        border-radius: 12px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.08);
    }}
    h1 {{
        margin-top: 0;
        color: #1a252f;
        border-bottom: 2px solid #3498db;
        padding-bottom: 12px;
    }}
    .subtitle {{
        color: #555;
        margin-bottom: 20px;
        line-height: 1.6;
    }}
    .level-badge {{
        display: inline-block;
        background: #2563eb;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 0.9rem;
    }}
    .interactive-box {{
        background: #eef2f7;
        padding: 20px;
        border-radius: 8px;
        margin-bottom: 30px;
    }}
    .interactive-box textarea {{
        width: 100%;
        box-sizing: border-box;
        padding: 12px;
        font-size: 1.1rem;
        border: 1px solid #ccc;
        border-radius: 6px;
        font-family: inherit;
    }}
    .rendered-display {{
        margin-top: 14px;
        padding: 18px;
        background: #fff;
        border-radius: 6px;
        min-height: 80px;
        font-family: 'FF14FourthWorld';
        font-size: 2.8rem;
        line-height: 1.4;
        word-break: break-all;
        border: 1px dashed #3498db;
        color: #2c3e50;
    }}
    table {{
        width: 100%;
        border-collapse: collapse;
        margin-top: 20px;
    }}
    th, td {{
        padding: 12px 14px;
        border-bottom: 1px solid #e1e8ed;
    }}
    th {{
        background-color: #f8fafc;
        color: #4a5568;
        font-size: 0.9rem;
    }}
    .preview-img {{
        max-height: 52px;
        image-rendering: pixelated;
        border: 1px solid #ddd;
        border-radius: 4px;
        background: #fff;
        padding: 2px;
    }}
    .preview-svg {{
        max-height: 52px;
        border: 1px solid #ddd;
        border-radius: 4px;
        background: #fff;
        padding: 2px;
    }}
    .badge {{
        display: inline-block;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: bold;
    }}
    .badge-upper {{
        background: #e3f2fd;
        color: #1976d2;
    }}
    .badge-lower {{
        background: #e8f5e9;
        color: #388e3c;
    }}
    .sample-tags {{
        margin-top: 8px;
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
    }}
    .sample-btn {{
        background: #fff;
        border: 1px solid #cbd5e1;
        padding: 4px 10px;
        border-radius: 14px;
        font-size: 0.85rem;
        cursor: pointer;
        transition: 0.2s;
    }}
    .sample-btn:hover {{
        background: #3498db;
        color: #fff;
        border-color: #3498db;
    }}
    .chart-container {{
        margin: 24px 0;
        background: #fafafa;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }}
    .chart-container img {{
        max-width: 100%;
        height: auto;
        border-radius: 6px;
    }}
</style>
</head>
<body>
<div class="container">
    <h1>FF14 第四世界文字 フォントプレビュー</h1>
    <div class="subtitle">
        現在のスムージング設定: <span class="level-badge">Level {smooth_level} / 100</span><br>
        （0 = 元画像そのままの完全生データ ／ 50 = ささくれを除去しつつ原画の筆致を残した中間設定 ／ 100 = 最大平滑化）
    </div>

    <div class="interactive-box">
        <h3>試し打ちテスト（リアルタイム入力）</h3>
        <p style="margin: 4px 0 10px 0; font-size: 0.9rem; color: #555;">
            下のテキストボックスに入力した文字が、生成された第四世界フォントで表示されます。
        </p>
        <textarea id="testInput" rows="2">Unmoored Isle</textarea>
        <div class="sample-tags">
            <span style="font-size: 0.85rem; color: #666; align-self: center;">サンプル単語:</span>
            <button class="sample-btn" onclick="setInput('Unmoored Isle')">Unmoored Isle</button>
            <button class="sample-btn" onclick="setInput('red rose')">red rose</button>
            <button class="sample-btn" onclick="setInput('iron soul')">iron soul</button>
            <button class="sample-btn" onclick="setInput('elder line')">elder line</button>
            <button class="sample-btn" onclick="setInput('moon order')">moon order</button>
            <button class="sample-btn" onclick="setInput('miles')">miles</button>
        </div>
        <div class="rendered-display" id="renderOutput"></div>
    </div>

    <div class="chart-container">
        <h3 style="text-align:left; margin-top:0;">スムージング強度（Level 0〜100）の比較チャート</h3>
        <p style="text-align:left; font-size:0.9rem; color:#666;">
            コマンド実行時に <code>--smooth [0〜100]</code> を指定することで、お好みの寄せ具合でフォントを再生成できます。
        </p>
        <img src="compare_levels.png" alt="Smooth levels comparison" />
    </div>

    <h3>登録文字一覧＆変換検証</h3>
    <table>
        <thead>
            <tr>
                <th>文字</th>
                <th>種別</th>
                <th>元画像 (data/)</th>
                <th>二値化マスク</th>
                <th>生成SVG</th>
                <th>TTFフォントレンダリング</th>
                <th>サイズ (Font座標)</th>
            </tr>
        </thead>
        <tbody>
            {''.join(rows_html)}
        </tbody>
    </table>
</div>

<script>
    const input = document.getElementById('testInput');
    const output = document.getElementById('renderOutput');
    function update() {{
        output.textContent = input.value;
    }}
    input.addEventListener('input', update);
    function setInput(text) {{
        input.value = text;
        update();
    }}
    update();
</script>
</body>
</html>
"""
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FF14 Fourth World Font Generation Pipeline")
    parser.add_argument(
        "-s", "--smooth",
        type=int,
        default=50,
        help="Smoothing level (0 = raw original, 50 = balanced default, 100 = max smooth)"
    )
    args = parser.parse_args()
    run_pipeline(smooth_level=args.smooth)
