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
# Character Sets & Glyph Mappings
# -------------------------------------------------------------
DIGIT_NAMES = {
    '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
    '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine'
}

SYMBOL_NAMES = {
    '!': 'exclam',
    '?': 'question',
    '-': 'hyphen',
}

# Full-width equivalents mapped to the same glyph
FULLWIDTH_MAP = {
    '！': '!',
    '？': '?',
    '－': '-',
    '—': '-',
    '–': '-',
    '　': ' ',
}

# Aliases for symbol file names
SYMBOL_FILENAME_ALIASES = {
    'exclamation': '!',
    'exclam': '!',
    'mark_exclamation': '!',
    'mark_exclam': '!',
    '!': '!',
    'question': '?',
    'question_mark': '?',
    'questionmark': '?',
    'mark_question': '?',
    '?': '?',
    'hyphen': '-',
    'dash': '-',
    'minus': '-',
    '-': '-',
}

# Target character list: a-z, A-Z, 0-9, !, ?, -
UPPERCASE_CHARS = [chr(c) for c in range(ord('A'), ord('Z') + 1)]  # 26
LOWERCASE_CHARS = [chr(c) for c in range(ord('a'), ord('z') + 1)]  # 26
DIGIT_CHARS = [chr(c) for c in range(ord('0'), ord('9') + 1)]      # 10
PUNCT_CHARS = ['!', '?', '-']                                      # 3
ALL_TARGET_CHARS = UPPERCASE_CHARS + LOWERCASE_CHARS + DIGIT_CHARS + PUNCT_CHARS

def get_glyph_name_for_char(char_str):
    if char_str.isalpha():
        return char_str  # 'A'-'Z' or 'a'-'z'
    if char_str in DIGIT_NAMES:
        return DIGIT_NAMES[char_str]
    if char_str in SYMBOL_NAMES:
        return SYMBOL_NAMES[char_str]
    if char_str == ' ':
        return 'space'
    return f"uni{ord(char_str):04X}"

def get_default_advance_width(char_str):
    if char_str.isupper():
        return 500
    if char_str.islower():
        return 450
    if char_str.isdigit():
        return 500
    if char_str == '!':
        return 300
    if char_str == '?':
        return 500
    if char_str == '-':
        return 400
    if char_str == ' ':
        return 300
    return 500

# -------------------------------------------------------------
# File name parser:
# - large_a.jpg / small_a.jpg / large-a.png / small-a.png
# - digit_0.png / num_0.jpg / 0.png
# - exclamation.png / question.png / hyphen.png
# -------------------------------------------------------------
def parse_char_info(filename):
    name, ext = os.path.splitext(filename)
    if ext.lower() not in [".jpg", ".jpeg", ".png"]:
        return None
        
    clean_name = name.lower().strip()
    
    # Check symbols
    for alias, sym_char in SYMBOL_FILENAME_ALIASES.items():
        if (clean_name == alias or 
            clean_name.endswith(f"_{alias}") or 
            clean_name.endswith(f"-{alias}") or
            clean_name.startswith(f"{alias}_") or
            clean_name.startswith(f"{alias}-")):
            is_large = not clean_name.startswith("small")
            glyph_name = get_glyph_name_for_char(sym_char)
            return sym_char, is_large, glyph_name
            
    # Check digits (e.g. digit_0, num_0, large_0, small_0, 0)
    m_digit = re.match(r"^(?:(?:digit|num|large|small)[_-])?([0-9])$", clean_name)
    if m_digit:
        d = m_digit.group(1)
        return d, True, get_glyph_name_for_char(d)
        
    # Check letters with prefix (e.g. large_a, small_d, upper_b, lower_c)
    m_alpha = re.match(r"^(large|small|upper|lower)[_-]([a-zA-Z])$", clean_name)
    if m_alpha:
        prefix, char_raw = m_alpha.groups()
        is_large = prefix in ["large", "upper"]
        char_str = char_raw.upper() if is_large else char_raw.lower()
        return char_str, is_large, get_glyph_name_for_char(char_str)
        
    # Single character filename (e.g. A.png, a.jpg, -.png)
    if len(name) == 1:
        c = name
        if c in ALL_TARGET_CHARS:
            is_large = c.isupper() if c.isalpha() else True
            return c, is_large, get_glyph_name_for_char(c)

    return None

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
    
    files = sorted([f for f in os.listdir(DATA_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    print(f"Found {len(files)} files in '{DATA_DIR}'.")
    
    processed_chars = {}   # char_str -> record
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
    
    # Space glyph
    space_pen = TTGlyphPen(None)
    ttf_glyphs["space"] = space_pen.glyph()
    ttf_metrics["space"] = (300, 0)
    cmap[ord(' ')] = "space"
    cmap[ord('\u3000')] = "space"  # Full-width space
    
    # 1. Process existing image files in data/
    for f in files:
        parsed = parse_char_info(f)
        if not parsed:
            print(f"  [Skip] Unrecognized file: {f}")
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
        
        # Map full-width equivalent if any
        for fw_char, hw_char in FULLWIDTH_MAP.items():
            if hw_char == char_str:
                cmap[ord(fw_char)] = glyph_name
                
        processed_chars[char_str] = {
            "char": char_str,
            "glyph_name": glyph_name,
            "has_image": True,
            "is_large": is_large,
            "orig_file": f,
            "clean_file": f"{base_name}.png",
            "svg_file": svg_filename,
            "width": int(gw),
            "height": int(gh),
            "adv_width": adv_w,
        }
        print(f"  Processed '{char_str}' ({'Upper' if is_large else 'Lower'}): size={gw:.1f}x{gh:.1f}, advance={adv_w}")

    # 2. Add blank glyphs for target characters that have no image yet
    # Target characters: a-z, A-Z, 0-9, !, ?, -
    blank_count = 0
    blank_records = {}
    for char_str in ALL_TARGET_CHARS:
        if char_str in processed_chars:
            continue
            
        glyph_name = get_glyph_name_for_char(char_str)
        adv_w = get_default_advance_width(char_str)
        
        # Blank glyph (empty contours)
        empty_pen = TTGlyphPen(None)
        ttf_glyphs[glyph_name] = empty_pen.glyph()
        ttf_metrics[glyph_name] = (adv_w, 0)
        cmap[ord(char_str)] = glyph_name
        
        # Map full-width equivalent if any
        for fw_char, hw_char in FULLWIDTH_MAP.items():
            if hw_char == char_str:
                cmap[ord(fw_char)] = glyph_name
                
        is_large = char_str.isupper() if char_str.isalpha() else True
        blank_records[char_str] = {
            "char": char_str,
            "glyph_name": glyph_name,
            "has_image": False,
            "is_large": is_large,
            "orig_file": "-",
            "clean_file": "-",
            "svg_file": "-",
            "width": 0,
            "height": 0,
            "adv_width": adv_w,
        }
        blank_count += 1
        
    print(f"\nCharacters with image: {len(processed_chars)}")
    print(f"Characters set to blank (pending image): {blank_count}")
    print(f"Total target glyphs registered: {len(processed_chars) + blank_count}")

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
        "version": f"Version 1.4 (Smooth {smooth_level}); 2026",
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
    build_html_preview(processed_chars, blank_records, html_path, smooth_level)
    print(f"Successfully generated preview dashboard: {html_path}")

    # Generate visual sample image
    try:
        font_large = ImageFont.truetype(font_path, 60)
        font_small = ImageFont.truetype(font_path, 36)
        font_label = ImageFont.load_default()

        canvas = Image.new("RGB", (960, 520), color=(252, 252, 254))
        draw = ImageDraw.Draw(canvas)

        draw.text((32, 20), f"FF14 Fourth World Font (Smooth Level: {smooth_level}/100)", fill=(40, 40, 50), font=font_label)
        draw.text((32, 45), f"Created Glyphs ({len(processed_chars)}): {' '.join(sorted(processed_chars.keys()))}", fill=(40, 120, 60), font=font_label)
        draw.text((32, 65), f"Blank Glyphs ({len(blank_records)}): a-zA-Z0-9!?- (missing images set to blank spaces)", fill=(120, 100, 40), font=font_label)
        
        # Render available characters
        draw.text((32, 95), " ".join(sorted(processed_chars.keys())), fill=(25, 30, 45), font=font_large)

        draw.text((32, 190), "Sample Text 1: 'Unmoored Isle' (all available)", fill=(100, 100, 110), font=font_label)
        draw.text((32, 215), "Unmoored Isle", fill=(25, 30, 45), font=font_large)

        draw.text((32, 305), "Sample Text 2: 'red rose / iron soul / moon order'", fill=(100, 100, 110), font=font_label)
        draw.text((32, 330), "red rose  iron soul  moon order", fill=(25, 30, 45), font=font_small)

        draw.text((32, 405), "Sample Text 3: 'FF14 Fourth World! (Missing letters rendered as blank)'", fill=(100, 100, 110), font=font_label)
        draw.text((32, 430), "FF14 Fourth World!", fill=(25, 30, 45), font=font_large)

        sample_img_path = os.path.join(OUT_DIR, "sample_render.png")
        canvas.save(sample_img_path)
        print(f"Successfully generated sample image: {sample_img_path}")
    except Exception as e:
        print(f"Sample image generation skipped: {e}")

# -------------------------------------------------------------
# HTML Preview Dashboard Generation
# -------------------------------------------------------------
def build_html_preview(processed_chars, blank_records, html_path, smooth_level=50):
    # Sort all target characters in a natural order: Upper -> Lower -> Digits -> Symbols
    categories = [
        ("大文字 (A-Z)", UPPERCASE_CHARS),
        ("小文字 (a-z)", LOWERCASE_CHARS),
        ("数字 (0-9)", DIGIT_CHARS),
        ("記号 (!, ?, -)", PUNCT_CHARS),
    ]
    
    table_sections_html = []
    
    for cat_title, char_list in categories:
        rows = []
        for ch in char_list:
            if ch in processed_chars:
                r = processed_chars[ch]
                badge_type = 'badge-upper' if r['is_large'] else 'badge-lower'
                type_label = '大文字' if r['is_large'] else ('小文字' if ch.isalpha() else '記号/数字')
                rows.append(f"""
                <tr class="row-created">
                    <td style="font-weight:bold; font-size:1.4rem; text-align:center;">{r['char']}</td>
                    <td style="text-align:center;"><span class="badge {badge_type}">{type_label}</span></td>
                    <td style="text-align:center;"><span class="badge badge-success">作成済</span></td>
                    <td style="text-align:center;"><img src="../data/{r['orig_file']}" class="preview-img" alt="{r['char']} original" /></td>
                    <td style="text-align:center;"><img src="cleaned_png/{r['clean_file']}" class="preview-img" alt="{r['char']} clean" /></td>
                    <td style="text-align:center;"><img src="svg/{r['svg_file']}" class="preview-svg" alt="{r['char']} svg" /></td>
                    <td style="font-family:'FF14FourthWorld'; font-size:2.4rem; text-align:center; color:#1a365d;">{r['char']}</td>
                    <td style="font-size:0.85rem; color:#666; text-align:center;">{r['width']}x{r['height']} (Adv:{r['adv_width']})</td>
                </tr>
                """)
            else:
                r = blank_records[ch]
                type_label = '大文字' if ch.isupper() else ('小文字' if ch.islower() else ('数字' if ch.isdigit() else '記号'))
                rows.append(f"""
                <tr class="row-blank">
                    <td style="font-weight:bold; font-size:1.4rem; text-align:center; color:#94a3b8;">{r['char']}</td>
                    <td style="text-align:center;"><span class="badge badge-neutral">{type_label}</span></td>
                    <td style="text-align:center;"><span class="badge badge-blank">素材待ち (空白)</span></td>
                    <td style="text-align:center; color:#94a3b8; font-size:0.85rem;">-</td>
                    <td style="text-align:center; color:#94a3b8; font-size:0.85rem;">-</td>
                    <td style="text-align:center; color:#94a3b8; font-size:0.85rem;">-</td>
                    <td style="font-family:'FF14FourthWorld'; font-size:2.4rem; text-align:center; background:#f8fafc;" title="空白グリフ（輪郭なし）">
                        <span style="display:inline-block; border-bottom:1px dashed #cbd5e1; width:24px; height:1.2em;"></span>
                    </td>
                    <td style="font-size:0.85rem; color:#94a3b8; text-align:center;">(Adv:{r['adv_width']})</td>
                </tr>
                """)
                
        table_sections_html.append(f"""
        <h4 style="margin: 24px 0 8px 0; color: #2c3e50; font-size: 1.1rem; border-left: 4px solid #3498db; padding-left: 8px;">{cat_title}</h4>
        <table>
            <thead>
                <tr>
                    <th style="width: 8%;">文字</th>
                    <th style="width: 10%;">種別</th>
                    <th style="width: 14%;">状態</th>
                    <th style="width: 14%;">元画像 (data/)</th>
                    <th style="width: 14%;">二値化マスク</th>
                    <th style="width: 14%;">生成SVG</th>
                    <th style="width: 14%;">TTFレンダリング</th>
                    <th style="width: 12%;">メトリクス</th>
                </tr>
            </thead>
            <tbody>
                {''.join(rows)}
            </tbody>
        </table>
        """)
        
    num_created = len(processed_chars)
    num_blank = len(blank_records)
    total_chars = num_created + num_blank

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
    .status-summary {{
        display: flex;
        gap: 16px;
        margin-bottom: 24px;
        flex-wrap: wrap;
    }}
    .status-card {{
        flex: 1;
        min-width: 200px;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }}
    .status-card .num {{
        font-size: 1.8rem;
        font-weight: bold;
        margin-top: 4px;
    }}
    .status-created {{ color: #16a34a; }}
    .status-blank {{ color: #eab308; }}
    .status-total {{ color: #2563eb; }}
    
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
        margin-top: 8px;
        margin-bottom: 24px;
    }}
    th, td {{
        padding: 10px 12px;
        border-bottom: 1px solid #e1e8ed;
    }}
    th {{
        background-color: #f8fafc;
        color: #4a5568;
        font-size: 0.85rem;
    }}
    .preview-img {{
        max-height: 48px;
        image-rendering: pixelated;
        border: 1px solid #ddd;
        border-radius: 4px;
        background: #fff;
        padding: 2px;
    }}
    .preview-svg {{
        max-height: 48px;
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
    .badge-upper {{ background: #e3f2fd; color: #1976d2; }}
    .badge-lower {{ background: #e8f5e9; color: #388e3c; }}
    .badge-neutral {{ background: #f1f5f9; color: #64748b; }}
    .badge-success {{ background: #dcfce7; color: #15803d; }}
    .badge-blank {{ background: #fef9c3; color: #a16207; }}
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
        （0 = 元画像そのままの完全生データ ／ 50 = 推奨中間設定 ／ 100 = 最大平滑化）
    </div>

    <div class="status-summary">
        <div class="status-card">
            <div style="font-size:0.85rem; color:#64748b;">作成済み文字 (実体あり)</div>
            <div class="num status-created">{num_created} 文字</div>
        </div>
        <div class="status-card">
            <div style="font-size:0.85rem; color:#64748b;">素材待ち文字 (空白グリフ)</div>
            <div class="num status-blank">{num_blank} 文字</div>
        </div>
        <div class="status-card">
            <div style="font-size:0.85rem; color:#64748b;">対象総文字数 (a-z, A-Z, 0-9, !?-)</div>
            <div class="num status-total">{total_chars} 文字</div>
        </div>
    </div>

    <div class="interactive-box">
        <h3>試し打ちテスト（リアルタイム入力）</h3>
        <p style="margin: 4px 0 10px 0; font-size: 0.9rem; color: #555;">
            下のテキストボックスに入力した文字が、生成された第四世界フォントで表示されます。<br>
            ※素材がない文字は指示通り「空白」として表示されます（エラーやトーフにはなりません）。
        </p>
        <textarea id="testInput" rows="2">Unmoored Isle (red rose, 12345!?)</textarea>
        <div class="sample-tags">
            <span style="font-size: 0.85rem; color: #666; align-self: center;">サンプル単語:</span>
            <button class="sample-btn" onclick="setInput('Unmoored Isle')">Unmoored Isle (作成済)</button>
            <button class="sample-btn" onclick="setInput('red rose')">red rose (作成済)</button>
            <button class="sample-btn" onclick="setInput('iron soul')">iron soul (作成済)</button>
            <button class="sample-btn" onclick="setInput('miles')">miles (作成済)</button>
            <button class="sample-btn" onclick="setInput('elder line')">elder line (作成済)</button>
            <button class="sample-btn" onclick="setInput('FF14 2026!?')">FF14 2026!? (空白確認)</button>
            <button class="sample-btn" onclick="setInput('A-Z a-z 0-9 ! ? -')">全文字テスト</button>
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
    <p style="font-size:0.9rem; color:#64748b; margin-top:0;">
        ※画像素材（data/*.jpg, png）が追加されると、自動的にベクター変換され「作成済」に切り替わります。
    </p>
    {''.join(table_sections_html)}
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
