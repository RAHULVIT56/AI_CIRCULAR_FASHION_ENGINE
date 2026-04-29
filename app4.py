import streamlit as st
import numpy as np
import os
import tempfile
import datetime

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

from PIL import Image, ImageDraw
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

import cv2


# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Smart Fabric Inspection System",
    layout="centered"
)


# =====================================================
# CONSTANTS
# =====================================================
IMG_SIZE = (224, 224)

FABRIC_CLASSES = [
    "Plain Weave",
    "Twill Weave",
    "Knitted Surface",
    "Dense Woven Surface",
    "Patterned Surface"
]

DEFECT_CLASSES = [
    "defect-free",
    "hole",
    "horizontal",
    "vertical",
    "lines",
    "pinched fabric",
    "needle mark",
    "broken stitch",
    "stain"
]


# =====================================================
# LOAD MODELS  (BUG FIX: removed duplicate decorator)
# =====================================================
@st.cache_resource
def load_models():

    base_path = os.path.dirname(__file__)

    fabric_path = os.path.join(base_path, "module1_fabric_surface_classifier.h5")
    defect_path = os.path.join(base_path, "best_model.h5")

    if not os.path.exists(fabric_path):
        st.error(f"❌ Fabric model not found: {fabric_path}")
        st.stop()

    if not os.path.exists(defect_path):
        st.error(f"❌ Defect model not found: {defect_path}")
        st.stop()

    fabric_model = load_model(fabric_path)
    defect_model = load_model(defect_path)

    return fabric_model, defect_model


fabric_model, defect_model = load_models()


# =====================================================
# PERFORMANCE FIX: Cache the GradCAM sub-model
# so it is not rebuilt on every image upload
# =====================================================
@st.cache_resource
def build_grad_model(_defect_model):
    for layer in reversed(_defect_model.layers):
        if len(layer.output.shape) == 4:
            return tf.keras.models.Model(
                [_defect_model.inputs],
                [_defect_model.get_layer(layer.name).output, _defect_model.output]
            )
    return None


grad_model = build_grad_model(defect_model)

if grad_model is None:
    st.error("❌ No convolutional layer found in defect model for GradCAM.")
    st.stop()


# =====================================================
# IMAGE PREPROCESS
# BUG FIX: always force RGB inside preprocess so it
# is safe regardless of how it is called
# =====================================================
def preprocess(img):
    img = img.convert("RGB")       # safe to call again, cheap
    img = img.resize(IMG_SIZE)
    arr = img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


# =====================================================
# MODULE 3: DECISION ENGINE
# (renamed to be more descriptive)
# =====================================================
def get_quality_decision(defect):

    rules = {
        "defect-free":   ("NONE",   "PASSED",   "Proceed to cutting",       "98%"),
        "needle mark":   ("MINOR",  "PASSED",   "Defect-aware cutting",     "90%"),
        "lines":         ("MINOR",  "PASSED",   "Defect-aware cutting",     "90%"),
        "horizontal":    ("MINOR",  "PASSED",   "Defect-aware cutting",     "90%"),
        "vertical":      ("MINOR",  "PASSED",   "Defect-aware cutting",     "90%"),
        "stain":         ("MEDIUM", "REWORK",   "Cleaning required",        "70%"),
        "broken stitch": ("MEDIUM", "REWORK",   "Repair stitching",         "75%"),
        "pinched fabric":("MEDIUM", "REWORK",   "Mechanical correction",    "72%"),
        "hole":          ("MAJOR",  "REJECTED", "Scrap fabric",             "0%"),
    }

    return rules.get(defect, ("UNKNOWN", "REJECTED", "Manual Check", "0%"))


# =====================================================
# MODULE 4: FABRIC UTILIZATION & CUTTING PLAN
# New module — computes usable area from GradCAM
# bounding box and recommends cutting zones
# =====================================================
def get_cutting_plan(img_width, img_height, box_coords, defect, decision):
    """
    Parameters
    ----------
    img_width, img_height : int
        Dimensions of the original image in pixels.
    box_coords : tuple (x1, y1, x2, y2) or None
        Bounding box of the detected defect in pixel coords.
        None when no defect region was found.
    defect : str
        Detected defect class name.
    decision : str
        Quality decision: "PASSED", "REWORK", or "REJECTED".

    Returns
    -------
    dict with keys:
        total_area          : int   (px²)
        defect_area         : int   (px²)
        usable_area         : int   (px²)
        usable_pct          : float (0–100)
        cutting_zones       : list of dicts {zone, x1,y1,x2,y2, label}
        recommendation      : str
        waste_category      : str   ("Minimal" / "Moderate" / "High" / "Total Loss")
    """

    total_area = img_width * img_height

    # ---- Defect area from bounding box ----
    if box_coords is not None:
        bx1, by1, bx2, by2 = box_coords
        defect_area = max(0, (bx2 - bx1) * (by2 - by1))
    else:
        defect_area = 0

    usable_area = total_area - defect_area
    usable_pct  = round((usable_area / total_area) * 100, 1)

    # ---- Waste category ----
    if usable_pct >= 95:
        waste_category = "Minimal"
    elif usable_pct >= 70:
        waste_category = "Moderate"
    elif usable_pct >= 40:
        waste_category = "High"
    else:
        waste_category = "Total Loss"

    # ---- Cutting zones ----
    # Divide the fabric into quadrants and flag which ones
    # overlap the defect bounding box.
    quadrants = {
        "Top-Left":     (0,                0,                img_width // 2, img_height // 2),
        "Top-Right":    (img_width // 2,   0,                img_width,      img_height // 2),
        "Bottom-Left":  (0,                img_height // 2,  img_width // 2, img_height),
        "Bottom-Right": (img_width // 2,   img_height // 2,  img_width,      img_height),
    }

    cutting_zones = []

    for name, (qx1, qy1, qx2, qy2) in quadrants.items():
        if box_coords is not None:
            bx1, by1, bx2, by2 = box_coords
            # Check overlap between quadrant and defect box
            overlap = (
                bx1 < qx2 and bx2 > qx1 and
                by1 < qy2 and by2 > qy1
            )
        else:
            overlap = False

        cutting_zones.append({
            "zone":    name,
            "x1": qx1, "y1": qy1, "x2": qx2, "y2": qy2,
            "defect":  overlap,
            "label":   "⚠️ Avoid" if overlap else "✅ Safe to cut",
        })

    # ---- Cutting recommendation ----
    if decision == "REJECTED":
        recommendation = (
            "Fabric is not suitable for any cutting. "
            "Recommend full scrap or recycle."
        )
    elif decision == "REWORK":
        safe_zones = [z["zone"] for z in cutting_zones if not z["defect"]]
        if safe_zones:
            recommendation = (
                f"Rework required before cutting. "
                f"Safe zones after rework: {', '.join(safe_zones)}."
            )
        else:
            recommendation = (
                "Defect spans entire fabric. Full rework needed before any cutting."
            )
    else:  # PASSED
        if defect_area == 0:
            recommendation = "No defect detected. Full fabric usable — proceed with standard cutting."
        else:
            safe_zones = [z["zone"] for z in cutting_zones if not z["defect"]]
            recommendation = (
                f"Defect area is small. Use defect-aware cutting. "
                f"Prioritise zones: {', '.join(safe_zones)}."
            )

    return {
        "total_area":     total_area,
        "defect_area":    defect_area,
        "usable_area":    usable_area,
        "usable_pct":     usable_pct,
        "cutting_zones":  cutting_zones,
        "recommendation": recommendation,
        "waste_category": waste_category,
    }


# =====================================================
# GRADCAM
# PERFORMANCE FIX: accepts pre-built grad_model
# BUG FIX: returns plain numpy array (not EagerTensor)
# =====================================================
def make_gradcam_heatmap(img_array, grad_model):

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_array)

        if isinstance(preds, (list, tuple)):
            preds = preds[0]

        class_idx = tf.argmax(preds[0])
        loss = preds[:, class_idx]

    grads = tape.gradient(loss, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_out_np = conv_out[0].numpy()
    pooled_np   = pooled.numpy()

    heatmap = conv_out_np @ pooled_np[..., np.newaxis]
    heatmap = np.squeeze(heatmap)
    heatmap = np.maximum(heatmap, 0)

    if np.max(heatmap) != 0:
        heatmap = heatmap / np.max(heatmap)

    # Always return a plain float32 numpy array
    return heatmap.astype(np.float32)


# =====================================================
# BOUNDING BOX
# Returns box coords as well as the drawn image
# so Module 4 can use them
# =====================================================
def draw_box(img, heatmap, threshold=0.6):
    """
    Returns
    -------
    drawn_img : PIL.Image  — image with bounding box drawn
    box_coords : tuple (x1, y1, x2, y2) or None
    """
    img = img.copy()     # always copy internally — BUG FIX
    h, w = img.size[1], img.size[0]

    hm_resized = cv2.resize(heatmap, (w, h))
    thresh = hm_resized > threshold
    coords = np.column_stack(np.where(thresh))

    if len(coords) == 0:
        return img, None

    y1, x1 = coords.min(0)
    y2, x2 = coords.max(0)

    draw = ImageDraw.Draw(img)
    draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=4)

    return img, (int(x1), int(y1), int(x2), int(y2))


# =====================================================
# OVERLAY
# BUG FIX: convert BGR→RGB before blending so
# heatmap colours are correct (was red/blue swapped)
# =====================================================
def overlay_heatmap(img, heatmap):
    img = img.copy()     # always copy internally

    hm_resized = cv2.resize(heatmap, (img.size[0], img.size[1]))
    hm_uint8   = np.uint8(255 * hm_resized)

    hm_bgr = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_JET)
    hm_rgb = cv2.cvtColor(hm_bgr, cv2.COLOR_BGR2RGB)   # BUG FIX: BGR→RGB

    img_np  = np.array(img)
    overlay = cv2.addWeighted(img_np, 0.6, hm_rgb, 0.4, 0)

    return Image.fromarray(overlay)


# =====================================================
# PDF  (now includes Module 4 data)
# BUG FIX: temp file is deleted after download button
# =====================================================
def generate_pdf(fabric, defect, severity, decision, action, util, plan):

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")

    c = canvas.Canvas(tmp.name, pagesize=A4)

    # ---- Header ----
    c.setFillColorRGB(0.1, 0.2, 0.6)
    c.setFont("Helvetica-Bold", 20)
    c.drawString(50, 810, "SMART FABRIC INSPECTION REPORT")

    c.setFillColorRGB(0.4, 0.4, 0.4)
    c.setFont("Helvetica", 10)
    c.drawString(50, 793, f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    c.setStrokeColorRGB(0.1, 0.2, 0.6)
    c.line(50, 785, 545, 785)

    # ---- Module 1 & 2 ----
    y = 760
    c.setFillColorRGB(0, 0, 0)
    c.setFont("Helvetica-Bold", 13)
    c.drawString(50, y, "MODULE 1 & 2 — Classification Results")
    y -= 25

    c.setFont("Helvetica", 11)
    fields = [
        ("Fabric Type",   fabric),
        ("Defect Type",   defect),
        ("Severity",      severity),
        ("Decision",      decision),
        ("Action",        action),
        ("Utilization",   util),
    ]
    for k, v in fields:
        c.setFont("Helvetica-Bold", 11)
        c.drawString(60, y, f"{k}:")
        c.setFont("Helvetica", 11)
        c.drawString(180, y, str(v))
        y -= 22

    # ---- Module 3 ----
    y -= 10
    c.setFont("Helvetica-Bold", 13)
    c.drawString(50, y, "MODULE 3 — Decision")
    y -= 22
    c.setFont("Helvetica", 11)
    c.drawString(60, y, f"Recommendation: {action}")
    y -= 22

    # ---- Module 4 ----
    y -= 10
    c.setFont("Helvetica-Bold", 13)
    c.drawString(50, y, "MODULE 4 — Cutting Plan & Utilization")
    y -= 22

    c.setFont("Helvetica", 11)
    m4_fields = [
        ("Total Area (px²)",   f"{plan['total_area']:,}"),
        ("Defect Area (px²)",  f"{plan['defect_area']:,}"),
        ("Usable Area (px²)",  f"{plan['usable_area']:,}"),
        ("Usable %",           f"{plan['usable_pct']}%"),
        ("Waste Category",     plan['waste_category']),
    ]
    for k, v in m4_fields:
        c.setFont("Helvetica-Bold", 11)
        c.drawString(60, y, f"{k}:")
        c.setFont("Helvetica", 11)
        c.drawString(220, y, str(v))
        y -= 22

    y -= 5
    c.setFont("Helvetica-Bold", 11)
    c.drawString(60, y, "Cutting Recommendation:")
    y -= 18
    c.setFont("Helvetica", 10)
    # Word-wrap the recommendation text
    words = plan['recommendation'].split()
    line  = ""
    for word in words:
        test = (line + " " + word).strip()
        if c.stringWidth(test, "Helvetica", 10) < 460:
            line = test
        else:
            c.drawString(70, y, line)
            y -= 15
            line = word
    if line:
        c.drawString(70, y, line)
        y -= 15

    y -= 10
    c.setFont("Helvetica-Bold", 11)
    c.drawString(60, y, "Cutting Zones:")
    y -= 18
    c.setFont("Helvetica", 10)
    for zone in plan['cutting_zones']:
        status = "AVOID" if zone['defect'] else "SAFE"
        c.drawString(70, y, f"  {zone['zone']}: {status}")
        y -= 15

    c.showPage()
    c.save()

    return tmp.name


# =====================================================
# UI
# =====================================================
st.title("🧵 Smart Fabric Inspection System")
st.markdown("### AI-Based Fabric Quality Control")
st.divider()

upload = st.file_uploader("Upload Fabric Image", ["jpg", "png", "jpeg"])


# =====================================================
# MAIN
# =====================================================
if upload:

    img = Image.open(upload).convert("RGB")
    st.image(img, caption="Uploaded Image")

    arr = preprocess(img)

    with st.spinner("Running AI analysis..."):

        # ---------- MODULE 1: Fabric Classification ----------
        f_pred = fabric_model.predict(arr, verbose=0)
        f_idx  = np.argmax(f_pred)
        fabric = FABRIC_CLASSES[f_idx]
        f_conf = f_pred[0][f_idx] * 100

        # ---------- MODULE 2: Defect Detection ----------
        d_pred = defect_model.predict(arr, verbose=0)
        d_idx  = np.argmax(d_pred)
        defect = DEFECT_CLASSES[d_idx]
        d_conf = d_pred[0][d_idx] * 100

        # ---------- MODULE 3: Decision Engine ----------
        severity, decision, action, util = get_quality_decision(defect)

        # ---------- GradCAM ----------
        heatmap = make_gradcam_heatmap(arr, grad_model)
        overlay = overlay_heatmap(img, heatmap)
        boxed, box_coords = draw_box(img, heatmap)

        # ---------- MODULE 4: Cutting Plan ----------
        plan = get_cutting_plan(
            img_width=img.size[0],
            img_height=img.size[1],
            box_coords=box_coords,
            defect=defect,
            decision=decision
        )


    # =====================================================
    # RESULTS — MODULE 1 & 2
    # =====================================================
    st.divider()
    st.subheader("📊 Module 1 & 2 — Classification Results")

    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Fabric Type", fabric, f"{f_conf:.1f}% confidence")
    with col_b:
        st.metric("Defect Type", defect, f"{d_conf:.1f}% confidence")


    # =====================================================
    # RESULTS — MODULE 3
    # =====================================================
    st.subheader("⚖️ Module 3 — Quality Decision")

    st.markdown(f"""
| Field | Value |
|-------|-------|
| **Severity** | {severity} |
| **Decision** | {decision} |
| **Action** | {action} |
| **Utilization** | {util} |
""")

    if decision == "PASSED":
        st.success("✅ APPROVED — Fabric passed quality check.")
    elif decision == "REWORK":
        st.warning("⚠️ REWORK REQUIRED — Defect can be corrected.")
    else:
        st.error("❌ REJECTED — Fabric does not meet quality standards.")


    # =====================================================
    # DEFECT LOCALIZATION (GradCAM)
    # =====================================================
    st.subheader("🔥 Defect Localization")

    col1, col2 = st.columns(2)
    with col1:
        st.image(overlay, caption="Heatmap Overlay")
    with col2:
        st.image(boxed, caption="Defect Bounding Box")

    if box_coords:
        x1, y1, x2, y2 = box_coords
        st.caption(f"Bounding box: top-left ({x1}, {y1})  →  bottom-right ({x2}, {y2})")
    else:
        st.caption("No distinct defect region detected in heatmap.")


    # =====================================================
    # MODULE 4 — CUTTING PLAN & UTILIZATION
    # =====================================================
    st.divider()
    st.subheader("✂️ Module 4 — Cutting Plan & Fabric Utilization")

    # Metrics row
    m4_col1, m4_col2, m4_col3, m4_col4 = st.columns(4)
    with m4_col1:
        st.metric("Total Area", f"{plan['total_area']:,} px²")
    with m4_col2:
        st.metric("Defect Area", f"{plan['defect_area']:,} px²")
    with m4_col3:
        st.metric("Usable Area", f"{plan['usable_area']:,} px²")
    with m4_col4:
        st.metric("Usable %", f"{plan['usable_pct']}%")

    # Waste category badge
    waste_color = {
        "Minimal":    "success",
        "Moderate":   "warning",
        "High":       "warning",
        "Total Loss": "error",
    }
    wc = plan['waste_category']
    if wc == "Minimal":
        st.success(f"♻️ Waste Category: **{wc}**")
    elif wc in ("Moderate", "High"):
        st.warning(f"♻️ Waste Category: **{wc}**")
    else:
        st.error(f"♻️ Waste Category: **{wc}**")

    # Cutting recommendation
    st.info(f"🪡 **Cutting Recommendation:** {plan['recommendation']}")

    # Cutting zones table
    st.markdown("#### Fabric Cutting Zones")
    zone_data = []
    for z in plan['cutting_zones']:
        zone_data.append({
            "Zone":          z["zone"],
            "Status":        z["label"],
            "Top-Left":      f"({z['x1']}, {z['y1']})",
            "Bottom-Right":  f"({z['x2']}, {z['y2']})",
        })
    st.table(zone_data)

    # Visual cutting map on the image
    st.markdown("#### Cutting Zone Map")
    zone_img = img.copy()
    draw = ImageDraw.Draw(zone_img)
    for z in plan['cutting_zones']:
        color = "red" if z["defect"] else "green"
        draw.rectangle(
            [(z["x1"] + 2, z["y1"] + 2), (z["x2"] - 2, z["y2"] - 2)],
            outline=color,
            width=3
        )
        label_pos = (z["x1"] + 8, z["y1"] + 8)
        draw.text(label_pos, z["zone"], fill=color)
    st.image(zone_img, caption="Green = Safe to cut  |  Red = Avoid (defect zone)")


    # =====================================================
    # PDF REPORT  (BUG FIX: temp file cleaned up after use)
    # =====================================================
    st.divider()
    pdf_path = generate_pdf(
        fabric, defect, severity, decision, action, util, plan
    )
    try:
        with open(pdf_path, "rb") as f:
            st.download_button(
                label="📄 Download Full Inspection Report (PDF)",
                data=f,
                file_name="fabric_inspection_report.pdf",
                mime="application/pdf"
            )
    finally:
        os.unlink(pdf_path)   # BUG FIX: always clean up temp file