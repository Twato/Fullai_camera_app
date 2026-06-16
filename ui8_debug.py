import os
import json
import threading
from threading import Event
from datetime import datetime
from time import sleep
import time
import traceback

import cv2
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

from config_test import (
    CAPTURE_DIR,
    ROI_DIR,
    OCR_DIR,
    PICAM1_INDEX,
    PICAM2_INDEX,
    USB_DEVICE,
    PREVIEW_SIZE,
    PREVIEW_INTERVAL
)

from camera_test import PiCameraTest, UsbCameraTest
from roi_test import crop_center, crop_by_box, preprocess_ocr, save_roi_files
from ocr_test import run_easyocr, load_ocr

os.makedirs(CAPTURE_DIR, exist_ok=True)
os.makedirs(ROI_DIR, exist_ok=True)
os.makedirs(OCR_DIR, exist_ok=True)

running = False
last_preview_time = 0
current_camera = None

result_ok_event = Event()
check_labels = {}
capture_state_label = None
capture_state_dot = None
result_product_slots = {}

debug_wait_counter = 0
current_review_context = "-"

delivery_no = ""
quantity = 0
current_index = 0


def dbg(msg):
    """
    Debug logger for UI/thread/event flow.
    Use this to see exactly where the program is waiting or skipping.
    """
    try:
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        th = threading.current_thread().name
        ev = result_ok_event.is_set()
        print(f"[DBG {ts} {th} event={ev} running={running}] {msg}", flush=True)
    except Exception as e:
        print("[DBG ERROR]", e, msg, flush=True)


def ui_call(name, func):
    """
    Run a UI callback with debug log and traceback if it fails.
    """
    try:
        dbg(f"UI_CALL START: {name}")
        func()
        dbg(f"UI_CALL END: {name}")
    except Exception:
        print(f"[UI ERROR] {name}", flush=True)
        traceback.print_exc()


def now_name(prefix):
    return datetime.now().strftime(f"{prefix}_%Y%m%d_%H%M%S_%f.jpg")

def set_status(text):
    """
    Update small status text in the right panel and the camera preview overlay.
    Uses a Canvas dot instead of emoji, so it works the same on Raspberry Pi.
    """
    try:
        status_label.config(text=text)

        color = "#f1c40f"  # yellow = waiting / default
        lower = text.lower()

        if "capture" in lower or "capturing" in lower or "opening" in lower:
            color = "#3498db"  # blue
        elif "ocr" in lower or "reading" in lower:
            color = "#e67e22"  # orange
        elif "ready" in lower:
            color = "#27ae60"  # green
        elif "stop" in lower or "error" in lower:
            color = "#c0392b"  # red

        if capture_state_label is not None:
            capture_state_label.config(text=text)

        if capture_state_dot is not None:
            capture_state_dot.delete("all")
            capture_state_dot.create_oval(2, 2, 12, 12, fill=color, outline=color)

        root.update_idletasks()
    except Exception:
        pass

    print(text)

def set_step(text):
    try:
        step_label.config(text=text)
        root.update_idletasks()
    except Exception:
        pass
    print("STEP:", text)

def update_preview(frame, is_bgr=False):
    global last_preview_time

    import time
    now = time.time()

    if now - last_preview_time < PREVIEW_INTERVAL:
        return

    last_preview_time = now

    try:
        display = cv2.resize(frame, PREVIEW_SIZE)

        if is_bgr:
            display = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)

        img = Image.fromarray(display)
        imgtk = ImageTk.PhotoImage(image=img)

        preview_label.imgtk = imgtk
        preview_label.config(image=imgtk, text="")

    except Exception as e:
        print("Preview error:", e)


def make_thumbnail_image(image_path, size=(120, 60)):
    """
    เน€เธยเน€เธเธเน€เธเธ…เน€เธโ€เน€เธย เน€เธเธ’เน€เธย ROI เน€เธยเน€เธเธ…เน€เธยเน€เธเธเน€เธโ€”เน€เธเธ“ thumbnail เน€เธเธเน€เธเธ“เน€เธเธเน€เธเธเน€เธเธ‘เน€เธยเน€เธยเน€เธเธเน€เธโ€เน€เธยเน€เธยเน€เธยเน€เธเธเน€เธยเน€เธยเน€เธเธ’ Result
    """
    try:
        if not image_path or not os.path.exists(image_path):
            return None

        img = cv2.imread(image_path)

        if img is None:
            return None

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, size)

        pil_img = Image.fromarray(img)
        return ImageTk.PhotoImage(image=pil_img)

    except Exception as e:
        print("Thumbnail error:", e)
        return None

def show_page(page):
    page_name = getattr(page, "_debug_name", str(page))
    dbg(f"show_page -> {page_name}")
    for p in [page_input, page_capture, page_result, page_ready_toc, page_complete]:
        p.pack_forget()
    page.pack(fill="both", expand=True)
    try:
        root.update_idletasks()
    except Exception:
        pass

def update_info():
    info_delivery.config(text=delivery_no or "-")
    info_en.config(text="-")
    info_progress.config(text=f"{current_index} / {quantity}")
    info_pack.config(text="-")

def set_progress(text):
    progress_label.config(text=text)

    try:
        if quantity > 0:
            percent = int((current_index / quantity) * 100)
        else:
            percent = 0

        progress_percent_label.config(text=f"{percent}%")
        progress_bar_fill.place(relx=0, rely=0, relwidth=percent / 100, relheight=1)
    except Exception:
        pass

    root.update_idletasks()

def set_checklist(active=None, done=None):
    for name, label in check_labels.items():
        if done and name in done:
            label.config(text=f"[OK]  {name}", fg="#229954", bg="#eafaf1")
        elif active == name:
            label.config(text=f"[RUN] {name}", fg="#b9770e", bg="#fff3cd")
        else:
            label.config(text=f"[ ]   {name}", fg=TEXT, bg=SOFT)

def capture_path(prefix):
    return os.path.join(CAPTURE_DIR, now_name(prefix))

def process_center_roi_ocr(image_path, prefix):
    img = cv2.imread(image_path)

    if img is None:
        return {
            "image_path": image_path,
            "error": "cannot_read_image",
            "raw": "",
            "clean": ""
        }

    crop, box = crop_center(img)
    processed = preprocess_ocr(crop)
    roi_path, processed_path = save_roi_files(image_path, crop, processed, prefix)

    ocr = run_easyocr(processed)

    return {
        "name": prefix,
        "source": "center_roi",
        "image_path": image_path,
        "roi_path": roi_path,
        "processed_path": processed_path,
        "box": box,
        "raw": ocr["raw"],
        "clean": ocr["clean"],
        "items": ocr["items"]
    }

def process_yolo_roi_ocr(image_path, detections):
    img = cv2.imread(image_path)

    if img is None:
        return []

    results = []

    for det in detections:
        name = det["name"]
        box = det["box"]

        crop, fixed_box = crop_by_box(img, box, pad=0)

        if crop is None or crop.size == 0:
            continue

        processed = preprocess_ocr(crop)

        prefix = f"toc_{name}"
        roi_path, processed_path = save_roi_files(image_path, crop, processed, prefix)

        ocr = run_easyocr(processed)

        results.append({
            "name": name,
            "source": "yolo_roi",
            "detect_conf": det["conf"],
            "image_path": image_path,
            "roi_path": roi_path,
            "processed_path": processed_path,
            "box": fixed_box,
            "raw": ocr["raw"],
            "clean": ocr["clean"],
            "items": ocr["items"]
        })

    return results

def add_result_row(title, data):
    """
    V5 compact result row:
    เน€เธยเน€เธเธเน€เธโ€เน€เธย ROI + OCR + RAW + source/conf เน€เธยเน€เธโ€ขเน€เธยเน€เธยเน€เธเธเน€เธยเน€เธเธเน€เธเธเน€เธยเน€เธยเน€เธยเน€เธเธ…เน€เธยเน€เธยเน€เธยเน€เธเธ
    """
    row = tk.Frame(
        result_list_frame,
        bg="#ffffff",
        padx=6,
        pady=5,
        relief="solid",
        borderwidth=1
    )
    row.pack(fill="x", padx=8, pady=4)

    # ROI image
    roi_path = data.get("roi_path", "")
    thumb = make_thumbnail_image(roi_path, size=(120, 60))

    img_box = tk.Label(
        row,
        bg="#17202a",
        width=120,
        height=60
    )
    img_box.pack(side="left", padx=(0, 8))

    if thumb is not None:
        img_box.imgtk = thumb
        img_box.config(image=thumb, text="")
    else:
        img_box.config(
            text="NO ROI",
            fg="white",
            font=("Arial", 8, "bold")
        )

    # Class / source
    info_box = tk.Frame(row, bg="#ffffff", width=115)
    info_box.pack(side="left", fill="y", padx=(0, 8))
    info_box.pack_propagate(False)

    tk.Label(
        info_box,
        text=title,
        bg="#ffffff",
        fg="#0b2a4a",
        font=("Arial", 11, "bold"),
        anchor="w"
    ).pack(anchor="w", pady=(2, 0))

    source = data.get("source", "-")
    detect_conf = data.get("detect_conf", None)

    if detect_conf is not None:
        source_text = f"{source} / DET {detect_conf:.2f}"
    else:
        source_text = source

    tk.Label(
        info_box,
        text=source_text,
        bg="#ffffff",
        fg="#7f8c8d",
        font=("Arial", 8),
        anchor="w"
    ).pack(anchor="w")

    # OCR clean
    clean_text_value = data.get("clean", "") or "(empty)"

    tk.Label(
        row,
        text=clean_text_value,
        bg="#eafaf1",
        fg="#145a32",
        font=("Arial", 16, "bold"),
        anchor="w",
        padx=10
    ).pack(side="left", fill="x", expand=True, padx=(0, 8))

    # Raw text
    raw_text = data.get("raw", "")

    tk.Label(
        row,
        text=f"RAW: {raw_text}",
        bg="#ffffff",
        fg="#34495e",
        font=("Arial", 9),
        anchor="w",
        justify="left",
        wraplength=220,
        width=30
    ).pack(side="left", padx=(0, 8))

    # Status
    is_read = clean_text_value != "(empty)"
    tk.Label(
        row,
        text="READ" if is_read else "EMPTY",
        bg="#27ae60" if is_read else "#e67e22",
        fg="white",
        font=("Arial", 10, "bold"),
        width=8,
        padx=4,
        pady=5
    ).pack(side="right")


def clear_result_list():
    # Keep this for old/table-style result rows if needed later.
    try:
        for w in result_list_frame.winfo_children():
            w.destroy()
    except Exception:
        pass


def clear_toc_cards():
    """
    Clear only temporary TOC cards.
    Product result cards are fixed widgets and are not destroyed.
    """
    try:
        for w in toc_result_area.winfo_children():
            w.destroy()
    except Exception:
        pass


def load_result_photo(image_path, size=(300, 86)):
    """
    Load ROI image from path and return PhotoImage.
    This is used to update fixed image labels quickly.
    """
    try:
        if not image_path or not os.path.exists(image_path):
            return None

        img = cv2.imread(image_path)
        if img is None:
            return None

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, size)
        pil_img = Image.fromarray(img)
        return ImageTk.PhotoImage(image=pil_img)

    except Exception as e:
        print("Result image load error:", e)
        return None


def update_result_image(label, image_path):
    """
    Update existing image label by image path.
    No card is destroyed; only the displayed PhotoImage changes.
    """
    photo = load_result_photo(image_path, size=(310, 86))

    if photo is not None:
        label.imgtk = photo
        label.config(image=photo, text="")
    else:
        label.imgtk = None
        label.config(
            image="",
            text="NO IMAGE",
            fg="white",
            font=("Arial", 11, "bold")
        )


def update_value_box(label, value):
    text = value or "(empty)"
    is_ok = text != "(empty)"
    label.config(
        text=text,
        bg=GREEN_BG if is_ok else "#f8f9fa",
        fg=GREEN if is_ok else "#7f8c8d"
    )


def make_result_field(parent, field_title):
    """
    Create one field block: title + ROI image + OCR value.
    Return dict with image label and value label for later update.
    """
    block = tk.Frame(parent, bg=CARD)
    block.pack(fill="x", pady=(0, 14))

    tk.Label(
        block,
        text=field_title,
        bg=CARD,
        fg=MUTED,
        font=("Arial", 10, "bold")
    ).pack(anchor="w", pady=(0, 5))

    img_box = tk.Label(
        block,
        bg=DARK,
        width=310,
        height=86
    )
    img_box.pack(fill="x", pady=(0, 8))

    value_box = tk.Label(
        block,
        text="-",
        bg="#f8f9fa",
        fg="#7f8c8d",
        font=("Arial", 15, "bold"),
        anchor="w",
        padx=12,
        pady=8
    )
    value_box.pack(fill="x")

    return {
        "image": img_box,
        "value": value_box
    }


def make_product_result_section(parent, key, title, icon_key, field1, field2):
    """
    Create fixed product result section.
    Each section has 2 ROI images and 2 value fields.
    """
    card = tk.Frame(
        parent,
        bg=CARD,
        highlightbackground=BORDER,
        highlightthickness=1,
        padx=16,
        pady=14
    )
    card.pack(side="left", fill="both", expand=True, padx=6)

    title_row = tk.Frame(card, bg=CARD)
    title_row.pack(fill="x", pady=(0, 12))

    icon_label(title_row, icon_key, bg=CARD).pack(side="left", padx=(0, 8))

    tk.Label(
        title_row,
        text=title,
        bg=CARD,
        fg=TEXT,
        font=("Arial", 16, "bold")
    ).pack(side="left")

    # ===== TOP FIELD =====
    top_part = tk.Frame(card, bg=CARD)
    top_part.pack(fill="x", anchor="n")

    first = make_result_field(top_part, field1)

    # ===== MIDDLE GAP =====
    # User selected 240 to place the second field around the middle of section.
    tk.Frame(
        card,
        bg=CARD,
        height=240
    ).pack(fill="x")

    # ===== MIDDLE FIELD =====
    bottom_part = tk.Frame(card, bg=CARD)
    bottom_part.pack(fill="x", anchor="n")

    second = make_result_field(bottom_part, field2)

    result_product_slots[key] = {
        "first": first,
        "second": second
    }


def update_product_result_section(key, image_path, first_value, second_value):
    section = result_product_slots.get(key)
    if not section:
        return

    update_result_image(section["first"]["image"], image_path)
    update_value_box(section["first"]["value"], first_value)

    update_result_image(section["second"]["image"], image_path)
    update_value_box(section["second"]["value"], second_value)


def add_toc_result_row(parent, row_no, title, image_data=None):
    """
    Compact TOC result row:
    - Always shows rows TOC -> TOC6
    - Image on the left, OCR result on the right
    - Fixed-pixel image area avoids Tkinter Label width/height text-unit bug.
    """
    data = image_data or {}
    clean_text = data.get("clean", "") or "(empty)"
    raw_text = data.get("raw", "") or ""
    source_text = data.get("source", "-")
    detect_conf = data.get("detect_conf", None)

    if detect_conf is not None:
        source_text = f"{source_text} / DET {detect_conf:.2f}"

    row = tk.Frame(
        parent,
        bg=CARD,
        highlightbackground=BORDER,
        highlightthickness=1,
        padx=10,
        pady=6
    )
    row.pack(fill="x", padx=12, pady=3)

    name_box = tk.Frame(row, bg=CARD, width=88)
    name_box.pack(side="left", fill="y", padx=(0, 10))
    name_box.pack_propagate(False)

    tk.Label(
        name_box,
        text=str(row_no),
        bg=BLUE,
        fg="white",
        font=("Arial", 10, "bold"),
        width=3,
        pady=2
    ).pack(anchor="w")

    tk.Label(
        name_box,
        text=title.upper(),
        bg=CARD,
        fg=TEXT,
        font=("Arial", 12, "bold"),
        anchor="w"
    ).pack(anchor="w", pady=(4, 0))

    roi_path = data.get("roi_path", "")
    photo = load_result_photo(roi_path, size=(260, 56))

    image_wrap = tk.Frame(row, bg=DARK, width=260, height=56)
    image_wrap.pack(side="left", padx=(0, 14))
    image_wrap.pack_propagate(False)

    img_box = tk.Label(
        image_wrap,
        bg=DARK,
        fg="white",
        text="NO ROI",
        font=("Arial", 9, "bold")
    )
    img_box.pack(fill="both", expand=True)

    if photo is not None:
        img_box.imgtk = photo
        img_box.config(image=photo, text="")
    else:
        img_box.imgtk = None
        img_box.config(image="", text="NO ROI")

    info_box = tk.Frame(row, bg=CARD)
    info_box.pack(side="left", fill="both", expand=True)

    top_line = tk.Frame(info_box, bg=CARD)
    top_line.pack(fill="x")

    tk.Label(
        top_line,
        text="OCR Result",
        bg=CARD,
        fg=MUTED,
        font=("Arial", 9, "bold")
    ).pack(side="left")

    tk.Label(
        top_line,
        text=source_text,
        bg=CARD,
        fg=MUTED,
        font=("Arial", 8),
        anchor="e"
    ).pack(side="right")

    value_box = tk.Label(
        info_box,
        text=clean_text,
        bg=GREEN_BG if clean_text != "(empty)" else "#f8f9fa",
        fg=GREEN if clean_text != "(empty)" else "#7f8c8d",
        font=("Arial", 14, "bold"),
        anchor="w",
        padx=10,
        pady=4
    )
    value_box.pack(fill="x", pady=(3, 2))

    tk.Label(
        info_box,
        text=f"RAW : {raw_text or '-'}",
        bg=CARD,
        fg=MUTED,
        font=("Arial", 9),
        anchor="w"
    ).pack(anchor="w")

def show_product_result(product_no, cam1_data, cam2_data):
    global current_review_context
    current_review_context = f"PRODUCT {product_no}"
    dbg(f"SHOW PRODUCT RESULT product={product_no}")
    show_page(page_result)

    result_title.config(text=f"OCR Result : Product {product_no}")
    result_status_badge.config(text="WAIT REVIEW", bg="#f1c40f", fg=TEXT)
    result_hint.config(
        text="Review ROI image and OCR result, then press OK / NEXT to continue."
    )

    # Show fixed 3-column Product Result UI.
    product_result_area.pack(fill="both", expand=True, padx=10, pady=(0, 12))
    toc_result_area.pack_forget()

    cam1_text = cam1_data.get("clean", "") or "(empty)"
    cam2_text = cam2_data.get("clean", "") or "(empty)"

    cam1_roi = cam1_data.get("roi_path", "")
    cam2_roi = cam2_data.get("roi_path", "")

    # Mock UI mapping for now:
    # Vendor uses CAM1 ROI twice: BOX ID image + Date Code image.
    # TOA uses CAM1 ROI twice: Lot No image + Date Code image.
    # TOB uses CAM2 ROI twice: Lot No image + Date Code image.
    update_product_result_section("vendor", cam1_roi, cam1_text, cam1_text)
    update_product_result_section("toa", cam1_roi, cam1_text, cam1_text)
    update_product_result_section("tob", cam2_roi, cam2_text, cam2_text)

    save_json({
        "type": "product",
        "product_no": product_no,
        "cam1": cam1_data,
        "cam2": cam2_data
    }, f"product_{product_no:03d}.json")


def show_toc_result(toc_items, fallback_data=None, toc_no=1, toc_total=1):
    global current_review_context
    current_review_context = f"TOC {toc_no}/{toc_total}"
    dbg(f"SHOW TOC RESULT toc={toc_no}/{toc_total} items={len(toc_items or [])} fallback={fallback_data is not None}")
    show_page(page_result)
    result_title.config(text=f"TOC Result : {toc_no} / {toc_total}")
    result_status_badge.config(text="WAIT REVIEW", bg="#f1c40f", fg=TEXT)

    product_result_area.pack_forget()
    toc_result_area.pack(fill="both", expand=True, padx=10, pady=(0, 12))
    clear_toc_cards()

    result_hint.config(
        text="Review TOC ROI from TOC to TOC6. Image is on the left and OCR result is on the right."
    )

    # Build lookup by detection name: toc, toc1, toc2, ... toc6
    item_map = {}
    for item in toc_items or []:
        name = str(item.get("name", "")).lower()
        item_map[name] = item

    expected_names = ["toc", "toc1", "toc2", "toc3", "toc4", "toc5", "toc6"]

    # If there is no YOLO item but fallback exists, show fallback on TOC row.
    if fallback_data is not None and not item_map:
        item_map["toc"] = fallback_data

    for idx, name in enumerate(expected_names, start=1):
        add_toc_result_row(
            toc_result_area,
            row_no=idx,
            title=name,
            image_data=item_map.get(name)
        )

    save_json({
        "type": "toc",
        "toc_no": toc_no,
        "toc_total": toc_total,
        "toc_items": toc_items,
        "fallback": fallback_data
    }, f"toc_result_{toc_no:02d}.json")

def save_json(data, filename):
    path = os.path.join(OCR_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print("Saved JSON:", path)

def start_job():
    global running, delivery_no, quantity, current_index

    delivery_no = delivery_entry.get().strip()
    qty_text = qty_entry.get().strip()

    if not delivery_no:
        messagebox.showwarning("Warning", "Please input Delivery No")
        return

    if not qty_text.isdigit():
        messagebox.showwarning("Warning", "Please input Quantity")
        return

    quantity = int(qty_text)

    if quantity < 1 or quantity > 100:
        messagebox.showwarning("Warning", "Quantity must be 1-100")
        return

    current_index = 0
    running = True
    result_ok_event.clear()
    dbg("START JOB: event cleared")

    update_info()
    show_page(page_capture)

    set_checklist(active="TOA", done=[])
    set_status("Waiting Object")
    set_progress(f"Product 0 / {quantity}")

    thread = threading.Thread(target=job_loop, daemon=True)
    thread.start()

def job_loop():
    global current_index, running

    set_status("Loading OCR Model...")
    load_ocr()
    set_status("OCR Ready")

    for i in range(quantity):
        if not running:
            return

        current_index = i + 1
        root.after(0, update_info)
        root.after(0, lambda i=i: set_progress(f"Product {i+1} / {quantity}"))

        ok = process_product(i + 1)

        if not ok:
            running = False
            return

        sleep(1.0)

    running = False
    root.after(0, lambda: show_page(page_ready_toc))

def process_product(product_no):
    global current_camera

    # =====================
    # CAM1
    # =====================
    root.after(0, lambda: show_page(page_capture))
    root.after(0, lambda: set_checklist(active="TOA", done=[]))
    root.after(0, lambda: set_step("CAM1 : Detect object by background"))
    root.after(0, lambda: set_status("CAM1 opening"))

    cam1_path = capture_path(f"cam1_product{product_no}")

    cam1 = PiCameraTest(
        index=PICAM1_INDEX,
        name="CAM1",
        status_cb=set_status,
        preview_cb=update_preview
    )
    current_camera = cam1

    try:
        cam1.open()
        bg = cam1.wait_background(lambda: running)

        if bg is None:
            return False

        detected = cam1.wait_object(bg, lambda: running)

        if not detected:
            return False

        cam1.focus_delay(lambda: running)
        cam1.capture_file(cam1_path)

    except Exception as e:
        root.after(0, lambda: messagebox.showerror("CAM1 Error", str(e)))
        return False

    finally:
        cam1.close()
        current_camera = None

    # =====================
    # CAM2
    # =====================
    if not running:
        return False

    root.after(0, lambda: set_checklist(active="TOB", done=["TOA"]))
    root.after(0, lambda: set_step("CAM2 : Capture directly"))
    root.after(0, lambda: set_status("CAM2 opening"))

    cam2_path = capture_path(f"cam2_product{product_no}")

    cam2 = PiCameraTest(
        index=PICAM2_INDEX,
        name="CAM2",
        status_cb=set_status,
        preview_cb=update_preview
    )
    current_camera = cam2

    try:
        cam2.open()
        cam2.focus_delay(lambda: running)
        cam2.capture_file(cam2_path)

    except Exception as e:
        root.after(0, lambda: messagebox.showerror("CAM2 Error", str(e)))
        return False

    finally:
        cam2.close()
        current_camera = None

    # =====================
    # OCR
    # =====================
    if not running:
        return False

    root.after(0, lambda: set_step("OCR : Center ROI CAM1/CAM2"))
    root.after(0, lambda: set_status("OCR reading"))

    cam1_data = process_center_roi_ocr(cam1_path, "cam1")
    cam2_data = process_center_roi_ocr(cam2_path, "cam2")

    dbg(f"PRODUCT {product_no}: OCR done, clear event before showing result")
    result_ok_event.clear()
    root.after(0, lambda: ui_call(f"set_checklist PRODUCT {product_no}", lambda: set_checklist(active=None, done=["TOA", "TOB"])))
    root.after(0, lambda: ui_call(f"show_product_result {product_no}", lambda: show_product_result(product_no, cam1_data, cam2_data)))

    # เน€เธเธเน€เธเธเน€เธยเน€เธเธเน€เธยเน€เธยเน€เธเธเน€เธยเน€เธยเน€เธยเน€เธยเน€เธยเน€เธโ€ OK / NEXT เน€เธยเน€เธยเน€เธเธเน€เธยเน€เธยเน€เธยเน€เธเธเน€เธเธเน€เธยเน€เธยเน€เธโ€“เน€เธยเน€เธเธ’เน€เธเธเน€เธโ€ขเน€เธเธ‘เน€เธเธเน€เธโ€“เน€เธเธ‘เน€เธโ€เน€เธยเน€เธย
    wait_result_ok()

    return running

def continue_toc():
    global running

    running = True
    result_ok_event.clear()
    dbg("CONTINUE TOC: event cleared")
    show_page(page_capture)
    set_checklist(active="TOC", done=["TOA", "TOB"])
    update_info()

    thread = threading.Thread(target=toc_loop, daemon=True)
    thread.start()

def toc_loop():
    global running

    # 6 products = 1 TOC capture.
    # Example: quantity 7 -> 2 TOC captures.
    toc_total = (quantity + 5) // 6
    dbg(f"TOC LOOP START quantity={quantity} toc_total={toc_total}")

    for toc_index in range(toc_total):
        if not running:
            dbg(f"TOC LOOP STOP before round {toc_index + 1}, running=False")
            return

        toc_no = toc_index + 1
        dbg(f"TOC LOOP ROUND START {toc_no}/{toc_total}")

        root.after(0, lambda: ui_call("show_page capture before TOC", lambda: show_page(page_capture)))
        root.after(0, lambda n=toc_no, total=toc_total: ui_call("set_progress TOC", lambda: set_progress(f"TOC {n} / {total}")))
        root.after(0, lambda: ui_call("set_checklist TOC active", lambda: set_checklist(active="TOC", done=["TOA", "TOB"])))
        root.after(0, lambda n=toc_no, total=toc_total: ui_call("set_step TOC waiting", lambda: set_step(f"CAM3 : Waiting TOC {n}/{total}")))
        root.after(0, lambda n=toc_no, total=toc_total: ui_call("set_status TOC waiting", lambda: set_status(f"CAM3 Waiting TOC {n}/{total}")))

        ok = process_toc(toc_no=toc_no, toc_total=toc_total)

        dbg(f"TOC LOOP ROUND END {toc_no}/{toc_total} ok={ok} running={running}")

        if not ok:
            running = False
            return

        sleep(0.5)

    running = False
    dbg("TOC LOOP COMPLETE -> page_complete")
    root.after(0, lambda: ui_call("show_page complete", lambda: show_page(page_complete)))

def process_toc(toc_no=1, toc_total=1):
    global current_camera

    dbg(f"PROCESS TOC START {toc_no}/{toc_total}")
    root.after(0, lambda: ui_call("set_step TOC open", lambda: set_step(f"CAM3 : TOC {toc_no}/{toc_total} YOLO detect -> capture -> crop ROI -> OCR")))
    root.after(0, lambda: ui_call("set_status TOC open", lambda: set_status(f"CAM3 opening TOC {toc_no}/{toc_total}")))

    toc_path = capture_path(f"cam3_toc_{toc_no}")

    cam3 = UsbCameraTest(
        device=USB_DEVICE,
        name="CAM3",
        status_cb=set_status,
        preview_cb=update_preview
    )

    current_camera = cam3

    try:
        dbg(f"TOC {toc_no}/{toc_total}: open camera")
        cam3.open()
        dbg(f"TOC {toc_no}/{toc_total}: capture_direct_or_yolo start")
        capture_result = cam3.capture_direct_or_yolo(toc_path, lambda: running)
        dbg(f"TOC {toc_no}/{toc_total}: capture_direct_or_yolo returned {bool(capture_result)}")

        if not capture_result:
            return False

    except Exception as e:
        dbg(f"TOC {toc_no}/{toc_total}: CAM3 error {e}")
        root.after(0, lambda: messagebox.showerror("CAM3 Error", str(e)))
        return False

    finally:
        cam3.close()
        current_camera = None
        dbg(f"TOC {toc_no}/{toc_total}: camera closed")

    image_path = capture_result["image_path"]
    detections = capture_result.get("detections", [])

    root.after(0, lambda: ui_call("set_step TOC OCR", lambda: set_step(f"TOC {toc_no}/{toc_total} OCR : YOLO ROI count = {len(detections)}")))
    root.after(0, lambda: ui_call("set_status TOC OCR", lambda: set_status(f"TOC {toc_no}/{toc_total} OCR reading from YOLO ROI")))

    dbg(f"TOC {toc_no}/{toc_total}: OCR start roi_count={len(detections)}")
    toc_items = process_yolo_roi_ocr(image_path, detections)
    dbg(f"TOC {toc_no}/{toc_total}: OCR done item_count={len(toc_items)}")

    fallback = None

    if not toc_items:
        dbg(f"TOC {toc_no}/{toc_total}: fallback center ROI start")
        root.after(0, lambda: ui_call("set_status TOC fallback", lambda: set_status("No YOLO ROI OCR, fallback center ROI")))
        fallback = process_center_roi_ocr(image_path, f"toc_center_{toc_no}")
        dbg(f"TOC {toc_no}/{toc_total}: fallback done")

    dbg(f"TOC {toc_no}/{toc_total}: CLEAR EVENT before show result")
    result_ok_event.clear()

    root.after(0, lambda: ui_call("set_checklist TOC done", lambda: set_checklist(active=None, done=["TOA", "TOB", "TOC"])))
    root.after(0, lambda: ui_call(f"show_toc_result {toc_no}/{toc_total}", lambda: show_toc_result(toc_items, fallback, toc_no, toc_total)))

    dbg(f"TOC {toc_no}/{toc_total}: WAIT_RESULT start")
    wait_result_ok(f"TOC {toc_no}/{toc_total}")
    dbg(f"TOC {toc_no}/{toc_total}: WAIT_RESULT released")

    return running

def stop_job():
    global running, current_camera

    dbg("STOP requested")
    running = False
    result_ok_event.set()

    try:
        if current_camera is not None:
            current_camera.close()
    except Exception:
        pass

    set_status("Stop Requested")

def on_result_ok():
    dbg(f"OK/NEXT pressed on {current_review_context}")
    result_ok_event.set()


def wait_result_ok(context="-"):
    global debug_wait_counter

    debug_wait_counter += 1
    wait_id = debug_wait_counter
    dbg(f"WAIT[{wait_id}] ENTER context={context}")

    last_log = time.time()

    while running and not result_ok_event.is_set():
        now = time.time()
        if now - last_log >= 1.0:
            dbg(f"WAIT[{wait_id}] still waiting context={context}")
            last_log = now
        sleep(0.1)

    dbg(f"WAIT[{wait_id}] EXIT context={context} event={result_ok_event.is_set()} running={running}")

    result_ok_event.clear()
    dbg(f"WAIT[{wait_id}] event cleared after release")

def reset_to_input():
    global running, current_index

    running = False
    current_index = 0

    delivery_entry.delete(0, tk.END)
    qty_entry.delete(0, tk.END)

    info_delivery.config(text="-")
    info_en.config(text="-")
    info_progress.config(text="-")
    info_pack.config(text="-")

    show_page(page_input)

def on_close():
    stop_job()
    root.destroy()

# =========================================================
# UI
# =========================================================
root = tk.Tk()
root.title("AI Camera TEST UI - YOLO ROI OCR")
root.geometry("1200x720")
root.resizable(False, False)

# =====================
# THEME
# =====================
BG = "#eef3f7"
CARD = "#ffffff"
TEXT = "#17202a"
MUTED = "#6c7a89"
BLUE = "#0b5cab"
BLUE_DARK = "#084b8a"
RED = "#c0392b"
RED_DARK = "#922b21"
BORDER = "#dfe6e9"
SOFT = "#f7f9fa"
SOFT_2 = "#f2f5f7"
GREEN = "#229954"
GREEN_BG = "#eafaf1"
YELLOW_TEXT = "#b9770e"
YELLOW_BG = "#fff3cd"
ORANGE = "#e67e22"
DARK = "#17202a"

root.configure(bg=BG)

# =====================
# ICONS
# =====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ICON_DIR = os.path.join(BASE_DIR, "assets", "icons")
icons = {}

def load_icon_file(filename, size=(24, 24)):
    """
    Load PNG icon from assets/icons.
    If icon is missing, return None so UI can still run.
    """
    path = os.path.join(ICON_DIR, filename)
    if not os.path.exists(path):
        print("Icon not found:", path)
        return None

    try:
        img = Image.open(path).convert("RGBA")
        try:
            img = img.resize(size, Image.Resampling.LANCZOS)
        except Exception:
            img = img.resize(size)
        return ImageTk.PhotoImage(img)
    except Exception as e:
        print("Icon load error:", path, e)
        return None

def load_icons():
    # Your current downloaded filenames
    icons["lot"] = load_icon_file("assignment.png")
    icons["delivery"] = load_icon_file("description.png")
    icons["barcode"] = load_icon_file("label.png")
    icons["progress"] = load_icon_file("analytics.png")
    icons["pack"] = load_icon_file("package.png")
    icons["camera"] = load_icon_file("photo.png")
    icons["status"] = load_icon_file("dashboard.png")
    icons["check"] = load_icon_file("checklist.png")
    icons["setting"] = load_icon_file("settings.png")
    icons["info"] = load_icon_file("info.png")
    icons["cancel"] = load_icon_file("cancel.png")
    icons["stop"] = load_icon_file("stop.png")

    # Optional icon. If you later rename refresh icon to refresh.png, it will be used.
    icons["refresh"] = load_icon_file("refresh.png")

def icon_label(parent, key, bg=CARD, size=None):
    icon = icons.get(key)
    if icon is not None:
        return tk.Label(parent, image=icon, bg=bg)

    # fallback when icon file missing
    fallback = key.upper()[:4]
    return tk.Label(
        parent,
        text=fallback,
        bg=SOFT,
        fg=BLUE,
        font=("Arial", 8, "bold"),
        width=5,
        height=2
    )

def title_with_icon(parent, icon_key, text, bg=CARD):
    frame = tk.Frame(parent, bg=bg)
    frame.pack(fill="x")

    icon_label(frame, icon_key, bg=bg).pack(side="left", padx=(0, 10))

    tk.Label(
        frame,
        text=text,
        bg=bg,
        fg=TEXT,
        font=("Arial", 14, "bold")
    ).pack(side="left")

    return frame

def flat_button(parent, text, bg, fg="white", command=None, icon_key=None, width=None):
    kwargs = {
        "text": text,
        "bg": bg,
        "fg": fg,
        "activebackground": RED_DARK if bg == RED else BLUE_DARK,
        "activeforeground": "white",
        "font": ("Arial", 10, "bold"),
        "relief": "flat",
        "command": command,
        "cursor": "hand2",
        "padx": 10,
        "pady": 5
    }

    if width is not None:
        kwargs["width"] = width

    icon = icons.get(icon_key) if icon_key else None
    if icon is not None:
        kwargs["image"] = icon
        kwargs["compound"] = "left"

    return tk.Button(parent, **kwargs)

load_icons()

# Hidden one-pixel top bar, kept only for page packing consistency
top = tk.Frame(root, bg=BG, height=1)
top.pack(side="top", fill="x")
top.pack_propagate(False)

# Placeholder labels before real capture-page labels are created
info_delivery = tk.Label(root)
info_en = tk.Label(root)
info_progress = tk.Label(root)
info_pack = tk.Label(root)

# =========================================================
# PAGE INPUT
# =========================================================
page_input = tk.Frame(root, bg=BG)

input_container = tk.Frame(page_input, bg=BG)
input_container.pack(fill="both", expand=True, padx=42, pady=34)

# Left information panel
input_left = tk.Frame(input_container, bg="#0b2a4a", padx=34, pady=30)
input_left.pack(side="left", fill="both", expand=True)

tk.Label(
    input_left,
    text="AI Camera Validation",
    font=("Arial", 26, "bold"),
    bg="#0b2a4a",
    fg="white"
).pack(anchor="w", pady=(22, 8))

tk.Label(
    input_left,
    text="TEST MODE",
    font=("Arial", 14, "bold"),
    bg="#0b2a4a",
    fg="#f1c40f"
).pack(anchor="w")

tk.Label(
    input_left,
    text="Capture  ->  ROI  ->  OCR  ->  Review",
    font=("Arial", 15),
    bg="#0b2a4a",
    fg="#d6eaf8"
).pack(anchor="w", pady=(24, 8))

tk.Label(
    input_left,
    text="Use this screen to test camera flow, YOLO ROI and EasyOCR before integrating with the main system.",
    font=("Arial", 12),
    bg="#0b2a4a",
    fg="#d6eaf8",
    wraplength=430,
    justify="left"
).pack(anchor="w", pady=(4, 20))

for item in [
    "CAM1 : Background detect + center ROI OCR",
    "CAM2 : Direct capture + center ROI OCR",
    "CAM3 : YOLO detect + crop ROI OCR",
    "Result page : Review ROI and press OK / NEXT"
]:
    tk.Label(
        input_left,
        text="[OK] " + item,
        font=("Arial", 12, "bold"),
        bg="#0b2a4a",
        fg="white",
        anchor="w"
    ).pack(anchor="w", pady=6)

# Right input card
input_card = tk.Frame(input_container, bg=CARD, padx=34, pady=32, highlightbackground=BORDER, highlightthickness=1)
input_card.pack(side="right", fill="y", padx=(28, 0))
input_card.pack_propagate(False)
input_card.config(width=380)

tk.Label(
    input_card,
    text="Start Test Job",
    font=("Arial", 22, "bold"),
    bg=CARD,
    fg=TEXT
).pack(anchor="w", pady=(14, 4))

tk.Label(
    input_card,
    text="Input data to start capture test.",
    font=("Arial", 11),
    bg=CARD,
    fg=MUTED
).pack(anchor="w", pady=(0, 22))

tk.Label(
    input_card,
    text="Delivery No",
    bg=CARD,
    fg="#34495e",
    font=("Arial", 11, "bold")
).pack(anchor="w")

delivery_entry = tk.Entry(
    input_card,
    font=("Arial", 18),
    justify="center",
    width=22,
    relief="solid",
    borderwidth=1
)
delivery_entry.pack(fill="x", pady=(6, 18), ipady=5)

tk.Label(
    input_card,
    text="Quantity",
    bg=CARD,
    fg="#34495e",
    font=("Arial", 11, "bold")
).pack(anchor="w")

qty_entry = tk.Entry(
    input_card,
    font=("Arial", 18),
    justify="center",
    width=10,
    relief="solid",
    borderwidth=1
)
qty_entry.pack(anchor="w", pady=(6, 24), ipady=5)

tk.Button(
    input_card,
    text="START TEST",
    font=("Arial", 14, "bold"),
    bg=BLUE,
    fg="white",
    activebackground=BLUE_DARK,
    activeforeground="white",
    width=22,
    relief="flat",
    command=start_job
).pack(fill="x", ipady=6, pady=(4, 12))

tk.Label(
    input_card,
    text="Quantity = number of products for CAM1 / CAM2 capture test.",
    bg=CARD,
    fg=MUTED,
    font=("Arial", 9),
    wraplength=300,
    justify="left"
).pack(anchor="w", pady=(8, 0))

# =========================================================
# PAGE CAPTURE
# =========================================================
page_capture = tk.Frame(root, bg=BG)

capture_container = tk.Frame(page_capture, bg=BG)
capture_container.pack(fill="both", expand=True, padx=14, pady=14)

# =====================
# LOT INFORMATION
# =====================
lot_frame = tk.Frame(
    capture_container,
    bg=CARD,
    highlightbackground=BORDER,
    highlightthickness=1,
    padx=18,
    pady=12
)
lot_frame.pack(fill="x", pady=(0, 10))

lot_top = tk.Frame(lot_frame, bg=CARD)
lot_top.pack(fill="x")

lot_title = tk.Frame(lot_top, bg=CARD)
lot_title.pack(side="left")

icon_label(lot_title, "lot", bg=CARD).pack(side="left", padx=(0, 10))

tk.Label(
    lot_title,
    text="Lot Information",
    bg=CARD,
    fg=TEXT,
    font=("Arial", 14, "bold")
).pack(side="left")

flat_button(
    lot_top,
    text="  Cancel Job",
    bg=RED,
    fg="white",
    command=reset_to_input,
    icon_key="cancel",
    width=130
).pack(side="right")

lot_info_row = tk.Frame(lot_frame, bg=CARD)
lot_info_row.pack(fill="x", pady=(14, 4))

def lot_info_box(parent, icon_key, title, width=14):
    box = tk.Frame(parent, bg=CARD)
    box.pack(side="left", padx=(0, 46))

    icon_label(box, icon_key, bg=CARD).pack(side="left", padx=(0, 10))

    text_box = tk.Frame(box, bg=CARD)
    text_box.pack(side="left")

    tk.Label(
        text_box,
        text=title,
        bg=CARD,
        fg=MUTED,
        font=("Arial", 9, "bold")
    ).pack(anchor="w")

    value = tk.Label(
        text_box,
        text="-",
        bg=CARD,
        fg=TEXT,
        font=("Arial", 14, "bold"),
        width=width,
        anchor="w"
    )
    value.pack(anchor="w", pady=(2, 0))

    return value

info_delivery = lot_info_box(lot_info_row, "delivery", "Delivery No", 14)
info_en = lot_info_box(lot_info_row, "barcode", "EN", 10)
info_progress = lot_info_box(lot_info_row, "progress", "Progress", 10)
info_pack = lot_info_box(lot_info_row, "pack", "Pack Type", 12)

# =====================
# BODY AREA
# =====================
body_frame = tk.Frame(capture_container, bg=BG)
body_frame.pack(fill="both", expand=True)

# LEFT : CAPTURE IMAGE
capture_left = tk.Frame(
    body_frame,
    bg=CARD,
    highlightbackground=BORDER,
    highlightthickness=1,
    padx=18,
    pady=14
)
capture_left.pack(side="left", fill="both", expand=True, padx=(0, 10))

title_with_icon(capture_left, "camera", "Capture Images", bg=CARD)

tk.Label(
    capture_left,
    text="Preview",
    bg=CARD,
    fg=MUTED,
    font=("Arial", 9, "bold")
).pack(anchor="w", pady=(8, 4))

preview_frame = tk.Frame(
    capture_left,
    bg=DARK,
    width=760,
    height=430,
    highlightbackground=DARK,
    highlightthickness=2
)
preview_frame.pack(pady=(4, 10))
preview_frame.pack_propagate(False)

preview_label = tk.Label(
    preview_frame,
    text="Camera Preview",
    bg=DARK,
    fg="white",
    font=("Arial", 16)
)
preview_label.place(x=0, y=0, relwidth=1, relheight=1)

# Preview overlay
preview_overlay = tk.Frame(preview_frame, bg=DARK)
preview_overlay.place(x=10, y=8)

capture_state_dot = tk.Canvas(
    preview_overlay,
    width=14,
    height=14,
    bg=DARK,
    highlightthickness=0
)
capture_state_dot.pack(side="left", padx=(0, 5))
capture_state_dot.create_oval(2, 2, 12, 12, fill="#f1c40f", outline="#f1c40f")

capture_state_label = tk.Label(
    preview_overlay,
    text="Waiting Object",
    bg=DARK,
    fg="white",
    font=("Arial", 11, "bold"),
    padx=2,
    pady=2
)
capture_state_label.pack(side="left")

reset_btn = tk.Button(
    capture_left,
    text="  Reset Background",
    bg="#dfe6e9",
    fg=TEXT,
    activebackground="#cfd8dc",
    activeforeground=TEXT,
    font=("Arial", 10, "bold"),
    relief="flat",
    height=2,
    cursor="hand2"
)

if icons.get("refresh") is not None:
    reset_btn.config(image=icons["refresh"], compound="left")

reset_btn.pack(fill="x", pady=(8, 0))

# RIGHT : STATUS
capture_right = tk.Frame(
    body_frame,
    bg=CARD,
    highlightbackground=BORDER,
    highlightthickness=1,
    padx=18,
    pady=14,
    width=330
)
capture_right.pack(side="right", fill="y")
capture_right.pack_propagate(False)

title_with_icon(capture_right, "status", "Status", bg=CARD)

# Product block
product_header = tk.Frame(capture_right, bg=CARD)
product_header.pack(fill="x", pady=(18, 2))

icon_label(product_header, "pack", bg=CARD).pack(side="left", padx=(0, 8))

tk.Label(
    product_header,
    text="Product",
    bg=CARD,
    fg=TEXT,
    font=("Arial", 10, "bold")
).pack(side="left")

product_row = tk.Frame(capture_right, bg=CARD)
product_row.pack(fill="x")

progress_label = tk.Label(
    product_row,
    text="Product 0 / 0",
    bg=CARD,
    fg=TEXT,
    font=("Arial", 10)
)
progress_label.pack(side="left")

progress_percent_label = tk.Label(
    product_row,
    text="0%",
    bg=CARD,
    fg=TEXT,
    font=("Arial", 10)
)
progress_percent_label.pack(side="right")

progress_bar_bg = tk.Frame(
    capture_right,
    bg="#c4c4c8",
    height=18
)
progress_bar_bg.pack(fill="x", pady=(4, 20))
progress_bar_bg.pack_propagate(False)

progress_bar_fill = tk.Frame(
    progress_bar_bg,
    bg=BLUE
)
progress_bar_fill.place(relx=0, rely=0, relwidth=0, relheight=1)

# Checklist block
check_header = tk.Frame(capture_right, bg=CARD)
check_header.pack(fill="x", pady=(4, 8))

icon_label(check_header, "check", bg=CARD).pack(side="left", padx=(0, 8))

tk.Label(
    check_header,
    text="Checklist",
    bg=CARD,
    fg=TEXT,
    font=("Arial", 10, "bold")
).pack(side="left")

checklist_frame = tk.Frame(capture_right, bg=CARD)
checklist_frame.pack(fill="x")

def make_check_item(name, text):
    item = tk.Label(
        checklist_frame,
        text=text,
        bg=SOFT,
        fg=TEXT,
        font=("Arial", 12, "bold"),
        anchor="w",
        padx=14,
        pady=9,
        relief="flat",
        borderwidth=0
    )
    item.pack(fill="x", pady=5)
    check_labels[name] = item

make_check_item("TOA", "[ ]   TOA")
make_check_item("TOB", "[ ]   TOB")
make_check_item("TOC", "[ ]   TOC")

# Step / Status detail block
step_status_box = tk.Frame(capture_right, bg=SOFT, padx=12, pady=10)
step_status_box.pack(fill="x", pady=(16, 10))

step_header = tk.Frame(step_status_box, bg=SOFT)
step_header.pack(fill="x")

icon_label(step_header, "setting", bg=SOFT).pack(side="left", padx=(0, 8))

tk.Label(
    step_header,
    text="Step",
    bg=SOFT,
    fg="#34495e",
    font=("Arial", 9, "bold")
).pack(side="left")

step_label = tk.Label(
    step_status_box,
    text="-",
    bg=SOFT,
    fg=TEXT,
    font=("Arial", 10),
    wraplength=280,
    justify="left"
)
step_label.pack(anchor="w", pady=(2, 8))

status_detail_header = tk.Frame(step_status_box, bg=SOFT)
status_detail_header.pack(fill="x")

icon_label(status_detail_header, "info", bg=SOFT).pack(side="left", padx=(0, 8))

tk.Label(
    status_detail_header,
    text="Status",
    bg=SOFT,
    fg="#34495e",
    font=("Arial", 9, "bold")
).pack(side="left")

status_label = tk.Label(
    step_status_box,
    text="-",
    bg=SOFT,
    fg=TEXT,
    font=("Arial", 10),
    wraplength=280,
    justify="left"
)
status_label.pack(anchor="w", pady=(2, 0))

stop_capture_btn = tk.Button(
    capture_right,
    text="  STOP",
    bg=RED,
    fg="white",
    activebackground=RED_DARK,
    activeforeground="white",
    font=("Arial", 13, "bold"),
    relief="flat",
    cursor="hand2",
    image=icons["stop"] if icons.get("stop") is not None else None,
    compound="left",
    padx=12,
    pady=10,
    command=stop_job
)
stop_capture_btn.pack(side="bottom", fill="x", padx=10, pady=(8, 18))

# =========================================================
# PAGE RESULT
# =========================================================
page_result = tk.Frame(root, bg=BG)

result_header = tk.Frame(page_result, bg=BG)
result_header.pack(fill="x", padx=16, pady=(12, 8))

result_left_header = tk.Frame(result_header, bg=BG)
result_left_header.pack(side="left")

result_title = tk.Label(
    result_left_header,
    text="OCR Result",
    font=("Arial", 22, "bold"),
    bg=BG,
    fg=TEXT
)
result_title.pack(side="left")

result_status_badge = tk.Label(
    result_left_header,
    text="WAIT REVIEW",
    bg="#f1c40f",
    fg=TEXT,
    font=("Arial", 12, "bold"),
    padx=14,
    pady=6
)
result_status_badge.pack(side="left", padx=14)

result_button_frame = tk.Frame(result_header, bg=BG)
result_button_frame.pack(side="right")

ok_next_btn = tk.Button(
    result_button_frame,
    text="OK / NEXT",
    bg="#27ae60",
    fg="white",
    activebackground="#1e8449",
    activeforeground="white",
    font=("Arial", 12, "bold"),
    relief="flat",
    cursor="hand2",
    padx=18,
    pady=8,
    command=on_result_ok
)
ok_next_btn.pack(side="right", padx=(8, 0))

stop_result_btn = tk.Button(
    result_button_frame,
    text="  STOP",
    bg=RED,
    fg="white",
    activebackground=RED_DARK,
    activeforeground="white",
    font=("Arial", 12, "bold"),
    relief="flat",
    cursor="hand2",
    image=icons["stop"] if icons.get("stop") is not None else None,
    compound="left",
    padx=14,
    pady=8,
    command=stop_job
)
stop_result_btn.pack(side="right", padx=(8, 0))

result_hint = tk.Label(
    page_result,
    text="Review ROI image and OCR result, then press OK / NEXT to continue.",
    bg=BG,
    fg=MUTED,
    font=("Arial", 12)
)
result_hint.pack(anchor="w", padx=18, pady=(0, 10))

# Fixed Product Result area: created once, then only image/value labels are updated.
product_result_area = tk.Frame(page_result, bg=BG)
product_result_area.pack(fill="both", expand=True, padx=10, pady=(0, 12))

make_product_result_section(
    product_result_area,
    key="vendor",
    title="Vendor Label",
    icon_key="delivery",
    field1="BOX ID",
    field2="Date Code"
)

make_product_result_section(
    product_result_area,
    key="toa",
    title="TOA Label",
    icon_key="barcode",
    field1="Lot No",
    field2="Date Code"
)

make_product_result_section(
    product_result_area,
    key="tob",
    title="TOB Label",
    icon_key="pack",
    field1="Lot No",
    field2="Date Code"
)

# Temporary TOC result area. Hidden during product review.
toc_result_area = tk.Frame(page_result, bg=BG)

# Keep this hidden frame so older helper functions do not error.
result_list_frame = tk.Frame(page_result, bg=BG)

# =========================================================
# READY TOC
# =========================================================
page_ready_toc = tk.Frame(root, bg=BG)

toc_card = tk.Frame(page_ready_toc, bg=CARD, padx=35, pady=35, highlightbackground=BORDER, highlightthickness=1)
toc_card.place(relx=0.5, rely=0.45, anchor="center", width=540, height=280)

tk.Label(
    toc_card,
    text="CAM1 / CAM2 Test Completed",
    font=("Arial", 21, "bold"),
    bg=CARD,
    fg="#229954"
).pack(pady=15)

tk.Label(
    toc_card,
    text="Press continue to test CAM3 TOC YOLO ROI OCR",
    font=("Arial", 13),
    bg=CARD,
    fg="#34495e"
).pack(pady=8)

tk.Button(
    toc_card,
    text="CONTINUE TO TOC",
    bg=BLUE,
    fg="white",
    font=("Arial", 14, "bold"),
    width=20,
    relief="flat",
    command=continue_toc
).pack(pady=20)

# =========================================================
# COMPLETE
# =========================================================
page_complete = tk.Frame(root, bg=BG)

complete_card = tk.Frame(page_complete, bg=CARD, padx=35, pady=35, highlightbackground=BORDER, highlightthickness=1)
complete_card.place(relx=0.5, rely=0.45, anchor="center", width=520, height=260)

tk.Label(
    complete_card,
    text="TEST COMPLETED",
    font=("Arial", 24, "bold"),
    bg=CARD,
    fg="#229954"
).pack(pady=25)

tk.Button(
    complete_card,
    text="OK",
    bg=BLUE,
    fg="white",
    font=("Arial", 14, "bold"),
    width=16,
    relief="flat",
    command=reset_to_input
).pack(pady=10)


# Debug page names
page_input._debug_name = "page_input"
page_capture._debug_name = "page_capture"
page_result._debug_name = "page_result"
page_ready_toc._debug_name = "page_ready_toc"
page_complete._debug_name = "page_complete"

root.protocol("WM_DELETE_WINDOW", on_close)
show_page(page_input)
root.mainloop()
