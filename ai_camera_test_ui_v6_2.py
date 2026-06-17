import os
import json
import threading
from threading import Event
from datetime import datetime
from time import sleep
import time
import traceback
import math

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
    PREVIEW_INTERVAL,
)

from camera_test import PiCameraTest, UsbCameraTest
from roi_test import crop_center, crop_by_box, preprocess_ocr, save_roi_files
from ocr_test import run_easyocr, load_ocr

os.makedirs(CAPTURE_DIR, exist_ok=True)
os.makedirs(ROI_DIR, exist_ok=True)
os.makedirs(OCR_DIR, exist_ok=True)

# =========================================================
# GLOBAL STATE
# =========================================================
running = False
last_preview_time = 0
current_camera = None

# Action event is used for OK/NEXT, Restart The Process, Cancel Job.
# action_value can be: None, "next", "restart", "cancel"
action_event = Event()
action_value = None

debug_wait_counter = 0
current_review_context = "-"
current_screen_mode = "input"  # input, capture_inner, result_inner, capture_outer, result_outer, ready_toc, complete

check_labels = {}
capture_state_label = None
capture_state_dot = None
result_product_slots = {}

# Job data
en_no = ""
delivery_no = ""
item_no = "-"       # Test mode: placeholder. Future: from picking list.
pack_type = "Tray"  # Fixed for this test UI.
quantity = 0        # Inner Box total.
current_index = 0   # Current Inner Box index.
current_outer_index = 0
outer_total = 0     # ceil(quantity / 6)


def dbg(msg):
    try:
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        th = threading.current_thread().name
        ev = action_event.is_set()
        print(f"[DBG {ts} {th} action={action_value} event={ev} running={running}] {msg}", flush=True)
    except Exception as e:
        print("[DBG ERROR]", e, msg, flush=True)


def ui_call(name, func):
    try:
        dbg(f"UI_CALL START: {name}")
        func()
        dbg(f"UI_CALL END: {name}")
    except Exception:
        print(f"[UI ERROR] {name}", flush=True)
        traceback.print_exc()


def now_name(prefix):
    return datetime.now().strftime(f"{prefix}_%Y%m%d_%H%M%S_%f.jpg")


def capture_path(prefix):
    return os.path.join(CAPTURE_DIR, now_name(prefix))


def current_process_alive():
    """Used by camera loops. False means current capture should stop/restart/cancel."""
    return running and action_value is None


def set_action(action):
    """Set user action: next, restart, or cancel."""
    global action_value, running, current_camera

    dbg(f"ACTION requested: {action}")
    action_value = action
    action_event.set()

    if action == "cancel":
        running = False

    # Force camera loops to unblock quickly.
    try:
        if current_camera is not None:
            current_camera.close()
    except Exception:
        pass


def clear_action():
    global action_value
    action_value = None
    action_event.clear()


def set_status(text):
    try:
        status_label.config(text=text)

        color = "#f1c40f"
        lower = text.lower()
        if "capture" in lower or "capturing" in lower or "opening" in lower:
            color = "#3498db"
        elif "ocr" in lower or "reading" in lower:
            color = "#e67e22"
        elif "ready" in lower:
            color = "#27ae60"
        elif "stop" in lower or "error" in lower or "cancel" in lower:
            color = "#c0392b"

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
    try:
        if not image_path or not os.path.exists(image_path):
            return None
        img = cv2.imread(image_path)
        if img is None:
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, size)
        return ImageTk.PhotoImage(image=Image.fromarray(img))
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


def get_current_count(mode):
    if mode == "outer":
        return current_outer_index, outer_total
    return current_index, quantity


def update_info(mode="inner"):
    idx, total = get_current_count(mode)
    box_title = "Outer Box" if mode == "outer" else "Inner Box"

    info_delivery.config(text=delivery_no or "-")
    info_item.config(text=item_no or "-")
    info_en.config(text=en_no or "-")
    info_box_title.config(text=box_title)
    info_box.config(text=f"{idx} / {total}" if total else "-")
    info_pack.config(text=pack_type)


def set_progress(mode="inner"):
    idx, total = get_current_count(mode)
    label = "Outer Box" if mode == "outer" else "Inner Box"
    progress_title_label.config(text=label)
    progress_label.config(text=f"{idx} / {total}" if total else "0 / 0")

    try:
        percent = int((idx / total) * 100) if total > 0 else 0
        progress_percent_label.config(text=f"{percent}%")
        progress_bar_fill.place(relx=0, rely=0, relwidth=percent / 100, relheight=1)
    except Exception:
        pass

    root.update_idletasks()


def set_checklist(active=None, done=None, mode="inner"):
    """
    Checklist count rule:
    - TOA / TOB are Inner Box processes, so their count follows Inner Box.
      During Outer Box capture, TOA / TOB are already finished, so show quantity/quantity.
    - TOC is Outer Box process, so its count follows Outer Box.
      During Inner Box capture, show the first planned Outer Box count, e.g. 1/1 or 1/6.
    """
    done = done or []

    if mode == "outer":
        inner_count = f"{quantity}/{quantity}" if quantity else "-/-"
        toc_idx = current_outer_index if current_outer_index > 0 else 1
        toc_count = f"{toc_idx}/{outer_total}" if outer_total else "-/-"
    else:
        inner_count = f"{current_index}/{quantity}" if quantity else "-/-"
        toc_idx = 1 if outer_total > 0 else 0
        toc_count = f"{toc_idx}/{outer_total}" if outer_total else "-/-"

    for name, label in check_labels.items():
        count_text = toc_count if name == "TOC" else inner_count

        if name in done:
            label.config(text=f"[OK]  {name:<4} {count_text}", fg=GREEN, bg=GREEN_BG)
        elif active == name:
            label.config(text=f"[RUN] {name:<4} {count_text}", fg=YELLOW_TEXT, bg=YELLOW_BG)
        else:
            label.config(text=f"[ ]   {name:<4} {count_text}", fg=TEXT, bg=SOFT)


def setup_capture_page(mode="inner", active="TOA", done=None):
    global current_screen_mode
    current_screen_mode = "capture_outer" if mode == "outer" else "capture_inner"
    update_info(mode)
    set_progress(mode)
    set_checklist(active=active, done=done or [], mode=mode)
    show_page(page_capture)


# =========================================================
# OCR PROCESS HELPERS
# =========================================================
def process_center_roi_ocr(image_path, prefix):
    img = cv2.imread(image_path)
    if img is None:
        return {"name": prefix, "image_path": image_path, "error": "cannot_read_image", "raw": "", "clean": ""}

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
        "items": ocr["items"],
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
            "items": ocr["items"],
        })

    return results


def save_json(data, filename):
    path = os.path.join(OCR_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print("Saved JSON:", path)


# =========================================================
# RESULT UI HELPERS
# =========================================================
def load_result_photo(image_path, size=(300, 86)):
    try:
        if not image_path or not os.path.exists(image_path):
            return None
        img = cv2.imread(image_path)
        if img is None:
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, size)
        return ImageTk.PhotoImage(image=Image.fromarray(img))
    except Exception as e:
        print("Result image load error:", e)
        return None


def update_result_image(label, image_path, size=(340, 70)):
    photo = load_result_photo(image_path, size=size)
    if photo is not None:
        label.imgtk = photo
        label.config(image=photo, text="")
    else:
        label.imgtk = None
        label.config(image="", text="NO IMAGE", fg="white", font=("Arial", 10, "bold"))


def update_value_box(label, value):
    text = value or "(empty)"
    is_ok = text != "(empty)"
    label.config(text=text, bg=GREEN_BG if is_ok else "#f8f9fa", fg=GREEN if is_ok else "#7f8c8d")


def make_result_field(parent, field_title, value_title):
    block = tk.Frame(parent, bg=CARD)
    block.pack(fill="x", pady=(0, 18))

    tk.Label(block, text=field_title, bg=CARD, fg=MUTED, font=("Arial", 10, "bold")).pack(anchor="w", pady=(0, 5))

    img_box = tk.Label(block, bg=DARK, width=340, height=70)
    img_box.pack(fill="x", pady=(0, 8))

    row = tk.Frame(block, bg=CARD)
    row.pack(fill="x")

    tk.Label(row, text=value_title, bg=CARD, fg=TEXT, font=("Arial", 11, "bold"), width=10, anchor="w").pack(side="left")

    value_box = tk.Label(row, text="-", bg="#f8f9fa", fg="#7f8c8d", font=("Arial", 14, "bold"), anchor="w", padx=10, pady=7)
    value_box.pack(side="left", fill="x", expand=True)

    return {"image": img_box, "value": value_box}


def make_product_result_section(parent, key, title, icon_key, field1, value1, field2, value2):
    card = tk.Frame(parent, bg=CARD, highlightbackground=BORDER, highlightthickness=1, padx=14, pady=12)
    card.pack(side="left", fill="both", expand=True, padx=6)

    title_row = tk.Frame(card, bg=CARD)
    title_row.pack(fill="x", pady=(0, 12))
    icon_label(title_row, icon_key, bg=CARD).pack(side="left", padx=(0, 8))
    tk.Label(title_row, text=title, bg=CARD, fg=TEXT, font=("Arial", 15, "bold")).pack(side="left")

    first = make_result_field(card, field1, value1)

    # Keep this blank area for future validation data / extra fields.
    # User wants this area reserved; 240 keeps spacing close to the original V5 layout.
    tk.Frame(card, bg=CARD, height=240).pack(fill="x")

    second = make_result_field(card, field2, value2)

    result_product_slots[key] = {"first": first, "second": second}


def update_product_result_section(key, first_image_path, first_value, second_image_path, second_value):
    section = result_product_slots.get(key)
    if not section:
        return

    update_result_image(section["first"]["image"], first_image_path, size=(340, 70))
    update_value_box(section["first"]["value"], first_value)

    update_result_image(section["second"]["image"], second_image_path, size=(340, 70))
    update_value_box(section["second"]["value"], second_value)


def clear_toc_cards():
    try:
        for w in toc_result_area.winfo_children():
            w.destroy()
    except Exception:
        pass


def add_toc_result_row(parent, row_no, title, image_data=None):
    data = image_data or {}
    clean_text = data.get("clean", "") or "(empty)"
    raw_text = data.get("raw", "") or ""
    source_text = data.get("source", "-")
    detect_conf = data.get("detect_conf", None)
    if detect_conf is not None:
        source_text = f"{source_text} / DET {detect_conf:.2f}"

    row = tk.Frame(parent, bg=CARD, highlightbackground=BORDER, highlightthickness=1, padx=10, pady=6)
    row.pack(fill="x", padx=12, pady=3)

    name_box = tk.Frame(row, bg=CARD, width=88)
    name_box.pack(side="left", fill="y", padx=(0, 10))
    name_box.pack_propagate(False)

    tk.Label(name_box, text=str(row_no), bg=BLUE, fg="white", font=("Arial", 10, "bold"), width=3, pady=2).pack(anchor="w")
    tk.Label(name_box, text=title.upper(), bg=CARD, fg=TEXT, font=("Arial", 12, "bold"), anchor="w").pack(anchor="w", pady=(4, 0))

    roi_path = data.get("roi_path", "")
    photo = load_result_photo(roi_path, size=(260, 56))

    image_wrap = tk.Frame(row, bg=DARK, width=260, height=56)
    image_wrap.pack(side="left", padx=(0, 14))
    image_wrap.pack_propagate(False)

    img_box = tk.Label(image_wrap, bg=DARK, fg="white", text="NO ROI", font=("Arial", 9, "bold"))
    img_box.pack(fill="both", expand=True)
    if photo is not None:
        img_box.imgtk = photo
        img_box.config(image=photo, text="")

    info_box = tk.Frame(row, bg=CARD)
    info_box.pack(side="left", fill="both", expand=True)

    top_line = tk.Frame(info_box, bg=CARD)
    top_line.pack(fill="x")
    tk.Label(top_line, text="OCR Result", bg=CARD, fg=MUTED, font=("Arial", 9, "bold")).pack(side="left")
    tk.Label(top_line, text=source_text, bg=CARD, fg=MUTED, font=("Arial", 8), anchor="e").pack(side="right")

    value_box = tk.Label(
        info_box,
        text=clean_text,
        bg=GREEN_BG if clean_text != "(empty)" else "#f8f9fa",
        fg=GREEN if clean_text != "(empty)" else "#7f8c8d",
        font=("Arial", 14, "bold"),
        anchor="w",
        padx=10,
        pady=4,
    )
    value_box.pack(fill="x", pady=(3, 2))

    tk.Label(info_box, text=f"RAW : {raw_text or '-'}", bg=CARD, fg=MUTED, font=("Arial", 9), anchor="w").pack(anchor="w")


def show_product_result(product_no, cam1_data, cam2_data):
    global current_review_context, current_screen_mode
    current_screen_mode = "result_inner"
    current_review_context = f"INNER BOX {product_no}/{quantity}"
    dbg(f"SHOW PRODUCT RESULT product={product_no}/{quantity}")

    show_page(page_result)
    result_title.config(text=f"OCR Result : Inner Box {product_no} / {quantity}")
    result_status_badge.config(text="WAIT REVIEW", bg="#f1c40f", fg=TEXT)
    result_hint.config(text="Review ROI image and OCR result, then press OK / NEXT to continue.")

    product_result_area.pack(fill="both", expand=True, padx=10, pady=(0, 12))
    toc_result_area.pack_forget()

    cam1_text = cam1_data.get("clean", "") or "(empty)"
    cam2_text = cam2_data.get("clean", "") or "(empty)"
    cam1_roi = cam1_data.get("roi_path", "")
    cam2_roi = cam2_data.get("roi_path", "")

    # Test mapping: real field-level ROI can be added later.
    update_product_result_section("vendor", cam1_roi, cam1_text, cam1_roi, cam1_text)
    update_product_result_section("toa", cam1_roi, cam1_text, cam1_roi, cam1_text)
    update_product_result_section("tob", cam2_roi, cam2_text, cam2_roi, cam2_text)

    save_json({"type": "inner_box", "inner_box_no": product_no, "inner_box_total": quantity, "cam1": cam1_data, "cam2": cam2_data}, f"inner_box_{product_no:03d}.json")


def show_toc_result(toc_items, fallback_data=None, toc_no=1, toc_total=1):
    global current_review_context, current_screen_mode
    current_screen_mode = "result_outer"
    current_review_context = f"OUTER BOX {toc_no}/{toc_total}"
    dbg(f"SHOW TOC RESULT outer={toc_no}/{toc_total} items={len(toc_items or [])} fallback={fallback_data is not None}")

    show_page(page_result)
    result_title.config(text=f"OCR Result : Outer Box {toc_no} / {toc_total}")
    result_status_badge.config(text="WAIT REVIEW", bg="#f1c40f", fg=TEXT)

    product_result_area.pack_forget()
    toc_result_area.pack(fill="both", expand=True, padx=10, pady=(0, 12))
    clear_toc_cards()

    result_hint.config(text="Review TOC ROI from TOC to TOC6. Image is on the left and OCR result is on the right.")

    item_map = {}
    for item in toc_items or []:
        name = str(item.get("name", "")).lower()
        item_map[name] = item

    expected_names = ["toc", "toc1", "toc2", "toc3", "toc4", "toc5", "toc6"]
    if fallback_data is not None and not item_map:
        item_map["toc"] = fallback_data

    for idx, name in enumerate(expected_names, start=1):
        add_toc_result_row(toc_result_area, row_no=idx, title=name, image_data=item_map.get(name))

    save_json({"type": "outer_box", "outer_box_no": toc_no, "outer_box_total": toc_total, "toc_items": toc_items, "fallback": fallback_data}, f"outer_box_{toc_no:02d}.json")


# =========================================================
# JOB FLOW
# =========================================================
def start_job():
    global running, en_no, delivery_no, quantity, current_index, current_outer_index, outer_total, item_no, action_value

    en_no = en_entry.get().strip()
    delivery_no = delivery_entry.get().strip()
    qty_text = qty_entry.get().strip()
    item_no = "-"

    if not delivery_no:
        messagebox.showwarning("Warning", "Please input Delivary NO")
        return
    if not qty_text.isdigit():
        messagebox.showwarning("Warning", "Please input Quantity")
        return

    quantity = int(qty_text)
    if quantity < 1 or quantity > 100:
        messagebox.showwarning("Warning", "Quantity must be 1-100")
        return

    outer_total = math.ceil(quantity / 6)
    current_index = 0
    current_outer_index = 0
    running = True
    clear_action()

    dbg("START JOB")
    setup_capture_page(mode="inner", active="TOA", done=[])
    set_status("Loading OCR Model...")

    thread = threading.Thread(target=job_loop, daemon=True)
    thread.start()


def job_loop():
    global current_index, running

    root.after(0, lambda: set_status("Loading OCR Model..."))
    load_ocr()
    root.after(0, lambda: set_status("OCR Ready"))

    for i in range(1, quantity + 1):
        if not running:
            return

        current_index = i
        root.after(0, lambda i=i: setup_capture_page(mode="inner", active="TOA", done=[]))

        ok = process_product(i)
        if not ok:
            if action_value == "cancel":
                root.after(0, reset_to_input)
            running = False
            return

        sleep(0.4)

    running = False
    root.after(0, lambda: show_page(page_ready_toc))


def process_product(product_no):
    global current_camera

    while running:
        clear_action()
        root.after(0, lambda: setup_capture_page(mode="inner", active="TOA", done=[]))
        root.after(0, lambda: set_step("CAM1 : Detect object by background"))
        root.after(0, lambda: set_status("CAM1 opening"))

        cam1_path = capture_path(f"cam1_inner{product_no}")
        cam2_path = capture_path(f"cam2_inner{product_no}")

        # ===================== CAM1 =====================
        cam1 = PiCameraTest(index=PICAM1_INDEX, name="CAM1", status_cb=set_status, preview_cb=update_preview)
        current_camera = cam1
        try:
            cam1.open()
            bg = cam1.wait_background(current_process_alive)
            if bg is None:
                cam1.close()
                current_camera = None
                if action_value == "restart":
                    continue
                return False

            detected = cam1.wait_object(bg, current_process_alive)
            if not detected:
                cam1.close()
                current_camera = None
                if action_value == "restart":
                    continue
                return False

            cam1.focus_delay(current_process_alive)
            if action_value is not None or not running:
                cam1.close()
                current_camera = None
                if action_value == "restart":
                    continue
                return False

            cam1.capture_file(cam1_path)
        except Exception as e:
            root.after(0, lambda e=e: messagebox.showerror("CAM1 Error", str(e)))
            if action_value == "restart":
                continue
            return False
        finally:
            try:
                cam1.close()
            except Exception:
                pass
            current_camera = None

        # ===================== CAM2 =====================
        if not running:
            return False
        if action_value == "restart":
            continue

        root.after(0, lambda: setup_capture_page(mode="inner", active="TOB", done=["TOA"]))
        root.after(0, lambda: set_step("CAM2 : Capture directly"))
        root.after(0, lambda: set_status("CAM2 opening"))

        cam2 = PiCameraTest(index=PICAM2_INDEX, name="CAM2", status_cb=set_status, preview_cb=update_preview)
        current_camera = cam2
        try:
            cam2.open()
            cam2.focus_delay(current_process_alive)
            if action_value is not None or not running:
                cam2.close()
                current_camera = None
                if action_value == "restart":
                    continue
                return False
            cam2.capture_file(cam2_path)
        except Exception as e:
            root.after(0, lambda e=e: messagebox.showerror("CAM2 Error", str(e)))
            if action_value == "restart":
                continue
            return False
        finally:
            try:
                cam2.close()
            except Exception:
                pass
            current_camera = None

        # ===================== OCR + REVIEW =====================
        if not running:
            return False
        if action_value == "restart":
            continue

        root.after(0, lambda: set_step("OCR : Center ROI CAM1/CAM2"))
        root.after(0, lambda: set_status("OCR reading"))

        cam1_data = process_center_roi_ocr(cam1_path, "cam1")
        cam2_data = process_center_roi_ocr(cam2_path, "cam2")

        clear_action()
        root.after(0, lambda: setup_capture_page(mode="inner", active=None, done=["TOA", "TOB"]))
        root.after(0, lambda: show_product_result(product_no, cam1_data, cam2_data))

        action = wait_review_action(f"Inner Box {product_no}/{quantity}")
        if action == "next":
            return True
        if action == "restart":
            continue
        return False

    return False


def continue_toc():
    global running, current_outer_index

    running = True
    current_outer_index = 0
    clear_action()
    dbg("CONTINUE TOC")

    setup_capture_page(mode="outer", active="TOC", done=["TOA", "TOB"])
    thread = threading.Thread(target=toc_loop, daemon=True)
    thread.start()


def toc_loop():
    global running, current_outer_index

    dbg(f"TOC LOOP START quantity={quantity} outer_total={outer_total}")

    for toc_no in range(1, outer_total + 1):
        if not running:
            return

        current_outer_index = toc_no
        root.after(0, lambda: setup_capture_page(mode="outer", active="TOC", done=["TOA", "TOB"]))

        ok = process_toc(toc_no=toc_no, toc_total=outer_total)
        if not ok:
            if action_value == "cancel":
                root.after(0, reset_to_input)
            running = False
            return

        sleep(0.4)

    running = False
    root.after(0, lambda: show_page(page_complete))


def process_toc(toc_no=1, toc_total=1):
    global current_camera

    while running:
        clear_action()
        root.after(0, lambda: setup_capture_page(mode="outer", active="TOC", done=["TOA", "TOB"]))
        root.after(0, lambda: set_step(f"CAM3 : Outer Box {toc_no}/{toc_total} YOLO detect -> capture -> crop ROI -> OCR"))
        root.after(0, lambda: set_status(f"CAM3 opening Outer Box {toc_no}/{toc_total}"))

        toc_path = capture_path(f"cam3_outer{toc_no}")
        cam3 = UsbCameraTest(device=USB_DEVICE, name="CAM3", status_cb=set_status, preview_cb=update_preview)
        current_camera = cam3

        try:
            cam3.open()
            capture_result = cam3.capture_direct_or_yolo(toc_path, current_process_alive)
            if not capture_result:
                cam3.close()
                current_camera = None
                if action_value == "restart":
                    continue
                return False
        except Exception as e:
            root.after(0, lambda e=e: messagebox.showerror("CAM3 Error", str(e)))
            if action_value == "restart":
                continue
            return False
        finally:
            try:
                cam3.close()
            except Exception:
                pass
            current_camera = None

        if not running:
            return False
        if action_value == "restart":
            continue

        image_path = capture_result["image_path"]
        detections = capture_result.get("detections", [])

        root.after(0, lambda: set_step(f"Outer Box {toc_no}/{toc_total} OCR : YOLO ROI count = {len(detections)}"))
        root.after(0, lambda: set_status(f"Outer Box {toc_no}/{toc_total} OCR reading from YOLO ROI"))

        toc_items = process_yolo_roi_ocr(image_path, detections)
        fallback = None
        if not toc_items:
            root.after(0, lambda: set_status("No YOLO ROI OCR, fallback center ROI"))
            fallback = process_center_roi_ocr(image_path, f"toc_center_{toc_no}")

        clear_action()
        root.after(0, lambda: setup_capture_page(mode="outer", active=None, done=["TOA", "TOB", "TOC"]))
        root.after(0, lambda: show_toc_result(toc_items, fallback, toc_no, toc_total))

        action = wait_review_action(f"Outer Box {toc_no}/{toc_total}")
        if action == "next":
            return True
        if action == "restart":
            continue
        return False

    return False


def wait_review_action(context="-"):
    global debug_wait_counter

    debug_wait_counter += 1
    wait_id = debug_wait_counter
    dbg(f"WAIT[{wait_id}] ENTER context={context}")
    last_log = time.time()

    while running and not action_event.is_set():
        now = time.time()
        if now - last_log >= 1.0:
            dbg(f"WAIT[{wait_id}] still waiting context={context}")
            last_log = now
        sleep(0.1)

    action = action_value or "cancel"
    dbg(f"WAIT[{wait_id}] EXIT context={context} action={action} running={running}")
    return action


def on_result_ok():
    dbg(f"OK/NEXT pressed on {current_review_context}")
    set_action("next")


def restart_process():
    # Keep current Inner Box / Outer Box index. Only redo current capture + OCR.
    dbg(f"Restart The Process pressed on {current_screen_mode}")
    set_action("restart")


def cancel_job():
    dbg("Cancel Job pressed")
    set_action("cancel")
    root.after(0, reset_to_input)


def reset_to_input():
    global running, current_index, current_outer_index, delivery_no, en_no, quantity, outer_total

    running = False
    clear_action()
    current_index = 0
    current_outer_index = 0
    quantity = 0
    outer_total = 0
    delivery_no = ""
    en_no = ""

    try:
        en_entry.delete(0, tk.END)
        delivery_entry.delete(0, tk.END)
        qty_entry.delete(0, tk.END)
    except Exception:
        pass

    show_page(page_input)


def exit_app():
    global running
    running = False
    try:
        if current_camera is not None:
            current_camera.close()
    except Exception:
        pass
    root.destroy()


def on_close():
    exit_app()


# =========================================================
# UI
# =========================================================
root = tk.Tk()
root.title("AI Camera TEST UI - YOLO ROI OCR")
root.geometry("1200x720")
root.resizable(False, False)

# ===================== THEME =====================
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
GREEN = "#229954"
GREEN_BG = "#eafaf1"
YELLOW_TEXT = "#b9770e"
YELLOW_BG = "#fff3cd"
DARK = "#17202a"

root.configure(bg=BG)

# ===================== ICONS =====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ICON_DIR = os.path.join(BASE_DIR, "assets", "icons")
icons = {}


def load_icon_file(filename, size=(22, 22)):
    path = os.path.join(ICON_DIR, filename)
    if not os.path.exists(path):
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
    icons["refresh"] = load_icon_file("refresh.png")


def icon_label(parent, key, bg=CARD):
    icon = icons.get(key)
    if icon is not None:
        return tk.Label(parent, image=icon, bg=bg)
    return tk.Label(parent, text=key.upper()[:4], bg=SOFT, fg=BLUE, font=("Arial", 8, "bold"), width=5, height=2)


def title_with_icon(parent, icon_key, text, bg=CARD):
    frame = tk.Frame(parent, bg=bg)
    frame.pack(fill="x")
    icon_label(frame, icon_key, bg=bg).pack(side="left", padx=(0, 10))
    tk.Label(frame, text=text, bg=bg, fg=TEXT, font=("Arial", 14, "bold")).pack(side="left")
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
        "pady": 5,
    }
    if width is not None:
        kwargs["width"] = width
    icon = icons.get(icon_key) if icon_key else None
    if icon is not None:
        kwargs["image"] = icon
        kwargs["compound"] = "left"
    return tk.Button(parent, **kwargs)


def add_action_button(parent, text, bg, fg="white", command=None, width_px=170, height_px=36, frame_bg=None):
    """Create same-size action buttons so Restart / Cancel / OK are aligned."""
    if frame_bg is None:
        frame_bg = parent.cget("bg")
    wrap = tk.Frame(parent, bg=frame_bg, width=width_px, height=height_px)
    wrap.pack(side="left", padx=(0, 8))
    wrap.pack_propagate(False)

    active_bg = RED_DARK if bg == RED else ("#1e8449" if bg == "#27ae60" else "#aeb4b8")
    btn = tk.Button(
        wrap,
        text=text,
        bg=bg,
        fg=fg,
        activebackground=active_bg,
        activeforeground="white" if fg == "white" else fg,
        font=("Arial", 10, "bold"),
        relief="flat",
        cursor="hand2",
        command=command,
        padx=0,
        pady=0,
    )
    btn.pack(fill="both", expand=True)
    return btn


load_icons()

top = tk.Frame(root, bg=BG, height=1)
top.pack(side="top", fill="x")
top.pack_propagate(False)

# Placeholder labels
info_delivery = tk.Label(root)
info_item = tk.Label(root)
info_en = tk.Label(root)
info_box_title = tk.Label(root)
info_box = tk.Label(root)
info_pack = tk.Label(root)
progress_title_label = tk.Label(root)
progress_label = tk.Label(root)
progress_percent_label = tk.Label(root)
status_label = tk.Label(root)
step_label = tk.Label(root)

# =========================================================
# PAGE 1: INPUT
# =========================================================
page_input = tk.Frame(root, bg=BG)

input_card = tk.Frame(page_input, bg=CARD, padx=52, pady=44, highlightbackground=BORDER, highlightthickness=1)
input_card.place(relx=0.5, rely=0.50, anchor="center", width=650, height=560)

tk.Label(input_card, text="Start Test Job", font=("Arial", 22, "bold"), bg=CARD, fg=TEXT).pack(anchor="w", pady=(0, 18))


def make_input(parent, title):
    tk.Label(parent, text=title, bg=CARD, fg=TEXT, font=("Arial", 12, "bold")).pack(anchor="w", pady=(0, 5))
    ent = tk.Entry(parent, font=("Arial", 18), relief="solid", borderwidth=1)
    ent.pack(fill="x", ipady=5, pady=(0, 18))
    return ent


en_entry = make_input(input_card, "EN")
delivery_entry = make_input(input_card, "Delivary NO")
qty_entry = make_input(input_card, "Quantity")

tk.Button(input_card, text="START TEST", font=("Arial", 14, "bold"), bg=BLUE, fg="white", activebackground=BLUE_DARK, activeforeground="white", relief="flat", command=start_job).pack(fill="x", ipady=7, pady=(4, 12))
tk.Button(input_card, text="EXIT APP", font=("Arial", 14, "bold"), bg=RED, fg="white", activebackground=RED_DARK, activeforeground="white", relief="flat", command=exit_app).pack(fill="x", ipady=7)

# =========================================================
# PAGE 2/4: CAPTURE
# =========================================================
page_capture = tk.Frame(root, bg=BG)
capture_container = tk.Frame(page_capture, bg=BG)
capture_container.pack(fill="both", expand=True, padx=14, pady=14)

# Section 1 header
header_frame = tk.Frame(capture_container, bg=CARD, highlightbackground=BORDER, highlightthickness=1, padx=16, pady=10)
header_frame.pack(fill="x", pady=(0, 10))

header_top = tk.Frame(header_frame, bg=CARD)
header_top.pack(fill="x")

info_left = tk.Frame(header_top, bg=CARD)
info_left.pack(side="left", fill="x", expand=True)

header_buttons = tk.Frame(header_top, bg=CARD)
header_buttons.pack(side="right")

add_action_button(header_buttons, "Restart The Process", bg="#bfc3c7", fg=TEXT, command=restart_process, width_px=170, height_px=36)
add_action_button(header_buttons, "Cancel Job", bg=RED, fg="white", command=cancel_job, width_px=170, height_px=36)
add_action_button(header_buttons, "OK / NEXT", bg="#27ae60", fg="white", command=on_result_ok, width_px=170, height_px=36)

info_row1 = tk.Frame(info_left, bg=CARD)
info_row1.pack(fill="x", pady=(0, 8))
info_row2 = tk.Frame(info_left, bg=CARD)
info_row2.pack(fill="x")


def header_info_inline(parent, title, value_width=18):
    """Header row 1: show title and value on the same line."""
    box = tk.Frame(parent, bg=CARD)
    box.pack(side="left", padx=(0, 60))
    tk.Label(box, text=f"{title} :", bg=CARD, fg=MUTED, font=("Arial", 10, "bold"), anchor="w").pack(side="left")
    val = tk.Label(box, text="-", bg=CARD, fg=TEXT, font=("Arial", 13, "bold"), width=value_width, anchor="w")
    val.pack(side="left", padx=(6, 0))
    return val


def header_info_box(parent, title, value_width=16):
    box = tk.Frame(parent, bg=CARD)
    box.pack(side="left", padx=(0, 55))
    tk.Label(box, text=title, bg=CARD, fg=MUTED, font=("Arial", 9, "bold"), anchor="w").pack(anchor="w")
    val = tk.Label(box, text="-", bg=CARD, fg=TEXT, font=("Arial", 13, "bold"), width=value_width, anchor="w")
    val.pack(anchor="w", pady=(2, 0))
    return val


info_delivery = header_info_inline(info_row1, "Delivary No", 22)
info_item = header_info_inline(info_row1, "Item No", 14)
info_en = header_info_box(info_row2, "EN", 14)

inner_box_container = tk.Frame(info_row2, bg=CARD)
inner_box_container.pack(side="left", padx=(0, 55))
info_box_title = tk.Label(inner_box_container, text="Inner Box", bg=CARD, fg=MUTED, font=("Arial", 9, "bold"), anchor="w")
info_box_title.pack(anchor="w")
info_box = tk.Label(inner_box_container, text="-", bg=CARD, fg=TEXT, font=("Arial", 13, "bold"), width=14, anchor="w")
info_box.pack(anchor="w", pady=(2, 0))

info_pack = header_info_box(info_row2, "Pack Type", 12)

# Body
body_frame = tk.Frame(capture_container, bg=BG)
body_frame.pack(fill="both", expand=True)

# Section 2 capture left
capture_left = tk.Frame(body_frame, bg=CARD, highlightbackground=BORDER, highlightthickness=1, padx=18, pady=14)
capture_left.pack(side="left", fill="both", expand=True, padx=(0, 10))

title_with_icon(capture_left, "camera", "Capture Images", bg=CARD)
tk.Label(capture_left, text="Preview", bg=CARD, fg=MUTED, font=("Arial", 9, "bold")).pack(anchor="w", pady=(8, 4))

preview_frame = tk.Frame(capture_left, bg=DARK, width=760, height=430, highlightbackground=DARK, highlightthickness=2)
preview_frame.pack(pady=(4, 10))
preview_frame.pack_propagate(False)

preview_label = tk.Label(preview_frame, text="Camera Preview", bg=DARK, fg="white", font=("Arial", 16))
preview_label.place(x=0, y=0, relwidth=1, relheight=1)

preview_overlay = tk.Frame(preview_frame, bg=DARK)
preview_overlay.place(x=10, y=8)

capture_state_dot = tk.Canvas(preview_overlay, width=14, height=14, bg=DARK, highlightthickness=0)
capture_state_dot.pack(side="left", padx=(0, 5))
capture_state_dot.create_oval(2, 2, 12, 12, fill="#f1c40f", outline="#f1c40f")

capture_state_label = tk.Label(preview_overlay, text="Waiting Object", bg=DARK, fg="white", font=("Arial", 11, "bold"), padx=2, pady=2)
capture_state_label.pack(side="left")

reset_btn = tk.Button(capture_left, text="Restart The Process", bg="#dfe6e9", fg=TEXT, activebackground="#cfd8dc", activeforeground=TEXT, font=("Arial", 10, "bold"), relief="flat", height=2, cursor="hand2", command=restart_process)
if icons.get("refresh") is not None:
    reset_btn.config(image=icons["refresh"], compound="left")
reset_btn.pack(fill="x", pady=(8, 0))

# Section 3 right status
capture_right = tk.Frame(body_frame, bg=CARD, highlightbackground=BORDER, highlightthickness=1, padx=18, pady=14, width=330)
capture_right.pack(side="right", fill="y")
capture_right.pack_propagate(False)

title_with_icon(capture_right, "status", "Status", bg=CARD)

product_header = tk.Frame(capture_right, bg=CARD)
product_header.pack(fill="x", pady=(18, 2))
icon_label(product_header, "pack", bg=CARD).pack(side="left", padx=(0, 8))
progress_title_label = tk.Label(product_header, text="Inner Box", bg=CARD, fg=TEXT, font=("Arial", 10, "bold"))
progress_title_label.pack(side="left")

product_row = tk.Frame(capture_right, bg=CARD)
product_row.pack(fill="x")
progress_label = tk.Label(product_row, text="0 / 0", bg=CARD, fg=TEXT, font=("Arial", 10))
progress_label.pack(side="left")
progress_percent_label = tk.Label(product_row, text="0%", bg=CARD, fg=TEXT, font=("Arial", 10))
progress_percent_label.pack(side="right")

progress_bar_bg = tk.Frame(capture_right, bg="#c4c4c8", height=18)
progress_bar_bg.pack(fill="x", pady=(4, 20))
progress_bar_bg.pack_propagate(False)
progress_bar_fill = tk.Frame(progress_bar_bg, bg=BLUE)
progress_bar_fill.place(relx=0, rely=0, relwidth=0, relheight=1)

check_header = tk.Frame(capture_right, bg=CARD)
check_header.pack(fill="x", pady=(4, 8))
icon_label(check_header, "check", bg=CARD).pack(side="left", padx=(0, 8))
tk.Label(check_header, text="Checklist", bg=CARD, fg=TEXT, font=("Arial", 10, "bold")).pack(side="left")

checklist_frame = tk.Frame(capture_right, bg=CARD)
checklist_frame.pack(fill="x")


def make_check_item(name):
    item = tk.Label(checklist_frame, text=f"[ ]   {name}", bg=SOFT, fg=TEXT, font=("Arial", 12, "bold"), anchor="w", padx=14, pady=9, relief="flat")
    item.pack(fill="x", pady=5)
    check_labels[name] = item


make_check_item("TOA")
make_check_item("TOB")
make_check_item("TOC")

step_status_box = tk.Frame(capture_right, bg=SOFT, padx=12, pady=10)
step_status_box.pack(fill="x", pady=(16, 10))

step_header = tk.Frame(step_status_box, bg=SOFT)
step_header.pack(fill="x")
icon_label(step_header, "setting", bg=SOFT).pack(side="left", padx=(0, 8))
tk.Label(step_header, text="Step", bg=SOFT, fg="#34495e", font=("Arial", 9, "bold")).pack(side="left")

step_label = tk.Label(step_status_box, text="-", bg=SOFT, fg=TEXT, font=("Arial", 10), wraplength=280, justify="left")
step_label.pack(anchor="w", pady=(2, 8))

status_detail_header = tk.Frame(step_status_box, bg=SOFT)
status_detail_header.pack(fill="x")
icon_label(status_detail_header, "info", bg=SOFT).pack(side="left", padx=(0, 8))
tk.Label(status_detail_header, text="Status", bg=SOFT, fg="#34495e", font=("Arial", 9, "bold")).pack(side="left")

status_label = tk.Label(step_status_box, text="-", bg=SOFT, fg=TEXT, font=("Arial", 10), wraplength=280, justify="left")
status_label.pack(anchor="w", pady=(2, 0))

# =========================================================
# PAGE 3/5: RESULT
# =========================================================
page_result = tk.Frame(root, bg=BG)

result_header = tk.Frame(page_result, bg=BG)
result_header.pack(fill="x", padx=16, pady=(12, 8))

result_left_header = tk.Frame(result_header, bg=BG)
result_left_header.pack(side="left")

result_title = tk.Label(result_left_header, text="OCR Result", font=("Arial", 22, "bold"), bg=BG, fg=TEXT)
result_title.pack(side="left")

result_status_badge = tk.Label(result_left_header, text="WAIT REVIEW", bg="#f1c40f", fg=TEXT, font=("Arial", 12, "bold"), padx=14, pady=6)
result_status_badge.pack(side="left", padx=14)

result_button_frame = tk.Frame(result_header, bg=BG)
result_button_frame.pack(side="right")

cancel_result_btn = add_action_button(result_button_frame, "Cancel Job", bg=RED, fg="white", command=cancel_job, width_px=170, height_px=38, frame_bg=BG)
restart_result_btn = add_action_button(result_button_frame, "Restart The Process", bg="#bfc3c7", fg=TEXT, command=restart_process, width_px=170, height_px=38, frame_bg=BG)
ok_next_btn = add_action_button(result_button_frame, "OK / NEXT", bg="#27ae60", fg="white", command=on_result_ok, width_px=170, height_px=38, frame_bg=BG)

result_hint = tk.Label(page_result, text="Review ROI image and OCR result, then press OK / NEXT to continue.", bg=BG, fg=MUTED, font=("Arial", 12))
result_hint.pack(anchor="w", padx=18, pady=(0, 10))

product_result_area = tk.Frame(page_result, bg=BG)
product_result_area.pack(fill="both", expand=True, padx=10, pady=(0, 12))

make_product_result_section(product_result_area, key="vendor", title="RESULT : Vendor Label", icon_key="delivery", field1="BOX ID", value1="BOX ID :", field2="Date Code", value2="Date Code :")
make_product_result_section(product_result_area, key="toa", title="RESULT : TOA Label", icon_key="barcode", field1="Lot No", value1="Lot No :", field2="Date Code", value2="Date Code :")
make_product_result_section(product_result_area, key="tob", title="RESULT : TOB Label", icon_key="pack", field1="Lot No", value1="Lot No :", field2="Date Code", value2="Date Code :")

toc_result_area = tk.Frame(page_result, bg=BG)

# =========================================================
# READY TOC
# =========================================================
page_ready_toc = tk.Frame(root, bg=BG)

toc_card = tk.Frame(page_ready_toc, bg=CARD, padx=35, pady=35, highlightbackground=BORDER, highlightthickness=1)
toc_card.place(relx=0.5, rely=0.45, anchor="center", width=560, height=300)

tk.Label(toc_card, text="Inner Box Test Completed", font=("Arial", 21, "bold"), bg=CARD, fg=GREEN).pack(pady=15)
tk.Label(toc_card, text="Press continue to test CAM3 Outer Box / TOC YOLO ROI OCR", font=("Arial", 13), bg=CARD, fg="#34495e", wraplength=460).pack(pady=8)
tk.Button(toc_card, text="CONTINUE TO OUTER BOX", bg=BLUE, fg="white", font=("Arial", 14, "bold"), width=24, relief="flat", command=continue_toc).pack(pady=20)
tk.Button(toc_card, text="CANCEL JOB", bg=RED, fg="white", font=("Arial", 12, "bold"), width=24, relief="flat", command=cancel_job).pack(pady=(0, 8))

# =========================================================
# COMPLETE
# =========================================================
page_complete = tk.Frame(root, bg=BG)

complete_card = tk.Frame(page_complete, bg=CARD, padx=35, pady=35, highlightbackground=BORDER, highlightthickness=1)
complete_card.place(relx=0.5, rely=0.45, anchor="center", width=520, height=260)

tk.Label(complete_card, text="TEST COMPLETED", font=("Arial", 24, "bold"), bg=CARD, fg=GREEN).pack(pady=25)
tk.Button(complete_card, text="OK", bg=BLUE, fg="white", font=("Arial", 14, "bold"), width=16, relief="flat", command=reset_to_input).pack(pady=10)

# Debug page names
page_input._debug_name = "page_input"
page_capture._debug_name = "page_capture"
page_result._debug_name = "page_result"
page_ready_toc._debug_name = "page_ready_toc"
page_complete._debug_name = "page_complete"

root.protocol("WM_DELETE_WINDOW", on_close)
show_page(page_input)
root.mainloop()
