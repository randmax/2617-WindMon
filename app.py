from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from PIL import Image, ImageTk
from tkinter import filedialog, messagebox, ttk

matplotlib.use("TkAgg")


APP_TITLE = "WindMon Tekercselés Kamera Monitor"
OUTPUT_DIR = Path("captures")
WAVELENGTH_START = 380
WAVELENGTH_END = 780
SPECTRUM_SAMPLES = 90
MARKER_COLORS = [
    "#ff3b30",
    "#007aff",
    "#34c759",
    "#ff9500",
    "#af52de",
    "#00c7be",
    "#ff2d55",
    "#ffd60a",
    "#64d2ff",
    "#30d158",
]


@dataclass
class SamplePoint:
    # Egy képen kijelölt mérési pont és az abból becsült spektrum.
    x: int
    y: int
    rgb: tuple[int, int, int]
    marker_color: str
    spectrum: np.ndarray


class CameraRecorderApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("960x640")
        self.root.minsize(860, 560)

        OUTPUT_DIR.mkdir(exist_ok=True)

        self.preview_size = (860, 500)
        self.capture_size = (960, 540)
        self.preview_after_id: str | None = None

        self.camera_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="Kamera keresése...")

        # Kamera- és képadatok: az élőkép BGR-ben érkezik az OpenCV-ből,
        # a feldolgozás és megjelenítés viszont RGB képpel dolgozik.
        self.available_cameras: list[tuple[int, str]] = []
        self.current_camera_index: int | None = None
        self.capture: cv2.VideoCapture | None = None
        self.latest_frame_bgr: np.ndarray | None = None
        self.captured_image_rgb: np.ndarray | None = None
        self.filtered_image_rgb: np.ndarray | None = None
        self.sample_points: list[SamplePoint] = []

        # Feldolgozó ablak és spektrumgrafikon állapota.
        self.processor_window: tk.Toplevel | None = None
        self.processor_canvas: tk.Canvas | None = None
        self.processor_figure: Figure | None = None
        self.processor_axis = None
        self.processor_plot_canvas: FigureCanvasTkAgg | None = None
        self.processor_marker_annotation = None
        self.processor_marker_artist = None
        self.processor_marker_line = None
        self.processor_plot_connection_id: int | None = None
        self.processor_lines: list = []
        self.filtered_window: tk.Toplevel | None = None
        self.filtered_canvas: tk.Canvas | None = None
        self.loaded_filter_min_values: np.ndarray | None = None
        self.loaded_filter_max_values: np.ndarray | None = None
        self.loaded_filter_name: str | None = None

        # A képek nézeti állapota: nagyítás és eltolás külön az eredeti
        # és a szűrt képhez, hogy a két ablak egymástól független maradjon.
        self.processor_zoom = 1.0
        self.processor_pan_x = 0.0
        self.processor_pan_y = 0.0
        self.filtered_zoom = 1.0
        self.filtered_pan_x = 0.0
        self.filtered_pan_y = 0.0
        self.image_pan_start: tuple[str, int, int, float, float] | None = None

        # Teljes képernyős spektrumablak állapota.
        self.fullscreen_window: tk.Toplevel | None = None
        self.fullscreen_figure: Figure | None = None
        self.fullscreen_axis = None
        self.fullscreen_plot_canvas: FigureCanvasTkAgg | None = None
        self.fullscreen_marker_annotation = None
        self.fullscreen_marker_artist = None
        self.fullscreen_marker_line = None
        self.fullscreen_plot_connection_id: int | None = None
        self.fullscreen_lines: list = []

        self.capture_photo: ImageTk.PhotoImage | None = None
        self.preview_photo: ImageTk.PhotoImage | None = None
        self.filtered_photo: ImageTk.PhotoImage | None = None

        self.menu_camera = tk.Menu(self.root, tearoff=False)

        self._build_layout()
        self._build_menubar()

        self.refresh_cameras()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.after(100, self.update_preview_loop)

    def _build_layout(self) -> None:
        root_frame = ttk.Frame(self.root, padding=12)
        root_frame.pack(fill=tk.BOTH, expand=True)
        root_frame.columnconfigure(0, weight=1)
        root_frame.rowconfigure(1, weight=1)

        control_frame = ttk.Frame(root_frame)
        control_frame.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        control_frame.columnconfigure(4, weight=1)

        ttk.Label(control_frame, text="Aktív kamera:").grid(row=0, column=0, padx=(0, 8))
        self.camera_combo = ttk.Combobox(
            control_frame,
            textvariable=self.camera_var,
            state="readonly",
            width=35,
        )
        self.camera_combo.grid(row=0, column=1, padx=(0, 8))
        self.camera_combo.bind("<<ComboboxSelected>>", self.on_camera_selected)

        ttk.Button(control_frame, text="Kamerák frissítése", command=self.refresh_cameras).grid(
            row=0, column=2, padx=(0, 8)
        )
        ttk.Button(control_frame, text="Kép rögzítése", command=self.capture_image).grid(
            row=0, column=3, padx=(0, 8)
        )
        ttk.Button(control_frame, text="Feldolgozó ablak megnyitása", command=self.open_processor_window).grid(
            row=0, column=4, sticky="e"
        )

        ttk.Label(control_frame, textvariable=self.status_var).grid(
            row=1,
            column=0,
            columnspan=5,
            sticky="w",
            pady=(10, 0),
        )

        preview_group = ttk.LabelFrame(root_frame, text="Előkép", padding=10)
        preview_group.grid(row=1, column=0, sticky="nsew")
        preview_group.columnconfigure(0, weight=1)
        preview_group.rowconfigure(1, weight=1)

        ttk.Label(
            preview_group,
            text="Előkép a kiválasztott kameráról. A rögzített kép külön feldolgozó ablakban jelenik meg.",
        ).grid(row=0, column=0, sticky="w", pady=(0, 8))

        self.preview_label = ttk.Label(preview_group)
        self.preview_label.grid(row=1, column=0, sticky="nsew")
        self.preview_label.bind("<Configure>", lambda _event: self.refresh_preview_image())

    def _build_menubar(self) -> None:
        menubar = tk.Menu(self.root)
        file_menu = tk.Menu(menubar, tearoff=False)
        file_menu.add_command(label="RGB kép mentése...", command=self.save_captured_image)
        file_menu.add_command(label="Feldolgozó ablak megnyitása", command=self.open_processor_window)
        file_menu.add_separator()
        file_menu.add_command(label="Kilépés", command=self.on_close)

        camera_menu = tk.Menu(menubar, tearoff=False)
        camera_menu.add_command(label="Kamerák újrakeresése", command=self.refresh_cameras)
        camera_menu.add_cascade(label="Kamera választása", menu=self.menu_camera)

        menubar.add_cascade(label="Fájl", menu=file_menu)
        menubar.add_cascade(label="Kamera", menu=camera_menu)
        self.root.config(menu=menubar)

    def _build_processor_window(self) -> None:
        if self.processor_window is not None and self.processor_window.winfo_exists():
            self.processor_window.lift()
            return

        # A feldolgozó ablak bal oldalon a képet, jobb oldalon a kijelölt
        # pontokból számolt spektrumgörbéket mutatja.
        self.processor_window = tk.Toplevel(self.root)
        self.processor_window.title("Feldolgozó ablak")
        self.processor_window.geometry("1460x860")
        self.processor_window.minsize(1180, 720)
        self.processor_window.protocol("WM_DELETE_WINDOW", self.on_processor_window_close)

        container = ttk.Frame(self.processor_window, padding=12)
        container.pack(fill=tk.BOTH, expand=True)
        container.columnconfigure(0, weight=3)
        container.columnconfigure(1, weight=2)
        container.rowconfigure(1, weight=1)

        toolbar = ttk.Frame(container)
        toolbar.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 12))
        toolbar.columnconfigure(7, weight=1)
        ttk.Button(toolbar, text="Pontok törlése", command=self.clear_points).grid(row=0, column=0, padx=(0, 8))
        ttk.Button(toolbar, text="RGB kép mentése", command=self.save_captured_image).grid(row=0, column=1, padx=(0, 8))
        ttk.Button(toolbar, text="Spektrum teljes képernyőn", command=self.open_fullscreen_spectrum_window).grid(
            row=0, column=2, padx=(0, 8)
        )
        ttk.Button(toolbar, text="Szűrő alkalmazása", command=self.apply_marker_range_filter).grid(
            row=0, column=3, padx=(0, 8)
        )
        ttk.Button(toolbar, text="Szűrő mentése", command=self.save_filter_band).grid(row=0, column=4, padx=(0, 8))
        ttk.Button(toolbar, text="Szűrő betöltése", command=self.load_filter_band).grid(row=0, column=5, padx=(0, 8))
        ttk.Button(toolbar, text="Betöltött szűrő törlése", command=self.clear_loaded_filter_band).grid(
            row=0, column=6, padx=(0, 8)
        )
        ttk.Label(
            toolbar,
            text="Bal kattintás: pont hozzáadása, jobb kattintás: pont törlése.",
        ).grid(row=0, column=7, sticky="e")

        image_group = ttk.LabelFrame(container, text="Rögzített RGB kép", padding=10)
        image_group.grid(row=1, column=0, sticky="nsew", padx=(0, 10))
        image_group.columnconfigure(0, weight=1)
        image_group.rowconfigure(0, weight=1)

        self.processor_canvas = tk.Canvas(
            image_group,
            width=self.capture_size[0],
            height=self.capture_size[1],
            bg="#1f1f1f",
            highlightthickness=1,
            highlightbackground="#707070",
            cursor="crosshair",
        )
        self.processor_canvas.grid(row=0, column=0, sticky="nsew")
        self.processor_canvas.bind("<Button-1>", self.on_capture_canvas_click)

        # Egérgörgővel nagyítás, középső gombbal pásztázás.
        # A kattintott pontok koordinátája a nagyított nézetből is
        # visszaszámolódik az eredeti képpixelre.
        self.processor_canvas.bind("<MouseWheel>", lambda event: self.on_image_mouse_wheel(event, "processor"))
        self.processor_canvas.bind("<Button-4>", lambda event: self.on_image_mouse_wheel(event, "processor", 1))
        self.processor_canvas.bind("<Button-5>", lambda event: self.on_image_mouse_wheel(event, "processor", -1))
        self.processor_canvas.bind("<ButtonPress-2>", lambda event: self.start_image_pan(event, "processor"))
        self.processor_canvas.bind("<B2-Motion>", self.on_image_pan)
        self.processor_canvas.bind("<Button-3>", self.remove_capture_point)
        self.processor_canvas.bind("<Configure>", lambda _event: self.redraw_capture_canvas())

        plot_group = ttk.LabelFrame(container, text="RGB pontok becsült spektruma", padding=10)
        plot_group.grid(row=1, column=1, sticky="nsew")
        plot_group.columnconfigure(0, weight=1)
        plot_group.rowconfigure(0, weight=1)

        plot_host = ttk.Frame(plot_group)
        plot_host.grid(row=0, column=0, sticky="nsew")

        self.processor_figure = Figure(figsize=(5.5, 4.8), dpi=100)
        self.processor_axis = self.processor_figure.add_subplot(111)
        self.processor_plot_canvas = FigureCanvasTkAgg(self.processor_figure, master=plot_host)
        self.processor_plot_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.connect_plot_events("processor")

        self.redraw_capture_canvas()
        self.update_spectrum_plot()

    def open_fullscreen_spectrum_window(self) -> None:
        if self.captured_image_rgb is None:
            messagebox.showinfo("Nincs rögzített kép", "Előbb készíts egy képet a kamerából.")
            return

        if self.fullscreen_window is not None and self.fullscreen_window.winfo_exists():
            self.fullscreen_window.lift()
            self.refresh_fullscreen_plot()
            return

        self.fullscreen_window = tk.Toplevel(self.root)
        self.fullscreen_window.title("Teljes képernyős spektrum")
        self.fullscreen_window.attributes("-fullscreen", True)
        self.fullscreen_window.protocol("WM_DELETE_WINDOW", self.close_fullscreen_spectrum_window)

        container = ttk.Frame(self.fullscreen_window, padding=10)
        container.pack(fill=tk.BOTH, expand=True)
        container.columnconfigure(0, weight=1)
        container.rowconfigure(1, weight=1)

        toolbar = ttk.Frame(container)
        toolbar.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        toolbar.columnconfigure(2, weight=1)
        ttk.Button(toolbar, text="Kilépés a teljes képernyőből", command=self.close_fullscreen_spectrum_window).grid(
            row=0, column=0, padx=(0, 8)
        )
        ttk.Button(toolbar, text="Pontok törlése", command=self.clear_points).grid(row=0, column=1, padx=(0, 8))
        ttk.Label(toolbar, text="Kattints a görbékre a hullámhossz és intenzitás leolvasásához.").grid(
            row=0, column=2, sticky="e"
        )

        plot_host = ttk.Frame(container)
        plot_host.grid(row=1, column=0, sticky="nsew")

        self.fullscreen_figure = Figure(figsize=(12, 7), dpi=100)
        self.fullscreen_axis = self.fullscreen_figure.add_subplot(111)
        self.fullscreen_plot_canvas = FigureCanvasTkAgg(self.fullscreen_figure, master=plot_host)
        self.fullscreen_plot_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.connect_plot_events("fullscreen")
        self.refresh_fullscreen_plot()

    def refresh_fullscreen_plot(self) -> None:
        if self.fullscreen_window is None or not self.fullscreen_window.winfo_exists():
            return

        self.fullscreen_window.update_idletasks()
        self.update_spectrum_plot()
        if self.fullscreen_plot_canvas is not None:
            self.fullscreen_plot_canvas.draw()

    def close_fullscreen_spectrum_window(self) -> None:
        if self.fullscreen_window is not None and self.fullscreen_window.winfo_exists():
            self.fullscreen_window.destroy()

        self.fullscreen_window = None
        self.fullscreen_figure = None
        self.fullscreen_axis = None
        self.fullscreen_plot_canvas = None
        self.fullscreen_marker_annotation = None
        self.fullscreen_marker_artist = None
        self.fullscreen_marker_line = None
        self.fullscreen_plot_connection_id = None
        self.fullscreen_lines = []

    def refresh_cameras(self) -> None:
        self.status_var.set("Kamerák keresése folyamatban...")
        self.available_cameras = self.discover_cameras()
        labels = [label for _, label in self.available_cameras]

        self.camera_combo["values"] = labels
        self.menu_camera.delete(0, tk.END)

        for index, label in self.available_cameras:
            self.menu_camera.add_command(
                label=label,
                command=lambda idx=index, lbl=label: self.select_camera(idx, lbl),
            )

        if labels:
            if self.current_camera_index is None:
                first_index, first_label = self.available_cameras[0]
                self.select_camera(first_index, first_label)
            else:
                matching = [item for item in self.available_cameras if item[0] == self.current_camera_index]
                if matching:
                    index, label = matching[0]
                    self.camera_var.set(label)
                    self.status_var.set(f"Kamera elérhető: {label}")
                else:
                    first_index, first_label = self.available_cameras[0]
                    self.select_camera(first_index, first_label)
        else:
            self.camera_var.set("")
            self.status_var.set("Nem található elérhető kamera.")
            self.release_camera()

    def discover_cameras(self, max_devices: int = 8) -> list[tuple[int, str]]:
        cameras: list[tuple[int, str]] = []
        for index in range(max_devices):
            # Windows alatt a DirectShow backend általában gyorsabban és
            # stabilabban nyitja meg az USB kamerákat.
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            if cap is None or not cap.isOpened():
                continue

            ok, _ = cap.read()
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            cap.release()

            if not ok:
                continue

            role = "Laptop webkamera" if index == 0 else f"USB / külső kamera {index}"
            cameras.append((index, f"{role} [ID {index}] - {width}x{height}"))

        return cameras

    def on_camera_selected(self, _event: tk.Event) -> None:
        selected_label = self.camera_var.get()
        matching = [item for item in self.available_cameras if item[1] == selected_label]
        if matching:
            index, label = matching[0]
            self.select_camera(index, label)

    def select_camera(self, index: int, label: str) -> None:
        self.release_camera()
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            self.status_var.set(f"A kamera nem nyitható meg: {label}")
            return

        self.configure_camera_resolution(cap)
        self.capture = cap
        self.current_camera_index = index
        self.camera_var.set(label)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        self.status_var.set(f"Aktív kamera: {label}, felbontás: {width}x{height}")

    def configure_camera_resolution(self, cap: cv2.VideoCapture) -> None:
        # Először nagyobb felbontásokat kérünk, 16:9-es és 4:3-as módokkal is.
        # Ha a kamera nem támogatja őket, a driver a legközelebbi elérhető
        # módra áll vissza.
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        for width, height in (
            (1920, 1080),
            (1600, 1200),
            (1280, 1024),
            (1280, 960),
            (1280, 720),
            (1024, 768),
            (800, 600),
            (640, 480),
        ):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            if actual_w >= width * 0.9 and actual_h >= height * 0.9:
                break

    def release_camera(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def update_preview_loop(self) -> None:
        if self.capture is not None and self.capture.isOpened():
            ok, frame_bgr = self.capture.read()
            if ok:
                self.latest_frame_bgr = frame_bgr
                self.refresh_preview_image()
            else:
                self.status_var.set("Nem sikerült képkockát olvasni a kamerából.")

        self.preview_after_id = self.root.after(30, self.update_preview_loop)

    def refresh_preview_image(self) -> None:
        if self.latest_frame_bgr is None:
            return

        preview_rgb = cv2.cvtColor(self.latest_frame_bgr, cv2.COLOR_BGR2RGB)
        preview_size = self.get_preview_display_size()
        preview_image = self.resize_for_display(preview_rgb, preview_size)
        self.preview_photo = ImageTk.PhotoImage(Image.fromarray(preview_image))
        self.preview_label.configure(image=self.preview_photo)

    def get_preview_display_size(self) -> tuple[int, int]:
        width = self.preview_label.winfo_width()
        height = self.preview_label.winfo_height()
        if width <= 1 or height <= 1:
            return self.preview_size
        return width, height

    def resize_for_display(self, image_rgb: np.ndarray, size: tuple[int, int]) -> np.ndarray:
        target_w, target_h = size
        src_h, src_w = image_rgb.shape[:2]
        scale = min(target_w / src_w, target_h / src_h)
        resized = cv2.resize(
            image_rgb,
            (max(1, int(src_w * scale)), max(1, int(src_h * scale))),
            interpolation=cv2.INTER_AREA,
        )

        # A képet aránytartóan illesztjük egy sötét háttérre, így nem torzul.
        canvas = np.full((target_h, target_w, 3), 20, dtype=np.uint8)
        offset_x = (target_w - resized.shape[1]) // 2
        offset_y = (target_h - resized.shape[0]) // 2
        canvas[offset_y : offset_y + resized.shape[0], offset_x : offset_x + resized.shape[1]] = resized
        return canvas

    def calculate_display_geometry(self, image_rgb: np.ndarray, size: tuple[int, int]) -> tuple[int, int, int, int]:
        target_w, target_h = size
        src_h, src_w = image_rgb.shape[:2]
        scale = min(target_w / src_w, target_h / src_h)
        disp_w = max(1, int(src_w * scale))
        disp_h = max(1, int(src_h * scale))
        offset_x = (target_w - disp_w) // 2
        offset_y = (target_h - disp_h) // 2
        return disp_w, disp_h, offset_x, offset_y

    def get_canvas_size(self, canvas: tk.Canvas) -> tuple[int, int]:
        width = canvas.winfo_width()
        height = canvas.winfo_height()
        if width <= 1 or height <= 1:
            return self.capture_size
        return width, height

    def get_image_view_state(self, target: str) -> tuple[float, float, float]:
        if target == "processor":
            return self.processor_zoom, self.processor_pan_x, self.processor_pan_y
        return self.filtered_zoom, self.filtered_pan_x, self.filtered_pan_y

    def set_image_view_state(self, target: str, zoom: float, pan_x: float, pan_y: float) -> None:
        if target == "processor":
            self.processor_zoom = zoom
            self.processor_pan_x = pan_x
            self.processor_pan_y = pan_y
        else:
            self.filtered_zoom = zoom
            self.filtered_pan_x = pan_x
            self.filtered_pan_y = pan_y

    def reset_image_view(self, target: str) -> None:
        self.set_image_view_state(target, 1.0, 0.0, 0.0)

    def clamp_image_pan(
        self,
        image_rgb: np.ndarray,
        canvas: tk.Canvas,
        zoom: float,
        pan_x: float,
        pan_y: float,
    ) -> tuple[float, float]:
        canvas_w, canvas_h = self.get_canvas_size(canvas)
        src_h, src_w = image_rgb.shape[:2]
        base_scale = min(canvas_w / src_w, canvas_h / src_h)
        disp_w = src_w * base_scale * zoom
        disp_h = src_h * base_scale * zoom
        base_offset_x = (canvas_w - disp_w) / 2.0
        base_offset_y = (canvas_h - disp_h) / 2.0

        # Nem engedjük teljesen kihúzni a képet a vászon alól.
        if disp_w <= canvas_w:
            pan_x = 0.0
        else:
            pan_x = min(-base_offset_x, max(canvas_w - disp_w - base_offset_x, pan_x))

        if disp_h <= canvas_h:
            pan_y = 0.0
        else:
            pan_y = min(-base_offset_y, max(canvas_h - disp_h - base_offset_y, pan_y))

        return pan_x, pan_y

    def get_zoomed_image_geometry(
        self,
        image_rgb: np.ndarray,
        canvas: tk.Canvas,
        target: str,
    ) -> tuple[float, float, float, float]:
        zoom, pan_x, pan_y = self.get_image_view_state(target)
        pan_x, pan_y = self.clamp_image_pan(image_rgb, canvas, zoom, pan_x, pan_y)
        self.set_image_view_state(target, zoom, pan_x, pan_y)

        canvas_w, canvas_h = self.get_canvas_size(canvas)
        src_h, src_w = image_rgb.shape[:2]
        base_scale = min(canvas_w / src_w, canvas_h / src_h)
        scale = base_scale * zoom
        disp_w = src_w * scale
        disp_h = src_h * scale
        offset_x = (canvas_w - disp_w) / 2.0 + pan_x
        offset_y = (canvas_h - disp_h) / 2.0 + pan_y
        return disp_w, disp_h, offset_x, offset_y

    def draw_zoomed_image(
        self,
        image_rgb: np.ndarray,
        canvas: tk.Canvas,
        target: str,
    ) -> ImageTk.PhotoImage | None:
        disp_w, disp_h, offset_x, offset_y = self.get_zoomed_image_geometry(image_rgb, canvas, target)
        canvas_w, canvas_h = self.get_canvas_size(canvas)
        src_h, src_w = image_rgb.shape[:2]
        scale_x = disp_w / src_w
        scale_y = disp_h / src_h

        # Nagyításkor csak a látható képrészletet méretezzük át, így nagy
        # zoomnál sem kell óriási bitmapet létrehozni.
        visible_left = max(0.0, offset_x)
        visible_top = max(0.0, offset_y)
        visible_right = min(float(canvas_w), offset_x + disp_w)
        visible_bottom = min(float(canvas_h), offset_y + disp_h)
        if visible_left >= visible_right or visible_top >= visible_bottom:
            return None

        src_left = max(0, int(np.floor((visible_left - offset_x) / scale_x)))
        src_top = max(0, int(np.floor((visible_top - offset_y) / scale_y)))
        src_right = min(src_w, int(np.ceil((visible_right - offset_x) / scale_x)))
        src_bottom = min(src_h, int(np.ceil((visible_bottom - offset_y) / scale_y)))
        if src_left >= src_right or src_top >= src_bottom:
            return None

        crop = image_rgb[src_top:src_bottom, src_left:src_right]
        draw_w = max(1, int(round((src_right - src_left) * scale_x)))
        draw_h = max(1, int(round((src_bottom - src_top) * scale_y)))
        interpolation = cv2.INTER_LINEAR if scale_x > 1.0 or scale_y > 1.0 else cv2.INTER_AREA
        display_image = cv2.resize(crop, (draw_w, draw_h), interpolation=interpolation)
        photo = ImageTk.PhotoImage(Image.fromarray(display_image))

        draw_x = int(round(offset_x + src_left * scale_x))
        draw_y = int(round(offset_y + src_top * scale_y))
        canvas.create_image(draw_x, draw_y, anchor=tk.NW, image=photo)
        return photo

    def on_image_mouse_wheel(self, event: tk.Event, target: str, direction: int | None = None) -> str:
        image_rgb = self.captured_image_rgb if target == "processor" else self.filtered_image_rgb
        canvas = self.processor_canvas if target == "processor" else self.filtered_canvas
        if image_rgb is None or canvas is None or not canvas.winfo_exists():
            return "break"

        old_zoom, _, _ = self.get_image_view_state(target)
        if direction is None:
            direction = 1 if getattr(event, "delta", 0) > 0 else -1

        new_zoom = old_zoom * (1.18 if direction > 0 else 1 / 1.18)
        new_zoom = min(12.0, max(1.0, new_zoom))
        if abs(new_zoom - old_zoom) < 0.001:
            return "break"

        # A nagyítás középpontja az egérmutató alatti képpont marad.
        _, _, old_offset_x, old_offset_y = self.get_zoomed_image_geometry(image_rgb, canvas, target)
        canvas_w, canvas_h = self.get_canvas_size(canvas)
        src_h, src_w = image_rgb.shape[:2]
        old_scale = min(canvas_w / src_w, canvas_h / src_h) * old_zoom
        image_x = (event.x - old_offset_x) / old_scale
        image_y = (event.y - old_offset_y) / old_scale

        new_scale = min(canvas_w / src_w, canvas_h / src_h) * new_zoom
        new_disp_w = src_w * new_scale
        new_disp_h = src_h * new_scale
        base_offset_x = (canvas_w - new_disp_w) / 2.0
        base_offset_y = (canvas_h - new_disp_h) / 2.0
        new_pan_x = event.x - image_x * new_scale - base_offset_x
        new_pan_y = event.y - image_y * new_scale - base_offset_y
        new_pan_x, new_pan_y = self.clamp_image_pan(image_rgb, canvas, new_zoom, new_pan_x, new_pan_y)
        self.set_image_view_state(target, new_zoom, new_pan_x, new_pan_y)

        if target == "processor":
            self.redraw_capture_canvas()
        else:
            self.redraw_filtered_canvas()

        return "break"

    def start_image_pan(self, event: tk.Event, target: str) -> str:
        zoom, pan_x, pan_y = self.get_image_view_state(target)
        if zoom <= 1.0:
            return "break"

        self.image_pan_start = (target, event.x, event.y, pan_x, pan_y)
        return "break"

    def on_image_pan(self, event: tk.Event) -> str:
        if self.image_pan_start is None:
            return "break"

        target, start_x, start_y, start_pan_x, start_pan_y = self.image_pan_start
        image_rgb = self.captured_image_rgb if target == "processor" else self.filtered_image_rgb
        canvas = self.processor_canvas if target == "processor" else self.filtered_canvas
        if image_rgb is None or canvas is None or not canvas.winfo_exists():
            return "break"

        zoom, _, _ = self.get_image_view_state(target)
        pan_x = start_pan_x + event.x - start_x
        pan_y = start_pan_y + event.y - start_y
        pan_x, pan_y = self.clamp_image_pan(image_rgb, canvas, zoom, pan_x, pan_y)
        self.set_image_view_state(target, zoom, pan_x, pan_y)

        if target == "processor":
            self.redraw_capture_canvas()
        else:
            self.redraw_filtered_canvas()

        return "break"

    def capture_image(self) -> None:
        if self.latest_frame_bgr is None:
            messagebox.showwarning("Nincs kép", "Először nyiss meg egy kamerát, és várj előképre.")
            return

        self.captured_image_rgb = cv2.cvtColor(self.latest_frame_bgr.copy(), cv2.COLOR_BGR2RGB)
        self.sample_points.clear()
        self.reset_image_view("processor")
        self.open_processor_window()
        self.redraw_capture_canvas()
        self.update_spectrum_plot()
        self.status_var.set("RGB kép rögzítve. A feldolgozó ablakban helyezhetsz el pontokat.")

    def open_processor_window(self) -> None:
        if self.captured_image_rgb is None:
            messagebox.showinfo("Nincs rögzített kép", "Előbb készíts egy képet a kamerából.")
            return

        self._build_processor_window()

    def redraw_capture_canvas(self) -> None:
        if self.processor_canvas is None or not self.processor_canvas.winfo_exists():
            return

        self.processor_canvas.delete("all")

        if self.captured_image_rgb is None:
            self.processor_canvas.create_text(
                self.capture_size[0] // 2,
                self.capture_size[1] // 2,
                text="Itt jelenik meg a rögzített RGB kép",
                fill="#d0d0d0",
                font=("Segoe UI", 16, "bold"),
            )
            return

        disp_w, disp_h, offset_x, offset_y = self.get_zoomed_image_geometry(
            self.captured_image_rgb,
            self.processor_canvas,
            "processor",
        )
        self.capture_photo = self.draw_zoomed_image(self.captured_image_rgb, self.processor_canvas, "processor")

        scale_x = disp_w / self.captured_image_rgb.shape[1]
        scale_y = disp_h / self.captured_image_rgb.shape[0]

        # A markerjelölések mindig az eredeti képpont-koordinátából
        # rajzolódnak újra, ezért zoom és pásztázás közben is a helyükön maradnak.
        for idx, point in enumerate(self.sample_points, start=1):
            canvas_x = int(point.x * scale_x) + offset_x
            canvas_y = int(point.y * scale_y) + offset_y
            self.processor_canvas.create_oval(
                canvas_x - 6,
                canvas_y - 6,
                canvas_x + 6,
                canvas_y + 6,
                fill=self.rgb_to_hex(point.rgb),
                outline="#ffffff",
                width=2,
            )
            self.processor_canvas.create_rectangle(
                canvas_x + 8,
                canvas_y - 16,
                canvas_x + 30,
                canvas_y + 4,
                fill="#101010",
                outline="#ffffff",
                width=1,
            )
            self.processor_canvas.create_text(
                canvas_x + 19,
                canvas_y - 6,
                text=str(idx),
                fill=point.marker_color,
                font=("Segoe UI", 10, "bold"),
                anchor=tk.CENTER,
            )

    def on_capture_canvas_click(self, event: tk.Event) -> None:
        if self.captured_image_rgb is None:
            return

        # A vászon-koordinátát visszavetítjük az eredeti kép koordinátarendszerébe.
        disp_w, disp_h, offset_x, offset_y = self.get_zoomed_image_geometry(
            self.captured_image_rgb,
            self.processor_canvas,
            "processor",
        )

        if not (offset_x <= event.x < offset_x + disp_w and offset_y <= event.y < offset_y + disp_h):
            return

        src_h, src_w = self.captured_image_rgb.shape[:2]
        rel_x = (event.x - offset_x) / disp_w
        rel_y = (event.y - offset_y) / disp_h
        img_x = min(src_w - 1, max(0, int(rel_x * src_w)))
        img_y = min(src_h - 1, max(0, int(rel_y * src_h)))

        rgb = tuple(int(v) for v in self.captured_image_rgb[img_y, img_x])
        point = SamplePoint(
            x=img_x,
            y=img_y,
            rgb=rgb,
            marker_color=self.get_marker_color(len(self.sample_points)),
            spectrum=self.estimate_spectrum(rgb),
        )
        self.sample_points.append(point)
        self.redraw_capture_canvas()
        self.update_spectrum_plot()

    def remove_capture_point(self, event: tk.Event) -> str:
        if self.captured_image_rgb is None or not self.sample_points:
            return "break"

        hit = self.find_nearest_sample_point_on_canvas(event.x, event.y)
        if hit is None:
            return "break"

        self.sample_points.pop(hit)
        self.redraw_capture_canvas()
        self.update_spectrum_plot()
        self.status_var.set("Marker pont törölve.")
        return "break"

    def find_nearest_sample_point_on_canvas(self, event_x: int, event_y: int) -> int | None:
        if self.captured_image_rgb is None or self.processor_canvas is None:
            return None

        disp_w, disp_h, offset_x, offset_y = self.get_zoomed_image_geometry(
            self.captured_image_rgb,
            self.processor_canvas,
            "processor",
        )
        scale_x = disp_w / self.captured_image_rgb.shape[1]
        scale_y = disp_h / self.captured_image_rgb.shape[0]
        hit_radius = 12.0
        nearest_index = None
        nearest_distance = None

        for index, point in enumerate(self.sample_points):
            canvas_x = point.x * scale_x + offset_x
            canvas_y = point.y * scale_y + offset_y
            distance = float(np.hypot(canvas_x - event_x, canvas_y - event_y))
            if distance <= hit_radius and (nearest_distance is None or distance < nearest_distance):
                nearest_index = index
                nearest_distance = distance

        return nearest_index

    def estimate_spectrum(self, rgb: tuple[int, int, int]) -> np.ndarray:
        wavelengths = np.linspace(WAVELENGTH_START, WAVELENGTH_END, SPECTRUM_SAMPLES)
        r, g, b = [channel / 255.0 for channel in rgb]
        luminance = np.clip(0.2126 * r + 0.7152 * g + 0.0722 * b, 0.05, 1.0)

        # Egyszerű RGB-alapú becslés: a három színcsatornát széles,
        # Gauss-jellegű spektrumgörbékkel közelítjük.
        red_curve = np.exp(-0.5 * ((wavelengths - 620.0) / 36.0) ** 2)
        green_curve = np.exp(-0.5 * ((wavelengths - 540.0) / 30.0) ** 2)
        blue_curve = np.exp(-0.5 * ((wavelengths - 460.0) / 24.0) ** 2)

        spectrum = (r * red_curve) + (g * green_curve) + (b * blue_curve)
        spectrum *= luminance
        max_value = float(np.max(spectrum))
        if max_value > 0:
            spectrum = spectrum / max_value
        return spectrum

    def update_spectrum_plot(self) -> None:
        self.render_plot(self.processor_axis, self.processor_figure, self.processor_plot_canvas, "processor")
        self.render_plot(self.fullscreen_axis, self.fullscreen_figure, self.fullscreen_plot_canvas, "fullscreen")

    def render_plot(self, axis, figure: Figure | None, canvas: FigureCanvasTkAgg | None, target: str) -> None:
        if axis is None or figure is None or canvas is None:
            return

        # Minden frissítéskor újrarajzoljuk a görbéket, hogy a pontlista
        # és a teljes képernyős nézet biztosan szinkronban maradjon.
        axis.clear()
        axis.set_title("Pontonkénti spektrum")
        axis.set_xlabel("Hullámhossz (nm)")
        axis.set_ylabel("Intenzitás")
        axis.set_xlim(WAVELENGTH_START, WAVELENGTH_END)
        axis.set_ylim(0.0, 1.05)
        axis.grid(True, alpha=0.25)

        wavelengths = np.linspace(WAVELENGTH_START, WAVELENGTH_END, SPECTRUM_SAMPLES)
        if self.loaded_filter_min_values is not None and self.loaded_filter_max_values is not None:
            axis.fill_between(
                wavelengths,
                self.loaded_filter_min_values,
                self.loaded_filter_max_values,
                color="#8a8a8a",
                alpha=0.25,
                linewidth=0,
                label="Betöltött szűrősáv",
                zorder=0,
            )

        lines: list = []
        for index, point in enumerate(self.sample_points, start=1):
            line, = axis.plot(
                wavelengths,
                point.spectrum,
                color=point.marker_color,
                linewidth=2.4,
                label=f"Pont {index} RGB{point.rgb}",
                zorder=2,
            )
            lines.append(line)

        if self.sample_points or self.loaded_filter_min_values is not None:
            axis.legend(loc="upper right", fontsize=8)
        else:
            axis.text(
                0.5,
                0.5,
                "A rögzített képen elhelyezett pontok spektruma itt jelenik meg.",
                transform=axis.transAxes,
                ha="center",
                va="center",
                fontsize=10,
            )

        self.set_lines_for_target(target, lines)
        self.clear_marker_for_target(target)
        figure.tight_layout()
        canvas.draw()

    def connect_plot_events(self, target: str) -> None:
        canvas = self.get_canvas_for_target(target)
        if canvas is None or self.get_connection_id_for_target(target) is not None:
            return

        connection_id = canvas.mpl_connect(
            "button_press_event",
            lambda event, plot_target=target: self.on_plot_click(event, plot_target),
        )
        self.set_connection_id_for_target(target, connection_id)

    def on_plot_click(self, event, target: str) -> None:
        axis = self.get_axis_for_target(target)
        canvas = self.get_canvas_for_target(target)
        lines = self.get_lines_for_target(target)
        if axis is None or canvas is None or event.inaxes != axis or event.xdata is None or event.ydata is None:
            return

        nearest = self.find_nearest_curve_point(float(event.xdata), float(event.ydata), lines)
        if nearest is None:
            return

        line, wavelength, intensity = nearest
        self.update_marker_for_target(target, axis, canvas, line, wavelength, intensity)

    def find_nearest_curve_point(self, x_value: float, y_value: float, lines: list) -> tuple | None:
        best_match = None
        best_distance = None

        for line in lines:
            x_data = np.asarray(line.get_xdata(), dtype=float)
            y_data = np.asarray(line.get_ydata(), dtype=float)
            if x_data.size == 0:
                continue

            x_span = max(1.0, float(np.max(x_data) - np.min(x_data)))
            y_span = max(1.0, float(np.max(y_data) - np.min(y_data)))
            # Normalizált távolságot használunk, hogy a nm- és intenzitástengely
            # eltérő skálája ne torzítsa a legközelebbi pont keresését.
            distances = ((x_data - x_value) / x_span) ** 2 + ((y_data - y_value) / y_span) ** 2
            index = int(np.argmin(distances))
            distance = float(distances[index])

            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_match = (line, float(x_data[index]), float(y_data[index]))

        return best_match

    def update_marker_for_target(self, target: str, axis, canvas: FigureCanvasTkAgg, line, wavelength: float, intensity: float) -> None:
        marker_artist = self.get_marker_artist_for_target(target)
        marker_annotation = self.get_marker_annotation_for_target(target)
        marker_line = self.get_marker_line_for_target(target)
        color = line.get_color()

        if marker_artist is None:
            marker_artist = axis.scatter([wavelength], [intensity], s=80, color=color, edgecolors="#ffffff", linewidths=1.5, zorder=5)
            self.set_marker_artist_for_target(target, marker_artist)
        else:
            marker_artist.set_offsets(np.array([[wavelength, intensity]]))
            marker_artist.set_color(color)
            marker_artist.set_edgecolors("#ffffff")

        if marker_line is None:
            marker_line = axis.axvline(wavelength, color=color, linestyle="--", linewidth=1.1, alpha=0.7)
            self.set_marker_line_for_target(target, marker_line)
        else:
            marker_line.set_xdata([wavelength, wavelength])
            marker_line.set_color(color)

        label = f"{line.get_label()}\nλ={wavelength:.1f} nm\nI={intensity:.3f}"
        if marker_annotation is None:
            marker_annotation = axis.annotate(
                label,
                xy=(wavelength, intensity),
                xytext=(14, 14),
                textcoords="offset points",
                bbox={"boxstyle": "round,pad=0.35", "fc": "#101010", "ec": color, "alpha": 0.92},
                color="#ffffff",
                fontsize=9,
                arrowprops={"arrowstyle": "->", "color": color},
            )
            self.set_marker_annotation_for_target(target, marker_annotation)
        else:
            marker_annotation.xy = (wavelength, intensity)
            marker_annotation.set_text(label)
            marker_annotation.set_bbox({"boxstyle": "round,pad=0.35", "fc": "#101010", "ec": color, "alpha": 0.92})
            if marker_annotation.arrow_patch is not None:
                marker_annotation.arrow_patch.set_color(color)

        canvas.draw()

    def get_current_marker_filter_band(self) -> tuple[np.ndarray, np.ndarray] | None:
        if len(self.sample_points) < 2:
            return None

        marker_spectrums = np.array([point.spectrum for point in self.sample_points], dtype=np.float32)
        return np.min(marker_spectrums, axis=0), np.max(marker_spectrums, axis=0)

    def apply_marker_range_filter(self) -> None:
        if self.captured_image_rgb is None:
            messagebox.showinfo("Nincs kép", "Előbb rögzíteni kell egy RGB képet.")
            return

        current_band = self.get_current_marker_filter_band()
        if current_band is not None:
            min_values, max_values = current_band
            filter_source = "aktuális markerek"
        elif self.loaded_filter_min_values is not None and self.loaded_filter_max_values is not None:
            min_values = self.loaded_filter_min_values
            max_values = self.loaded_filter_max_values
            filter_source = f"betöltött szűrő: {self.loaded_filter_name}"
        else:
            messagebox.showinfo(
                "Nincs szűrősáv",
                "A szűréshez jelölj ki legalább két pontot, vagy tölts be egy korábban mentett szűrőt.",
            )
            return

        # A kijelölt pontok minden hullámhosszon meghatározzák a megengedett
        # intenzitássávot. Egy pixel csak akkor marad látható, ha a teljes
        # becsült spektruma ebben a sávban fut.
        image_spectrum = self.estimate_image_spectrum()
        mask = np.all((image_spectrum >= min_values) & (image_spectrum <= max_values), axis=2)
        filtered_image = np.zeros_like(self.captured_image_rgb)
        filtered_image[mask] = self.captured_image_rgb[mask]

        self.open_filtered_window(filtered_image, int(np.count_nonzero(mask)), filter_source)

    def save_filter_band(self) -> None:
        filter_band = self.get_current_marker_filter_band()
        if filter_band is None:
            messagebox.showinfo("Kevés marker", "A szűrő mentéséhez legalább két pontot jelölj ki a képen.")
            return

        min_values, max_values = filter_band
        default_name = f"windmon_szuro_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        target = filedialog.asksaveasfilename(
            title="Szűrő mentése",
            defaultextension=".json",
            initialdir=OUTPUT_DIR,
            initialfile=default_name,
            filetypes=[("WindMon szűrő", "*.json"), ("JSON fájl", "*.json")],
        )
        if not target:
            return

        wavelengths = np.linspace(WAVELENGTH_START, WAVELENGTH_END, SPECTRUM_SAMPLES)
        payload = {
            "format": "windmon-filter-band",
            "version": 1,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "wavelength_start": WAVELENGTH_START,
            "wavelength_end": WAVELENGTH_END,
            "spectrum_samples": SPECTRUM_SAMPLES,
            "wavelengths": wavelengths.tolist(),
            "min_values": min_values.astype(float).tolist(),
            "max_values": max_values.astype(float).tolist(),
        }

        try:
            Path(target).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError as exc:
            messagebox.showerror("Mentési hiba", f"Nem sikerült menteni a szűrőt:\n{exc}")
            return

        self.status_var.set(f"Szűrő mentve: {target}")

    def load_filter_band(self) -> None:
        target = filedialog.askopenfilename(
            title="Szűrő betöltése",
            initialdir=OUTPUT_DIR,
            filetypes=[("WindMon szűrő", "*.json"), ("JSON fájl", "*.json"), ("Minden fájl", "*.*")],
        )
        if not target:
            return

        try:
            payload = json.loads(Path(target).read_text(encoding="utf-8"))
            min_values = np.array(payload["min_values"], dtype=np.float32)
            max_values = np.array(payload["max_values"], dtype=np.float32)
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            messagebox.showerror("Betöltési hiba", f"Nem sikerült betölteni a szűrőt:\n{exc}")
            return

        if min_values.shape != (SPECTRUM_SAMPLES,) or max_values.shape != (SPECTRUM_SAMPLES,):
            messagebox.showerror("Érvénytelen szűrő", "A betöltött szűrő mintaszáma nem egyezik a program beállításával.")
            return

        if np.any(min_values > max_values):
            messagebox.showerror("Érvénytelen szűrő", "A betöltött szűrőben van olyan pont, ahol a minimum nagyobb a maximumnál.")
            return

        self.loaded_filter_min_values = min_values
        self.loaded_filter_max_values = max_values
        self.loaded_filter_name = Path(target).name
        self.update_spectrum_plot()
        self.status_var.set(f"Szűrő betöltve: {target}")

    def clear_loaded_filter_band(self) -> None:
        if self.loaded_filter_min_values is None and self.loaded_filter_max_values is None:
            self.status_var.set("Nincs betöltött szűrő.")
            return

        self.loaded_filter_min_values = None
        self.loaded_filter_max_values = None
        self.loaded_filter_name = None
        self.update_spectrum_plot()
        self.status_var.set("A betöltött szűrő törölve lett.")

    def estimate_image_spectrum(self) -> np.ndarray:
        if self.captured_image_rgb is None:
            return np.empty((0, 0, 0), dtype=np.float32)

        # A teljes kép spektrumát egyszerre, NumPy-vektorizáltan számoljuk,
        # mert pixelenkénti Python ciklussal ez érezhetően lassabb lenne.
        wavelengths = np.linspace(WAVELENGTH_START, WAVELENGTH_END, SPECTRUM_SAMPLES)
        image = self.captured_image_rgb.astype(np.float32) / 255.0
        r = image[:, :, 0]
        g = image[:, :, 1]
        b = image[:, :, 2]

        red_curve = np.exp(-0.5 * ((wavelengths - 620.0) / 36.0) ** 2).astype(np.float32)
        green_curve = np.exp(-0.5 * ((wavelengths - 540.0) / 30.0) ** 2).astype(np.float32)
        blue_curve = np.exp(-0.5 * ((wavelengths - 460.0) / 24.0) ** 2).astype(np.float32)

        spectrum = (
            (r[:, :, np.newaxis] * red_curve)
            + (g[:, :, np.newaxis] * green_curve)
            + (b[:, :, np.newaxis] * blue_curve)
        )
        max_spectrum = np.max(spectrum, axis=2, keepdims=True)
        return np.divide(
            spectrum,
            max_spectrum,
            out=np.zeros_like(spectrum, dtype=np.float32),
            where=max_spectrum > 0,
        )

    def open_filtered_window(
        self,
        filtered_image: np.ndarray,
        matched_pixels: int,
        filter_source: str,
    ) -> None:
        if self.filtered_window is not None and self.filtered_window.winfo_exists():
            self.filtered_window.destroy()

        self.filtered_window = tk.Toplevel(self.root)
        self.filtered_window.title("Szűrt kép")
        self.filtered_window.geometry("1120x720")
        self.filtered_window.minsize(900, 620)
        self.filtered_window.protocol("WM_DELETE_WINDOW", self.close_filtered_window)
        self.filtered_image_rgb = filtered_image
        self.reset_image_view("filtered")

        container = ttk.Frame(self.filtered_window, padding=12)
        container.pack(fill=tk.BOTH, expand=True)
        container.columnconfigure(0, weight=1)
        container.rowconfigure(1, weight=1)

        info_text = (
            f"Szűrés: teljes {WAVELENGTH_START}-{WAVELENGTH_END} nm tartomány, "
            f"forrás: {filter_source}, találat: {matched_pixels} pixel"
        )
        ttk.Label(container, text=info_text).grid(row=0, column=0, sticky="w", pady=(0, 10))

        self.filtered_canvas = tk.Canvas(
            container,
            width=self.capture_size[0],
            height=self.capture_size[1],
            bg="#1f1f1f",
            highlightthickness=1,
            highlightbackground="#707070",
        )
        self.filtered_canvas.grid(row=1, column=0, sticky="nsew")
        self.filtered_canvas.bind("<MouseWheel>", lambda event: self.on_image_mouse_wheel(event, "filtered"))
        self.filtered_canvas.bind("<Button-4>", lambda event: self.on_image_mouse_wheel(event, "filtered", 1))
        self.filtered_canvas.bind("<Button-5>", lambda event: self.on_image_mouse_wheel(event, "filtered", -1))
        self.filtered_canvas.bind("<ButtonPress-2>", lambda event: self.start_image_pan(event, "filtered"))
        self.filtered_canvas.bind("<B2-Motion>", self.on_image_pan)
        self.filtered_canvas.bind("<ButtonPress-3>", lambda event: self.start_image_pan(event, "filtered"))
        self.filtered_canvas.bind("<B3-Motion>", self.on_image_pan)
        self.filtered_canvas.bind("<Configure>", lambda _event: self.redraw_filtered_canvas())

        self.redraw_filtered_canvas()
        self.status_var.set(info_text)

    def redraw_filtered_canvas(self) -> None:
        if (
            self.filtered_canvas is None
            or not self.filtered_canvas.winfo_exists()
            or self.filtered_image_rgb is None
        ):
            return

        self.filtered_canvas.delete("all")
        self.filtered_photo = self.draw_zoomed_image(self.filtered_image_rgb, self.filtered_canvas, "filtered")

    def save_captured_image(self) -> None:
        if self.captured_image_rgb is None:
            messagebox.showinfo("Nincs kép", "Előbb rögzíteni kell egy RGB képet.")
            return

        default_name = f"tekercseles_rgb_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        target = filedialog.asksaveasfilename(
            title="RGB kép mentése",
            defaultextension=".png",
            initialdir=OUTPUT_DIR,
            initialfile=default_name,
            filetypes=[("PNG kép", "*.png"), ("JPEG kép", "*.jpg *.jpeg")],
        )
        if not target:
            return

        image_bgr = cv2.cvtColor(self.captured_image_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(target, image_bgr)
        self.status_var.set(f"RGB kép mentve: {target}")

    def clear_points(self) -> None:
        self.sample_points.clear()
        self.redraw_capture_canvas()
        self.update_spectrum_plot()
        self.status_var.set("A pontok törölve lettek.")

    def close_filtered_window(self) -> None:
        if self.filtered_window is not None and self.filtered_window.winfo_exists():
            self.filtered_window.destroy()

        self.filtered_window = None
        self.filtered_canvas = None
        self.filtered_image_rgb = None
        self.filtered_photo = None
        self.image_pan_start = None

    def on_processor_window_close(self) -> None:
        if self.processor_window is not None and self.processor_window.winfo_exists():
            self.processor_window.destroy()

        self.processor_window = None
        self.processor_canvas = None
        self.processor_figure = None
        self.processor_axis = None
        self.processor_plot_canvas = None
        self.processor_marker_annotation = None
        self.processor_marker_artist = None
        self.processor_marker_line = None
        self.processor_plot_connection_id = None
        self.processor_lines = []

    def on_close(self) -> None:
        if self.preview_after_id is not None:
            self.root.after_cancel(self.preview_after_id)
            self.preview_after_id = None

        self.close_fullscreen_spectrum_window()
        self.close_filtered_window()
        self.on_processor_window_close()
        self.release_camera()
        self.root.destroy()

    @staticmethod
    def rgb_to_hex(rgb: tuple[int, int, int]) -> str:
        return "#{:02x}{:02x}{:02x}".format(*rgb)

    @staticmethod
    def get_marker_color(index: int) -> str:
        return MARKER_COLORS[index % len(MARKER_COLORS)]

    def set_lines_for_target(self, target: str, lines: list) -> None:
        if target == "processor":
            self.processor_lines = lines
        else:
            self.fullscreen_lines = lines

    def get_lines_for_target(self, target: str) -> list:
        return self.processor_lines if target == "processor" else self.fullscreen_lines

    def get_axis_for_target(self, target: str):
        return self.processor_axis if target == "processor" else self.fullscreen_axis

    def get_canvas_for_target(self, target: str) -> FigureCanvasTkAgg | None:
        return self.processor_plot_canvas if target == "processor" else self.fullscreen_plot_canvas

    def get_connection_id_for_target(self, target: str) -> int | None:
        return self.processor_plot_connection_id if target == "processor" else self.fullscreen_plot_connection_id

    def set_connection_id_for_target(self, target: str, value: int) -> None:
        if target == "processor":
            self.processor_plot_connection_id = value
        else:
            self.fullscreen_plot_connection_id = value

    def get_marker_annotation_for_target(self, target: str):
        return self.processor_marker_annotation if target == "processor" else self.fullscreen_marker_annotation

    def set_marker_annotation_for_target(self, target: str, value) -> None:
        if target == "processor":
            self.processor_marker_annotation = value
        else:
            self.fullscreen_marker_annotation = value

    def get_marker_artist_for_target(self, target: str):
        return self.processor_marker_artist if target == "processor" else self.fullscreen_marker_artist

    def set_marker_artist_for_target(self, target: str, value) -> None:
        if target == "processor":
            self.processor_marker_artist = value
        else:
            self.fullscreen_marker_artist = value

    def get_marker_line_for_target(self, target: str):
        return self.processor_marker_line if target == "processor" else self.fullscreen_marker_line

    def set_marker_line_for_target(self, target: str, value) -> None:
        if target == "processor":
            self.processor_marker_line = value
        else:
            self.fullscreen_marker_line = value

    def clear_marker_for_target(self, target: str) -> None:
        marker_annotation = self.get_marker_annotation_for_target(target)
        marker_artist = self.get_marker_artist_for_target(target)
        marker_line = self.get_marker_line_for_target(target)

        if marker_annotation is not None:
            marker_annotation.remove()
            self.set_marker_annotation_for_target(target, None)
        if marker_artist is not None:
            marker_artist.remove()
            self.set_marker_artist_for_target(target, None)
        if marker_line is not None:
            marker_line.remove()
            self.set_marker_line_for_target(target, None)


def main() -> None:
    root = tk.Tk()
    style = ttk.Style(root)
    if "vista" in style.theme_names():
        style.theme_use("vista")
    CameraRecorderApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
