from __future__ import annotations

import json
import threading
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
if hasattr(cv2, "setLogLevel"):
    cv2.setLogLevel(2)


APP_TITLE = "WindMon Tekercselés Kamera Monitor"
OUTPUT_DIR = Path("captures")
WAVELENGTH_START = 380
WAVELENGTH_END = 780
SPECTRUM_SAMPLES = 90
CAMERA_BACKEND = cv2.CAP_ANY
PREVIEW_INTERVAL_MS = 60
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


@dataclass
class MonoSamplePoint:
    # Egy monokróm képen kijelölt mérési pont és intenzitás.
    x: int
    y: int
    intensity: float
    marker_color: str


class CameraRecorderApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("960x640")
        self.root.minsize(860, 560)

        OUTPUT_DIR.mkdir(exist_ok=True)

        self.preview_size = (420, 500)
        self.capture_size = (960, 540)
        self.preview_after_id: str | None = None
        self.camera_loading = False

        self.camera_var = tk.StringVar(value="")
        self.mono_camera_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="Kamera keresése...")

        # Kamera- és képadatok: az élőkép BGR-ben érkezik az OpenCV-ből,
        # a feldolgozás és megjelenítés viszont RGB képpel dolgozik.
        self.available_cameras: list[tuple[int, str]] = []
        self.current_camera_index: int | None = None
        self.current_mono_camera_index: int | None = None
        self.capture: cv2.VideoCapture | None = None
        self.mono_capture: cv2.VideoCapture | None = None
        self.latest_frame_bgr: np.ndarray | None = None
        self.latest_mono_frame_bgr: np.ndarray | None = None
        self.captured_image_rgb: np.ndarray | None = None
        self.captured_mono_image_rgb: np.ndarray | None = None
        self.captured_mono_gray: np.ndarray | None = None
        self.filtered_image_rgb: np.ndarray | None = None
        self.filtered_original_rgb: np.ndarray | None = None
        self.sample_points: list[SamplePoint] = []
        self.mono_sample_points: list[MonoSamplePoint] = []

        # Feldolgozó ablak és spektrumgrafikon állapota.
        self.processor_window: tk.Toplevel | None = None
        self.processor_notebook: ttk.Notebook | None = None
        self.processor_canvas: tk.Canvas | None = None
        self.processor_figure: Figure | None = None
        self.processor_axis = None
        self.processor_plot_canvas: FigureCanvasTkAgg | None = None
        self.processor_marker_annotation = None
        self.processor_marker_artist = None
        self.processor_marker_line = None
        self.processor_plot_connection_id: int | None = None
        self.processor_lines: list = []
        self.mono_processor_canvas: tk.Canvas | None = None
        self.mono_processor_window: tk.Toplevel | None = None
        self.mono_processor_figure: Figure | None = None
        self.mono_processor_axis = None
        self.mono_processor_plot_canvas: FigureCanvasTkAgg | None = None
        self.alignment_canvas: tk.Canvas | None = None
        self.alignment_photo: ImageTk.PhotoImage | None = None
        self.alignment_dx = tk.IntVar(value=0)
        self.alignment_dy = tk.IntVar(value=0)
        self.alignment_scale = tk.DoubleVar(value=1.0)
        self.alignment_alpha = tk.DoubleVar(value=0.45)
        self.alignment_ref_mode = tk.StringVar(value="rgb")
        self.alignment_rgb_points: list[tuple[float, float]] = []
        self.alignment_mono_points: list[tuple[float, float]] = []
        self.alignment_transform_override: np.ndarray | None = None
        self.alignment_use_piecewise_warp = False
        self.alignment_zoom = 1.0
        self.alignment_pan_x = 0.0
        self.alignment_pan_y = 0.0
        self.alignment_pan_start: tuple[int, int, float, float] | None = None
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
        self.mono_processor_zoom = 1.0
        self.mono_processor_pan_x = 0.0
        self.mono_processor_pan_y = 0.0
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
        self.mono_capture_photo: ImageTk.PhotoImage | None = None
        self.preview_photo: ImageTk.PhotoImage | None = None
        self.mono_preview_photo: ImageTk.PhotoImage | None = None
        self.filtered_photo: ImageTk.PhotoImage | None = None

        self.menu_camera = tk.Menu(self.root, tearoff=False)
        self.menu_mono_camera = tk.Menu(self.root, tearoff=False)

        self._build_layout()
        self._build_menubar()

        self.root.after(100, self.refresh_cameras)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.after(100, self.update_preview_loop)

    def _build_layout(self) -> None:
        root_frame = ttk.Frame(self.root, padding=12)
        root_frame.pack(fill=tk.BOTH, expand=True)
        root_frame.columnconfigure(0, weight=1)
        root_frame.rowconfigure(1, weight=1)

        control_frame = ttk.Frame(root_frame)
        control_frame.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        control_frame.columnconfigure(3, weight=1)

        ttk.Label(control_frame, text="Színes kamera:").grid(row=0, column=0, padx=(0, 8))
        self.camera_combo = ttk.Combobox(
            control_frame,
            textvariable=self.camera_var,
            state="readonly",
            width=35,
        )
        self.camera_combo.grid(row=0, column=1, padx=(0, 8))
        self.camera_combo.bind("<<ComboboxSelected>>", self.on_camera_selected)

        ttk.Label(control_frame, text="Monokróm kamera:").grid(row=1, column=0, padx=(0, 8), pady=(8, 0))
        self.mono_camera_combo = ttk.Combobox(
            control_frame,
            textvariable=self.mono_camera_var,
            state="readonly",
            width=35,
        )
        self.mono_camera_combo.grid(row=1, column=1, padx=(0, 8), pady=(8, 0))
        self.mono_camera_combo.bind("<<ComboboxSelected>>", self.on_mono_camera_selected)

        ttk.Button(control_frame, text="Kamerák frissítése", command=self.refresh_cameras).grid(
            row=0, column=2, padx=(0, 8)
        )
        ttk.Button(control_frame, text="Kamerák felcserélése", command=self.swap_camera_roles).grid(
            row=1, column=2, padx=(0, 8), pady=(8, 0)
        )
        ttk.Button(control_frame, text="Kép rögzítése", command=self.capture_image).grid(
            row=0, column=3, padx=(0, 8), sticky="e"
        )
        ttk.Button(control_frame, text="Feldolgozó ablak megnyitása", command=self.open_processor_window).grid(
            row=1, column=3, sticky="e", pady=(8, 0)
        )

        ttk.Label(control_frame, textvariable=self.status_var).grid(
            row=2,
            column=0,
            columnspan=4,
            sticky="w",
            pady=(10, 0),
        )
        self.camera_progress = ttk.Progressbar(control_frame, mode="indeterminate")
        self.camera_progress.grid(row=3, column=0, columnspan=4, sticky="ew", pady=(8, 0))
        self.camera_progress.grid_remove()

        preview_group = ttk.LabelFrame(root_frame, text="Előkép", padding=10)
        preview_group.grid(row=1, column=0, sticky="nsew")
        preview_group.columnconfigure(0, weight=1)
        preview_group.columnconfigure(1, weight=1)
        preview_group.rowconfigure(1, weight=1)

        ttk.Label(
            preview_group,
            text="RGB előkép",
        ).grid(row=0, column=0, sticky="w", pady=(0, 8))

        self.preview_label = ttk.Label(preview_group)
        self.preview_label.grid(row=1, column=0, sticky="nsew", padx=(0, 5))
        self.preview_label.bind("<Configure>", lambda _event: self.refresh_preview_image())

        ttk.Label(preview_group, text="Monokróm előkép").grid(row=0, column=1, sticky="w", pady=(0, 8), padx=(10, 0))
        self.mono_preview_label = ttk.Label(preview_group)
        self.mono_preview_label.grid(row=1, column=1, sticky="nsew", padx=(5, 0))
        self.mono_preview_label.bind("<Configure>", lambda _event: self.refresh_preview_image())

    def _build_menubar(self) -> None:
        menubar = tk.Menu(self.root)
        file_menu = tk.Menu(menubar, tearoff=False)
        file_menu.add_command(label="RGB kép mentése...", command=self.save_captured_image)
        file_menu.add_command(label="Feldolgozó ablak megnyitása", command=self.open_processor_window)
        file_menu.add_separator()
        file_menu.add_command(label="Kilépés", command=self.on_close)

        camera_menu = tk.Menu(menubar, tearoff=False)
        camera_menu.add_command(label="Kamerák újrakeresése", command=self.refresh_cameras)
        camera_menu.add_cascade(label="Színes kamera választása", menu=self.menu_camera)
        camera_menu.add_cascade(label="Monokróm kamera választása", menu=self.menu_mono_camera)

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

        self.processor_notebook = ttk.Notebook(self.processor_window)
        self.processor_notebook.pack(fill=tk.BOTH, expand=True)

        container = ttk.Frame(self.processor_notebook, padding=12)
        self.processor_notebook.add(container, text="RGB kép")
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
        if self.camera_loading:
            return

        self.camera_loading = True
        self.status_var.set("Kamerák betöltése folyamatban...")
        self.camera_progress.grid()
        self.camera_progress.start(12)
        threading.Thread(target=self.refresh_cameras_worker, daemon=True).start()
        return

    def refresh_cameras_sync_unused(self) -> None:
        self.status_var.set("Kamerák keresése folyamatban...")
        self.available_cameras = self.discover_cameras()
        labels = [label for _, label in self.available_cameras]

        self.camera_combo["values"] = labels
        self.mono_camera_combo["values"] = labels
        self.menu_camera.delete(0, tk.END)
        self.menu_mono_camera.delete(0, tk.END)

        for index, label in self.available_cameras:
            self.menu_camera.add_command(
                label=label,
                command=lambda idx=index, lbl=label: self.select_camera(idx, lbl),
            )
            self.menu_mono_camera.add_command(
                label=label,
                command=lambda idx=index, lbl=label: self.select_mono_camera(idx, lbl),
            )

        if labels:
            matching = [item for item in self.available_cameras if item[0] == self.current_camera_index]
            if matching:
                self.camera_var.set(matching[0][1])
                self.status_var.set(f"Kamera elérhető: {matching[0][1]}")
            else:
                first_index, first_label = self.available_cameras[0]
                self.select_camera(first_index, first_label)

            self.ensure_mono_camera_selected()
        else:
            self.camera_var.set("")
            self.mono_camera_var.set("")
            self.status_var.set("Nem található elérhető kamera.")
            self.release_camera()
            self.release_mono_camera()

    def refresh_cameras_worker(self) -> None:
        cameras = self.discover_cameras()
        color_choice, mono_choice = self.choose_default_camera_roles(cameras)
        color_cap = self.open_configured_camera(color_choice[0]) if color_choice is not None else None
        mono_cap = self.open_configured_camera(mono_choice[0]) if mono_choice is not None else None
        self.root.after(0, lambda: self.apply_camera_refresh(cameras, color_choice, mono_choice, color_cap, mono_cap))

    def apply_camera_refresh(
        self,
        cameras: list[tuple[int, str]],
        color_choice: tuple[int, str] | None,
        mono_choice: tuple[int, str] | None,
        color_cap: cv2.VideoCapture | None,
        mono_cap: cv2.VideoCapture | None,
    ) -> None:
        self.camera_loading = False
        self.camera_progress.stop()
        self.camera_progress.grid_remove()
        self.available_cameras = cameras
        labels = [label for _, label in self.available_cameras]

        self.camera_combo["values"] = labels
        self.mono_camera_combo["values"] = labels
        self.menu_camera.delete(0, tk.END)
        self.menu_mono_camera.delete(0, tk.END)

        for index, label in self.available_cameras:
            self.menu_camera.add_command(
                label=label,
                command=lambda idx=index, lbl=label: self.select_camera(idx, lbl),
            )
            self.menu_mono_camera.add_command(
                label=label,
                command=lambda idx=index, lbl=label: self.select_mono_camera(idx, lbl),
            )

        if labels:
            self.release_camera()
            self.release_mono_camera()
            self.capture = color_cap
            self.mono_capture = mono_cap
            self.current_camera_index = color_choice[0] if color_choice is not None and color_cap is not None else None
            self.current_mono_camera_index = mono_choice[0] if mono_choice is not None and mono_cap is not None else None
            self.camera_var.set(color_choice[1] if color_choice is not None and color_cap is not None else "")
            self.mono_camera_var.set(mono_choice[1] if mono_choice is not None and mono_cap is not None else "")
            self.status_var.set("Kamerák betöltve." if self.capture is not None else "A kamerák nem nyithatók meg.")
        else:
            self.camera_var.set("")
            self.mono_camera_var.set("")
            self.status_var.set("Nem található elérhető kamera.")
            self.release_camera()
            self.release_mono_camera()

    def choose_default_camera_roles(
        self,
        cameras: list[tuple[int, str]],
    ) -> tuple[tuple[int, str] | None, tuple[int, str] | None]:
        if not cameras:
            return None, None

        color = next((item for item in cameras if item[0] == self.current_camera_index), cameras[0])
        mono_candidates = [item for item in cameras if item[0] != color[0]]
        mono = next((item for item in mono_candidates if item[0] == self.current_mono_camera_index), None)
        if mono is None and mono_candidates:
            mono = mono_candidates[0]
        return color, mono

    def discover_cameras(self, max_devices: int = 2) -> list[tuple[int, str]]:
        cameras: list[tuple[int, str]] = []
        for index in range(1, max_devices + 1):
            # Windows alatt a DirectShow backend általában gyorsabban és
            # stabilabban nyitja meg az USB kamerákat.
            cap = self.open_camera(index)
            if cap is None or not cap.isOpened():
                continue

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            cap.release()

            role = "Laptop webkamera" if index == 0 else f"USB / külső kamera {index}"
            cameras.append((index, f"{role} [ID {index}] - {width}x{height}"))
            if len(cameras) >= 2:
                break

        if len(cameras) < 2:
            cap = self.open_camera(0)
            if cap is not None and cap.isOpened():
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
                cap.release()
                cameras.append((0, f"Laptop webkamera [ID 0] - {width}x{height}"))

        return cameras

    def open_camera(self, index: int) -> cv2.VideoCapture:
        return cv2.VideoCapture(index, CAMERA_BACKEND)

    def open_configured_camera(self, index: int) -> cv2.VideoCapture | None:
        cap = self.open_camera(index)
        if cap is None or not cap.isOpened():
            return None
        self.configure_camera_resolution(cap)
        return cap

    def on_camera_selected(self, _event: tk.Event) -> None:
        selected_label = self.camera_var.get()
        matching = [item for item in self.available_cameras if item[1] == selected_label]
        if matching:
            index, label = matching[0]
            self.select_camera(index, label)
            if self.current_mono_camera_index == index:
                self.current_mono_camera_index = None
                self.ensure_mono_camera_selected()

    def on_mono_camera_selected(self, _event: tk.Event) -> None:
        selected_label = self.mono_camera_var.get()
        matching = [item for item in self.available_cameras if item[1] == selected_label]
        if matching:
            index, label = matching[0]
            if index == self.current_camera_index:
                messagebox.showinfo("Kamera foglalt", "Válassz másik kamerát a monokróm képhez.")
                self.ensure_mono_camera_selected()
                return
            self.select_mono_camera(index, label)

    def swap_camera_roles(self) -> None:
        if self.current_camera_index is None or self.current_mono_camera_index is None:
            self.status_var.set("A kamerák felcseréléséhez két aktív kamera kell.")
            return

        self.capture, self.mono_capture = self.mono_capture, self.capture
        self.current_camera_index, self.current_mono_camera_index = (
            self.current_mono_camera_index,
            self.current_camera_index,
        )
        self.latest_frame_bgr, self.latest_mono_frame_bgr = self.latest_mono_frame_bgr, self.latest_frame_bgr

        color_label = self.get_camera_label(self.current_camera_index)
        mono_label = self.get_camera_label(self.current_mono_camera_index)
        self.camera_var.set(color_label)
        self.mono_camera_var.set(mono_label)
        self.status_var.set("A színes és monokróm kamera fel lett cserélve.")

    def get_camera_label(self, index: int | None) -> str:
        if index is None:
            return ""
        matching = [item for item in self.available_cameras if item[0] == index]
        if matching:
            return matching[0][1]
        return f"Kamera [ID {index}]"

    def ensure_mono_camera_selected(self) -> None:
        if not self.available_cameras:
            return

        matching = [
            item for item in self.available_cameras
            if item[0] == self.current_mono_camera_index and item[0] != self.current_camera_index
        ]
        if matching:
            self.mono_camera_var.set(matching[0][1])
            return

        candidates = [item for item in self.available_cameras if item[0] != self.current_camera_index]
        if not candidates:
            self.release_mono_camera()
            self.current_mono_camera_index = None
            self.mono_camera_var.set("")
            self.status_var.set("Csak egy kamera érhető el; monokróm párhuzamos kép nincs.")
            return

        index, label = candidates[0]
        self.select_mono_camera(index, label)

    def select_camera(self, index: int, label: str) -> None:
        if index == self.current_camera_index and self.capture is not None and self.capture.isOpened():
            self.camera_var.set(label)
            return

        self.release_camera()
        self.latest_frame_bgr = None
        cap = self.open_camera(index)
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

    def select_mono_camera(self, index: int, label: str) -> None:
        if index == self.current_camera_index:
            return
        if index == self.current_mono_camera_index and self.mono_capture is not None and self.mono_capture.isOpened():
            self.mono_camera_var.set(label)
            return

        self.release_mono_camera()
        self.latest_mono_frame_bgr = None
        cap = self.open_camera(index)
        if not cap.isOpened():
            self.status_var.set(f"A monokróm kamera nem nyitható meg: {label}")
            return

        self.configure_camera_resolution(cap)
        self.mono_capture = cap
        self.current_mono_camera_index = index
        self.mono_camera_var.set(label)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        self.status_var.set(f"Monokróm kamera: {label}, felbontás: {width}x{height}")

    def configure_camera_resolution(self, cap: cv2.VideoCapture) -> None:
        # Először nagyobb felbontásokat kérünk, 16:9-es és 4:3-as módokkal is.
        # Ha a kamera nem támogatja őket, a driver a legközelebbi elérhető
        # módra áll vissza.
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    def release_camera(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def release_mono_camera(self) -> None:
        if self.mono_capture is not None:
            self.mono_capture.release()
            self.mono_capture = None

    def update_preview_loop(self) -> None:
        if self.capture is not None and self.capture.isOpened():
            ok, frame_bgr = self.capture.read()
            if ok:
                self.latest_frame_bgr = frame_bgr
            else:
                self.status_var.set("Nem sikerült képkockát olvasni a kamerából.")

        if self.mono_capture is not None and self.mono_capture.isOpened():
            ok, frame_bgr = self.mono_capture.read()
            if ok:
                self.latest_mono_frame_bgr = frame_bgr
            else:
                self.status_var.set("Nem sikerült képkockát olvasni a monokróm kamerából.")

        self.refresh_preview_image()
        self.preview_after_id = self.root.after(PREVIEW_INTERVAL_MS, self.update_preview_loop)

    def refresh_preview_image(self) -> None:
        if self.latest_frame_bgr is not None:
            preview_rgb = cv2.cvtColor(self.latest_frame_bgr, cv2.COLOR_BGR2RGB)
            preview_size = self.get_preview_display_size(self.preview_label)
            preview_image = self.resize_for_display(preview_rgb, preview_size)
            self.preview_photo = ImageTk.PhotoImage(Image.fromarray(preview_image))
            self.preview_label.configure(image=self.preview_photo)

        if self.latest_mono_frame_bgr is not None:
            mono_rgb = self.frame_bgr_to_gray_rgb(self.latest_mono_frame_bgr)
            preview_size = self.get_preview_display_size(self.mono_preview_label)
            preview_image = self.resize_for_display(mono_rgb, preview_size)
            self.mono_preview_photo = ImageTk.PhotoImage(Image.fromarray(preview_image))
            self.mono_preview_label.configure(image=self.mono_preview_photo)

    def get_preview_display_size(self, label: ttk.Label) -> tuple[int, int]:
        width = label.winfo_width()
        height = label.winfo_height()
        if width <= 1 or height <= 1:
            return self.preview_size
        return width, height

    def frame_bgr_to_gray_rgb(self, frame_bgr: np.ndarray) -> np.ndarray:
        gray = self.frame_to_gray(frame_bgr)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    def frame_to_gray(self, frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 2:
            return frame
        if frame.shape[2] == 1:
            return frame[:, :, 0]
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

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
        if target == "mono_processor":
            return self.mono_processor_zoom, self.mono_processor_pan_x, self.mono_processor_pan_y
        return self.filtered_zoom, self.filtered_pan_x, self.filtered_pan_y

    def set_image_view_state(self, target: str, zoom: float, pan_x: float, pan_y: float) -> None:
        if target == "processor":
            self.processor_zoom = zoom
            self.processor_pan_x = pan_x
            self.processor_pan_y = pan_y
        elif target == "mono_processor":
            self.mono_processor_zoom = zoom
            self.mono_processor_pan_x = pan_x
            self.mono_processor_pan_y = pan_y
        else:
            self.filtered_zoom = zoom
            self.filtered_pan_x = pan_x
            self.filtered_pan_y = pan_y

    def reset_image_view(self, target: str) -> None:
        self.set_image_view_state(target, 1.0, 0.0, 0.0)

    def get_image_for_target(self, target: str) -> np.ndarray | None:
        if target == "processor":
            return self.captured_image_rgb
        if target == "mono_processor":
            return self.captured_mono_image_rgb
        return self.filtered_image_rgb

    def get_image_canvas_for_target(self, target: str) -> tk.Canvas | None:
        if target == "processor":
            return self.processor_canvas
        if target == "mono_processor":
            return self.mono_processor_canvas
        return self.filtered_canvas

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
        image_rgb = self.get_image_for_target(target)
        canvas = self.get_image_canvas_for_target(target)
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
        elif target == "mono_processor":
            self.redraw_mono_capture_canvas()
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
        image_rgb = self.get_image_for_target(target)
        canvas = self.get_image_canvas_for_target(target)
        if image_rgb is None or canvas is None or not canvas.winfo_exists():
            return "break"

        zoom, _, _ = self.get_image_view_state(target)
        pan_x = start_pan_x + event.x - start_x
        pan_y = start_pan_y + event.y - start_y
        pan_x, pan_y = self.clamp_image_pan(image_rgb, canvas, zoom, pan_x, pan_y)
        self.set_image_view_state(target, zoom, pan_x, pan_y)

        if target == "processor":
            self.redraw_capture_canvas()
        elif target == "mono_processor":
            self.redraw_mono_capture_canvas()
        else:
            self.redraw_filtered_canvas()

        return "break"

    def capture_image(self) -> None:
        if self.latest_frame_bgr is None:
            messagebox.showwarning("Nincs kép", "Először nyiss meg egy kamerát, és várj előképre.")
            return

        self.captured_image_rgb = cv2.cvtColor(self.latest_frame_bgr.copy(), cv2.COLOR_BGR2RGB)
        if self.latest_mono_frame_bgr is not None:
            self.captured_mono_image_rgb = self.frame_bgr_to_gray_rgb(self.latest_mono_frame_bgr.copy())
            self.captured_mono_gray = self.frame_to_gray(self.latest_mono_frame_bgr.copy())
        else:
            self.captured_mono_image_rgb = None
            self.captured_mono_gray = None
        self.sample_points.clear()
        self.mono_sample_points.clear()
        self.reset_alignment()
        self.reset_image_view("processor")
        self.reset_image_view("mono_processor")
        self.open_processor_window()
        self.open_mono_processor_window()
        self.redraw_capture_canvas()
        self.redraw_mono_capture_canvas()
        self.update_spectrum_plot()
        self.update_mono_intensity_plot()
        self.status_var.set("RGB és monokróm kép rögzítve. A feldolgozó ablakokban helyezhetsz el pontokat.")

    def open_processor_window(self) -> None:
        if self.captured_image_rgb is None:
            messagebox.showinfo("Nincs rögzített kép", "Előbb készíts egy képet a kamerából.")
            return

        self._build_processor_window()
        self.open_mono_processor_window()

    def open_mono_processor_window(self) -> None:
        if self.captured_mono_image_rgb is None or self.captured_mono_gray is None:
            return

        if self.processor_window is None or not self.processor_window.winfo_exists():
            return

        if self.mono_processor_window is not None and self.mono_processor_window.winfo_exists():
            if self.processor_notebook is not None:
                self.processor_notebook.select(self.mono_processor_window)
            return

        if self.processor_notebook is None:
            return

        container = ttk.Frame(self.processor_notebook, padding=12)
        self.processor_notebook.add(container, text="Monokróm kép")
        self.mono_processor_window = container
        container.columnconfigure(0, weight=3)
        container.columnconfigure(1, weight=2)
        container.rowconfigure(1, weight=1)

        toolbar = ttk.Frame(container)
        toolbar.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 12))
        toolbar.columnconfigure(3, weight=1)
        ttk.Button(toolbar, text="Pontok törlése", command=self.clear_mono_points).grid(row=0, column=0, padx=(0, 8))
        ttk.Button(toolbar, text="Intenzitás szűrő alkalmazása", command=self.apply_mono_intensity_filter).grid(
            row=0, column=1, padx=(0, 8)
        )
        ttk.Label(toolbar, text="Bal kattintás: pont hozzáadása, jobb kattintás: pont törlése.").grid(
            row=0, column=3, sticky="e"
        )

        image_group = ttk.LabelFrame(container, text="Rögzített monokróm kép", padding=10)
        image_group.grid(row=1, column=0, sticky="nsew", padx=(0, 10))
        image_group.columnconfigure(0, weight=1)
        image_group.rowconfigure(0, weight=1)

        self.mono_processor_canvas = tk.Canvas(
            image_group,
            width=self.capture_size[0],
            height=self.capture_size[1],
            bg="#1f1f1f",
            highlightthickness=1,
            highlightbackground="#707070",
            cursor="crosshair",
        )
        self.mono_processor_canvas.grid(row=0, column=0, sticky="nsew")
        self.mono_processor_canvas.bind("<Button-1>", self.on_mono_capture_canvas_click)
        self.mono_processor_canvas.bind("<Button-3>", self.remove_mono_capture_point)
        self.mono_processor_canvas.bind("<MouseWheel>", lambda event: self.on_image_mouse_wheel(event, "mono_processor"))
        self.mono_processor_canvas.bind("<Button-4>", lambda event: self.on_image_mouse_wheel(event, "mono_processor", 1))
        self.mono_processor_canvas.bind("<Button-5>", lambda event: self.on_image_mouse_wheel(event, "mono_processor", -1))
        self.mono_processor_canvas.bind("<ButtonPress-2>", lambda event: self.start_image_pan(event, "mono_processor"))
        self.mono_processor_canvas.bind("<B2-Motion>", self.on_image_pan)
        self.mono_processor_canvas.bind("<Configure>", lambda _event: self.redraw_mono_capture_canvas())

        plot_group = ttk.LabelFrame(container, text="Monokróm pontok intenzitása", padding=10)
        plot_group.grid(row=1, column=1, sticky="nsew")
        plot_group.columnconfigure(0, weight=1)
        plot_group.rowconfigure(0, weight=1)

        plot_host = ttk.Frame(plot_group)
        plot_host.grid(row=0, column=0, sticky="nsew")
        self.mono_processor_figure = Figure(figsize=(4.5, 4.8), dpi=100)
        self.mono_processor_axis = self.mono_processor_figure.add_subplot(111)
        self.mono_processor_plot_canvas = FigureCanvasTkAgg(self.mono_processor_figure, master=plot_host)
        self.mono_processor_plot_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.redraw_mono_capture_canvas()
        self.update_mono_intensity_plot()
        self.build_alignment_tab()

    def build_alignment_tab(self) -> None:
        if (
            self.processor_notebook is None
            or self.captured_image_rgb is None
            or self.captured_mono_image_rgb is None
            or self.alignment_canvas is not None
        ):
            return

        container = ttk.Frame(self.processor_notebook, padding=12)
        self.processor_notebook.add(container, text="Igazítás")
        container.columnconfigure(0, weight=1)
        container.rowconfigure(1, weight=1)

        toolbar = ttk.Frame(container)
        toolbar.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        toolbar.columnconfigure(13, weight=1)

        ttk.Radiobutton(toolbar, text="RGB referencia", variable=self.alignment_ref_mode, value="rgb").grid(
            row=0, column=0, padx=(0, 8)
        )
        ttk.Radiobutton(toolbar, text="Monokróm referencia", variable=self.alignment_ref_mode, value="mono").grid(
            row=0, column=1, padx=(0, 12)
        )
        ttk.Button(toolbar, text="Monokróm igazítása", command=self.apply_reference_alignment).grid(
            row=0, column=2, padx=(0, 8)
        )
        ttk.Button(toolbar, text="Pontok törlése", command=self.clear_alignment_points).grid(row=0, column=3, padx=(0, 12))
        ttk.Button(toolbar, text="Bal", command=lambda: self.nudge_alignment(-5, 0)).grid(row=0, column=4, padx=(0, 4))
        ttk.Button(toolbar, text="Jobb", command=lambda: self.nudge_alignment(5, 0)).grid(row=0, column=5, padx=(0, 8))
        ttk.Button(toolbar, text="Fel", command=lambda: self.nudge_alignment(0, -5)).grid(row=0, column=6, padx=(0, 4))
        ttk.Button(toolbar, text="Le", command=lambda: self.nudge_alignment(0, 5)).grid(row=0, column=7, padx=(0, 8))
        ttk.Button(toolbar, text="Alaphelyzet", command=self.reset_alignment).grid(row=0, column=8, padx=(0, 12))
        ttk.Label(toolbar, text="Méret").grid(row=0, column=9, padx=(0, 4))
        ttk.Scale(toolbar, from_=0.80, to=1.20, variable=self.alignment_scale, command=lambda _v: self.redraw_alignment_canvas()).grid(
            row=0, column=10, sticky="ew", padx=(0, 12)
        )
        ttk.Label(toolbar, text="Átlátszóság").grid(row=0, column=11, padx=(0, 4))
        ttk.Scale(toolbar, from_=0.15, to=0.85, variable=self.alignment_alpha, command=lambda _v: self.redraw_alignment_canvas()).grid(
            row=0, column=13, sticky="ew"
        )

        self.alignment_canvas = tk.Canvas(
            container,
            width=self.capture_size[0],
            height=self.capture_size[1],
            bg="#1f1f1f",
            highlightthickness=1,
            highlightbackground="#707070",
        )
        self.alignment_canvas.grid(row=1, column=0, sticky="nsew")
        self.alignment_canvas.bind("<Configure>", lambda _event: self.redraw_alignment_canvas())
        self.alignment_canvas.bind("<Button-1>", self.on_alignment_canvas_click)
        self.alignment_canvas.bind("<Button-3>", self.remove_alignment_point)
        self.alignment_canvas.bind("<MouseWheel>", self.on_alignment_mouse_wheel)
        self.alignment_canvas.bind("<Button-4>", lambda event: self.on_alignment_mouse_wheel(event, 1))
        self.alignment_canvas.bind("<Button-5>", lambda event: self.on_alignment_mouse_wheel(event, -1))
        self.alignment_canvas.bind("<ButtonPress-2>", self.start_alignment_pan)
        self.alignment_canvas.bind("<B2-Motion>", self.on_alignment_pan)
        self.alignment_transform_override = None
        self.alignment_use_piecewise_warp = False
        self.redraw_alignment_canvas()

    def nudge_alignment(self, dx: int, dy: int) -> None:
        self.alignment_dx.set(self.alignment_dx.get() + dx)
        self.alignment_dy.set(self.alignment_dy.get() + dy)
        self.redraw_alignment_canvas()

    def reset_alignment(self) -> None:
        self.alignment_dx.set(0)
        self.alignment_dy.set(0)
        self.alignment_scale.set(1.0)
        self.alignment_alpha.set(0.45)
        self.alignment_rgb_points.clear()
        self.alignment_mono_points.clear()
        self.alignment_transform_override = None
        self.alignment_use_piecewise_warp = False
        self.alignment_zoom = 1.0
        self.alignment_pan_x = 0.0
        self.alignment_pan_y = 0.0
        self.alignment_pan_start = None
        self.redraw_alignment_canvas()

    def clear_alignment_points(self) -> None:
        self.alignment_rgb_points.clear()
        self.alignment_mono_points.clear()
        self.alignment_transform_override = None
        self.alignment_use_piecewise_warp = False
        self.redraw_alignment_canvas()
        self.status_var.set("Az igazítási referencia pontok törölve lettek.")

    def on_alignment_canvas_click(self, event: tk.Event) -> None:
        if self.alignment_canvas is None:
            return

        base_x, base_y = self.screen_to_alignment_base(float(event.x), float(event.y))
        mode = self.alignment_ref_mode.get()
        if mode == "rgb":
            self.alignment_rgb_points.append((base_x, base_y))
            self.status_var.set(f"RGB referencia pont hozzáadva: {len(self.alignment_rgb_points)}")
        else:
            mono_x, mono_y = self.canvas_to_mono_alignment_point(base_x, base_y)
            self.alignment_mono_points.append((mono_x, mono_y))
            self.status_var.set(f"Monokróm referencia pont hozzáadva: {len(self.alignment_mono_points)}")

        self.alignment_transform_override = None
        self.redraw_alignment_canvas()

    def apply_reference_alignment(self) -> None:
        pair_count = min(len(self.alignment_rgb_points), len(self.alignment_mono_points))
        if pair_count > 0:
            transform = self.calculate_reference_alignment_transform(pair_count)
            if transform is None:
                messagebox.showinfo("Nem számolható igazítás", "Adj meg legalább egy érvényes referencia pontpárt.")
                return

            self.alignment_transform_override = transform
            self.alignment_use_piecewise_warp = pair_count >= 3
            self.alignment_dx.set(0)
            self.alignment_dy.set(0)
            self.alignment_scale.set(1.0)
            self.redraw_alignment_canvas()
            self.status_var.set(f"Monokróm kép igazítva {pair_count} referencia pontpár alapján.")
            return
        if pair_count == 0:
            messagebox.showinfo("Nincs pontpár", "Adj meg legalább egy RGB és egy monokróm referencia pontot.")
            return

        self.redraw_alignment_canvas()
        self.status_var.set(f"Igazítás újraszámolva {pair_count} referencia pontpár alapján.")

    def remove_alignment_point(self, event: tk.Event) -> str:
        base_x, base_y = self.screen_to_alignment_base(float(event.x), float(event.y))
        hit = self.find_nearest_alignment_point(base_x, base_y)
        if hit is None:
            return "break"

        point_type, index = hit
        if point_type == "rgb":
            self.alignment_rgb_points.pop(index)
        else:
            self.alignment_mono_points.pop(index)
        self.alignment_transform_override = None
        self.alignment_use_piecewise_warp = False
        self.redraw_alignment_canvas()
        self.status_var.set("Igazítási referencia pont törölve.")
        return "break"

    def find_nearest_alignment_point(self, x: float, y: float) -> tuple[str, int] | None:
        transform = self.get_alignment_transform()
        if transform is None:
            scale = float(self.alignment_scale.get())
            transform = np.array(
                [[scale, 0.0, float(self.alignment_dx.get())], [0.0, scale, float(self.alignment_dy.get())]],
                dtype=np.float32,
            )

        nearest: tuple[str, int] | None = None
        nearest_distance: float | None = None
        hit_radius = 14.0 / max(1.0, self.alignment_zoom)

        for index, point in enumerate(self.alignment_rgb_points):
            distance = float(np.hypot(point[0] - x, point[1] - y))
            if distance <= hit_radius and (nearest_distance is None or distance < nearest_distance):
                nearest = ("rgb", index)
                nearest_distance = distance

        for index, point in enumerate(self.alignment_mono_points):
            if self.alignment_use_piecewise_warp and index < len(self.alignment_rgb_points):
                mapped_x, mapped_y = self.alignment_rgb_points[index]
            else:
                mono_point = np.array([point[0], point[1], 1.0])
                mapped_x, mapped_y = transform @ mono_point
            distance = float(np.hypot(mapped_x - x, mapped_y - y))
            if distance <= hit_radius and (nearest_distance is None or distance < nearest_distance):
                nearest = ("mono", index)
                nearest_distance = distance

        return nearest

    def canvas_to_mono_alignment_point(self, x: float, y: float) -> tuple[float, float]:
        transform = self.get_alignment_transform()
        if transform is None:
            scale = max(0.001, float(self.alignment_scale.get()))
            return (x - float(self.alignment_dx.get())) / scale, (y - float(self.alignment_dy.get())) / scale

        matrix = np.vstack([transform, [0.0, 0.0, 1.0]])
        try:
            inverse = np.linalg.inv(matrix)
        except np.linalg.LinAlgError:
            return x, y
        point = inverse @ np.array([x, y, 1.0], dtype=np.float32)
        return float(point[0]), float(point[1])

    def get_alignment_transform(self) -> np.ndarray | None:
        if self.alignment_transform_override is not None:
            transform = self.alignment_transform_override.copy()
            manual_scale = float(self.alignment_scale.get())
            if abs(manual_scale - 1.0) > 0.001:
                transform[:, :2] *= manual_scale
            transform[0, 2] += float(self.alignment_dx.get())
            transform[1, 2] += float(self.alignment_dy.get())
            return transform

        pair_count = min(len(self.alignment_rgb_points), len(self.alignment_mono_points))
        if pair_count == 0:
            return None

        src = np.array(self.alignment_mono_points[:pair_count], dtype=np.float32)
        dst = np.array(self.alignment_rgb_points[:pair_count], dtype=np.float32)

        if pair_count == 1:
            dx = float(dst[0, 0] - src[0, 0] + self.alignment_dx.get())
            dy = float(dst[0, 1] - src[0, 1] + self.alignment_dy.get())
            scale = float(self.alignment_scale.get())
            return np.array([[scale, 0.0, dx], [0.0, scale, dy]], dtype=np.float32)

        transform, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
        if transform is None:
            return None

        transform = transform.astype(np.float32)
        transform[0, 2] += float(self.alignment_dx.get())
        transform[1, 2] += float(self.alignment_dy.get())
        manual_scale = float(self.alignment_scale.get())
        if abs(manual_scale - 1.0) > 0.001:
            transform[:, :2] *= manual_scale
        return transform

    def calculate_reference_alignment_transform(self, pair_count: int) -> np.ndarray | None:
        src = np.array(self.alignment_mono_points[:pair_count], dtype=np.float32)
        dst = np.array(self.alignment_rgb_points[:pair_count], dtype=np.float32)

        if pair_count == 1:
            dx = float(dst[0, 0] - src[0, 0])
            dy = float(dst[0, 1] - src[0, 1])
            return np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)

        if pair_count == 2:
            return self.calculate_two_point_similarity(src, dst)

        transform, _ = cv2.estimateAffine2D(src, dst, method=cv2.LMEDS)
        if transform is not None:
            return transform.astype(np.float32)
        transform, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
        return transform.astype(np.float32) if transform is not None else None

    def calculate_two_point_similarity(self, src: np.ndarray, dst: np.ndarray) -> np.ndarray | None:
        src_vec = src[1] - src[0]
        dst_vec = dst[1] - dst[0]
        src_len = float(np.hypot(src_vec[0], src_vec[1]))
        dst_len = float(np.hypot(dst_vec[0], dst_vec[1]))
        if src_len <= 0.001:
            return None

        scale = dst_len / src_len
        src_angle = float(np.arctan2(src_vec[1], src_vec[0]))
        dst_angle = float(np.arctan2(dst_vec[1], dst_vec[0]))
        angle = dst_angle - src_angle
        cos_a = np.cos(angle) * scale
        sin_a = np.sin(angle) * scale
        matrix = np.array([[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0]], dtype=np.float32)
        mapped_first = matrix[:, :2] @ src[0]
        matrix[0, 2] = dst[0, 0] - mapped_first[0]
        matrix[1, 2] = dst[0, 1] - mapped_first[1]
        return matrix

    def get_alignment_view_origin(self, base_w: int, base_h: int, view_w: int, view_h: int) -> tuple[float, float]:
        if self.alignment_zoom <= 1.0:
            self.alignment_pan_x = 0.0
            self.alignment_pan_y = 0.0
            return 0.0, 0.0

        visible_w = view_w / self.alignment_zoom
        visible_h = view_h / self.alignment_zoom
        max_x = max(0.0, base_w - visible_w)
        max_y = max(0.0, base_h - visible_h)
        self.alignment_pan_x = min(max_x, max(0.0, self.alignment_pan_x))
        self.alignment_pan_y = min(max_y, max(0.0, self.alignment_pan_y))
        return self.alignment_pan_x, self.alignment_pan_y

    def screen_to_alignment_base(self, x: float, y: float) -> tuple[float, float]:
        if self.alignment_canvas is None:
            return x, y

        canvas_w, canvas_h = self.get_canvas_size(self.alignment_canvas)
        origin_x, origin_y = self.get_alignment_view_origin(canvas_w, canvas_h, canvas_w, canvas_h)
        return origin_x + x / self.alignment_zoom, origin_y + y / self.alignment_zoom

    def alignment_base_to_screen(self, x: float, y: float) -> tuple[float, float]:
        if self.alignment_canvas is None:
            return x, y

        canvas_w, canvas_h = self.get_canvas_size(self.alignment_canvas)
        origin_x, origin_y = self.get_alignment_view_origin(canvas_w, canvas_h, canvas_w, canvas_h)
        return (x - origin_x) * self.alignment_zoom, (y - origin_y) * self.alignment_zoom

    def on_alignment_mouse_wheel(self, event: tk.Event, direction: int | None = None) -> str:
        if self.alignment_canvas is None:
            return "break"

        if direction is None:
            direction = 1 if getattr(event, "delta", 0) > 0 else -1

        old_zoom = self.alignment_zoom
        new_zoom = old_zoom * (1.18 if direction > 0 else 1 / 1.18)
        new_zoom = min(10.0, max(1.0, new_zoom))
        if abs(new_zoom - old_zoom) < 0.001:
            return "break"

        canvas_w, canvas_h = self.get_canvas_size(self.alignment_canvas)
        old_origin_x, old_origin_y = self.get_alignment_view_origin(canvas_w, canvas_h, canvas_w, canvas_h)
        base_x = old_origin_x + event.x / old_zoom
        base_y = old_origin_y + event.y / old_zoom
        self.alignment_zoom = new_zoom
        self.alignment_pan_x = base_x - event.x / new_zoom
        self.alignment_pan_y = base_y - event.y / new_zoom
        self.get_alignment_view_origin(canvas_w, canvas_h, canvas_w, canvas_h)
        self.redraw_alignment_canvas()
        return "break"

    def start_alignment_pan(self, event: tk.Event) -> str:
        if self.alignment_zoom <= 1.0:
            return "break"
        self.alignment_pan_start = (event.x, event.y, self.alignment_pan_x, self.alignment_pan_y)
        return "break"

    def on_alignment_pan(self, event: tk.Event) -> str:
        if self.alignment_pan_start is None or self.alignment_canvas is None:
            return "break"

        start_x, start_y, start_pan_x, start_pan_y = self.alignment_pan_start
        self.alignment_pan_x = start_pan_x - (event.x - start_x) / self.alignment_zoom
        self.alignment_pan_y = start_pan_y - (event.y - start_y) / self.alignment_zoom
        canvas_w, canvas_h = self.get_canvas_size(self.alignment_canvas)
        self.get_alignment_view_origin(canvas_w, canvas_h, canvas_w, canvas_h)
        self.redraw_alignment_canvas()
        return "break"

    def redraw_alignment_canvas(self) -> None:
        if (
            self.alignment_canvas is None
            or not self.alignment_canvas.winfo_exists()
            or self.captured_image_rgb is None
            or self.captured_mono_image_rgb is None
        ):
            return

        canvas_w, canvas_h = self.get_canvas_size(self.alignment_canvas)
        rgb_base = self.resize_for_display(self.captured_image_rgb, (canvas_w, canvas_h))
        mono_fit = self.resize_for_display(self.captured_mono_image_rgb, (canvas_w, canvas_h))

        transform = self.get_alignment_transform()
        if transform is None:
            scale = float(self.alignment_scale.get())
            dx = float(self.alignment_dx.get())
            dy = float(self.alignment_dy.get())
            transform = np.array([[scale, 0.0, dx], [0.0, scale, dy]], dtype=np.float32)

        if self.alignment_use_piecewise_warp:
            shifted_mono = self.warp_mono_piecewise(mono_fit, canvas_w, canvas_h)
        else:
            shifted_mono = cv2.warpAffine(
                mono_fit,
                transform,
                (canvas_w, canvas_h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0),
            )

        alpha = float(self.alignment_alpha.get())
        overlay = cv2.addWeighted(rgb_base, 1.0 - alpha, shifted_mono, alpha, 0.0)
        display_overlay = self.crop_alignment_view(overlay, canvas_w, canvas_h)
        self.alignment_photo = ImageTk.PhotoImage(Image.fromarray(display_overlay))
        self.alignment_canvas.delete("all")
        self.alignment_canvas.create_image(0, 0, anchor=tk.NW, image=self.alignment_photo)
        self.draw_alignment_reference_points(transform)

    def crop_alignment_view(self, image_rgb: np.ndarray, canvas_w: int, canvas_h: int) -> np.ndarray:
        if self.alignment_zoom <= 1.0:
            return image_rgb

        origin_x, origin_y = self.get_alignment_view_origin(image_rgb.shape[1], image_rgb.shape[0], canvas_w, canvas_h)
        crop_w = max(1, int(round(canvas_w / self.alignment_zoom)))
        crop_h = max(1, int(round(canvas_h / self.alignment_zoom)))
        x0 = min(image_rgb.shape[1] - crop_w, max(0, int(round(origin_x))))
        y0 = min(image_rgb.shape[0] - crop_h, max(0, int(round(origin_y))))
        crop = image_rgb[y0 : y0 + crop_h, x0 : x0 + crop_w]
        return cv2.resize(crop, (canvas_w, canvas_h), interpolation=cv2.INTER_LINEAR)

    def warp_mono_piecewise(self, mono_fit: np.ndarray, canvas_w: int, canvas_h: int) -> np.ndarray:
        pair_count = min(len(self.alignment_rgb_points), len(self.alignment_mono_points))
        if pair_count < 3:
            transform = self.get_alignment_transform()
            if transform is None:
                return mono_fit
            return cv2.warpAffine(mono_fit, transform, (canvas_w, canvas_h), flags=cv2.INTER_LINEAR)

        src_points = [tuple(point) for point in self.alignment_mono_points[:pair_count]]
        dst_points = [tuple(point) for point in self.alignment_rgb_points[:pair_count]]
        anchors = [
            (0.0, 0.0),
            (canvas_w - 1.0, 0.0),
            (canvas_w - 1.0, canvas_h - 1.0),
            (0.0, canvas_h - 1.0),
            ((canvas_w - 1.0) / 2.0, 0.0),
            (canvas_w - 1.0, (canvas_h - 1.0) / 2.0),
            ((canvas_w - 1.0) / 2.0, canvas_h - 1.0),
            (0.0, (canvas_h - 1.0) / 2.0),
        ]
        src_points.extend(anchors)
        dst_points.extend(anchors)

        triangles = self.calculate_delaunay_triangles(dst_points, canvas_w, canvas_h)
        if not triangles:
            transform = self.get_alignment_transform()
            if transform is None:
                return mono_fit
            return cv2.warpAffine(mono_fit, transform, (canvas_w, canvas_h), flags=cv2.INTER_LINEAR)

        output = np.zeros_like(mono_fit)
        for triangle in triangles:
            src_triangle = np.float32([src_points[index] for index in triangle])
            dst_triangle = np.float32([dst_points[index] for index in triangle])
            self.warp_triangle(mono_fit, output, src_triangle, dst_triangle)
        return output

    def calculate_delaunay_triangles(
        self,
        points: list[tuple[float, float]],
        width: int,
        height: int,
    ) -> list[tuple[int, int, int]]:
        subdiv = cv2.Subdiv2D((0, 0, width, height))
        for point in points:
            x = min(width - 1.0, max(0.0, point[0]))
            y = min(height - 1.0, max(0.0, point[1]))
            try:
                subdiv.insert((float(x), float(y)))
            except cv2.error:
                continue

        triangles: list[tuple[int, int, int]] = []
        seen: set[tuple[int, int, int]] = set()
        for triangle in subdiv.getTriangleList():
            coords = [(float(triangle[i]), float(triangle[i + 1])) for i in range(0, 6, 2)]
            if not all(0 <= x < width and 0 <= y < height for x, y in coords):
                continue

            indices = tuple(self.find_nearest_point_index(point, points) for point in coords)
            if len(set(indices)) != 3:
                continue

            key = tuple(sorted(indices))
            if key in seen:
                continue
            seen.add(key)
            triangles.append(indices)
        return triangles

    def find_nearest_point_index(self, point: tuple[float, float], points: list[tuple[float, float]]) -> int:
        distances = [float(np.hypot(point[0] - candidate[0], point[1] - candidate[1])) for candidate in points]
        return int(np.argmin(distances))

    def warp_triangle(
        self,
        source: np.ndarray,
        target: np.ndarray,
        src_triangle: np.ndarray,
        dst_triangle: np.ndarray,
    ) -> None:
        src_rect = cv2.boundingRect(src_triangle)
        dst_rect = cv2.boundingRect(dst_triangle)
        src_x, src_y, src_w, src_h = src_rect
        dst_x, dst_y, dst_w, dst_h = dst_rect
        if src_w <= 0 or src_h <= 0 or dst_w <= 0 or dst_h <= 0:
            return

        src_h_img, src_w_img = source.shape[:2]
        dst_h_img, dst_w_img = target.shape[:2]
        if src_x < 0 or src_y < 0 or src_x + src_w > src_w_img or src_y + src_h > src_h_img:
            return
        if dst_x < 0 or dst_y < 0 or dst_x + dst_w > dst_w_img or dst_y + dst_h > dst_h_img:
            return

        src_offset = np.float32([[p[0] - src_x, p[1] - src_y] for p in src_triangle])
        dst_offset = np.float32([[p[0] - dst_x, p[1] - dst_y] for p in dst_triangle])
        matrix = cv2.getAffineTransform(src_offset, dst_offset)
        source_crop = source[src_y : src_y + src_h, src_x : src_x + src_w]
        warped = cv2.warpAffine(
            source_crop,
            matrix,
            (dst_w, dst_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

        mask = np.zeros((dst_h, dst_w, 3), dtype=np.float32)
        cv2.fillConvexPoly(mask, np.int32(dst_offset), (1.0, 1.0, 1.0), lineType=cv2.LINE_AA)
        target_crop = target[dst_y : dst_y + dst_h, dst_x : dst_x + dst_w].astype(np.float32)
        blended = target_crop * (1.0 - mask) + warped.astype(np.float32) * mask
        target[dst_y : dst_y + dst_h, dst_x : dst_x + dst_w] = np.clip(blended, 0, 255).astype(np.uint8)

    def draw_alignment_reference_points(self, transform: np.ndarray) -> None:
        if self.alignment_canvas is None:
            return

        pair_count = max(len(self.alignment_rgb_points), len(self.alignment_mono_points))
        for index in range(pair_count):
            if index < len(self.alignment_rgb_points):
                x, y = self.alignment_rgb_points[index]
                screen_x, screen_y = self.alignment_base_to_screen(x, y)
                self.draw_alignment_marker(screen_x, screen_y, index + 1, "#ff3b30", "R")
            if index < len(self.alignment_mono_points):
                if self.alignment_use_piecewise_warp and index < len(self.alignment_rgb_points):
                    x, y = self.alignment_rgb_points[index]
                else:
                    mono_point = np.array([self.alignment_mono_points[index][0], self.alignment_mono_points[index][1], 1.0])
                    x, y = transform @ mono_point
                screen_x, screen_y = self.alignment_base_to_screen(float(x), float(y))
                self.draw_alignment_marker(screen_x, screen_y, index + 1, "#34c759", "M")

    def draw_alignment_marker(self, x: float, y: float, index: int, color: str, prefix: str) -> None:
        if self.alignment_canvas is None:
            return

        radius = 7
        self.alignment_canvas.create_oval(
            x - radius,
            y - radius,
            x + radius,
            y + radius,
            outline=color,
            width=3,
        )
        self.alignment_canvas.create_text(
            x + 14,
            y - 12,
            text=f"{prefix}{index}",
            fill=color,
            font=("Segoe UI", 10, "bold"),
            anchor=tk.W,
        )

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

    def redraw_mono_capture_canvas(self) -> None:
        if self.mono_processor_canvas is None or not self.mono_processor_canvas.winfo_exists():
            return

        self.mono_processor_canvas.delete("all")
        if self.captured_mono_image_rgb is None:
            return

        disp_w, disp_h, offset_x, offset_y = self.get_zoomed_image_geometry(
            self.captured_mono_image_rgb,
            self.mono_processor_canvas,
            "mono_processor",
        )
        self.mono_capture_photo = self.draw_zoomed_image(
            self.captured_mono_image_rgb,
            self.mono_processor_canvas,
            "mono_processor",
        )

        scale_x = disp_w / self.captured_mono_image_rgb.shape[1]
        scale_y = disp_h / self.captured_mono_image_rgb.shape[0]
        for idx, point in enumerate(self.mono_sample_points, start=1):
            canvas_x = int(point.x * scale_x) + offset_x
            canvas_y = int(point.y * scale_y) + offset_y
            gray = int(round(point.intensity * 255.0))
            self.mono_processor_canvas.create_oval(
                canvas_x - 6,
                canvas_y - 6,
                canvas_x + 6,
                canvas_y + 6,
                fill=self.rgb_to_hex((gray, gray, gray)),
                outline=point.marker_color,
                width=2,
            )
            self.mono_processor_canvas.create_text(
                canvas_x + 18,
                canvas_y - 8,
                text=str(idx),
                fill=point.marker_color,
                font=("Segoe UI", 10, "bold"),
            )

    def on_mono_capture_canvas_click(self, event: tk.Event) -> None:
        if self.captured_mono_image_rgb is None or self.captured_mono_gray is None or self.mono_processor_canvas is None:
            return

        disp_w, disp_h, offset_x, offset_y = self.get_zoomed_image_geometry(
            self.captured_mono_image_rgb,
            self.mono_processor_canvas,
            "mono_processor",
        )
        if not (offset_x <= event.x < offset_x + disp_w and offset_y <= event.y < offset_y + disp_h):
            return

        src_h, src_w = self.captured_mono_gray.shape[:2]
        rel_x = (event.x - offset_x) / disp_w
        rel_y = (event.y - offset_y) / disp_h
        img_x = min(src_w - 1, max(0, int(rel_x * src_w)))
        img_y = min(src_h - 1, max(0, int(rel_y * src_h)))
        intensity = float(self.captured_mono_gray[img_y, img_x]) / 255.0
        self.mono_sample_points.append(
            MonoSamplePoint(
                x=img_x,
                y=img_y,
                intensity=intensity,
                marker_color=self.get_marker_color(len(self.mono_sample_points)),
            )
        )
        self.redraw_mono_capture_canvas()
        self.update_mono_intensity_plot()

    def remove_mono_capture_point(self, event: tk.Event) -> str:
        if self.captured_mono_image_rgb is None or not self.mono_sample_points:
            return "break"

        hit = self.find_nearest_mono_sample_point_on_canvas(event.x, event.y)
        if hit is None:
            return "break"

        self.mono_sample_points.pop(hit)
        self.redraw_mono_capture_canvas()
        self.update_mono_intensity_plot()
        self.status_var.set("Monokróm marker pont törölve.")
        return "break"

    def find_nearest_mono_sample_point_on_canvas(self, event_x: int, event_y: int) -> int | None:
        if self.captured_mono_image_rgb is None or self.mono_processor_canvas is None:
            return None

        disp_w, disp_h, offset_x, offset_y = self.get_zoomed_image_geometry(
            self.captured_mono_image_rgb,
            self.mono_processor_canvas,
            "mono_processor",
        )
        scale_x = disp_w / self.captured_mono_image_rgb.shape[1]
        scale_y = disp_h / self.captured_mono_image_rgb.shape[0]
        nearest_index = None
        nearest_distance = None
        for index, point in enumerate(self.mono_sample_points):
            canvas_x = point.x * scale_x + offset_x
            canvas_y = point.y * scale_y + offset_y
            distance = float(np.hypot(canvas_x - event_x, canvas_y - event_y))
            if distance <= 12.0 and (nearest_distance is None or distance < nearest_distance):
                nearest_index = index
                nearest_distance = distance
        return nearest_index

    def update_mono_intensity_plot(self) -> None:
        if self.mono_processor_axis is None or self.mono_processor_figure is None or self.mono_processor_plot_canvas is None:
            return

        axis = self.mono_processor_axis
        axis.clear()
        axis.set_title("Pontonkénti intenzitás")
        axis.set_xlabel("Pont")
        axis.set_ylabel("Intenzitás")
        axis.set_ylim(0.0, 1.05)
        axis.grid(True, axis="y", alpha=0.25)

        if self.mono_sample_points:
            xs = np.arange(1, len(self.mono_sample_points) + 1)
            values = [point.intensity for point in self.mono_sample_points]
            colors = [point.marker_color for point in self.mono_sample_points]
            axis.bar(xs, values, color=colors)
            axis.axhspan(min(values), max(values), color="#8a8a8a", alpha=0.18)
            axis.set_xticks(xs)
        else:
            axis.text(
                0.5,
                0.5,
                "A monokróm képen kijelölt pontok intenzitása itt jelenik meg.",
                transform=axis.transAxes,
                ha="center",
                va="center",
                fontsize=10,
            )

        self.mono_processor_figure.tight_layout()
        self.mono_processor_plot_canvas.draw()

    def get_current_mono_intensity_band(self) -> tuple[float, float] | None:
        if len(self.mono_sample_points) < 2:
            return None
        values = [point.intensity for point in self.mono_sample_points]
        return min(values), max(values)

    def apply_mono_intensity_filter(self) -> None:
        if self.captured_mono_gray is None or self.captured_mono_image_rgb is None:
            messagebox.showinfo("Nincs kép", "Előbb rögzíteni kell egy monokróm képet.")
            return

        intensity_band = self.get_current_mono_intensity_band()
        if intensity_band is None:
            messagebox.showinfo("Keves marker", "Az intenzitas szureshez legalabb ket pontot jelolj ki.")
            return

        min_value, max_value = intensity_band
        image_intensity = self.captured_mono_gray.astype(np.float32) / 255.0
        mask = (image_intensity >= min_value) & (image_intensity <= max_value)
        filtered_image = np.zeros_like(self.captured_mono_image_rgb)
        filtered_image[mask] = self.captured_mono_image_rgb[mask]
        source = f"monokróm intenzitás {min_value:.3f}-{max_value:.3f}"
        self.open_filtered_window(
            filtered_image,
            int(np.count_nonzero(mask)),
            source,
            "monokróm intenzitástartomány",
        )

    def clear_mono_points(self) -> None:
        self.mono_sample_points.clear()
        self.redraw_mono_capture_canvas()
        self.update_mono_intensity_plot()
        self.status_var.set("A monokróm pontok törölve lettek.")

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
        filter_label: str | None = None,
    ) -> None:
        if self.filtered_window is not None and self.filtered_window.winfo_exists():
            self.filtered_window.destroy()

        self.filtered_window = tk.Toplevel(self.root)
        self.filtered_window.title("Szűrt kép")
        self.filtered_window.geometry("1120x720")
        self.filtered_window.minsize(900, 620)
        self.filtered_window.protocol("WM_DELETE_WINDOW", self.close_filtered_window)
        self.filtered_image_rgb = filtered_image
        self.filtered_original_rgb = filtered_image.copy()
        self.reset_image_view("filtered")

        container = ttk.Frame(self.filtered_window, padding=12)
        container.pack(fill=tk.BOTH, expand=True)
        container.columnconfigure(0, weight=1)
        container.rowconfigure(2, weight=1)

        if filter_label is None:
            filter_label = f"teljes {WAVELENGTH_START}-{WAVELENGTH_END} nm tartomány"
        info_text = f"Szűrés: {filter_label}, forrás: {filter_source}, találat: {matched_pixels} pixel"
        ttk.Label(container, text=info_text).grid(row=0, column=0, sticky="w", pady=(0, 10))

        toolbar = ttk.Frame(container)
        toolbar.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        ttk.Button(toolbar, text="Nem fekete pixelek fehérítése", command=self.whiten_filtered_pixels).grid(
            row=0, column=0, padx=(0, 8)
        )
        ttk.Button(toolbar, text="Színes szűrt kép visszaállítása", command=self.restore_colored_filtered_pixels).grid(
            row=0, column=1, padx=(0, 8)
        )

        self.filtered_canvas = tk.Canvas(
            container,
            width=self.capture_size[0],
            height=self.capture_size[1],
            bg="#1f1f1f",
            highlightthickness=1,
            highlightbackground="#707070",
        )
        self.filtered_canvas.grid(row=2, column=0, sticky="nsew")
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

    def whiten_filtered_pixels(self) -> None:
        if self.filtered_image_rgb is None:
            return

        mask = np.any(self.filtered_image_rgb > 0, axis=2)
        self.filtered_image_rgb[mask] = (255, 255, 255)
        self.redraw_filtered_canvas()
        self.status_var.set("A szűrt képen a nem fekete pixelek fehérre lettek állítva.")

    def restore_colored_filtered_pixels(self) -> None:
        if self.filtered_original_rgb is None:
            return

        self.filtered_image_rgb = self.filtered_original_rgb.copy()
        self.redraw_filtered_canvas()
        self.status_var.set("A szűrt kép visszaállt az eredeti színes nézetre.")

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
        self.filtered_original_rgb = None
        self.filtered_photo = None
        self.image_pan_start = None

    def on_processor_window_close(self) -> None:
        if self.processor_window is not None and self.processor_window.winfo_exists():
            self.processor_window.destroy()

        self.processor_window = None
        self.processor_notebook = None
        self.processor_canvas = None
        self.processor_figure = None
        self.processor_axis = None
        self.processor_plot_canvas = None
        self.processor_marker_annotation = None
        self.processor_marker_artist = None
        self.processor_marker_line = None
        self.processor_plot_connection_id = None
        self.processor_lines = []
        self.mono_processor_window = None
        self.mono_processor_canvas = None
        self.mono_processor_figure = None
        self.mono_processor_axis = None
        self.mono_processor_plot_canvas = None
        self.mono_capture_photo = None
        self.alignment_canvas = None
        self.alignment_photo = None
        self.reset_alignment()

    def on_mono_processor_window_close(self) -> None:
        if self.mono_processor_window is not None and self.mono_processor_window.winfo_exists():
            self.mono_processor_window.destroy()

        self.mono_processor_window = None
        self.mono_processor_canvas = None
        self.mono_processor_figure = None
        self.mono_processor_axis = None
        self.mono_processor_plot_canvas = None
        self.mono_capture_photo = None
        if self.image_pan_start is not None and self.image_pan_start[0] == "mono_processor":
            self.image_pan_start = None

    def on_close(self) -> None:
        if self.preview_after_id is not None:
            self.root.after_cancel(self.preview_after_id)
            self.preview_after_id = None

        self.close_fullscreen_spectrum_window()
        self.close_filtered_window()
        self.on_mono_processor_window_close()
        self.on_processor_window_close()
        self.release_camera()
        self.release_mono_camera()
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
