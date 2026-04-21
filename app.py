from __future__ import annotations

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


APP_TITLE = "WindMon Tekercseles Kamera Monitor"
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
        self.status_var = tk.StringVar(value="Kamera keresese...")

        self.available_cameras: list[tuple[int, str]] = []
        self.current_camera_index: int | None = None
        self.capture: cv2.VideoCapture | None = None
        self.latest_frame_bgr: np.ndarray | None = None
        self.captured_image_rgb: np.ndarray | None = None
        self.sample_points: list[SamplePoint] = []

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

        ttk.Label(control_frame, text="Aktiv kamera:").grid(row=0, column=0, padx=(0, 8))
        self.camera_combo = ttk.Combobox(
            control_frame,
            textvariable=self.camera_var,
            state="readonly",
            width=35,
        )
        self.camera_combo.grid(row=0, column=1, padx=(0, 8))
        self.camera_combo.bind("<<ComboboxSelected>>", self.on_camera_selected)

        ttk.Button(control_frame, text="Kamerak frissitese", command=self.refresh_cameras).grid(
            row=0, column=2, padx=(0, 8)
        )
        ttk.Button(control_frame, text="Kep rogzitese", command=self.capture_image).grid(
            row=0, column=3, padx=(0, 8)
        )
        ttk.Button(control_frame, text="Feldolgozo ablak megnyitasa", command=self.open_processor_window).grid(
            row=0, column=4, sticky="e"
        )

        ttk.Label(control_frame, textvariable=self.status_var).grid(
            row=1,
            column=0,
            columnspan=5,
            sticky="w",
            pady=(10, 0),
        )

        preview_group = ttk.LabelFrame(root_frame, text="Elokep", padding=10)
        preview_group.grid(row=1, column=0, sticky="nsew")
        preview_group.columnconfigure(0, weight=1)
        preview_group.rowconfigure(1, weight=1)

        ttk.Label(
            preview_group,
            text="Elokep a kivalasztott kamerarol. A rogzitett kep kulon feldolgozo ablakban jelenik meg.",
        ).grid(row=0, column=0, sticky="w", pady=(0, 8))

        self.preview_label = ttk.Label(preview_group)
        self.preview_label.grid(row=1, column=0, sticky="nsew")

    def _build_menubar(self) -> None:
        menubar = tk.Menu(self.root)
        file_menu = tk.Menu(menubar, tearoff=False)
        file_menu.add_command(label="RGB kep mentese...", command=self.save_captured_image)
        file_menu.add_command(label="Feldolgozo ablak megnyitasa", command=self.open_processor_window)
        file_menu.add_separator()
        file_menu.add_command(label="Kilepes", command=self.on_close)

        camera_menu = tk.Menu(menubar, tearoff=False)
        camera_menu.add_command(label="Kamerak ujrakeresese", command=self.refresh_cameras)
        camera_menu.add_cascade(label="Kamera valasztasa", menu=self.menu_camera)

        menubar.add_cascade(label="Fajl", menu=file_menu)
        menubar.add_cascade(label="Kamera", menu=camera_menu)
        self.root.config(menu=menubar)

    def _build_processor_window(self) -> None:
        if self.processor_window is not None and self.processor_window.winfo_exists():
            self.processor_window.lift()
            return

        self.processor_window = tk.Toplevel(self.root)
        self.processor_window.title("Feldolgozo ablak")
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
        toolbar.columnconfigure(4, weight=1)
        ttk.Button(toolbar, text="Pontok torlese", command=self.clear_points).grid(row=0, column=0, padx=(0, 8))
        ttk.Button(toolbar, text="RGB kep mentese", command=self.save_captured_image).grid(row=0, column=1, padx=(0, 8))
        ttk.Button(toolbar, text="Spektrum teljes kepernyon", command=self.open_fullscreen_spectrum_window).grid(
            row=0, column=2, padx=(0, 8)
        )
        ttk.Button(toolbar, text="Szuro alkalmazasa", command=self.apply_marker_range_filter).grid(
            row=0, column=3, padx=(0, 8)
        )
        ttk.Label(
            toolbar,
            text="Kattints a kepre pontokhoz, a grafikonra markeres leolvasashoz.",
        ).grid(row=0, column=4, sticky="e")

        image_group = ttk.LabelFrame(container, text="Rogzitett RGB kep", padding=10)
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

        plot_group = ttk.LabelFrame(container, text="RGB pontok becsult spektruma", padding=10)
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
            messagebox.showinfo("Nincs rogzitett kep", "Elobb keszits egy kepet a kamerabol.")
            return

        if self.fullscreen_window is not None and self.fullscreen_window.winfo_exists():
            self.fullscreen_window.lift()
            self.refresh_fullscreen_plot()
            return

        self.fullscreen_window = tk.Toplevel(self.root)
        self.fullscreen_window.title("Teljes kepernyos spektrum")
        self.fullscreen_window.attributes("-fullscreen", True)
        self.fullscreen_window.protocol("WM_DELETE_WINDOW", self.close_fullscreen_spectrum_window)

        container = ttk.Frame(self.fullscreen_window, padding=10)
        container.pack(fill=tk.BOTH, expand=True)
        container.columnconfigure(0, weight=1)
        container.rowconfigure(1, weight=1)

        toolbar = ttk.Frame(container)
        toolbar.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        toolbar.columnconfigure(2, weight=1)
        ttk.Button(toolbar, text="Kilepes a teljes kepernyobol", command=self.close_fullscreen_spectrum_window).grid(
            row=0, column=0, padx=(0, 8)
        )
        ttk.Button(toolbar, text="Pontok torlese", command=self.clear_points).grid(row=0, column=1, padx=(0, 8))
        ttk.Label(toolbar, text="Kattints a gorbekre a hullamhossz es intenzitas leolvasasahoz.").grid(
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
        self.status_var.set("Kamerak keresese folyamatban...")
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
                    self.status_var.set(f"Kamera elerheto: {label}")
                else:
                    first_index, first_label = self.available_cameras[0]
                    self.select_camera(first_index, first_label)
        else:
            self.camera_var.set("")
            self.status_var.set("Nem talalhato elerheto kamera.")
            self.release_camera()

    def discover_cameras(self, max_devices: int = 8) -> list[tuple[int, str]]:
        cameras: list[tuple[int, str]] = []
        for index in range(max_devices):
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            if cap is None or not cap.isOpened():
                continue

            ok, _ = cap.read()
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            cap.release()

            if not ok:
                continue

            role = "Laptop webkamera" if index == 0 else f"USB / kulso kamera {index}"
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
            self.status_var.set(f"A kamera nem nyithato meg: {label}")
            return

        self.capture = cap
        self.current_camera_index = index
        self.camera_var.set(label)
        self.status_var.set(f"Aktiv kamera: {label}")

    def release_camera(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def update_preview_loop(self) -> None:
        if self.capture is not None and self.capture.isOpened():
            ok, frame_bgr = self.capture.read()
            if ok:
                self.latest_frame_bgr = frame_bgr
                preview_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                preview_image = self.resize_for_display(preview_rgb, self.preview_size)
                self.preview_photo = ImageTk.PhotoImage(Image.fromarray(preview_image))
                self.preview_label.configure(image=self.preview_photo)
            else:
                self.status_var.set("Nem sikerult kepkockat olvasni a kamerabol.")

        self.preview_after_id = self.root.after(30, self.update_preview_loop)

    def resize_for_display(self, image_rgb: np.ndarray, size: tuple[int, int]) -> np.ndarray:
        target_w, target_h = size
        src_h, src_w = image_rgb.shape[:2]
        scale = min(target_w / src_w, target_h / src_h)
        resized = cv2.resize(
            image_rgb,
            (max(1, int(src_w * scale)), max(1, int(src_h * scale))),
            interpolation=cv2.INTER_AREA,
        )

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

    def capture_image(self) -> None:
        if self.latest_frame_bgr is None:
            messagebox.showwarning("Nincs kep", "Eloszor nyiss meg egy kamerat, es varj elokepre.")
            return

        self.captured_image_rgb = cv2.cvtColor(self.latest_frame_bgr.copy(), cv2.COLOR_BGR2RGB)
        self.sample_points.clear()
        self.open_processor_window()
        self.redraw_capture_canvas()
        self.update_spectrum_plot()
        self.status_var.set("RGB kep rogzitve. A feldolgozo ablakban helyezhetsz el pontokat.")

    def open_processor_window(self) -> None:
        if self.captured_image_rgb is None:
            messagebox.showinfo("Nincs rogzitett kep", "Elobb keszits egy kepet a kamerabol.")
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
                text="Itt jelenik meg a rogzitett RGB kep",
                fill="#d0d0d0",
                font=("Segoe UI", 16, "bold"),
            )
            return

        display_image = self.resize_for_display(self.captured_image_rgb, self.capture_size)
        self.capture_photo = ImageTk.PhotoImage(Image.fromarray(display_image))
        self.processor_canvas.create_image(0, 0, anchor=tk.NW, image=self.capture_photo)

        disp_w, disp_h, offset_x, offset_y = self.calculate_display_geometry(self.captured_image_rgb, self.capture_size)
        scale_x = disp_w / self.captured_image_rgb.shape[1]
        scale_y = disp_h / self.captured_image_rgb.shape[0]

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

        disp_w, disp_h, offset_x, offset_y = self.calculate_display_geometry(self.captured_image_rgb, self.capture_size)

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

    def estimate_spectrum(self, rgb: tuple[int, int, int]) -> np.ndarray:
        wavelengths = np.linspace(WAVELENGTH_START, WAVELENGTH_END, SPECTRUM_SAMPLES)
        r, g, b = [channel / 255.0 for channel in rgb]
        luminance = np.clip(0.2126 * r + 0.7152 * g + 0.0722 * b, 0.05, 1.0)

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

        axis.clear()
        axis.set_title("Pontonkenti spektrum")
        axis.set_xlabel("Hullamhossz (nm)")
        axis.set_ylabel("Intenzitas")
        axis.set_xlim(WAVELENGTH_START, WAVELENGTH_END)
        axis.set_ylim(0.0, 1.05)
        axis.grid(True, alpha=0.25)

        wavelengths = np.linspace(WAVELENGTH_START, WAVELENGTH_END, SPECTRUM_SAMPLES)
        lines: list = []
        for index, point in enumerate(self.sample_points, start=1):
            line, = axis.plot(
                wavelengths,
                point.spectrum,
                color=point.marker_color,
                linewidth=2.4,
                label=f"Pont {index} RGB{point.rgb}",
            )
            lines.append(line)

        if self.sample_points:
            axis.legend(loc="upper right", fontsize=8)
        else:
            axis.text(
                0.5,
                0.5,
                "A rogzitett kepen elhelyezett pontok spektruma itt jelenik meg.",
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

    def apply_marker_range_filter(self) -> None:
        if self.captured_image_rgb is None:
            messagebox.showinfo("Nincs kep", "Elobb rogziteni kell egy RGB kepet.")
            return

        if len(self.sample_points) < 2:
            messagebox.showinfo("Keves marker", "A szureshez legalabb ket pontot jelolj ki a kepen.")
            return

        marker_spectrums = np.array([point.spectrum for point in self.sample_points], dtype=np.float32)
        min_values = np.min(marker_spectrums, axis=0)
        max_values = np.max(marker_spectrums, axis=0)

        image_spectrum = self.estimate_image_spectrum()
        mask = np.all((image_spectrum >= min_values) & (image_spectrum <= max_values), axis=2)
        filtered_image = np.zeros_like(self.captured_image_rgb)
        filtered_image[mask] = self.captured_image_rgb[mask]

        self.open_filtered_window(filtered_image, int(np.count_nonzero(mask)))

    def estimate_image_spectrum(self) -> np.ndarray:
        if self.captured_image_rgb is None:
            return np.empty((0, 0, 0), dtype=np.float32)

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
    ) -> None:
        if self.filtered_window is not None and self.filtered_window.winfo_exists():
            self.filtered_window.destroy()

        self.filtered_window = tk.Toplevel(self.root)
        self.filtered_window.title("Szurt kep")
        self.filtered_window.geometry("1120x720")
        self.filtered_window.minsize(900, 620)
        self.filtered_window.protocol("WM_DELETE_WINDOW", self.close_filtered_window)

        container = ttk.Frame(self.filtered_window, padding=12)
        container.pack(fill=tk.BOTH, expand=True)
        container.columnconfigure(0, weight=1)
        container.rowconfigure(1, weight=1)

        info_text = f"Szures: teljes {WAVELENGTH_START}-{WAVELENGTH_END} nm tartomany, talalat: {matched_pixels} pixel"
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

        display_image = self.resize_for_display(filtered_image, self.capture_size)
        self.filtered_photo = ImageTk.PhotoImage(Image.fromarray(display_image))
        self.filtered_canvas.create_image(0, 0, anchor=tk.NW, image=self.filtered_photo)
        self.status_var.set(info_text)

    def save_captured_image(self) -> None:
        if self.captured_image_rgb is None:
            messagebox.showinfo("Nincs kep", "Elobb rogziteni kell egy RGB kepet.")
            return

        default_name = f"tekercseles_rgb_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        target = filedialog.asksaveasfilename(
            title="RGB kep mentese",
            defaultextension=".png",
            initialdir=OUTPUT_DIR,
            initialfile=default_name,
            filetypes=[("PNG kep", "*.png"), ("JPEG kep", "*.jpg *.jpeg")],
        )
        if not target:
            return

        image_bgr = cv2.cvtColor(self.captured_image_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(target, image_bgr)
        self.status_var.set(f"RGB kep mentve: {target}")

    def clear_points(self) -> None:
        self.sample_points.clear()
        self.redraw_capture_canvas()
        self.update_spectrum_plot()
        self.status_var.set("A pontok torolve lettek.")

    def close_filtered_window(self) -> None:
        if self.filtered_window is not None and self.filtered_window.winfo_exists():
            self.filtered_window.destroy()

        self.filtered_window = None
        self.filtered_canvas = None
        self.filtered_photo = None

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
