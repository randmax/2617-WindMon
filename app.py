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


@dataclass
class SamplePoint:
    x: int
    y: int
    rgb: tuple[int, int, int]
    hex_color: str
    spectrum: np.ndarray


class CameraRecorderApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("1420x860")
        self.root.minsize(1200, 760)

        OUTPUT_DIR.mkdir(exist_ok=True)

        self.preview_size = (800, 450)
        self.capture_size = (960, 540)
        self.preview_after_id: str | None = None

        self.camera_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="Kamera keresese...")
        self.recording_var = tk.StringVar(value="Nincs aktiv felvetel")

        self.available_cameras: list[tuple[int, str]] = []
        self.current_camera_index: int | None = None
        self.capture: cv2.VideoCapture | None = None
        self.latest_frame_bgr: np.ndarray | None = None
        self.captured_image_rgb: np.ndarray | None = None
        self.recording_writer: cv2.VideoWriter | None = None
        self.recording_path: Path | None = None
        self.sample_points: list[SamplePoint] = []

        self.menu_camera = tk.Menu(self.root, tearoff=False)

        self._build_layout()
        self._build_menubar()
        self._setup_plot()

        self.refresh_cameras()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.after(100, self.update_preview_loop)

    def _build_layout(self) -> None:
        root_frame = ttk.Frame(self.root, padding=12)
        root_frame.pack(fill=tk.BOTH, expand=True)
        root_frame.columnconfigure(0, weight=3)
        root_frame.columnconfigure(1, weight=2)
        root_frame.rowconfigure(1, weight=1)

        control_frame = ttk.Frame(root_frame)
        control_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 12))
        control_frame.columnconfigure(7, weight=1)

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
        ttk.Button(control_frame, text="Felvetel inditasa", command=self.start_recording).grid(
            row=0, column=3, padx=(0, 8)
        )
        ttk.Button(control_frame, text="Felvetel leallitasa", command=self.stop_recording).grid(
            row=0, column=4, padx=(0, 8)
        )
        ttk.Button(control_frame, text="RGB kep rogzitese", command=self.capture_image).grid(
            row=0, column=5, padx=(0, 8)
        )
        ttk.Button(control_frame, text="Pontok torlese", command=self.clear_points).grid(
            row=0, column=6, padx=(0, 8)
        )

        ttk.Label(control_frame, textvariable=self.status_var).grid(row=1, column=0, columnspan=4, sticky="w", pady=(10, 0))
        ttk.Label(control_frame, textvariable=self.recording_var).grid(row=1, column=4, columnspan=4, sticky="e", pady=(10, 0))

        preview_group = ttk.LabelFrame(root_frame, text="Elokep es rogzitett RGB kep", padding=10)
        preview_group.grid(row=1, column=0, sticky="nsew", padx=(0, 10))
        preview_group.columnconfigure(0, weight=1)
        preview_group.columnconfigure(1, weight=1)
        preview_group.rowconfigure(1, weight=1)

        ttk.Label(preview_group, text="Elokep").grid(row=0, column=0, sticky="w", pady=(0, 8))
        ttk.Label(preview_group, text="Rogzitett kep pontkijelolessel").grid(row=0, column=1, sticky="w", pady=(0, 8))

        self.preview_label = ttk.Label(preview_group)
        self.preview_label.grid(row=1, column=0, sticky="nsew", padx=(0, 10))

        self.capture_canvas = tk.Canvas(
            preview_group,
            width=self.capture_size[0],
            height=self.capture_size[1],
            bg="#1f1f1f",
            highlightthickness=1,
            highlightbackground="#707070",
            cursor="crosshair",
        )
        self.capture_canvas.grid(row=1, column=1, sticky="nsew")
        self.capture_canvas.bind("<Button-1>", self.on_capture_canvas_click)

        plot_group = ttk.LabelFrame(root_frame, text="RGB pontok becsult spektruma", padding=10)
        plot_group.grid(row=1, column=1, sticky="nsew")
        plot_group.columnconfigure(0, weight=1)
        plot_group.rowconfigure(0, weight=1)

        self.plot_host = ttk.Frame(plot_group)
        self.plot_host.grid(row=0, column=0, sticky="nsew")

    def _build_menubar(self) -> None:
        menubar = tk.Menu(self.root)
        file_menu = tk.Menu(menubar, tearoff=False)
        file_menu.add_command(label="RGB kep mentese...", command=self.save_captured_image)
        file_menu.add_separator()
        file_menu.add_command(label="Kilepes", command=self.on_close)

        camera_menu = tk.Menu(menubar, tearoff=False)
        camera_menu.add_command(label="Kamerak ujrakeresese", command=self.refresh_cameras)
        camera_menu.add_cascade(label="Kamera valasztasa", menu=self.menu_camera)

        menubar.add_cascade(label="Fajl", menu=file_menu)
        menubar.add_cascade(label="Kamera", menu=camera_menu)
        self.root.config(menu=menubar)

    def _setup_plot(self) -> None:
        self.figure = Figure(figsize=(5.5, 4.8), dpi=100)
        self.axis = self.figure.add_subplot(111)
        self.axis.set_title("Pontonkenti spektrum")
        self.axis.set_xlabel("Hullamhossz (nm)")
        self.axis.set_ylabel("Intenzitas")
        self.axis.set_xlim(WAVELENGTH_START, WAVELENGTH_END)
        self.axis.set_ylim(0.0, 1.05)
        self.axis.grid(True, alpha=0.25)

        self.plot_canvas = FigureCanvasTkAgg(self.figure, master=self.plot_host)
        self.plot_canvas.draw()
        self.plot_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

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

                if self.recording_writer is not None:
                    self.recording_writer.write(frame_bgr)
            else:
                self.status_var.set("Nem sikerult kepkockat olvasni a kamerabol.")

        self.preview_after_id = self.root.after(30, self.update_preview_loop)

    def resize_for_display(self, image_rgb: np.ndarray, size: tuple[int, int]) -> np.ndarray:
        target_w, target_h = size
        src_h, src_w = image_rgb.shape[:2]
        scale = min(target_w / src_w, target_h / src_h)
        resized = cv2.resize(image_rgb, (max(1, int(src_w * scale)), max(1, int(src_h * scale))), interpolation=cv2.INTER_AREA)

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
        self.redraw_capture_canvas()
        self.update_spectrum_plot()
        self.status_var.set("RGB kep rogzitve. Kattints a kepre pontok hozzaadasahoz.")

    def redraw_capture_canvas(self) -> None:
        self.capture_canvas.delete("all")

        if self.captured_image_rgb is None:
            self.capture_canvas.create_text(
                self.capture_size[0] // 2,
                self.capture_size[1] // 2,
                text="Itt jelenik meg a rogzitett RGB kep",
                fill="#d0d0d0",
                font=("Segoe UI", 16, "bold"),
            )
            return

        display_image = self.resize_for_display(self.captured_image_rgb, self.capture_size)
        self.capture_photo = ImageTk.PhotoImage(Image.fromarray(display_image))
        self.capture_canvas.create_image(0, 0, anchor=tk.NW, image=self.capture_photo)

        disp_w, disp_h, offset_x, offset_y = self.calculate_display_geometry(self.captured_image_rgb, self.capture_size)
        scale_x = disp_w / self.captured_image_rgb.shape[1]
        scale_y = disp_h / self.captured_image_rgb.shape[0]

        for idx, point in enumerate(self.sample_points, start=1):
            canvas_x = int(point.x * scale_x) + offset_x
            canvas_y = int(point.y * scale_y) + offset_y
            self.capture_canvas.create_oval(
                canvas_x - 6,
                canvas_y - 6,
                canvas_x + 6,
                canvas_y + 6,
                fill=point.hex_color,
                outline="#ffffff",
                width=2,
            )
            self.capture_canvas.create_text(
                canvas_x + 12,
                canvas_y - 10,
                text=str(idx),
                fill=point.hex_color,
                font=("Segoe UI", 10, "bold"),
                anchor=tk.NW,
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
            hex_color=self.rgb_to_hex(rgb),
            spectrum=self.estimate_spectrum(rgb),
        )
        self.sample_points.append(point)
        self.redraw_capture_canvas()
        self.update_spectrum_plot()

    def estimate_spectrum(self, rgb: tuple[int, int, int]) -> np.ndarray:
        wavelengths = np.linspace(WAVELENGTH_START, WAVELENGTH_END, SPECTRUM_SAMPLES)
        r, g, b = [channel / 255.0 for channel in rgb]
        luminance = np.clip(0.2126 * r + 0.7152 * g + 0.0722 * b, 0.05, 1.0)

        # RGB kepbol csak becsult spektrum allithato elo.
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
        self.axis.clear()
        self.axis.set_title("Pontonkenti spektrum")
        self.axis.set_xlabel("Hullamhossz (nm)")
        self.axis.set_ylabel("Intenzitas")
        self.axis.set_xlim(WAVELENGTH_START, WAVELENGTH_END)
        self.axis.set_ylim(0.0, 1.05)
        self.axis.grid(True, alpha=0.25)

        wavelengths = np.linspace(WAVELENGTH_START, WAVELENGTH_END, SPECTRUM_SAMPLES)
        for index, point in enumerate(self.sample_points, start=1):
            self.axis.plot(
                wavelengths,
                point.spectrum,
                color=point.hex_color,
                linewidth=2.2,
                label=f"Pont {index} RGB{point.rgb}",
            )

        if self.sample_points:
            self.axis.legend(loc="upper right", fontsize=8)
        else:
            self.axis.text(
                0.5,
                0.5,
                "A rogzitett kepen elhelyezett pontok spektruma itt jelenik meg.",
                transform=self.axis.transAxes,
                ha="center",
                va="center",
                fontsize=10,
            )

        self.figure.tight_layout()
        self.plot_canvas.draw()

    def start_recording(self) -> None:
        if self.latest_frame_bgr is None:
            messagebox.showwarning("Nincs kamera", "Eloszor valassz kamerat, hogy elinduljon az elokep.")
            return

        if self.recording_writer is not None:
            messagebox.showinfo("Felvetel", "A felvetel mar folyamatban van.")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.recording_path = OUTPUT_DIR / f"tekercseles_{timestamp}.avi"
        height, width = self.latest_frame_bgr.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(str(self.recording_path), fourcc, 20.0, (width, height))

        if not writer.isOpened():
            messagebox.showerror("Hiba", "A video fajl nem hozhato letre.")
            return

        self.recording_writer = writer
        self.recording_var.set(f"Felvetel aktiv: {self.recording_path.name}")

    def stop_recording(self) -> None:
        if self.recording_writer is None:
            return

        self.recording_writer.release()
        saved_path = self.recording_path
        self.recording_writer = None
        self.recording_path = None
        self.recording_var.set("Nincs aktiv felvetel")
        if saved_path is not None:
            self.status_var.set(f"Video mentve ide: {saved_path}")

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

    def on_close(self) -> None:
        if self.preview_after_id is not None:
            self.root.after_cancel(self.preview_after_id)
            self.preview_after_id = None

        self.stop_recording()
        self.release_camera()
        self.root.destroy()

    @staticmethod
    def rgb_to_hex(rgb: tuple[int, int, int]) -> str:
        return "#{:02x}{:02x}{:02x}".format(*rgb)


def main() -> None:
    root = tk.Tk()
    style = ttk.Style(root)
    if "vista" in style.theme_names():
        style.theme_use("vista")
    app = CameraRecorderApp(root)
    app.redraw_capture_canvas()
    app.update_spectrum_plot()
    root.mainloop()


if __name__ == "__main__":
    main()
