import numpy as np
import tkinter as tk
from tkinter import Canvas, Button
from PIL import Image, ImageTk, ImageEnhance


def normalize_image_for_display(img_16bit: np.ndarray) -> Image.Image:
    """
    Normaliza una imagen de 16 bits a 8 bits para visualización.

    Parameters
    ----------
    img_16bit : np.ndarray
        Imagen original (16 bits).

    Returns
    -------
    PIL.Image.Image
        Imagen lista para mostrar en pantalla (8 bits).
    """
    # Escalar entre 0 y 255 (evita dividir entre 0)
    min_val = np.min(img_16bit)
    max_val = np.max(img_16bit)
    if max_val - min_val == 0:
        norm_img = np.zeros_like(img_16bit, dtype=np.uint8)
    else:
        norm_img = ((img_16bit - min_val) / (max_val - min_val) * 255).astype(np.uint8)

    return Image.fromarray(norm_img)


class ROISelector(tk.Tk):
    def __init__(self, image, display_max_size=800):
        super().__init__()
        # GUI Title
        self.title("Region of Interest")

        # Open image
        self.image = normalize_image_for_display(image)
        self.original_width, self.original_height = self.image.size

        # Geometry of the window
        self.display_max_size = display_max_size
        self.scale = min(self.display_max_size / self.original_width, self.display_max_size / self.original_height)

        self.display_width = int(self.original_width * self.scale)
        self.display_height = int(self.original_height * self.scale)

        self.display_image = self.image.resize(
            (self.display_width, self.display_height),
            Image.Resampling.LANCZOS
        )
        self.tk_image = ImageTk.PhotoImage(self.display_image)

        # Canvas para mostrar imagen
        self.canvas = Canvas(self, width=self.display_width, height=self.display_height)
        self.canvas.pack()
        self.canvas_image_id = self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)

        # ROI
        self.start_x = 0
        self.start_y = 0
        self.rect = self.canvas.create_rectangle(
            self.start_x,
            self.start_y,
            self.start_x,
            self.start_y,
            outline="red",
            width=3
        )
        self.roi_coords = (0, 0, 0, 0)

        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)

        self.button = Button(self, text="Set ROI", command=self.set_roi)
        self.button.pack(pady=10)

        self.adjust_button = Button(self, text="Adjust Brightness/Contrast", command=self.open_adjust_window)
        self.adjust_button.pack(pady=10)

        # State variables for brightness and contrast
        self.brightness = 1.0
        self.contrast = 1.0


    def on_mouse_down(self, event):
        self.start_x = self.canvas.canvasx(event.x)
        self.start_y = self.canvas.canvasy(event.y)

        if self.rect:
            self.canvas.delete(self.rect)

        self.rect = self.canvas.create_rectangle(
            self.start_x,
            self.start_y,
            self.start_x,
            self.start_y,
            outline="red",
            width=3
        )

    def on_mouse_drag(self, event):
        curr_x = self.canvas.canvasx(event.x)
        curr_y = self.canvas.canvasy(event.y)
        self.canvas.coords(self.rect, self.start_x, self.start_y, curr_x, curr_y)

    def on_mouse_up(self, event):
        end_x = self.canvas.canvasx(event.x)
        end_y = self.canvas.canvasy(event.y)

        # Sort coordinates to ensure correct rectangle
        x0, x1 = sorted([self.start_x, end_x])
        y0, y1 = sorted([self.start_y, end_y])

        # Convert to original scale
        real_x0 = int(x0 / self.scale)
        real_x1 = int(x1 / self.scale)
        real_y0 = int(y0 / self.scale)
        real_y1 = int(y1 / self.scale)

        self.roi_coords = (real_y0, real_y1, real_x0, real_x1)


    def set_roi(self):
        y0, y1, x0, x1 = self.roi_coords
        # ROI coordinates saved in self.roi_coords - no need to print


    def open_adjust_window(self):
        window = tk.Toplevel(self)
        window.title("Adjust Brightness/Contrast")

        tk.Label(window, text="Brightness").pack()
        brightness_slider = tk.Scale(
            window,
            from_=0.1,
            to=10.0,
            resolution=0.1,
            orient=tk.HORIZONTAL, length=1000,
            command=self.update_brightness
        )
        brightness_slider.set(self.brightness)
        brightness_slider.pack()


        tk.Label(window, text="Contrast").pack()
        contrast_slider = tk.Scale(
            window,
            from_=0.1,
            to=10.0,
            resolution=0.1,
            orient=tk.HORIZONTAL,
            length=1000,
            command=self.update_contrast
        )
        contrast_slider.set(self.contrast)
        contrast_slider.pack()


    def update_brightness(self, val):
        self.brightness = float(val)
        self.refresh_display_image()


    def update_contrast(self, val):
        self.contrast = float(val)
        self.refresh_display_image()


    def refresh_display_image(self):
        img = self.display_image.copy()

        # Apply brightness and contrast
        enhancer_b = ImageEnhance.Brightness(img)
        img = enhancer_b.enhance(self.brightness)

        enhancer_c = ImageEnhance.Contrast(img)
        img = enhancer_c.enhance(self.contrast)

        self.tk_image = ImageTk.PhotoImage(img)
        self.canvas.itemconfig(self.canvas_image_id, image=self.tk_image)


def cropImage(image: np.ndarray):
    app = ROISelector(image)
    app.mainloop()
    return app.roi_coords

