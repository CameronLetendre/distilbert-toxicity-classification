"""Transparent, click-through, always-on-top overlay for drawing bbox + label
on top of the live game window (Windows)."""
import ctypes
import tkinter as tk

GWL_EXSTYLE = -20
WS_EX_LAYERED = 0x00080000
WS_EX_TRANSPARENT = 0x00000020

LABEL_COLORS = {
    "E": "#FF3030",
    "I": "#FF8C00",
    "A": "#FFD700",
    "O": "#00C800",
}

_TRANSPARENT_KEY = "#000001"


class ScreenOverlay:
    def __init__(self):
        self.root = tk.Tk()
        self.root.attributes("-fullscreen", True)
        self.root.attributes("-topmost", True)
        self.root.attributes("-transparentcolor", _TRANSPARENT_KEY)
        self.canvas = tk.Canvas(
            self.root,
            bg=_TRANSPARENT_KEY,
            highlightthickness=0,
            borderwidth=0,
        )
        self.canvas.pack(fill="both", expand=True)
        self.root.update_idletasks()

        hwnd = ctypes.windll.user32.GetParent(self.root.winfo_id())
        styles = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
        ctypes.windll.user32.SetWindowLongW(
            hwnd, GWL_EXSTYLE, styles | WS_EX_LAYERED | WS_EX_TRANSPARENT
        )
        self.root.update()

    def update(self, results, offset_x, offset_y):
        self.canvas.delete("all")
        for (_, _, bbox), (label, conf) in results:
            x1, y1, x2, y2 = bbox
            sx1 = int(offset_x + x1)
            sy1 = int(offset_y + y1)
            sx2 = int(offset_x + x2)
            sy2 = int(offset_y + y2)
            color = LABEL_COLORS.get(label, "#FFFFFF")
            self.canvas.create_rectangle(
                sx1, sy1, sx2, sy2, outline=color, width=2
            )
            self.canvas.create_text(
                sx1, max(sy1 - 10, 0),
                text=f"{label} {conf:.0%}",
                fill=color,
                anchor="w",
                font=("Segoe UI", 10, "bold"),
            )
        self.root.update_idletasks()
        self.root.update()

    def clear(self):
        self.canvas.delete("all")
        self.root.update_idletasks()
        self.root.update()

    def close(self):
        try:
            self.root.destroy()
        except tk.TclError:
            pass
