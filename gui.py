"""
Tkinter GUI — Keyword Spotting Client
Talks to the FastAPI server at http://127.0.0.1:8000
"""

import tkinter as tk
from tkinter import font as tkfont
import threading
import requests
import json
import time

SERVER = "http://127.0.0.1:8000"

# ── Colour palette ────────────────────────────────────────────────────────────
BG          = "#0f1117"
SURFACE     = "#1a1d27"
BORDER      = "#2a2d3a"
ACCENT_IDLE = "#3b82f6"      # blue  – "Begin"
ACCENT_LIVE = "#ef4444"      # red   – "End"
TEXT_PRI    = "#f1f5f9"
TEXT_SEC    = "#64748b"
TAG_COLORS  = {              # per-keyword accent pills
    "marvin": "#a855f7",
    "go":     "#22c55e",
    "stop":   "#ef4444",
    "left":   "#f97316",
    "right":  "#f97316",
    "yes":    "#22c55e",
    "no":     "#ef4444",
}
DEFAULT_TAG = "#3b82f6"

FONT_MONO = ("Courier New", 11)


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Keyword Spotter")
        self.configure(bg=BG)
        self.resizable(False, False)

        self._listening   = False
        self._sse_thread  = None
        self._sse_stop    = threading.Event()

        self._build_ui()
        self._check_server()          # non-blocking connectivity check

    # ── UI construction ────────────────────────────────────────────────────────
    def _build_ui(self):
        pad = dict(padx=20, pady=14)

        # ── Header ────────────────────────────────────────────────────────────
        hdr = tk.Frame(self, bg=BG)
        hdr.pack(fill="x", **pad)

        tk.Label(
            hdr, text="🎙  Keyword Spotter",
            bg=BG, fg=TEXT_PRI,
            font=("Courier New", 18, "bold"),
        ).pack(side="left")

        self._status_dot = tk.Label(hdr, text="●", bg=BG, fg=TEXT_SEC, font=("Courier New", 18))
        self._status_dot.pack(side="right")
        self._status_lbl = tk.Label(hdr, text="idle", bg=BG, fg=TEXT_SEC,
                                    font=("Courier New", 11))
        self._status_lbl.pack(side="right", padx=(0, 6))

        # ── Divider ───────────────────────────────────────────────────────────
        tk.Frame(self, bg=BORDER, height=1).pack(fill="x", padx=20)

        # ── Toggle button ─────────────────────────────────────────────────────
        btn_frame = tk.Frame(self, bg=BG)
        btn_frame.pack(pady=22)

        self._btn = tk.Button(
            btn_frame,
            text="▶  Begin Listening",
            command=self._toggle,
            bg=ACCENT_IDLE, fg="white",
            activebackground="#2563eb", activeforeground="white",
            relief="flat", cursor="hand2",
            font=("Courier New", 13, "bold"),
            padx=28, pady=10,
            bd=0,
        )
        self._btn.pack()

        # ── Last detected keyword ──────────────────────────────────────────────
        kw_frame = tk.Frame(self, bg=SURFACE, bd=0, highlightthickness=1,
                            highlightbackground=BORDER)
        kw_frame.pack(fill="x", padx=20, pady=(0, 14))

        tk.Label(kw_frame, text="Last detected", bg=SURFACE, fg=TEXT_SEC,
                 font=("Courier New", 9)).pack(anchor="w", padx=14, pady=(10, 2))

        self._kw_label = tk.Label(
            kw_frame, text="—",
            bg=SURFACE, fg=TEXT_PRI,
            font=("Courier New", 28, "bold"),
        )
        self._kw_label.pack(anchor="w", padx=14)

        self._conf_label = tk.Label(
            kw_frame, text="",
            bg=SURFACE, fg=TEXT_SEC,
            font=("Courier New", 10),
        )
        self._conf_label.pack(anchor="w", padx=14, pady=(0, 10))

        # ── Log ───────────────────────────────────────────────────────────────
        log_frame = tk.Frame(self, bg=BG)
        log_frame.pack(fill="both", expand=True, padx=20, pady=(0, 20))

        tk.Label(log_frame, text="Detection log", bg=BG, fg=TEXT_SEC,
                 font=("Courier New", 9)).pack(anchor="w")

        txt_frame = tk.Frame(log_frame, bg=SURFACE, highlightthickness=1,
                             highlightbackground=BORDER)
        txt_frame.pack(fill="both", expand=True, pady=(4, 0))

        self._log = tk.Text(
            txt_frame,
            bg=SURFACE, fg=TEXT_PRI,
            font=FONT_MONO,
            relief="flat", bd=0,
            width=50, height=14,
            wrap="word",
            state="disabled",
            insertbackground=TEXT_PRI,
        )
        sb = tk.Scrollbar(txt_frame, command=self._log.yview, bg=SURFACE,
                          troughcolor=SURFACE, bd=0, width=8)
        self._log.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self._log.pack(side="left", fill="both", expand=True, padx=8, pady=6)

        # configure tag styles for keywords
        for kw, color in TAG_COLORS.items():
            self._log.tag_config(kw, foreground=color, font=("Courier New", 11, "bold"))
        self._log.tag_config("_default", foreground=DEFAULT_TAG,
                             font=("Courier New", 11, "bold"))
        self._log.tag_config("ts", foreground=TEXT_SEC)

    # ── Server health check ───────────────────────────────────────────────────
    def _check_server(self):
        def _ping():
            try:
                r = requests.get(f"{SERVER}/status", timeout=3)
                if r.status_code == 200:
                    data = r.json()
                    self._listening = data.get("is_listening", False)
                    self.after(0, self._sync_ui)
                    self._log_line("system", "Connected to server ✓")
                    if self._listening:
                        self._start_sse()
            except Exception as e:
                self._log_line("system", f"Server unreachable: {e}")

        threading.Thread(target=_ping, daemon=True).start()

    # ── Toggle listening ──────────────────────────────────────────────────────
    def _toggle(self):
        self._btn.configure(state="disabled")
        if not self._listening:
            threading.Thread(target=self._do_start, daemon=True).start()
        else:
            threading.Thread(target=self._do_stop, daemon=True).start()

    def _do_start(self):
        try:
            r = requests.post(f"{SERVER}/start", timeout=5)
            if r.status_code == 200:
                self._listening = True
                self._start_sse()
                self._log_line("system", "Listening started")
        except Exception as e:
            self._log_line("system", f"Error: {e}")
        self.after(0, self._sync_ui)

    def _do_stop(self):
        self._sse_stop.set()
        try:
            r = requests.post(f"{SERVER}/stop", timeout=5)
            if r.status_code == 200:
                self._listening = False
                self._log_line("system", "Listening stopped")
        except Exception as e:
            self._log_line("system", f"Error: {e}")
        self.after(0, self._sync_ui)

    # ── Sync button + status indicator ───────────────────────────────────────
    def _sync_ui(self):
        if self._listening:
            self._btn.configure(
                text="■  End Listening",
                bg=ACCENT_LIVE,
                activebackground="#dc2626",
                state="normal",
            )
            self._status_dot.configure(fg="#22c55e")
            self._status_lbl.configure(text="listening", fg="#22c55e")
        else:
            self._btn.configure(
                text="▶  Begin Listening",
                bg=ACCENT_IDLE,
                activebackground="#2563eb",
                state="normal",
            )
            self._status_dot.configure(fg=TEXT_SEC)
            self._status_lbl.configure(text="idle", fg=TEXT_SEC)

    # ── SSE subscriber ────────────────────────────────────────────────────────
    def _start_sse(self):
        self._sse_stop.clear()
        self._sse_thread = threading.Thread(target=self._sse_loop, daemon=True)
        self._sse_thread.start()

    def _sse_loop(self):
        try:
            with requests.get(f"{SERVER}/events", stream=True, timeout=None) as resp:
                for raw in resp.iter_lines():
                    if self._sse_stop.is_set():
                        break
                    if not raw:
                        continue
                    line = raw.decode("utf-8") if isinstance(raw, bytes) else raw
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    try:
                        event = json.loads(payload)
                    except json.JSONDecodeError:
                        continue

                    if event.get("type") == "keyword":
                        self.after(0, self._on_keyword, event)
        except Exception:
            pass   # connection closed or server gone

    def _on_keyword(self, event: dict):
        kw   = event.get("keyword", "?")
        conf = event.get("confidence", 0)
        ts   = event.get("timestamp", "")

        # Big display
        color = TAG_COLORS.get(kw, DEFAULT_TAG)
        self._kw_label.configure(text=kw.upper(), fg=color)
        self._conf_label.configure(text=f"{conf:.0%} confidence  ·  {ts}")

        # Log entry
        self._log_line(kw, f"{ts}  {kw.upper()}  {conf:.0%}")

    # ── Log helpers ───────────────────────────────────────────────────────────
    def _log_line(self, tag: str, text: str):
        """Append a line to the log text widget (thread-safe via after)."""
        self.after(0, self._insert_log, tag, text)

    def _insert_log(self, tag: str, text: str):
        self._log.configure(state="normal")
        if self._log.index("end-1c") != "1.0":
            self._log.insert("end", "\n")
        if tag in TAG_COLORS:
            self._log.insert("end", text, tag)
        elif tag == "system":
            self._log.insert("end", text, "ts")
        else:
            self._log.insert("end", text, "_default")
        self._log.configure(state="disabled")
        self._log.see("end")


if __name__ == "__main__":
    app = App()
    app.mainloop()
