"""Small GUI for running app.py at a booth.

    python3 launcher.py

Finds calib*.json and parts*.yaml in the working directory, remembers the last
selection in profiles.json, streams the log into the window, and stops app.py
with SIGINT so its finally block parks the beam.

If the launcher itself runs on the system Python, app.py is still started with
the virtualenv interpreter -- see app_python()."""
from __future__ import annotations
import glob
import json
import os
import queue
import signal
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import ttk
from paths import data_path, data_glob

CFG = data_path("profiles.json")


def app_python():
    """Interpreter to run app.py with.

    A pyenv build without tcl-tk has no tkinter, so this launcher often has to
    run on the system Python -- but app.py needs the virtualenv where the
    dependencies live. Hence two interpreters.

    Order: $APP_PYTHON, then a .venv beside this file or above it, then us.
    """
    env = os.environ.get("APP_PYTHON")
    if env and os.path.exists(env):
        return env
    here = os.path.dirname(os.path.abspath(__file__))
    for _ in range(4):
        for name in (".venv", "venv"):
            for exe in ("bin/python3", "bin/python"):
                c = os.path.join(here, name, exe)
                if os.path.exists(c):
                    return c
        parent = os.path.dirname(here)
        if parent == here:
            break
        here = parent
    return sys.executable


ENGINES = [
    ("mlx (로컬, 기본)", "mlx", ""),
    ("faster (로컬, 경량)", "faster", "small"),
    ("groq · turbo (클라우드)", "groq", "whisper-large-v3-turbo"),
    ("groq · large-v3 (정확도 우선)", "groq", "whisper-large-v3"),
]


def discover(pattern, fallback=None):
    out = data_glob(pattern)
    return out if out else ([fallback] if fallback else [])


class Launcher:
    def __init__(self, root):
        self.root = root
        root.title("Galvo Highlighter — 실행")
        root.geometry("620x560")
        self.proc = None
        self.q: queue.Queue[str] = queue.Queue()

        pad = dict(padx=10, pady=4, sticky="w")
        f = ttk.Frame(root, padding=12)
        f.pack(fill="both", expand=True)

        r = 0
        ttk.Label(f, text="실행 설정", font=("", 14, "bold")).grid(row=r, column=0,
                                                                columnspan=3, **pad)
        r += 1

        ttk.Label(f, text="캘리브 (표면)").grid(row=r, column=0, **pad)
        self.calib = ttk.Combobox(f, width=42, state="readonly",
                                  values=discover("calib*.json", "calib.json"))
        self.calib.grid(row=r, column=1, columnspan=2, **pad); r += 1

        ttk.Label(f, text="부위 사전").grid(row=r, column=0, **pad)
        self.parts = ttk.Combobox(f, width=42, state="readonly",
                                  values=discover("parts*.yaml", "parts_mockup.yaml"))
        self.parts.grid(row=r, column=1, columnspan=2, **pad); r += 1

        ttk.Label(f, text="음성 엔진").grid(row=r, column=0, **pad)
        self.engine = ttk.Combobox(f, width=42, state="readonly",
                                   values=[e[0] for e in ENGINES])
        self.engine.grid(row=r, column=1, columnspan=2, **pad); r += 1

        self.key_note = ttk.Label(f, text="", foreground="#b45309")
        self.key_note.grid(row=r, column=1, columnspan=2, **pad); r += 1
        self.engine.bind("<<ComboboxSelected>>", self._check_key)

        ttk.Label(f, text="카메라").grid(row=r, column=0, **pad)
        self.cam = ttk.Combobox(f, width=18, state="readonly",
                                values=["사용 안 함", "0", "1", "2"])
        self.cam.grid(row=r, column=1, **pad); r += 1

        self.sim = tk.BooleanVar(value=False)
        self.novoice = tk.BooleanVar(value=False)
        ttk.Checkbutton(f, text="시뮬레이터 (하드웨어 없이)",
                        variable=self.sim).grid(row=r, column=1, **pad); r += 1
        ttk.Checkbutton(f, text="음성 끄기 (핫키 전용 — 시연 백업)",
                        variable=self.novoice).grid(row=r, column=1, **pad); r += 1

        bf = ttk.Frame(f)
        bf.grid(row=r, column=0, columnspan=3, pady=10, sticky="w")
        self.btn_run = ttk.Button(bf, text="▶  실행", command=self.run, width=14)
        self.btn_run.pack(side="left", padx=(10, 6))
        self.btn_stop = ttk.Button(bf, text="■  정지", command=self.stop,
                                   width=14, state="disabled")
        self.btn_stop.pack(side="left", padx=6)
        ttk.Button(bf, text="↻ 파일 다시 찾기", command=self.refresh,
                   width=16).pack(side="left", padx=6)
        r += 1

        self.status = ttk.Label(f, text="대기 중", foreground="#555")
        self.status.grid(row=r, column=0, columnspan=3, **pad); r += 1

        ttk.Label(f, text="로그").grid(row=r, column=0, **pad); r += 1
        self.log = tk.Text(f, height=14, width=74, bg="#111", fg="#ddd",
                           insertbackground="#ddd")
        self.log.grid(row=r, column=0, columnspan=3, padx=10, pady=4)

        ttk.Label(f, text="실행 중 핫키(HUD 창):  1~9 부위 지정   space 파킹   q 종료",
                  foreground="#555").grid(row=r + 1, column=0, columnspan=3, **pad)

        self.load()
        self._check_key()
        self.write(f"[i] app.py 인터프리터: {app_python()}\n"
                   f"    (다르게 쓰려면  export APP_PYTHON=/경로/python)\n")
        self.root.after(120, self._drain)
        root.protocol("WM_DELETE_WINDOW", self.on_close)

    def load(self):
        d = {}
        if os.path.exists(CFG):
            try:
                d = json.load(open(CFG))
            except Exception:
                pass
        def pick(cb, val, default_idx=0):
            vals = cb["values"]
            if val in vals:
                cb.set(val)
            elif vals:
                cb.current(default_idx)
        pick(self.calib, d.get("calib"))
        pick(self.parts, d.get("parts"))
        pick(self.engine, d.get("engine"))
        pick(self.cam, d.get("cam"))
        self.sim.set(d.get("sim", False))
        self.novoice.set(d.get("novoice", False))

    def save(self):
        json.dump({"calib": self.calib.get(), "parts": self.parts.get(),
                   "engine": self.engine.get(), "cam": self.cam.get(),
                   "sim": self.sim.get(), "novoice": self.novoice.get()},
                  open(CFG, "w"), ensure_ascii=False, indent=1)

    def refresh(self):
        self.calib["values"] = discover("calib*.json", "calib.json")
        self.parts["values"] = discover("parts*.yaml", "parts_mockup.yaml")
        self.write("[i] 파일 목록 갱신\n")

    def _check_key(self, *_):
        eng = self._engine_tuple()
        if eng and eng[1] == "groq" and not os.environ.get("GROQ_API_KEY"):
            self.key_note.config(
                text="⚠ GROQ_API_KEY 미설정 — 터미널에서 export 후 이 앱을 다시 실행")
        else:
            self.key_note.config(text="")

    def _engine_tuple(self):
        for e in ENGINES:
            if e[0] == self.engine.get():
                return e
        return None


    def _parts_have_surface(self):
        """True when the chosen parts file drives calibration itself.

        A parts.yaml with surface: tags loads calib_<surface>.json per surface
        and app.py ignores --calib, so passing it would only mislead. Read as
        plain text: this launcher often runs on a bare system Python with no
        PyYAML installed.
        """
        try:
            with open(data_path(self.parts.get()), encoding="utf-8") as fh:
                return any(l.strip().startswith("surface:") for l in fh)
        except OSError:
            return False


    def run(self):
        if self.proc:
            return
        cmd = [app_python(), "app.py"]
        if self.parts.get():
            cmd += ["--parts", self.parts.get()]
        eng = self._engine_tuple()
        if eng:
            cmd += ["--engine", eng[1]]
            if eng[2]:
                cmd += ["--model", eng[2]]
        if self.sim.get():
            cmd += ["--sim"]
        if self.novoice.get():
            cmd += ["--no-voice"]
        if self.cam.get() and self.cam.get() != "사용 안 함":
            cmd += ["--cam", self.cam.get()]

        if (self.calib.get() and self.calib.get() != "calib.json"
                and not self._parts_have_surface()):
            cmd += ["--calib", self.calib.get()]

        self.save()
        self.write("$ " + " ".join(cmd) + "\n")
        try:
            self.proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, env=os.environ.copy())
        except Exception as e:
            self.write(f"[!] 실행 실패: {e}\n")
            return
        threading.Thread(target=self._reader, daemon=True).start()
        self.btn_run.config(state="disabled")
        self.btn_stop.config(state="normal")
        self.status.config(text="실행 중", foreground="#166534")

    def _reader(self):
        for line in self.proc.stdout:
            self.q.put(line)
        self.q.put("__END__")

    def _drain(self):
        try:
            while True:
                line = self.q.get_nowait()
                if line == "__END__":
                    self._finished()
                else:
                    self.write(line)
        except queue.Empty:
            pass
        self.root.after(120, self._drain)

    def _finished(self):
        self.proc = None
        self.btn_run.config(state="normal")
        self.btn_stop.config(state="disabled")
        self.status.config(text="종료됨 (빔 파킹 완료)", foreground="#555")

    def stop(self):
        if not self.proc:
            return
        self.write("[i] 종료 신호 전송 — app.py 가 파킹 후 종료합니다\n")
        try:
            self.proc.send_signal(signal.SIGINT)
        except Exception:
            self.proc.terminate()

    def write(self, txt):
        self.log.insert("end", txt)
        self.log.see("end")

    def on_close(self):
        if self.proc:
            self.stop()
            self.root.after(800, self.root.destroy)
        else:
            self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    Launcher(root)
    root.mainloop()
