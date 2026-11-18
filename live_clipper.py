"""
Live Clipper: clips a Twitch stream WHILE it is still live.

  1. Streamlink records the stream into 10-second chunks (a rolling buffer on disk).
  2. It reads Twitch chat anonymously (no login) and counts messages every 10 seconds.
  3. When chat explodes (a burst well above its recent average), FFmpeg cuts the moment.
  4. Whisper adds captions and you get a ready-to-post 9:16 clip, while the stream is still going.

Usage:
    python live_clipper.py CHANNEL
    python live_clipper.py CHANNEL --model small --spike 2.5 --before 25 --after 5

Needs: Python 3.10+, FFmpeg on PATH, and `pip install -r requirements.txt`.
It only works on channels that are LIVE. Start it before (or during) the stream.
Only clip and post streams you have permission to use.

GPU: clips are encoded on the GPU when there is one (AMD AMF, NVIDIA NVENC) and Whisper runs on
the GPU too (NVIDIA through CUDA, AMD through ZLUDA: run `python setup_zluda.py` once). Without a
usable GPU everything falls back to the CPU, just slower.
"""
import argparse, collections, csv, glob, json, os, queue, random, re, shutil, signal, socket, statistics, subprocess, sys, threading, time
from datetime import datetime

SEG = 10  # seconds per recorded chunk and per chat bucket
HERE = os.path.dirname(os.path.abspath(__file__))

def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)

# ---------------------------------------------------------------- GPU discovery
_gpus = None

def gpu_names():
    """What Windows/Linux reports as the video cards in this machine (cached)."""
    global _gpus
    if _gpus is None:
        out = ""
        for cmd in (["wmic", "path", "win32_VideoController", "get", "name"],  # wmic is gone from newer Windows
                    ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_VideoController).Name"],
                    ["lspci"]):
            try:
                out = subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout
            except (OSError, subprocess.SubprocessError):
                continue
            if out.strip(): break
        _gpus = [l.strip() for l in out.splitlines() if l.strip() and l.strip().lower() not in ("name", "device")]
    return _gpus

def has_gpu(word):
    return any(word in n.lower() for n in gpu_names())

def find_zluda():
    """ZLUDA (CUDA -> AMD translation layer): ./zluda next to this script, $ZLUDA_DIR, or a system install."""
    system = os.path.join(os.environ["ProgramFiles"], "zluda") if os.environ.get("ProgramFiles") else None
    for d in (os.path.join(HERE, "zluda"), os.environ.get("ZLUDA_DIR"), system):
        if d and os.path.exists(os.path.join(d, "zluda.exe")):
            return d
    return None

def find_hip_sdk():
    """A HIP SDK ZLUDA can use. It needs bin/rocblas.dll: ZLUDA's cuBLAS is a wrapper around rocBLAS."""
    cands = ([os.environ["HIP_PATH"]] if os.environ.get("HIP_PATH") else [])
    for pat in (r"C:\Program Files\AMD\ROCm\*", r"C:\Program Files\AMD\therock-dist-windows-*", "/opt/rocm", "/opt/rocm-*"):
        cands += sorted(glob.glob(pat), reverse=True)  # newest first
    for d in cands:
        if os.path.exists(os.path.join(d, "bin", "rocblas.dll")) or os.path.exists(os.path.join(d, "lib", "librocblas.so")):
            return d
    return None

def zluda_active():
    """Are we already running with ZLUDA's CUDA shim patched in?"""
    if os.environ.get("ZLUDA_RUNNING"):
        return True
    has_amd = has_gpu("amd") or has_gpu("radeon")
    return has_amd and not has_gpu("nvidia")

def relaunch_under_zluda(a):
    """An AMD GPU only looks like a CUDA device to Whisper once ZLUDA is loaded, so start again through it.

    Does nothing when there is nothing to do, and replaces this process when there is (the child inherits
    our console, so the user keeps seeing one continuous stream of logs).
    """
    if a.cpu_only or os.environ.get("ZLUDA_RUNNING") or not (has_gpu("amd") or has_gpu("radeon")):
        return False
    try:
        import ctranslate2
        if ctranslate2.get_cuda_device_count() > 0:  # a real CUDA device (NVIDIA) is already visible
            return False
    except Exception:
        pass
    z = find_zluda()
    if not z:
        log("AMD GPU found, but ZLUDA is not installed: captions will run on the CPU. Run `python setup_zluda.py`.")
        return False
    hip = find_hip_sdk()
    if not hip:
        log("AMD GPU found, but no usable HIP SDK (needs bin/rocblas.dll): captions will run on the CPU. See the README.")
        return False
    env = dict(os.environ, HIP_PATH=hip, ZLUDA_RUNNING="1",
               PATH=os.path.join(hip, "bin") + os.pathsep + os.environ.get("PATH", ""))
    log(f"AMD GPU: restarting under ZLUDA ({z}) so Whisper can use it")
    sys.stdout.flush()
    sys.exit(subprocess.call([os.path.join(z, "zluda.exe"), "--", sys.executable, os.path.abspath(__file__)] + sys.argv[1:], env=env))

# ---------------------------------------------------------------- video encoder
def probe_encoder(enc, opts):
    """The only reliable way to know an encoder works: encode a few frames with it."""
    try:
        r = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=30",
                            "-frames:v", "3", "-c:v", enc, *opts, "-f", "null", "-"], capture_output=True, text=True, timeout=60)
        return r.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False

def pick_encoder(pref="auto"):
    """GPU encoding when the GPU is there (AMD AMF / NVIDIA NVENC), software x264 otherwise."""
    table = {"amf": ("h264_amf", ["-quality", "speed", "-rc", "cqp", "-qp_i", "22", "-qp_p", "22", "-qp_b", "24"]),
             "nvenc": ("h264_nvenc", ["-preset", "p4", "-rc", "vbr", "-cq", "23", "-b:v", "0"]),
             "x264": ("libx264", ["-preset", "veryfast", "-crf", "20"])}
    for k in {"auto": ["amf", "nvenc", "x264"], "amf": ["amf"], "nvenc": ["nvenc"], "x264": ["x264"]}[pref]:
        enc, opts = table[k]
        if probe_encoder(enc, opts):
            log(f"encoder: {enc}" + ("" if k == "x264" else " (GPU)"))
            return enc, opts
        if pref != "auto":
            log(f"encoder: {enc} is not usable here, using software x264")
    return table["x264"]

# ---------------------------------------------------------------- recorder
class Recorder:
    """streamlink -> ffmpeg segmenter. Keeps a list of finished chunks with the wall-clock time they ended."""
    def __init__(self, channel, quality, workdir, keep_minutes):
        self.channel, self.quality, self.dir = channel, quality, workdir
        self.keep = keep_minutes * 60
        self.segs = []  # dicts: file, dur, end_wall
        self.lock = threading.Lock()
        self.stop = threading.Event()
        os.makedirs(self.dir, exist_ok=True)
        self.csv = os.path.join(self.dir, "segments.csv")

    def start(self):
        url = f"https://www.twitch.tv/{self.channel}"
        sl = [sys.executable, "-m", "streamlink", "--stdout", "--twitch-low-latency", "--loglevel", "warning", url, self.quality]
        ff = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", "pipe:0", "-map", "0:v:0?", "-map", "0:a:0?", "-c", "copy",
              "-f", "segment", "-segment_time", str(SEG), "-reset_timestamps", "1",
              "-segment_list", self.csv, "-segment_list_type", "csv", "-segment_list_flags", "+live",
              os.path.join(self.dir, "chunk_%06d.ts")]
        self.p_sl = subprocess.Popen(sl, stdout=subprocess.PIPE, stderr=sys.stderr)
        self.p_ff = subprocess.Popen(ff, stdin=self.p_sl.stdout)
        self.p_sl.stdout.close()
        threading.Thread(target=self._watch, daemon=True).start()

    def alive(self):
        return self.p_ff.poll() is None and self.p_sl.poll() is None

    def _watch(self):
        seen = 0
        while not self.stop.is_set():
            time.sleep(0.5)
            if not os.path.exists(self.csv):
                continue
            with open(self.csv, newline="") as f:
                rows = list(csv.reader(f))
            now = time.time()
            with self.lock:
                for r in rows[seen:]:
                    if len(r) < 3: continue
                    self.segs.append({"file": os.path.join(self.dir, r[0]), "dur": float(r[2]) - float(r[1]), "end_wall": now})
                seen = len(rows)
                # rolling buffer: delete old chunks from disk (never ones a clip may still need)
                while self.segs and now - self.segs[0]["end_wall"] > self.keep:
                    old = self.segs.pop(0)
                    try: os.remove(old["file"])
                    except OSError: pass

    def window(self, t0, t1):
        """chunks overlapping wall-clock [t0, t1] -> (files, wall time where the first file starts)"""
        with self.lock:
            pick = [s for s in self.segs if s["end_wall"] > t0 and s["end_wall"] - s["dur"] < t1]
        if not pick: return [], None
        return [s["file"] for s in pick], pick[0]["end_wall"] - pick[0]["dur"]

    def close(self):
        self.stop.set()
        for p in (getattr(self, "p_sl", None), getattr(self, "p_ff", None)):
            if p and p.poll() is None:
                p.terminate()

# ---------------------------------------------------------------- chat
class Chat:
    """Anonymous, read-only Twitch chat (IRC). No account or token needed."""
    def __init__(self, channel):
        self.channel = channel.lower()
        self.buckets = collections.deque(maxlen=60)  # (bucket_end_wall, count, sample msgs)
        self.cur, self.cur_msgs = 0, []
        self.lock = threading.Lock()
        self.stop = threading.Event()

    def start(self):
        threading.Thread(target=self._run, daemon=True).start()
        threading.Thread(target=self._tick, daemon=True).start()

    def _run(self):
        while not self.stop.is_set():
            try:
                s = socket.create_connection(("irc.chat.twitch.tv", 6667), timeout=30)
                s.sendall(f"PASS SCHMOOPIIE\r\nNICK justinfan{random.randint(10000, 99999)}\r\n".encode())
                buf, joined, quiet = "", False, 0
                while not self.stop.is_set():
                    try:
                        data = s.recv(65536)
                    except socket.timeout:
                        quiet += 1  # a quiet chat is normal: ping the server to keep the link up
                        s.sendall(b"PING :tmi.twitch.tv\r\n")
                        if quiet >= 4: break
                        continue
                    quiet = 0
                    if not data: break
                    buf += data.decode("utf-8", "ignore")
                    *lines, buf = buf.split("\r\n")
                    for ln in lines:
                        if " 376 " in ln and not joined:  # JOIN only after the welcome, or Twitch ignores it
                            s.sendall(f"JOIN #{self.channel}\r\n".encode()); joined = True
                            log(f"chat: reading #{self.channel} (anonymous, read-only, no login)")
                        if ln.startswith("PING"):
                            s.sendall(ln.replace("PING", "PONG", 1).encode() + b"\r\n")
                        elif " PRIVMSG " in ln:
                            m = re.match(r":(\w+)!.* PRIVMSG #\w+ :(.*)", ln)
                            with self.lock:
                                self.cur += 1
                                if m and len(self.cur_msgs) < 40: self.cur_msgs.append(f"{m.group(1)}: {m.group(2)[:80]}")
            except OSError as e:
                if not self.stop.is_set():
                    log("chat: reconnecting after", type(e).__name__); time.sleep(3)

    def _tick(self):
        nxt = time.time() + SEG
        while not self.stop.is_set():
            time.sleep(max(0, nxt - time.time()))
            with self.lock:
                self.buckets.append((nxt, self.cur, self.cur_msgs))
                self.cur, self.cur_msgs = 0, []
            nxt += SEG

# ---------------------------------------------------------------- clip maker
class Clipper:
    def __init__(self, args, rec, outdir):
        self.a, self.rec, self.out = args, rec, outdir
        self.q = queue.Queue()
        self.model = None
        self.busy = False
        threading.Thread(target=self._worker, daemon=True).start()

    def _load_whisper(self):
        if self.model is not None or self.a.no_captions: return
        from faster_whisper import WhisperModel
        try:
            import ctranslate2
            gpu = ctranslate2.get_cuda_device_count() > 0 and not self.a.cpu_only
        except Exception:
            gpu = False
        dev, ct = ("cuda", "float16") if gpu else ("cpu", "int8")
        how = " (AMD GPU via ZLUDA)" if gpu and zluda_active() else ""
        log(f"whisper: loading '{self.a.model}' on {dev}{how}")
        try:
            self.model = WhisperModel(self.a.model, device=dev, compute_type=ct)
        except Exception as e:
            log(f"whisper: GPU not usable ({type(e).__name__}), using CPU")
            self.model = WhisperModel(self.a.model, device="cpu", compute_type="int8")

    def transcribe(self, path):
        try:
            segs, _ = self.model.transcribe(path, word_timestamps=True, vad_filter=True)
            return [(w.word.strip(), w.start, w.end) for s in segs for w in (s.words or []) if w.word.strip()]
        except Exception as e:  # e.g. NVIDIA card present but CUDA libraries missing
            log(f"whisper: GPU failed ({type(e).__name__}), switching to CPU")
            from faster_whisper import WhisperModel
            self.model = WhisperModel(self.a.model, device="cpu", compute_type="int8")
            segs, _ = self.model.transcribe(path, word_timestamps=True, vad_filter=True)
            return [(w.word.strip(), w.start, w.end) for s in segs for w in (s.words or []) if w.word.strip()]

    def add(self, spike):
        self.q.put(spike)

    def _worker(self):
        while True:
            sp = self.q.get()
            self.busy = True
            # wait until the chunks AFTER the spike exist
            ready_at = sp["wall"] + self.a.after + SEG + 2
            while time.time() < ready_at: time.sleep(0.5)
            try:
                self.make(sp)
            except Exception as e:
                log("clip failed:", repr(e))
            finally:
                self.busy = False

    def make(self, sp):
        t0, t1 = sp["wall"] - self.a.before - self.a.delay, sp["wall"] + self.a.after - self.a.delay
        files, first_start = self.rec.window(t0, t1)
        if not files:
            log("clip skipped: no recorded video for that moment yet"); return
        name = f"{self.a.channel}_{datetime.fromtimestamp(sp['wall']).strftime('%Y%m%d_%H%M%S')}"
        d = os.path.join(self.out, name); os.makedirs(d, exist_ok=True)
        lst = os.path.join(d, "chunks.txt")
        with open(lst, "w") as f:
            for fp in files: f.write(f"file '{os.path.abspath(fp)}'\n")
        ss = max(0.0, t0 - first_start); dur = t1 - t0
        raw = os.path.join(d, "clip_16x9.mp4")
        run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst,
             "-ss", f"{ss:.2f}", "-t", f"{dur:.2f}", "-c:v", self.a.enc, *self.a.enc_opts, "-pix_fmt", "yuv420p",
             "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", raw])
        os.remove(lst)
        ass = None
        if not self.a.no_captions:
            self._load_whisper()
            words = self.transcribe(raw)
            write_srt(words, os.path.join(d, "captions.srt"))
            if words:
                ass = "captions.ass"; write_ass(words, os.path.join(d, ass))
        vert = os.path.join(d, "clip_9x16.mp4")
        # 9:16: blurred full-frame background + the full 16:9 picture in the middle + big captions
        vf = ("[0:v]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,boxblur=24:2,eq=brightness=-0.08[bg];"
              "[0:v]scale=1080:-2[fg];[bg][fg]overlay=(W-w)/2:(H-h)/2" + (f",ass={ass}" if ass else "") + "[v]")
        run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", "clip_16x9.mp4", "-filter_complex", vf, "-map", "[v]", "-map", "0:a?",
             "-c:v", self.a.enc, *self.a.enc_opts, "-pix_fmt", "yuv420p", "-c:a", "copy", "-movflags", "+faststart", "clip_9x16.mp4"], cwd=d)
        json.dump({"channel": self.a.channel, "spike_time": datetime.fromtimestamp(sp["wall"]).isoformat(timespec="seconds"),
                   "msgs_in_10s": sp["count"], "baseline_10s": sp["base"], "chat_sample": sp["msgs"][:15]},
                  open(os.path.join(d, "info.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
        log(f"CLIP READY -> {vert}")

def run(cmd, cwd=None):
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed: {r.stderr.strip()[-400:]}")

def chunks3(words, n=3):
    out, cur = [], []
    for w in words:
        cur.append(w)
        if len(cur) == n or re.search(r"[.!?,]$", w[0]):
            out.append(cur); cur = []
    if cur: out.append(cur)
    return out

def ts(t, sep):
    h, m, s = int(t // 3600), int(t % 3600 // 60), t % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", sep) if sep == "," else f"{h:d}:{m:02d}:{s:05.2f}"

def write_srt(words, path):
    with open(path, "w", encoding="utf-8") as f:
        for i, c in enumerate(chunks3(words), 1):
            f.write(f"{i}\n{ts(c[0][1], ',')} --> {ts(c[-1][2], ',')}\n{' '.join(w[0] for w in c)}\n\n")

def write_ass(words, path):
    head = """[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,Arial Black,84,&H00FFFFFF,&H0000D8FF,&H00000000,&H64000000,-1,0,0,0,100,100,0,0,1,7,2,2,60,60,440,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(head)
        cs = chunks3(words)
        for i, c in enumerate(cs):
            end = c[-1][2] + 0.15
            if i + 1 < len(cs): end = min(end, cs[i + 1][0][1])  # never overlap the next line
            # karaoke: each word turns yellow while it is spoken
            txt = "".join(f"{{\\kf{max(1, int((w[2] - w[1]) * 100))}}}{w[0].upper().replace('{', '').replace('}', '')} " for w in c)
            f.write(f"Dialogue: 0,{ts(c[0][1], '.')},{ts(end, '.')},Cap,,0,0,0,,{txt.strip()}\n")

# ---------------------------------------------------------------- main loop
def main():
    ap = argparse.ArgumentParser(description="Clip a live Twitch stream the moment chat explodes.")
    ap.add_argument("channel", help="Twitch channel name, e.g. xqc")
    ap.add_argument("--quality", default="720p60,720p,best", help="stream quality (default 720p60,720p,best)")
    ap.add_argument("--spike", type=float, default=2.5, help="chat must hit this many times its recent average (default 2.5)")
    ap.add_argument("--min-msgs", type=int, default=20, help="minimum messages in 10 s to count as a spike (default 20)")
    ap.add_argument("--before", type=float, default=25, help="seconds kept before the spike (default 25)")
    ap.add_argument("--after", type=float, default=5, help="seconds kept after the spike (default 5)")
    ap.add_argument("--delay", type=float, default=4, help="chat reacts this many seconds after the moment (default 4)")
    ap.add_argument("--cooldown", type=float, default=60, help="seconds between clips (default 60)")
    ap.add_argument("--model", default="small", help="Whisper model: tiny, base, small, medium, large-v3 (default small)")
    ap.add_argument("--no-captions", action="store_true", help="skip Whisper captions")
    ap.add_argument("--out", default="clips", help="output folder (default ./clips)")
    ap.add_argument("--buffer-minutes", type=float, default=5, help="minutes of video kept on disk (default 5)")
    ap.add_argument("--encoder", choices=("auto", "amf", "nvenc", "x264"), default="auto",
                    help="video encoder: auto picks AMD AMF, NVIDIA NVENC or software x264 (default auto)")
    ap.add_argument("--cpu-only", action="store_true", help="never touch the GPU: software encoding, CPU captions")
    ap.add_argument("--test-clip-after", type=float, default=0, help="TEST: also force one clip after N seconds")
    a = ap.parse_args()
    a.channel = a.channel.lower().lstrip("#").split("/")[-1]
    if not shutil.which("ffmpeg"):
        sys.exit("FFmpeg not found. Install it and make sure `ffmpeg` runs in a terminal.")
    relaunch_under_zluda(a)  # replaces this process when ZLUDA is what the GPU needs
    a.enc, a.enc_opts = pick_encoder("x264" if a.cpu_only else a.encoder)

    work = os.path.join(a.out, f".buffer_{a.channel}")
    shutil.rmtree(work, ignore_errors=True)
    rec = Recorder(a.channel, a.quality, work, a.buffer_minutes)
    chat = Chat(a.channel)
    clipper = Clipper(a, rec, a.out)
    log(f"watching twitch.tv/{a.channel} live. Clips go to {os.path.abspath(a.out)}  (Ctrl+C to stop)")
    rec.start(); chat.start()
    threading.Thread(target=clipper._load_whisper, daemon=True).start()

    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    start, last_clip, last_bucket, forced = time.time(), 0.0, None, False
    try:
        while not stop.is_set():
            time.sleep(1)
            if not rec.alive():
                log("stream ended or is offline. Live only: no rewind."); break
            with chat.lock:
                b = list(chat.buckets)
            if b and b[-1][0] != last_bucket:
                last_bucket = b[-1][0]
                end, cnt, msgs = b[-1]
                hist = [x[1] for x in b[:-1]][-30:]
                base = statistics.median(hist) if len(hist) >= 3 else None
                bar = "#" * min(40, cnt // 2)
                log(f"chat {cnt:4d} msgs/10s  avg {base if base is not None else '...':>4}  {bar}")
                if base is not None and cnt >= a.min_msgs and cnt >= a.spike * max(base, 1) and end - last_clip > a.cooldown:
                    last_clip = end
                    log(f"CHAT EXPLODED ({cnt} vs avg {base}) -> cutting the last {a.before:.0f}s + {a.after:.0f}s")
                    clipper.add({"wall": end, "count": cnt, "base": base, "msgs": msgs})
            if a.test_clip_after and not forced and time.time() - start > a.test_clip_after:
                forced = True
                log("TEST: forcing one clip now")
                with chat.lock:
                    msgs = chat.buckets[-1][2] if chat.buckets else []
                clipper.add({"wall": time.time(), "count": 0, "base": 0, "msgs": msgs})
    finally:
        # let a clip that is being made finish
        t = time.time()
        while (clipper.busy or not clipper.q.empty()) and time.time() - t < 120:
            time.sleep(1)
        chat.stop.set(); rec.close()
        log("stopped.")
        if os.environ.get("ZLUDA_RUNNING"):
            # ZLUDA v7 preview panics in its thread-local destructors while unloading; skip the teardown
            sys.stdout.flush(); sys.stderr.flush(); os._exit(0)

if __name__ == "__main__":
    main()
