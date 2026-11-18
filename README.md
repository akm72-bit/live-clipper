# Live Clipper 🎮

Clips a Twitch stream **while it is still live**. When chat explodes, you get a ready-to-post 9:16 clip with captions, usually under a minute after the moment.

From the @tokenstoviews reel *"Don't download the stream. Beat it."*

## How it works

1. **Streamlink** records the live stream into 10-second chunks. Only the last 5 minutes are kept on disk.
2. It **reads Twitch chat anonymously** (no login, no token) and counts messages every 10 seconds.
3. When chat jumps to **2.5× its recent average** (and at least 20 messages in 10 s), that's a moment.
4. **FFmpeg** cuts 25 s before the spike + 5 s after, and makes a 9:16 version (blurred background, full picture in the middle).
5. **Whisper** (runs on your own PC) adds word-by-word captions.

Your GPU does the heavy lifting when there is one: FFmpeg encodes the clips with **AMF** (AMD) or **NVENC** (NVIDIA), and Whisper captions run on the GPU too (**CUDA** on NVIDIA, **ZLUDA** on AMD). No GPU? Everything still works on the CPU, just slower.

All free. Nothing is uploaded anywhere, and no AI credits or API keys are used.

## Setup (once, about 10 minutes)

1. Install **Python 3.10 or newer**: https://www.python.org/downloads/ (on Windows, tick *"Add Python to PATH"*).
2. Install **FFmpeg**:
   - Windows: open a terminal and run `winget install Gyan.FFmpeg`
   - Mac: `brew install ffmpeg`
   - Linux: `sudo apt install ffmpeg`
3. Open a terminal **in this folder** and run:
   ```
   pip install -r requirements.txt
   ```
4. Check that it worked: `ffmpeg -version` and `python -m streamlink --version` should both print a version.

That's it: no GPU drivers, CUDA toolkit or extra setup needed, whatever card you have. An NVIDIA GPU is picked up automatically. On AMD, one extra command gets the GPU working — see below.

## Using your GPU (optional)

**NVIDIA:** nothing to do. Clips encode with NVENC and Whisper uses CUDA.

**AMD:** clips already encode on the GPU (AMF) with no setup at all. For GPU captions, Whisper needs to talk CUDA, which AMD doesn't speak — so [ZLUDA](https://github.com/vosen/ZLUDA) translates. Two pieces are needed:

1. A recent **AMD driver**, plus the **HIP SDK** (it is *not* part of the driver). Grab the official **AMD HIP SDK for Windows** — but note it doesn't support every card (Radeon RX 6000 cards like the 6600 XT are `gfx1032`, which is unsupported). For those, download a nightly `therock-dist-windows-gfx103X-*.tar.gz` from [here](https://therock-nightly-tarball.s3.amazonaws.com/index.html), unpack it and set `HIP_PATH` to that folder. Either way, the folder must contain `bin\rocblas.dll`.
2. **ZLUDA** itself, which the setup script fetches for you:

```
python setup_zluda.py
```

It installs ZLUDA into `./zluda` and checks the whole chain, telling you exactly what's missing if something is. After that, just run the clipper as usual: it notices the AMD GPU and restarts itself under ZLUDA, so Whisper captions on your GPU. If any piece is missing it says so and quietly captions on the CPU instead.

Worth knowing: ZLUDA compiles shaders on first use, so the first few clips are slow and later ones are quick.

## Use it

Start it **before or during** the stream:

```
python live_clipper.py CHANNEL_NAME
```

Example: `python live_clipper.py kaicenat`. Leave the window open. Every clip lands in `clips/<channel>_<date>_<time>/`:

| File | What it is |
|---|---|
| `clip_9x16.mp4` | Ready for Reels / Shorts / TikTok, captions burned in |
| `clip_16x9.mp4` | The original wide clip, no captions |
| `captions.srt` | Captions to edit or re-use |
| `info.json` | How big the chat spike was + a sample of what chat said |

Press **Ctrl+C** to stop.

## Tuning

| Option | Default | Use it when |
|---|---|---|
| `--spike 2.5` | 2.5 | Too many clips → raise it (3, 4). Too few → lower it (2). |
| `--min-msgs 20` | 20 | Small channel → lower it (8–10). |
| `--before 25 --after 5` | 25 / 5 | Want longer clips → `--before 40` |
| `--cooldown 60` | 60 s | Minimum gap between two clips |
| `--model small` | small | Better captions: `medium` or `large-v3` (slower). Weak PC: `base` |
| `--quality 720p60,720p,best` | 720p | `best` for 1080p (bigger files) |
| `--encoder auto` | auto | `x264` to force software encoding, `amf`/`nvenc` to force a GPU one |
| `--cpu-only` | off | Never touch the GPU (software encoding + CPU captions) |
| `--no-captions` | off | Skip Whisper completely |
| `--test-clip-after 30` | off | Test the setup: forces one clip after 30 s |

**Launch night tip:** huge streams have thousands of messages, so their "average" is already high. Start the clipper 10–15 minutes before the big moment so it learns the normal chat speed first.

## Good to know

- **Live only.** It can't go back in time. If the stream has already ended, there's nothing to clip.
- A chat spike usually means something happened, but not always (giveaways, raids, spam). Watch each clip before you post it.
- **Only post clips you're allowed to use.** Clip streamers who allow clipping, or your own streams, and give credit. Respect Twitch's rules and the streamer's wishes.
- Not affiliated with Twitch, Rockstar Games, Take-Two, Sony, Microsoft or Anthropic.

Built with Claude Code · follow **@tokenstoviews** for a new free AI build every week.
