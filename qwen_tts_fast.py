#!/usr/bin/env python3
"""
Fast local Qwen3-TTS playback.

The first normal invocation starts a small background daemon that loads the MLX
Qwen3-TTS model once and keeps it in memory. Later invocations send text to the
daemon and stream PCM audio to the local speaker.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import Lock
from typing import Any

import httpx
import numpy as np
import sounddevice as sd


ROOT = Path(__file__).resolve().parent
CACHE_DIR = Path.home() / ".cache" / "qwen-tts-fast"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
PID_FILE = CACHE_DIR / "daemon.pid"
LOG_FILE = CACHE_DIR / "daemon.log"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
DEFAULT_MODEL = "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-6bit"
DEFAULT_STREAM_INTERVAL = 0.5
DEFAULT_SAMPLE_RATE = 24000

LANG_CODES = {
    "Chinese": "zh",
    "English": "en",
    "Japanese": "ja",
    "Korean": "ko",
    "German": "de",
    "French": "fr",
    "Russian": "ru",
    "Portuguese": "pt",
    "Spanish": "es",
    "Italian": "it",
}

MODEL = None
MODEL_ID = DEFAULT_MODEL
GENERATE_LOCK = Lock()


def base_url(host: str, port: int) -> str:
    return f"http://{host}:{port}"


def contains_cjk(text: str) -> bool:
    return bool(re.search(r"[\u3400-\u9fff]", text))


def language_code(language: str | None, text: str) -> str:
    if language and language != "Auto":
        return LANG_CODES.get(language, language)
    return "zh" if contains_cjk(text) else "en"


def to_int16_bytes(audio: Any) -> bytes:
    arr = np.asarray(audio, dtype=np.float32)
    arr = np.clip(arr, -1.0, 1.0)
    return (arr * 32767).astype(np.int16).tobytes()


def get_model():
    global MODEL
    if MODEL is None:
        from mlx_audio.tts.utils import load_model

        print(f"Loading {MODEL_ID}...", flush=True)
        MODEL = load_model(MODEL_ID)
        print(f"Model ready: sample_rate={MODEL.sample_rate}", flush=True)
    return MODEL


class TTSHandler(BaseHTTPRequestHandler):
    server_version = "qwen-tts-fast/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"{self.address_string()} - {fmt % args}", flush=True)

    def send_json(self, status: int, body: dict[str, Any]) -> None:
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def read_json(self) -> dict[str, Any]:
        size = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(size)
        return json.loads(raw.decode("utf-8")) if raw else {}

    def do_GET(self) -> None:
        if self.path == "/health":
            model = get_model()
            self.send_json(
                200,
                {
                    "status": "healthy",
                    "model": MODEL_ID,
                    "backend": "mlx",
                    "sample_rate": model.sample_rate,
                },
            )
            return
        self.send_json(404, {"error": "not_found"})

    def do_POST(self) -> None:
        if self.path != "/speak":
            self.send_json(404, {"error": "not_found"})
            return

        headers_sent = False
        try:
            req = self.read_json()
            text = str(req.get("text", "")).strip()
            if not text:
                self.send_json(400, {"error": "missing text"})
                return

            model = get_model()
            speaker = str(req.get("speaker", "Serena")).lower()
            lang_code = language_code(req.get("language"), text)
            interval = float(req.get("stream_interval", DEFAULT_STREAM_INTERVAL))
            instruct = req.get("instruct")

            kwargs = {
                "text": text,
                "voice": speaker,
                "lang_code": lang_code,
                "verbose": False,
                "stream": True,
                "streaming_interval": interval,
                "split_pattern": "",
            }
            if instruct:
                kwargs["instruct"] = instruct

            start = time.monotonic()
            first_chunk_at = None
            chunks = 0
            samples = 0

            self.send_response(200)
            self.send_header("Content-Type", "audio/pcm")
            self.send_header("Transfer-Encoding", "chunked")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            headers_sent = True

            with GENERATE_LOCK:
                for result in model.generate(**kwargs):
                    payload = to_int16_bytes(result.audio)
                    if not payload:
                        continue
                    if first_chunk_at is None:
                        first_chunk_at = time.monotonic()
                    chunks += 1
                    samples += len(payload) // 2
                    self.wfile.write(f"{len(payload):x}\r\n".encode("ascii"))
                    self.wfile.write(payload)
                    self.wfile.write(b"\r\n")
                    self.wfile.flush()

            self.wfile.write(b"0\r\n\r\n")
            elapsed = time.monotonic() - start
            audio_s = samples / model.sample_rate if model.sample_rate else 0
            ttfb = (first_chunk_at - start) if first_chunk_at else 0
            rtf = elapsed / audio_s if audio_s else 0
            print(
                f"stream done: ttfb={ttfb:.3f}s total={elapsed:.2f}s "
                f"audio={audio_s:.2f}s rtf={rtf:.2f}x chunks={chunks}",
                flush=True,
            )
        except BrokenPipeError:
            return
        except Exception as exc:
            print(f"request failed: {exc}", file=sys.stderr, flush=True)
            if not headers_sent:
                self.send_json(500, {"error": str(exc)})


def run_daemon(host: str, port: int, model_id: str) -> None:
    global MODEL_ID
    MODEL_ID = model_id

    def _shutdown_handler(signum, frame):
        print(f"Received signal {signum}, shutting down...", flush=True)
        PID_FILE.unlink(missing_ok=True)
        os._exit(0)

    signal.signal(signal.SIGTERM, _shutdown_handler)
    signal.signal(signal.SIGINT, _shutdown_handler)

    get_model()
    PID_FILE.write_text(str(os.getpid()), encoding="utf-8")
    server = HTTPServer((host, port), TTSHandler)
    print(f"Serving qwen-tts-fast on {base_url(host, port)}", flush=True)
    server.serve_forever()


def health(host: str, port: int) -> dict[str, Any] | None:
    try:
        response = httpx.get(f"{base_url(host, port)}/health", timeout=1.0)
        response.raise_for_status()
        return response.json()
    except Exception:
        return None


def start_daemon(host: str, port: int, model_id: str, timeout: float) -> None:
    if health(host, port):
        return

    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    log = LOG_FILE.open("ab")
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--daemon",
        "--host",
        host,
        "--port",
        str(port),
        "--model",
        model_id,
    ]
    subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        stdout=log,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
        close_fds=True,
    )

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if health(host, port):
            return
        time.sleep(0.25)

    tail = ""
    if LOG_FILE.exists():
        tail = subprocess.run(
            ["tail", "-n", "80", str(LOG_FILE)],
            text=True,
            capture_output=True,
        ).stdout
    raise RuntimeError(f"daemon did not become healthy within {timeout}s\n{tail}")


def stop_daemon() -> None:
    if not PID_FILE.exists():
        print("not running")
        return
    pid = int(PID_FILE.read_text(encoding="utf-8").strip())
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        PID_FILE.unlink(missing_ok=True)
        print("not running")
        return
    for _ in range(20):
        time.sleep(0.1)
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
    else:
        try:
            os.kill(pid, signal.SIGKILL)
            time.sleep(0.5)
        except ProcessLookupError:
            pass
    PID_FILE.unlink(missing_ok=True)
    print(f"stopped daemon pid={pid}")


def decode_pcm(raw: bytes) -> np.ndarray:
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def play_text(
    text: str,
    host: str,
    port: int,
    speaker: str,
    language: str,
    instruct: str | None,
    stream_interval: float,
    sample_rate: int,
) -> None:
    payload = {
        "text": text,
        "speaker": speaker,
        "language": language,
        "instruct": instruct,
        "stream_interval": stream_interval,
    }
    started = time.monotonic()
    first_chunk = None
    total_bytes = 0
    leftover = b""

    with sd.OutputStream(samplerate=sample_rate, channels=1, dtype="float32", blocksize=2048) as stream:
        with httpx.stream("POST", f"{base_url(host, port)}/speak", json=payload, timeout=None) as response:
            response.raise_for_status()
            for chunk in response.iter_bytes():
                if not chunk:
                    continue
                if first_chunk is None:
                    first_chunk = time.monotonic()
                data = leftover + chunk
                usable_len = len(data) - (len(data) % 2)
                usable = data[:usable_len]
                leftover = data[usable_len:]
                if not usable:
                    continue
                total_bytes += len(usable)
                stream.write(decode_pcm(usable))

    elapsed = time.monotonic() - started
    ttfb = (first_chunk - started) if first_chunk else 0
    audio_s = total_bytes / 2 / sample_rate
    print(f"done: ttfb={ttfb:.3f}s total={elapsed:.2f}s audio={audio_s:.2f}s")


def main() -> int:
    parser = argparse.ArgumentParser(description="Fast local Qwen3-TTS playback")
    parser.add_argument("text", nargs="*", help='Text to speak. Use "-" to read stdin.')
    parser.add_argument("--speaker", "--voice", default="Serena")
    parser.add_argument("--language", default="Auto")
    parser.add_argument("--instruct", default=None)
    parser.add_argument("--stream-interval", type=float, default=DEFAULT_STREAM_INTERVAL)
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--start-timeout", type=float, default=45.0)
    parser.add_argument("--no-auto-start", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--stop", action="store_true")
    parser.add_argument("--daemon", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.daemon:
        run_daemon(args.host, args.port, args.model)
        return 0

    if args.stop:
        stop_daemon()
        return 0

    if args.status:
        state = health(args.host, args.port)
        if state:
            print(json.dumps(state, ensure_ascii=False, indent=2))
            return 0
        print("not running")
        return 1

    if args.text == ["-"]:
        text = sys.stdin.read().strip()
    else:
        text = " ".join(args.text).strip()
    if not text:
        parser.error("Please provide text to speak.")

    if not args.no_auto_start:
        start_daemon(args.host, args.port, args.model, args.start_timeout)
    elif not health(args.host, args.port):
        raise RuntimeError("daemon is not running")

    play_text(
        text=text,
        host=args.host,
        port=args.port,
        speaker=args.speaker,
        language=args.language,
        instruct=args.instruct,
        stream_interval=args.stream_interval,
        sample_rate=args.sample_rate,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
