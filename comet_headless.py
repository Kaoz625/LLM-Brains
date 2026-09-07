"""
Headless Comet launcher for Python/Playwright — the house replacement for
`playwright.chromium.launch(headless=True)` and for launching Markus's real
Google Chrome (`channel="chrome"`).

WHY THIS FILE EXISTS. Markus, 2026-09-07: "there should be no more use of
chromium this should be replaced with comet." skool_scraper.py was one of two
scripts caught red-handed spawning chrome-headless-shell / Chrome for Testing
this morning. This is the ONE shared helper — mirrors
~/.claude/skills/comet-browser/scripts/comet-headless.mjs — so the launch
dance does not get copy-pasted (and drift back to Chromium) into every script
that needs a browser.

THE RECIPE, measured on Comet 151.0.7922.247 (see
~/.claude/skills/comet-browser/references/headless.md):
  --headless=new              the ONLY flag that works. Bare --headless is
                               IGNORED by Comet — it boots the full visible
                               Perplexity onboarding AND PLAYS AUDIO OUT LOUD.
  --remote-debugging-port=N   an OS-assigned free port, never 9222 (his real,
                               visible Comet — never spawn onto it).
  --user-data-dir=<scratch>   his real profile is never touched.
  NEVER --disable-gpu         it silently turns h264 decoding off; canPlayType()
                               still answers "probably" so nothing catches it.

SPAWN AND POLL, THEN connect_over_cdp — not launch(). Playwright's launch()
returns as soon as it sees "DevTools listening on ws://..." on stderr, and
Comet prints that line BEFORE it can actually serve a page. That race is what
makes `chromium.launch({executablePath: comet, headless: True})` hang or flake
(measured 37% flaky, 5/8, over 8 identical runs with the flag forced via args).
Polling /json/version until it really answers is what makes this reliable
(measured 8/8).
"""

from __future__ import annotations

import http.client
import json
import os
import shutil
import signal
import subprocess
import tempfile
import time
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass

COMET_BIN = "/Applications/Comet.app/Contents/MacOS/Comet"

# His real, visible browser. Never spawn onto this port, never touch his real
# profile dir. Use ~/bin/comet-cdp.sh to make sure something is listening here.
HIS_PORT = 9222
HIS_CDP_URL = f"http://127.0.0.1:{HIS_PORT}"


@dataclass
class HeadlessComet:
    port: int
    ws_url: str
    proc: subprocess.Popen
    profile_dir: str

    @property
    def cdp_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def stop(self) -> None:
        # Negative pid = the process GROUP. Comet's renderer/GPU/network
        # helpers are children of this pid and SURVIVE a plain proc.kill() —
        # measured on the Node helper this mirrors: 3 descendants before, 2
        # still alive after a plain kill, 0 after killing the group.
        try:
            os.killpg(self.proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                pass
        try:
            shutil.rmtree(self.profile_dir, ignore_errors=True)
        except Exception:
            pass


def _free_port() -> int:
    import socket

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _cdp_version(port: int, timeout_s: float = 2.0) -> dict | None:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout_s)
    try:
        conn.request("GET", "/json/version")
        resp = conn.getresponse()
        if resp.status != 200:
            return None
        body = json.loads(resp.read())
        return body if body.get("webSocketDebuggerUrl") else None
    except Exception:
        return None
    finally:
        conn.close()


def spawn_headless_comet(timeout_s: float = 30.0) -> HeadlessComet:
    """Spawn a scratch-profile headless Comet and wait until its CDP answers.

    Caller MUST call .stop() (use a try/finally) — an unstopped headless Comet
    survives the script and keeps its port and ~200MB.
    """
    if not os.path.exists(COMET_BIN):
        raise RuntimeError(f"Comet not found at {COMET_BIN}. Install Comet — do NOT fall back to Chrome.")

    port = _free_port()
    if port == HIS_PORT:
        raise RuntimeError(f"refusing port {HIS_PORT}: that is Markus's visible Comet")

    profile_dir = tempfile.mkdtemp(prefix="comet-headless-py-")

    proc = subprocess.Popen(
        [
            COMET_BIN,
            "--headless=new",  # the load-bearing flag; bare --headless is ignored
            f"--remote-debugging-port={port}",
            f"--user-data-dir={profile_dir}",
            "--mute-audio",  # bare --headless once played an intro out loud
            "--no-first-run",
            "--no-default-browser-check",
            "--disable-extensions",
            # DO NOT ADD --disable-gpu — silently kills h264 decoding; see module docstring.
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,  # own process group, so stop() can kill the whole tree
    )

    deadline = time.time() + timeout_s
    version = None
    while time.time() < deadline:
        version = _cdp_version(port)
        if version:
            break
        time.sleep(0.25)

    if not version:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            pass
        shutil.rmtree(profile_dir, ignore_errors=True)
        raise RuntimeError(f"Comet did not open CDP on :{port} within {timeout_s}s. Binary: {COMET_BIN}")

    return HeadlessComet(port=port, ws_url=version["webSocketDebuggerUrl"], proc=proc, profile_dir=profile_dir)


@contextmanager
def headless_comet_browser(playwright, timeout_s: float = 30.0):
    """Sync helper: `with headless_comet_browser(p) as browser:` — for
    playwright.sync_api. Replaces `p.chromium.launch(headless=True)` and
    `p.chromium.launch(channel="chrome", headless=False)`.
    """
    handle = spawn_headless_comet(timeout_s=timeout_s)
    try:
        browser = playwright.chromium.connect_over_cdp(handle.cdp_url)
        try:
            yield browser
        finally:
            # Do not call browser.close() on a CDP connection tied to a process
            # we own — that races the SIGKILL below. Just disconnect.
            try:
                browser.close()
            except Exception:
                pass
    finally:
        handle.stop()


@asynccontextmanager
async def headless_comet_browser_async(playwright, timeout_s: float = 30.0):
    """Async helper: `async with headless_comet_browser_async(pw) as browser:`
    — for playwright.async_api. Replaces `await pw.chromium.launch(headless=True)`.
    """
    handle = spawn_headless_comet(timeout_s=timeout_s)
    try:
        browser = await playwright.chromium.connect_over_cdp(handle.cdp_url)
        try:
            yield browser, handle
        finally:
            try:
                await browser.close()
            except Exception:
                pass
    finally:
        handle.stop()


def visible_comet_available(timeout_s: float = 2.0) -> bool:
    """True if Markus's real, visible Comet already has CDP open on :9222
    (started via ~/bin/comet-cdp.sh). Never spawn to make this true."""
    return _cdp_version(HIS_PORT, timeout_s=timeout_s) is not None
