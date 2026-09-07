# HANDOFF — 2026-08-29, LLM-Brains

Working on: nothing in flight. This file was rewritten because register item
STL-2 said it was stale. **STL-2 is wrong about what it says** (see below) — but
the file WAS wrong about something else, and that is now corrected. No code
changed in this pass.

Last action: Verified the previous handoff's claims by compiling every `.py` in
the repo. Its headline claim — "all Python compiles except `orchestrator.py`" —
is **false**. Five files fail, not one, and they fail for the same reason.

Next step: the thing actually costing money every night is not the syntax. It is
this, at 03:00 daily, from `logs/dream_cycle.log`:

    Your credit balance is too low to access the Anthropic API.
    Done. Promoted: 0  Skipped: 3

`com.llmbrains.dreamcycle` is loaded and firing on schedule, `dream_cycle.py`
compiles and runs fine, and it promotes **nothing** because the key has no
credit. Either top up the Anthropic account or point `dream_cycle.py` at LiteLLM
on mac2 (`http://100.88.99.116:4000`), which is up. That is a Markus decision.

Key files:
  dream_cycle.py            the only thing on a timer. Compiles, runs, no credit.
  logs/dream_cycle.log      198 KB, last entry 2026-08-29 03:00
  src/compiler.py           EXISTS and compiles. See "STL-2 is wrong" below.
  orchestrator.py           broken — two scripts merged. So are four others.
  life_data_ingest.py       broken, same way
  rss_ingest.py             broken, same way
  media_store.py            broken, same way
  cross_fragment_lint.py    broken, same way

Blockers:
  - Anthropic credit balance is zero. Nightly dream cycle promotes 0 every run.
  - Five corrupted files need an author's judgment call, not a guess. Details below.

---

## STL-2 is wrong on both halves

The register says: *"LLM-Brains handoff names `src/compiler.py`, which does not
exist, for a commit that already landed."*

1. **The handoff did not name `src/compiler.py`.** Both versions in history are
   on record — `670d558` (RISK-2 / `orchestrator.py`) and `e076467` before it
   (PopSpot / Paperclip research). Neither mentions a compiler.
2. **`src/compiler.py` exists**, and it compiles clean:

       $ ls src/
       __init__.py  compiler.py  db_manager.py  embeddings.py
       pipeline.py  search.py  vault_parser.py
       $ python3 -m py_compile src/compiler.py   # exit 0

The phrase came from `~/.claude-team/chat/LLM-Brains.md:128`, a note about adding
OpenRouter key rotation *to* `src/compiler.py`. A sweep read that as a handoff
claim. It was never in a handoff.

**Do not close STL-2 as "handoff corrected".** Close it as "the register entry
was false; `src/compiler.py` is present and healthy" — and then fix the real
defect, which the previous handoff undercounted by four.

## The real defect: five files, not one

`orchestrator.py` was correctly described as two different scripts fused at the
byte level. What was missed is that **the same corruption hit four more files**.
The previous handoff said it had "verified all Python compiles except
orchestrator.py". Re-run, 2026-08-29, over all 28 `.py` files:

    FAILS: ./orchestrator.py
    FAILS: ./life_data_ingest.py
    FAILS: ./rss_ingest.py
    FAILS: ./media_store.py
    FAILS: ./cross_fragment_lint.py

The other 23 compile.

### The shape is identical every time

Two whole scripts are concatenated. The second one's `#!/usr/bin/env python3`
and its **opening** `"""` were eaten, so the second header's prose is parsed as
code and dies on its own em dash:

    File "life_data_ingest.py", line 591
      life_data_ingest.py — Personal life data ingestion from various export formats.
                          ^
    SyntaxError: invalid character '—' (U+2014)

Every boundary, measured:

| file | total lines | second copy starts | first `def` after it |
|---|---|---|---|
| `life_data_ingest.py` | 1388 | 591 | 632 |
| `rss_ingest.py` | 738 | 272 | 362 |
| `media_store.py` | 1195 | 539 | 614 |
| `cross_fragment_lint.py` | 1592 | 1065 | 1105 |
| `orchestrator.py` | 983 | interleaved, not appended — worse | — |

### Why you still must not guess-fix them

I checked whether the two halves are duplicates. **They are not.** In
`life_data_ingest.py` the halves share **zero** function names:

    before line 591:  _write_raw, ingest_apple_health, ingest_browser_history,
                      ingest_calendar, ingest_code, ingest_contacts,
                      ingest_document, ingest_email, ingest_google_takeout,
                      ingest_gpx, ingest_imessage, ingest_spotify,
                      ingest_twitter, ingest_all              (14)
    after line 591:   append_to_file, ensure_dirs, format_ics_date,
                      get_or_create_file, …                   (22)

So restoring the missing `"""` makes the file *compile* while leaving two
unrelated programs in one module, with the second silently shadowing nothing and
running nothing. That is not a fix, it is a quieter bug. The previous session's
reasoning was right — it just applied it to one file out of five.

### What makes this safe to leave for now

Nothing imports any of the five. Grepped, not assumed:

    grep -rn "import orchestrator|import media_store|import rss_ingest|
              import life_data_ingest|import cross_fragment_lint" --include='*.py' .
    (no hits)

All five are tracked in git, so nothing is at risk of being lost. The only thing
on a schedule is `dream_cycle.py`, which is clean. So these five are dead weight,
not a live outage — which is why the credit balance above outranks them.

## Repo state

    branch main, clean, level with origin/main at 670d558
    remote  https://github.com/Kaoz625/LLM-Brains.git

---

# HANDOFF — 2026-09-07, LLM-Brains (Chromium -> Comet)

Working on: Markus's order, "there should be no more use of chromium this
should be replaced with comet" — LLM-Brains was named as CONFIRMED CULPRIT #1
(skool_scraper.py caught live spawning chrome-headless-shell at 05:55, hung 14
min at 0% CPU on `--posts-only --max-pages 50`, killed by Markus).

Last action: Fixed all 3 real Chromium/Chrome launches in skool_scraper.py —
login() (was `p.chromium.launch(channel="chrome", headless=False)`),
discover_communities(), and scrape() — all now go through new
`comet_headless.py` (spawn Comet with `--headless=new`, poll
`/json/version`, `connect_over_cdp`; kills via process group). scrape() now
reuses ONE Comet CDP endpoint for crawl4ai's `BrowserConfig(use_managed_browser
=True, cdp_url=...)` instead of crawl4ai launching its own separate internal
Chromium (that double-launch was very likely PART of the hang/CPU picture).
Committed `86439d6`, pushed to `github.com/Kaoz625/LLM-Brains.git`.

Why it hung (my best-supported hypothesis, not fully proven): `crawler.arun()`
with no explicit page timeout, against Skool's persistent websocket/long-poll
connections, plus a SECOND Chromium (crawl4ai's own bundled one) fighting the
first (Playwright's) for the same site session — `waitUntil`-style "network
idle" never truly settles against a page that keeps a live socket open, so it
sits at ~0% CPU forever waiting on a network event that never completes. The
managed-browser fix (one browser, reused CDP, real Comet with real cookies
instead of a fresh automated Chromium Skool can fingerprint) addresses the
most likely trigger, but I did not reproduce the original hang under load to
prove it, so treat this as fixed-and-plausible, not fixed-and-proven.

Next step: Run `skool_scraper.py --posts-only --max-pages 50` for real (the
exact command that hung before) and confirm it completes instead of hanging.
I ran a much smaller smoke test only (see comet_headless.py's own inline
verification: example.com fetch, title "Example Domain", 559 bytes) — the
full scraper was not re-run end-to-end against Skool itself this session.

Key files:
  skool_scraper.py    the fixed culprit — 3 launch sites converted
  comet_headless.py   new helper (spawn/poll/connect/killpg), also copied
                       into nexgate's tools/ as a matching fix there

Blockers: none for the code fix. The "did the hang actually stop" question
needs one real `--posts-only --max-pages 50` run to close out for good.
