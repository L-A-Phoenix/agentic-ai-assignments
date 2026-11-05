# Encore — CSCI 391 Midterm Starter

A ListenBrainz music agent that remembers. The full assignment is in `Midterm.html`.

```
encore.py            the starter: complete every TODO (Part 8)
prompts/             procedural memory: system_v1.md (700-word limit) and CHANGELOG.md
traces/              one JSONL trace per session (Part 6)
memory_examples/     runnable examples and guides for every memory store (see its README)
tests/unit/          48 offline tests of the provided tools and helpers
tests/live/          28 tests against the real ListenBrainz and MusicBrainz APIs:
  test_live_apis.py        14  the provided tools, end to end (read-only)
  test_live_endpoints.py   10  every endpoint your Part 3 tools call (read-only)
  test_live_writes.py       4  write round trips on your account (opt-in, restored)
.env.example         copy to .env and fill in
```

## Setup

```
python -m pip install -r requirements.txt
cp .env.example .env            # Windows PowerShell: copy .env.example .env
python -m pytest tests/unit     # should pass before you change anything
```

## Tests

```
python -m pytest tests/unit     # offline: fake network and clock, under a second
python -m pytest tests/live     # real APIs, read-only, about a minute
python -m pytest                # both
```

`test_live_endpoints.py` is executable documentation: each test shows the URL,
parameters, and response shape one of your tools depends on. The LB Radio and
popularity tests need `LISTENBRAINZ_TOKEN`: ListenBrainz refuses anonymous calls
to those endpoints (HTTP 401), so your tools must send the token there too.

### Write round trips (opt-in)

```
ENCORE_WRITE_TESTS=1 python -m pytest tests/live/test_live_writes.py -v
# PowerShell:  $env:ENCORE_WRITE_TESTS=1; python -m pytest tests/live/test_live_writes.py -v
```

These use your token on `LISTENBRAINZ_USER`. They create a private playlist titled
`[Encore test] …`, add and remove a track, and delete it; then love one recording,
clear it, and restore whatever feedback you had before. Cleanup runs even when a
test fails, and leftover `[Encore test]` playlists are deleted on the next run.
If a test prints `CLEANUP FAILED`, fix that item by hand on listenbrainz.org.

### Export: manual check (not automated)

Exporting creates a real playlist in Spotify or Apple Music that no test can
remove, so check it once by hand after `export_playlist` works:

1. In ListenBrainz settings, confirm the service is connected with playlist permission.
2. Ask Encore to make a two-song playlist, then to export it; approve the prompt.
3. Confirm the playlist appears in your Spotify or Apple Music library, then delete it there.
4. Disconnect the service in ListenBrainz and export again: Encore must report a
   `fatal` error that tells you what to fix, and must not retry.

Record the date and result in your README.

You can also open a test file and press VS Code's Run button. The live tests skip
with a reason if the APIs can't be reached. Don't run them during class:
MusicBrainz limits requests per IP address, and the class shares one on campus.

These tests cover the provided code and must keep passing. If you change a
provided function on purpose, update its tests and explain why in this README.
Add tests for the tools you write in the same two layers.

## Your README (replace this section)

Before you submit, document: how to run Encore, your memory design, retention
policy and dedup threshold, the embedding model, and known limitations.
