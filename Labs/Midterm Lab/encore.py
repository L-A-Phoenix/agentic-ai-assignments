"""
Encore -- a ListenBrainz music agent that remembers.   (CSCI 391 midterm starter)

    python encore.py <online|offline>             # interactive session
    python encore.py <online|offline> --forget    # delete every memory for ENCORE_USER_ID

Copy chat() and model_for() from your Lab 3 agent and your router from Lab 4.
"""

import json
import os, sys
import random
import time
import jsonschema
from datetime import datetime, timezone, timedelta
from dataclasses import asdict, dataclass
from typing import Any, Optional

# Memory imports
import chromadb
import uuid

import requests
from dotenv import load_dotenv

load_dotenv()

LB_API = "https://api.listenbrainz.org/1"
MB_API = "https://musicbrainz.org/ws/2"
LB_TOKEN = os.getenv("LISTENBRAINZ_TOKEN")        # never log or store this
LB_USER = os.getenv("LISTENBRAINZ_USER")
ENCORE_USER_ID = os.getenv("ENCORE_USER_ID", LB_USER)  # deletion key for all memory
USER_AGENT = f"Encore-CSCI391/0.1 ( {os.getenv('CONTACT_EMAIL', 'you@umontana.edu')} )"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

MAX_STEPS = 12          # bounded loop (Lab 2)
WINDOW_SIZE = 20
READ_TIMEOUT_S = 5
WRITE_TIMEOUT_S = 15
SLOW_READ_TIMEOUT_S = 15  # override for reads measured to be slow (ListenBrainz listens)
MB_MIN_INTERVAL_S = 1.5  # MusicBrainz allows about 1 request/second per IP; leave headroom

_last_mb_call = 0.0
_lb_wait_until = 0.0

TIME_OFFSET = 0
# Memory
EMBEDDING_MODEL = "chromadb-default"
EMBEDDING_VERSION = "v1"
STALENESS_THRESHOLD = 30
client = chromadb.PersistentClient(path="./memory/chroma")

# Working Memory
HOT_STATE_FILE = "memory/hot_state.json"

def load_hot_state():
    default = {
        "active_playlist": None,
        "last_recording": None,
        "last_artist": None,
        "preferences": [],
        "pending_approval": None

    }
    if not os.path.exists(HOT_STATE_FILE):
        return default
    try:
        with open(HOT_STATE_FILE, "r")as file:
            return json.load(file)
    except(json.JSONDecodeError, OSError):
        return default

def save_hot_state(state):
    with open(HOT_STATE_FILE, "w") as file:
        json.dump(state, file, indent=2)


# ---Procedural Memory---


# ---Semantic Memory---
semantic_collection = client.get_or_create_collection(
    name="semantic",
    metadata={"hnsw:space": "cosine"},
)

def load_semantic_docs():
    if semantic_collection.count() > 0:
            return
    source_files = [
        ("docs/glossary.txt", "glossary"),
        ("docs/listenbrainz.txt", "listenbrainz-docs"),
        ("docs/musicbrainz.txt", "musicbrainz-docs")
    ]
    for path, source in source_files:
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf8") as f:
            text = f.read()
        chunks = text.split("\n\n")
        semantic_collection.upsert(
            ids=[str(uuid.uuid4()) for _ in chunks],
            documents=chunks,
            metadatas=[{
                "source": source,
                "embedding_model": EMBEDDING_MODEL,
                "embedding_version": EMBEDDING_VERSION
            } for _ in chunks])
        log_memory("write", "semantic", {
            "source": source,
            "chuncks": len(chunks)
        })

# ---Episodic Memory---
episodic_collection = client.get_or_create_collection(
    name="episodic",
    metadata={"hnsw:space": "cosine"},
)

def write_session_summary(all_steps: list[str], mode: str):
    step_text = "\n".join(str(step.get("content", "")) for step in all_steps)
    response = chat(SUMMARY_PROMPT + step_text, "cheap", mode, [SUMMARY_SCHEMA], FORCE_SUMMARY)
    if not response or "tool_calls" not in response:
        print("ERROR: Summary generation failed")
        return {
            "decisions": [],
            "preferences": [],
            "open_threads": [],
            "key_facts": [],
            "errors": [],
            "topics": []
        }
    summary = json.loads(response["tool_calls"][0]["function"]["arguments"])
    return summary

def store_session_summary(summary, quality_info):
    text_summary = ""
    for item in ["decisions","preferences","open_threads","key_facts","errors"]:
        text_summary += (f"{item.upper()}:\n")
        if summary[item]:
            text_summary += "\n".join(summary[item])
        text_summary += "\n"
    log_memory("write", "episodic", {
        "topics": summary["topics"],
        "quality": quality_info["quality"]
    })
    
    episodic_collection.upsert(
        ids=[str(uuid.uuid4())],
        documents=[text_summary],
        metadatas=[{
            "user_id": ENCORE_USER_ID,
            "timestamp": current_time().isoformat(),
            "topics": ",".join(summary["topics"]),
            "quality": quality_info["quality"],
            "issues": ";".join(quality_info["issues"]),
            "embedding_model": EMBEDDING_MODEL,
            "embedding_version": EMBEDDING_VERSION,
        }]
    )

def retrieve_memories(query: str):
    results = episodic_collection.query(
        query_texts=[query],
        n_results=3,
        where={"user_id": ENCORE_USER_ID}
    )
    if not results["documents"][0]:
        return ""
    rescored = []
    #! Recency Weighting
    for doc, meta, dist in zip(
        results["documents"][0], 
        results["metadatas"][0],
        results["distances"][0],
     ):
        similarity = 1 - dist
        ts = datetime.fromisoformat(meta["timestamp"])
        age_days = (current_time() - ts).days
        recency = 1/(1 + age_days)
        score = (0.8*similarity + 0.2*recency)
        
        rescored.append((score, doc, meta))
    rescored.sort(reverse=True, key=lambda x: x[0])

    memory_text = ""
    for score, doc, meta in rescored[:3]:
        quality = meta.get("quality","high")
        if quality == "low":
            memory_text += ("[LOW CONFIDENCE MEMORY]\n"
                            "Verify before using.\n")
        memory_text += (f"[{meta['timestamp']}]\n{doc}\n\n")

        if memory_is_stale(meta):
            memory_text += ("[STALE MEMORY]\n"
                            "This memory may be outdated. \n")

    version = meta.get("embedding_version")
    if version != EMBEDDING_VERSION:
        memory_text += (
            "[OLD EMBEDDING VERSION]\n"
            "Memory may require re-indexing.\n"
        )
    
    return memory_text

# Checks similarity on embeddings
def memory_exists(text, threshold=0.93):
    if episodic_collection.count() == 0:
        return False
    results = episodic_collection.query(
        query_texts=[text], 
        n_results=1,
        where={"user_id": ENCORE_USER_ID})
    similarity = (1-results["distances"][0][0])

    return similarity >= threshold


@dataclass
class ToolResult:                      # same contract as the Chapter 6 examples
    success: bool
    data: Optional[Any]
    error: Optional[str]
    error_category: Optional[str]      # "transient" | "input" | "resource" | "fatal"

    def to_json(self) -> str:
        return json.dumps(asdict(self))


def ok(data: Any) -> ToolResult:
    return ToolResult(True, data, None, None)

def fail(category: str, message: str) -> ToolResult:
    return ToolResult(False, None, message, category)

def classify_status(status: int) -> str:
    if status == 0 or status == 429 or status >= 500:
        return "transient"
    if status in (401, 403):
        return "fatal"
    if status == 404:
        return "resource"
    return "input"

def http(method: str, url: str, *, params=None, body=None, auth=False, timeout=READ_TIMEOUT_S):
    """One polite request. Returns (status, json_or_text, headers). Status 0 = no response."""
    global _last_mb_call, _lb_wait_until
    if url.startswith(MB_API):
        wait = _last_mb_call + MB_MIN_INTERVAL_S - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _last_mb_call = time.monotonic()
        params = {**(params or {}), "fmt": "json"}
    elif url.startswith(LB_API):
        wait = _lb_wait_until - time.monotonic()
        if wait > 0:                                  # honor the server's rate-limit headers
            time.sleep(wait)
    headers = {"User-Agent": USER_AGENT}
    if auth:
        headers["Authorization"] = f"Token {LB_TOKEN}"
    try:
        r = requests.request(method, url, params=params, json=body, headers=headers, timeout=timeout)
    except requests.RequestException as exc:
        return 0, f"{type(exc).__name__}", {}
    if url.startswith(LB_API) and r.headers.get("X-RateLimit-Remaining") == "0":
        _lb_wait_until = time.monotonic() + float(r.headers.get("X-RateLimit-Reset-In", "1"))
    try:
        return r.status_code, r.json(), r.headers
    except ValueError:
        return r.status_code, r.text, r.headers

# ---------------------------------------------------------------- worked examples
# Domain: Listening
def get_recent_listens(count: int = 25) -> str:
    if not 1 <= count <= 100:
        return fail("input", f"count must be between 1 and 100; got {count}.").to_json()
    status, body, _ = http("GET", f"{LB_API}/user/{LB_USER}/listens", params={"count": count},
                           timeout=SLOW_READ_TIMEOUT_S)
    if status != 200:
        return fail(classify_status(status), f"ListenBrainz listens request failed (HTTP {status}).").to_json()
    listens = []
    for item in body["payload"]["listens"]:
        meta = item["track_metadata"]
        listens.append({
            "listened_at": item["listened_at"],
            "artist": meta.get("artist_name"),
            "track": meta.get("track_name"),
            "recording_mbid": (meta.get("mbid_mapping") or {}).get("recording_mbid"),
        })
    return ok({"listens": listens}).to_json()

def get_feedback(score: str, count: int=25):
    if not score.strip():
        return fail("input", "Feedback score 'score' must not be empty.").to_json()
    if score not in ["love", "hate"]:
        return fail("input", "Score 'score' is invalid. Must be 'love' or 'hate'.").to_json()
    if not 1 <= count <= 100:
        return fail("input", "Count 'count' must be between 1 and 100.").to_json()
    status, body, _ = http("GET", f"{LB_API}/feedback/user/{LB_USER}/get-feedback", params={"score": score, "count": count}, auth=True)
    if status != 200:
        return fail(classify_status(status), f"Failed to get feedback (HTTP {status})").to_json()
    feedback = []
    for item in body.get("feedback", []):
        feedback.append({
            "recording_mbid": item.get("recording_mbid"),
            "recording_name": item.get("recording_name"),
            "artist_name": item.get("artist_name"),
            "score": item.get("score")
        })
    if not feedback:
        return fail("resource", f"No recordings found with feedback '{score}'.").to_json()
    return ok({"feedback": feedback}).to_json()

# Domain: Catalog
def search_recordings(title: str, artist: str = "", limit: int = 5) -> str:
    if not title.strip():
        return fail("input", "Song title 'title' must not be empty.").to_json()
    query = f'recording:"{title}"' + (f' AND artist:"{artist}"' if artist else "")
    status, body, _ = http("GET", f"{MB_API}/recording", params={"query": query, "limit": limit})
    if status != 200:
        return fail(classify_status(status), f"MusicBrainz search failed (HTTP {status}).").to_json()
    hits = [{
        "recording_mbid": r["id"],
        "title": r["title"],
        "artist": "".join(c["name"] + c.get("joinphrase", "") for c in r.get("artist-credit", [])),
        "length_s": (r.get("length") or 0) // 1000,
        "first_release": r.get("first-release-date"),
    } for r in body.get("recordings", [])]
    if not hits:
        return fail("resource", f"No MusicBrainz recording matches '{title}'"
                                f"{' by ' + artist if artist else ''}.").to_json()
    return ok({"recordings": hits}).to_json()

def lookup_artist(artist_mbid: str):
    if not artist_mbid.strip():
        return fail("input", "Artist mbid 'artist_mbid' must not be empty.").to_json()
    status, body, _ = http("GET", f"{MB_API}/artist/{artist_mbid}", params={"inc": "artist-rels+url-rels"})
    if status != 200:
        return fail(classify_status(status), f"MusicBrainz search failed (HTTP {status}).").to_json()
    if not body:
        return fail("resource", f"No MusicBrainz artist matches '{artist_mbid}'").to_json()

    result = {
        "artist_mbid": body["id"],
        "name": body["name"],
        "type": body.get("type"),
        "relations": body.get("relations",[])
    }
    return ok({"artist": result}).to_json()

#THIS NF
def get_top_recordings_for_artist(artist_mbid: str, limit: int=5):
    if not artist_mbid.strip():
        return fail("input", "Artist mbid 'artist_mbid' must not be empty.").to_json()
    status, body, _ = http("GET", f"{LB_API}/artist/{artist_mbid}/top-recordings", auth=True)
    if status != 200:
        return fail(classify_status(status), f"MusicBrainz search failed (HTTP {status}).").to_json()
    return ok({"recordings": body.get("payload", [])}).to_json()

def verify_recordings(recording_mbids: list[str]):
    if not recording_mbids:
        return fail("input", "Recording mbids 'recording_mbids' must not be empty.").to_json()
    verified_mbids = []
    for mbid in recording_mbids:
        status, body, _ = http("GET", f"{MB_API}/recording/{mbid}")
        if status == 200:
            verified_mbids.append({
                "mbid": mbid,
                "title": body.get("title")
            })
    if not verified_mbids:
        return fail("resource", f"No valid recordings in {recording_mbids}").to_json()
    return ok({"verified": verified_mbids}).to_json()


# Domain: Discovery
def get_fresh_releases():
    status, body, _ = http("GET", f"{LB_API}/user/{LB_USER}/fresh_releases")
    if status != 200:
        return fail(classify_status(status), f"Failed to get fresh release ({status})").to_json()
    return ok({"releases": body.get("payload", {})}).to_json()

def lb_radio(query: str, mode: str):
    if mode not in ["easy","medium","hard"]:
        return fail("input",f"Mode '{mode}' is invalid. Must be 'easy', 'medium', or 'hard'").to_json()
    status, body, _ = http("GET", f"{LB_API}/explore/lb-radio", params={"prompt": query,
                                                                        "mode": mode}, auth=True)
    if status != 200:
        return fail(classify_status(status), f"LB Radio request failed (HTTP {status})").to_json()
    return ok({
        "tracks": body.get("payload", [])
    }).to_json()
    
# Domain: Library
def find_playlists():
    status, body, _ = http("GET", f"{LB_API}/user/{LB_USER}/playlists",auth=True)
    if status != 200:
        return fail(classify_status(status), f"Playlist lookup failed ({status})").to_json()
    playlists = []
    for p in body.get("playlists", []):
        playlists.append({
            "playlist_id": p["identifier"],
            "title": p["title"]
        })
    return ok({"playlists": playlists}).to_json()

def create_playlist(playlist_title: str, description: str=""):
    if not playlist_title.strip():
        return fail("input","Playlist title 'title' must not be empty.").to_json()
    payload = {
        "playlist": {
            "title": playlist_title,
            "description": description
        }
    }
    status, body, _ = http("POST", 
                           f"{LB_API}/playlist/create",
                           body = payload,
                           auth = True)
    if status != 200:
        return fail(classify_status(status), f"Failed to create playlist (HTTP{status})").to_json()
    return ok({"playlist_id": body["playlist_mbid"]}).to_json()
    
def add_playlist_items(playlist_mbid: str, recording_mbids: list[str]):
    if not playlist_mbid.strip():
        return fail("input","Playlist mbid 'playlist_mbid' must not be empty.").to_json()
    if not recording_mbids:
        return fail("input","Recording mbids 'recording_mbids' field must not be empty").to_json()
    payload = {"recording_mbids": recording_mbids}
    status, body, _ = http("POST", 
                           f"{LB_API}/playlist/{playlist_mbid}/item/add",
                           body = payload,
                           auth = True)
    if status != 200:
        return fail(classify_status(status), f"Failed to add songs (HTTP{status})").to_json()
    return ok({"playlist_id": body["playlist_mbid"]}).to_json()
    
def remove_playlist_items(playlist_mbid: str, items_to_remove: list[str]):
    if not playlist_mbid.strip():
        return fail("input","Playlist mbid 'playlist_mbid' field must not be empty.").to_json()
    if not items_to_remove:
        return fail("input","Songs to remove 'items_to_remove' field must not be empty").to_json()
    results = []
    for item in items_to_remove:
        payload = {"item": item}
        status, body, _ = http("POST", 
                            f"{LB_API}/playlist/{playlist_mbid}/item/delete",
                            auth=True,
                            body=payload
                            )
        if status != 200:
            return fail(classify_status(status), f"Failed to remove songs (HTTP {status})").to_json()
        results.append(body)
    return ok({"removed_items": results}).to_json()
    
def set_feedback(recording_mbid: str, score: str):
    if not recording_mbid.strip():
        return fail("input", "Song mbid field 'recording_mbid' must not be empty.").to_json()
    if not score.strip():
        return fail("input", "Score field 'score' must not be empty.").to_json()
    if score not in ["love","hate","clear"]:
        return fail("input","Score 'score' is invalid. Must be 'love', 'hate', or 'clear'.").to_json()
    payload = {"recording_mbid": recording_mbid, "score": score}
    status, body, _ = http("POST", 
                           f"{LB_API}/feedback/recording-feedback",
                           auth=True,
                           body = payload)
    if status != 200:
        return fail(classify_status(status), f"Feedback update request failed (HTTP {status})").to_json()
    return ok(body).to_json()
    
def export_playlist(playlist_mbid: str, service: str):
    if not playlist_mbid.strip():
        return fail("input", "Playlist mbid field 'playlist_mbid' must not be empty.").to_json()
    if not service.strip():
        return fail("input", "Service 'service' must not be empty.").to_json()
    if service not in ["spotify", "apple_music"]:
        return fail("input", "Service 'service' is invalid. Must be 'spotify' or 'apple_music'.").to_json()
    status, body, _ = http("POST", f"{LB_API}/playlist/{playlist_mbid}/export/{service}", auth=True)
    if classify_status(status) == "fatal":
        return fail("fatal", "Spotify not connected to user").to_json()
    if status != 200:
        return fail(classify_status(status), f"Failed export request (HTTP {status}).").to_json()
    return ok(body).to_json()

def delete_playlist(playlist_title: str, playlist_mbid: str):
    if not playlist_mbid.strip():
        return fail("input", "Playlist mbid 'playlist_mbid' must not be empty.").to_json()
    if "[Encore]" not in playlist_title:
        return fail("input", "Playlist must be an Encore-created playlist to qualify for deletion.").to_json()
    status, body, _ = http("POST", 
                           f"{LB_API}/playlist/{playlist_mbid}/delete",
                           auth=True)
    if status != 200:
        return fail(classify_status(status), f"Failed to delete playlist (HTTP {status}).").to_json()
    return ok(body).to_json()

# Domain: Memory
def search_past_sessions(query: str):
    if not query.strip():
        return fail("input", "Query 'query' must not be empty.").to_json()
    results = episodic_collection.query(query_texts=[query],
                                        n_results=2,
                                        where={"user_id": ENCORE_USER_ID})
    docs = results["documents"][0]
    if not docs:
        return fail("resource", "No relevant session memories found.").to_json()
    memories = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        memories.append({"summary": doc, 
                       "topics": meta["topics"], 
                       "timestamp": meta["timestamp"],
                       "similarity": round(1 - dist, 3)})
    return ok({"memories": memories}).to_json()

def search_music_knowledge(query: str):
    if not query.strip():
        return fail("input", "Query 'query' must not be empty.").to_json()
    results = semantic_collection.query(query_texts=[query], n_results=5)
    docs = results["documents"][0]
    if not docs:
        return fail("resource", "No matching knowledge found.").to_json()
    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        chunks.append({"text": doc, 
                       "source": meta["source"], 
                       "similarity": round(1 - dist, 3)})
    return ok({"chunks": chunks}).to_json()
    
def note_preference(preference: str):
    if not preference.strip():
        return fail("input", "Preference 'preference' must not be empty.").to_json()
    state = load_hot_state()
    prefs = state.get("preferences", [])

    if preference not in prefs:
        prefs.append(preference)
    state["preferences"] = prefs
    save_hot_state(state)

    return ok({"stored": preference}).to_json()

# Domain: Control
def final_answer(status: str, answer: str, confidence: str, sources: str):
    return ok({
        "status": status,
        "answer": answer,
        "confidence": confidence,
        "sources": sources
    }).to_json()

# I don't really know what to do for fault injection
def inject_fault(error_type: str):
    if error_type == "transient":
        return fail("transient", "Injected transient error.").to_json()
    if error_type == "input":
        return fail("input", "Injected input error.").to_json()
    if error_type == "resource":
        return fail("resource", "Injected resource error.").to_json()
    if error_type == "fatal":
        return fail("fatal", "Injected fatal error.").to_json()
    return fail("unknown", "Injected unknown error '{error_type}'.").to_json()
    
# TODO 1: implement the remaining required tools in Part 3 (same ToolResult contract)

# TODO 2: write a JSON schema for every tool: descriptions say when NOT to use it,
#         bounded values are enums, write tools state their side effects (Part 3)

# ---Router Stuff---
ROUTER_PROMPT = """
You are a request router. Call classify_intent with every toolset
domain the user's request needs. A request may need more than one.

- listening: Get recent listens, feedback on loved/hated songs
- catalog: Search for songs and artists, look up relationships
- discovery: Find new releases, get recommendations from ListenBrainz Radio
- library: Find, create, edit, export playlists; set feedback
- memory: Search past conversations or music knowledge
- none: User is asking a general question that doesn't need tools (small talk, general knowledge)

Do not answer the request. Only classify it.
"""
CLASSIFIER_SCHEMA = {
    "type": "function",
    "function": {
        "name": "classify_intent",
        "description": "Report which toolset domains the request needs.",
        "parameters": {
            "type": "object",
            "properties": {
                "domains": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["listening", "catalog", "discovery", 
                                                         "library", "memory","none"]},
                    "description": "Every toolset domain the request needs."
                }
            },
            "required": ["domains"],  
        },
    },
}
FORCE_CLASSIFIER = {"type": "function", "function": {"name": "classify_intent"}}

def route_tools(user_input: str, mode):
    message = chat(ROUTER_PROMPT, "cheap", mode, [CLASSIFIER_SCHEMA], FORCE_CLASSIFIER, user_input)
    if not message or "tool_calls" not in message:
        print("ERROR: Router failed to classify intent")
        return ["none"]
    tool_arguments = json.loads(message["tool_calls"][0]["function"]["arguments"])
    domains = tool_arguments["domains"]
    not_validated = validate_tool(tool_arguments, CLASSIFIER_SCHEMA["function"]["parameters"])
    if not_validated is not None:
        print(f"ERROR: Router output failed validation: {not_validated}")
        return ["none"]
    tools = []
    for domain in domains:
        tools.extend(TOOLSETS[domain])
    if not domains:
        domains = ["none"]

    return domains

# ---SUMMARY STUFF---
FORCE_SUMMARY = {"type": "function", "function": {"name": "write_session_summary"}}
SUMMARY_SCHEMA = {
    "type": "function",
    "function": {
        "name": "write_session_summary",
        "description": "",
        "parameters": {
            "type": "object",
            "properties": {
                "decisions": {"type": "array", "items": {"type": "string"}},
                "preferences": {"type": "array", "items": {"type": "string"}},
                "open_threads": {"type": "array", "items": {"type": "string"}},
                "key_facts": {"type": "array", "items": {"type": "string"}},
                "errors": {"type": "array", "items": {"type": "string"}},
                "topics": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["decisions", "preferences", "open_threads", "key_facts", "errors", "topics"],
            "additionalProperties": False,
        }
    }
}
SUMMARY_PROMPT = """
Summarize the session.

Return:

- decisions
- preferences
- open_threads
- key_facts
- errors
- topics

Only include information explicitly discussed.
"""

COMPRESSION_PROMPT = """
Compress conversation history.

Preserve:
- decisions
- preferences
- open_threads
- important facts
- mistakes/errors

Discard:
- repetition
- unsuccessful reasoning
- transient tool outputs

Return the structured summary.
"""
# ---Quality Stuff---
FORCE_QUALITY = {"type": "function", "function": {"name":"quality_check"}}
QUALITY_SCHEMA = {
    "type": "function",
    "function": {
        "name": "quality_check",
        "description": "Evaluate whether a session summary is factually supported by the transcript.",
        "parameters": {
            "type": "object",
            "properties": {
                "quality": {
                    "type": "string",
                    "enum": ["high", "low"]
                },
                "issues": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "reason": {"type": "string"}
            },
            "required": [
                "quality",
                "issues",
                "reason"
            ],
            "additionalProperties": False
        }
    }
}

QUALITY_PROMPT = """
You are evaluating a session summary.

Check whether every fact in the summary is supported
by the session transcript.

Flag:
- invented preferences
- invented decisions
- invented facts
- missing context
- contradictions

Return:
quality = 'high' or 'low'
high: summary is supported by transcript.
low: summary contains unsupported or contradictory claims.
"""

def quality_check(summary: str, all_steps: list[str], mode: str):
    transcript = "\n".join(str(step.get("content", "")) for step in all_steps)

    payload = f"""
    TRANSCRIPT:
    {transcript}

    SUMMARY:
    {json.dumps(summary, indent=2)}
    """
    response = chat(
        QUALITY_PROMPT + payload,
        "cheap",
        mode = mode,
        tools = [QUALITY_SCHEMA],
        tool_choice = FORCE_QUALITY
    )

    if not response or "tool_calls" not in response:
        return {"quality": "low", "issues": ["Quality check failed"], "reason": "No response"}
    result = json.loads(
        response["tool_calls"][0]["function"]["arguments"]
    )
    return result

# ---REACT Stuff---
FORCE_FINAL = {"type": "function", "function": {"name": "final_answer"}}
TOOLSETS = {
    "listening": ["get_recent_listens", "get_feedback"],
    "catalog": ["search_recordings","lookup_artist", "get_top_recordings_for_artist", "verify_recordings"],
    "discovery": ["get_fresh_releases", "lb_radio"],
    "library": ["find_playlists","create_playlist","add_playlist_items","remove_playlist_items","set_feedback","export_playlist","delete_playlist"],
    "memory": ["search_past_sessions","search_music_knowledge","note_preference"],
    "none": [],
}
DOMAINS = list(TOOLSETS)

TOOLS = {
    "get_recent_listens": get_recent_listens,
    "get_feedback": get_feedback,
    "search_recordings": search_recordings,
    "lookup_artist": lookup_artist,
    "get_top_recordings_for_artist": get_top_recordings_for_artist,
    "verify_recordings": verify_recordings,
    "get_fresh_releases": get_fresh_releases,
    "lb_radio": lb_radio,
    "find_playlists": find_playlists,
    "create_playlist": create_playlist,
    "add_playlist_items": add_playlist_items,
    "remove_playlist_items": remove_playlist_items,
    "set_feedback": set_feedback,
    "export_playlist": export_playlist,
    "delete_playlist": delete_playlist,
    "search_past_sessions": search_past_sessions,
    "search_music_knowledge": search_music_knowledge,
    "note_preference": note_preference,
    "final_answer": final_answer,
}
WRITE_TOOLS = {
    "create_playlist": create_playlist,
    "add_playlist_items": add_playlist_items,
    "remove_playlist_items": remove_playlist_items,
    "set_feedback": set_feedback,
    "export_playlist": export_playlist,
    "delete_playlist": delete_playlist,
}
TOOL_SCHEMAS = {
    "get_recent_listens": {
        "type": "function",
        "function": {
            "name": "get_recent_listens",
            "description": "Returns songs the user has listened to recently. "
                            "Pass 'count' as an optional value (minimum:1 maximum:100). "
                            "Use when user requests songs that they have recently played or when you need listening history. "
                            "Do NOT use when the recency of songs is not relevant to the user's question.",
            "parameters": {
                "type": "object",
                "properties": {
                    "count": {"type": "integer",
                              "minimum": 1,
                              "maximum": 100,}
                },
                "additionalProperties": False,
            }
        }
    },
    "get_feedback": {
        "type": "function",
        "function": {
            "name": "get_feedback",
            "description": "Returns users loved or hated songs depending on 'score'. "
                            "Pass 'score' as a required value, 'limit' as an optional value. "
                            "Use when user asks about songs they have hated or loved. "
                            "Do NOT use when user does not specifically ask for songs they have scored.",
            "parameters": {
                "type": "object",
                "properties": {
                    "score": {"type": "string", "enum": ["love","hate"]},
                    "count": {"type": "integer"},
                },
                "required": ["score"],
                "additionalProperties": False,
            }
        }
    },
    "search_recordings": {
        "type": "function",
        "function": {
            "name": "search_recordings",
            "description": "Searches for songs by title with optional artist name and returns a list of relevant songs. "
                            "Pass 'title' as a required value, 'artist' as an optional value, and 'limit' as an optional value. "
                            "Use when user is looking for a specific song. "
                            "Do NOT use this if looking for an artist, use 'lookup_artist' instead.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "artist": {"type": "string"},
                    "limit": {"type": "integer"}
                },
                "required": ["title"],
                "additionalProperties": False,
            }
        }
    },
    "lookup_artist": {
        "type": "function",
        "function": {
            "name": "lookup_artist",
            "description": "Look up a MusicBrainz artist and relationships. Returns artist name, mbid, and type. "
                            "Returns information about the artist. "
                            "Pass 'artist_mbid'. "
                            "Use for questions about additional info on artist. "
                            "Do NOT use for song searches. ",
            "parameters": {
                "type": "object",
                "properties": {
                    "artist_mbid": {"type": "string"},
                },
                "required": ["artist_mbid"],
                "additionalProperties": False,
            }
        }
    },
    "get_top_recordings_for_artist": {
        "type": "function",
        "function": {
            "name": "get_top_recordings_for_artist",
            "description": "Retrieve the most popular songs for an artist. "
                            "Returns a list of 'limit' number of the artist most popular songs. "
                            "Pass 'artist_mbid' and 'limit' (defaults limit to 5). "
                            "Use when exploring an artist's catalog. "
                            "Do NOT use to verify song's existence.",
            "parameters": {
                "type": "object",
                "properties": {
                    "artist_mbid": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                "required": ["artist_mbid"],
                "additionalProperties": False,
            }
        }
    },
    "verify_recordings": {
        "type": "function",
        "function": {
            "name": "verify_recordings",
            "description": "Verify that songs MBIDs exist in MusicBrainz. Required BEFORE any playlist write. "
                            "Returns a list of validated songs from 'recording_mbids' if any exist. "
                            "Pass 'recording_mbids'. "
                            "Use before create_playlist, add_playlist_items, or export operations. "
                            "Do NOT use to search for recordings.",
            "parameters": {
                "type": "object",
                "properties": {
                    "recording_mbids": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["recording_mbids"],
                "additionalProperties": False,
            }
        }
    },
    "get_fresh_releases": {
        "type": "function",
        "function": {
            "name": "get_fresh_releases",
            "description": "Retrieve recent releases from artists the user follows or listens to. "
                            "Pass no arguments. "
                            "Use when the user asks about new music. "
                            "Do NOT use for playlist creation.",
            "parameters": {
                "type": "object",
                "properties": {}
            },
            "additionalProperties": False,
        }
    },
    "lb_radio": {
        "type": "function",
        "function": {
            "name": "lb_radio",
            "description": "Generate music recommendations from ListenBrainz. "
                            "Returns a list of recommendations. "
                            "Pass 'query' and 'mode' ('mode' controls recommendation diversity). "
                            "Use for discovery and playlist building. "
                            "Do NOT use if the user specifically requests their own listening history.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "mode": {"type": "string", "enum": ["easy","medium","hard"]}
                },
                "required": ["query","mode"],
                "additionalProperties": False,
            }
        }
    },
    "find_playlists": {
        "type": "function",
        "function": {
            "name": "find_playlists",
            "description": "Retrieve playlists from the user's account. "
                            "Returns a list of all user's playlists. "
                            "Pass no arguments. "
                            "Use before playlist creation to avoid duplicates or to retrieve specific playlist for user. "
                            "Do NOT use to modify playlists.",
            "parameters": {
                "type": "object",
                "properties": {}
            },
            "additionalProperties": False,
        }
    },
    "create_playlist": {
        "type": "function",
        "function": {
            "name": "create_playlist",
            "description": "Create a playlist in the user's account. Will create or update account content. Must call 'find_playlists' BEFORE creating "
                            "Returns the mbid of the created/updated playlist. "
                            "Pass 'title' and 'description' (description defaults to ''). "
                            "Use find_playlists before calling. "
                            "Do NOT use if approval has not been granted.",
            "parameters": {
                "type": "object",
                "properties": {
                    "playlist_title": {"type": "string"},
                    "description": {"type": "string"}
                },
                "required": ["playlist_title"],
                "additionalProperties": False,
            }
        }
    },
    "add_playlist_items": {
        "type": "function",
        "function": {
            "name": "add_playlist_items",
            "description": "Add tracks to a playlist. Side effect: modifies playlist contents. Must call 'verify_recordings' BEFORE use. "
                            "Returns updates playlist mbid. "
                            "Pass 'playlist_mbid' and 'recording_mbids' (Requires verified recording MBIDs). "
                            "Use when user asks to add playlist items. If MBIDs are unknown, call search_recordings and verify_recordings first. "
                            "Do NOT use before approval.",
            "parameters": {
                "type": "object",
                "properties": {
                    "playlist_mbid": {"type": "string"},
                    "recording_mbids": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["playlist_mbid", "recording_mbids"],
                "additionalProperties": False,
            }
        }
    },
    "remove_playlist_items": {
        "type": "function",
        "function": {
            "name": "remove_playlist_items",
            "description": "Remove tracks from a playlist. "
                            "Returns items removed from playlist. "
                            "Pass 'playlist_mbid', and 'items_to_remove'. "
                            "Use when the user wants tracks removed. "
                            "Do NOT use without approval.",
            "parameters": {
                "type": "object",
                "properties": {
                    "playlist_mbid": {"type": "string"},
                    "items_to_remove": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["playlist_mbid", "items_to_remove"],
                "additionalProperties": False,
            }
        }
    },
    "set_feedback": {
        "type": "function",
        "function": {
            "name": "set_feedback",
            "description": "Love, hate, or clear feedback on a recording. This operation is idempotent. "
                            "Pass 'recording_mbid' and 'score'. "
                            "Use when user wants to change preferences. "
                            "Do NOT use if the recording MBID is unknown.",
            "parameters": {
                "type": "object",
                "properties": {
                    "recording_mbid": {"type": "string"},
                    "score": {"type": "string", "enum": ["love","hate","clear"]}
                },
                "required": ["recording_mbid", "score"],
                "additionalProperties": False,
            }
        }
    },
    "export_playlist": {
        "type": "function",
        "function": {
            "name": "export_playlist",
            "description": "Export a playlist to a connected streaming service. Always requires approval. "
                            "Pass 'playlist_mbid' and 'service'. "
                            "Use when user wants to take content out of ListenBrainz. "
                            "Do NOT use if the service connection has not been verified.",
            "parameters": {
                "type": "object",
                "properties": {
                    "playlist_mbid": {"type": "string"},
                    "service": {"type": "string"}
                },
                "required": ["playlist_mbid", "service"],
                "additionalProperties": False,
            }
        }
    },
    "delete_playlist": {
        "type": "function",
        "function": {
            "name": "delete_playlist",
            "description": "Delete an Encore-created playlist. Always requires approval. "
                            "Pass 'playlist_title','playlist_mbid'. "
                            "Use when user wants to permanently delete an Encore-created playlist. "
                            "Do NOT use on user-created playlists. Refuse playlists that are not marked as Encore playlists.",
            "parameters": {
                "type": "object",
                "properties": {
                    "playlist_title": {"type": "string"},
                    "playlist_mbid": {"type": "string"},
                },
                "required": ["playlist_title","playlist_mbid"],
                "additionalProperties": False,
            }
        }
    },
    "search_past_sessions": {
        "type": "function",
        "function": {
            "name": "search_past_sessions",
            "description": "Search episodic memory for information from previous conversations with this user. "
                            "Returns past decisions, preferences, mistakes, open threads, and previously completed actions that were stored in session summaries. "
                            "Pass 'query'. "
                            "Use when the user refers to previous conversations with phrases such as 'same as last time', 'what did we do last week', 'continue working on that playlist', or similar references to prior sessions. "
                            "Do NOT use for ListenBrainz documentation, MusicBrainz documentation, music definitions, API questions, or general music knowledge.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                },
                "required": ["query"],
                "additionalProperties": False,
            }
        }
    },
    "search_music_knowledge": {
        "type": "function",
        "function": {
            "name": "search_music_knowledge",
            "description": "Search semantic memory containing ListenBrainz documentation, MusicBrainz documentation, and the music glossary. "
                            "Returns relevant knowledge chunks and their sources. "
                            "Pass 'query'. "
                            "Use when the user asks how ListenBrainz works, how MusicBrainz works, what a music term means, why a service behaves a certain way, or any question that requires factual knowledge rather than memory of previous conversations. "
                            "Do NOT use when the user is referring to a prior conversation, playlist, decision, preference, or action from an earlier session.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                },
                "required": ["query"],
                "additionalProperties": False,
            }
        }
    },
    "note_preference": {
        "type": "function",
        "function": {
            "name": "note_preference",
            "description": "Store a user preference in hot state for the current session. Preferences may include favorite genres, artists, moods, dislikes, playlist constraints, or listening goals. "
                            "Pass 'preference'. "
                            "Use when the user explicitly states something they like, dislike, want, or never want. Examples include 'I hate Christmas music', 'I prefer instrumental tracks', or 'I only want metal recommendations'. "
                            "Do NOT use for facts learned solely from tool results, assumptions about the user, or information that the user did not explicitly state.",
            "parameters": {
                "type": "object",
                "properties": {
                    "preference": {"type": "string"},
                },
                "required": ["preference"],
                "additionalProperties": False,
            }
        }
    },
    "final_answer": {
        "type": "function",
        "function": {
            "name": "final_answer",
            "description": "Provide the final response to the user and terminate "
                            "the current request. "
                            "Pass 'status' (one of the allowed values), 'answer' (a summary of what was found or done), 'confidence' (your certainty level), and 'sources' (MBIDs or knowledge chunks used). "
                            "Use only after the goal has been completed, progress is impossible, the step budget has been exceeded, or a policy restriction prevents further action. "
                            "Do NOT use if additional tool calls are still needed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "enum": [
                            "answered",
                            "answered_low_confidence",
                            "stuck",
                            "budget_exceeded",
                            "policy_halt"
                        ]
                    },
                    "answer": {"type": "string"},
                    "confidence": {
                        "type": "string",
                        "enum": ["high","medium","low"]
                    },
                    "sources": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                },
                "required": ["status","answer","confidence","sources"],
                "additionalProperties": False,
            }
        }
    },
}

def check_model_drift(mode, tier, runtime_model):
    pinned = os.environ.get(
        f"{'OPENROUTER' if mode == 'online' else 'LOCAL_LLM'}_MODEL_{tier.upper()}"
    )

    env_var_name = f"{'OPENROUTER' if mode == 'online' else 'LOCAL_LLM'}_MODEL_{tier.upper()}"
    print(f"DEBUG check_model_drift: env_var_name={env_var_name}")
    print(f"DEBUG check_model_drift: pinned={pinned}")
    print(f"DEBUG check_model_drift: runtime_model={runtime_model}")
    print(f"DEBUG check_model_drift: match={pinned == runtime_model}")
    if pinned is None:
        print(f"WARNING! '.env' has no pinned model for this tier ({tier}) at all -- there is nothing to drift-check against.")
        return True
    elif pinned != runtime_model:
        print(f"WARNING! Pinned Model {pinned} and Runtime Model {runtime_model} do not match!")
        return True
    return False

def build_windowed_context(all_steps, rolling_summary, mode):
    if len(all_steps) <= WINDOW_SIZE and get_tokens(all_steps) <= 6000:
        return all_steps, rolling_summary
    steps_to_summarize = all_steps[:-WINDOW_SIZE]
    steps_to_keep = all_steps[-WINDOW_SIZE:]
    summary_lines = []

    for step in steps_to_summarize:
        content = step.get("content", "")
        if isinstance(content, dict):
            summary_lines.append(json.dumps(content))
        else:
            summary_lines.append(str(content))
    transcript = "\n".join(summary_lines)
    payload = f"""
    EXISTING SUMMARY:
    {rolling_summary["content"]}

    NEW STEPS:
    {transcript}
    """
    response = chat(COMPRESSION_PROMPT + payload, "cheap", mode, [SUMMARY_SCHEMA], FORCE_SUMMARY)
    if "tool_call" not in response or "tool_calls" not in response:
        print("ERROR: Compression failed, returning uncompressed context")
        return all_steps, rolling_summary
    summary = json.loads(response["tool_calls"][0]["function"]["arguments"])
    rolling_summary = {
        "role": "system",
        "content": json.dumps(summary, indent=2)
    }
    return [rolling_summary] + steps_to_keep, rolling_summary

def build_messages(history, hot_state, memory_context, persistent_constraints):
    cleaned_history = []
    for msg in history:
        if msg.get("content") is not None:
            cleaned_history.append(msg)
    messages = [
        {"role": "system","content": persistent_constraints},
        {"role": "system","content": json.dumps(hot_state, indent=2)},
    ]
    if memory_context.strip():
        messages.append({"role": "system","content": memory_context})
    return messages + cleaned_history

def run_tool(tool_name: str, tool_arguments):
    tool_function = TOOLS[tool_name]
    failures = 0
    same_tool_count = 0
    last_tool = None

    for delay in backoff_delays():
        tool_result = tool_function(**tool_arguments)
        parsed = json.loads(tool_result)

        #! Termination Overrides
        if parsed.get("error_category") == "fatal":
            result = terminate(
                status="policy_halt",
                answer=parsed["error"],
                confidence="high"
            )
            print(json.dumps(result, indent=2))
            return
        failures = failures +1 if not parsed.get("success") else 0
        if tool_name == last_tool:
            same_tool_count += 1
        else:
            same_tool_count = 0
        last_tool = tool_name
        if failures >= 3:
            result = terminate(
                "stuck",
                f"Three consecutive tool failures running tool '{tool_name}'",
                confidence="low")
            print(json.dumps(result, indent=2))
            return 
        if same_tool_count >= 3:
            result = terminate(
                status="stuck",
                answer="Agent repeatedly selected the same tool without making progress.",
                confidence="low"
            )
            print(json.dumps(result, indent=2))
            return 
        if parsed.get("error_category") != "transient":
            return tool_result
        time.sleep(delay)
    return tool_result

def backoff_delays(attempts: int = 4, base: float = 1.0, cap: float = 30.0):
    """Exponential backoff with full jitter. The LOOP retries transient errors, never the model."""
    return [random.uniform(0, min(cap, base * 2 ** i)) for i in range(attempts)]

#! Trace logs
def log_tool_call(step, tool_name, arguments, result, latency_ms):
    os.makedirs("traces", exist_ok=True)
    log_path = f"traces/{SESSION_ID}.jsonl"
    time_now = current_time().isoformat()
    record = {
        "timestamp": time_now,
        "event": "tool_call",
        "step": step,
        "tool_name": tool_name,
        "arguments": arguments,
        "result": result,
        "latency_ms": latency_ms,
    }
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

def log_termination(status, reason):
    os.makedirs("traces", exist_ok=True)
    with open(f"traces/{SESSION_ID}.jsonl","a") as file:
        file.write(
            json.dumps({
                "timestamp": current_time().isoformat(),
                "event": "termination",
                "status": status,
                "reason": reason,
            }) + "\n")

def log_approval(step, tool_name: str, response: str, approved):
    os.makedirs("traces", exist_ok=True)
    with open(f"traces/{SESSION_ID}.jsonl","a") as file:
        file.write(
            json.dumps({
                "timestamp": current_time().isoformat(),
                "event": "approval",
                "step": step,
                "tool_name": tool_name,
                "response": response,
            }) + "\n"
        )

def log_memory(operation: str, memory_type: str, details):
    os.makedirs("traces", exist_ok=True)
    with open(f"traces/{SESSION_ID}.jsonl","a") as file:
        file.write(
            json.dumps({
                "timestamp": current_time().isoformat(),
                "event": "memory",
                "operation": operation,
                "memory_type": memory_type,
                "details": details,
            }) + "\n"
        )
def chat(prompt, tier, mode, tools, tool_choice, user_input=None):
    try:
        model, url, headers = model_for(tier, mode)
    except Exception as e:
        print("ERROR: Model failed failed drift check.")
        return
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": prompt},
        ],
        "tools": tools,
        "tool_choice": tool_choice,
        "temperature": 0,
    }
    if user_input:
        payload["messages"] = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_input},
        ]
    
    response = requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]

def model_for(tier: str, mode: str):
    if mode == "offline":
        model_tier = "LOCAL_LLM_MODEL_CHEAP" if tier=="cheap" else "LOCAL_LLM_MODEL_EXPENSIVE"
        url = os.getenv("LOCAL_LLM_URL","http://127.0.0.1:8080/v1/chat/completions")
        headers = {
            "Authorization": f"Bearer Local",
            "Content-Type": "application/json"
        }
    else:
        model_tier = "OPENROUTER_MODEL_CHEAP" if tier=="cheap" else "OPENROUTER_MODEL_EXPENSIVE"
        url =  os.getenv("OPENROUTER_URL","https://openrouter.ai/api/v1/chat/completions")
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        }
    model = os.getenv(model_tier)
    print(f"DEBUG: model_tier={model_tier}, model={model}")
    if not model:
        print(f"ERROR: Model not configured for tier {tier} in mode {mode}")
        return None, None, None  
    if check_model_drift(mode, tier, model):
        return None, None, None  

    return model, url, headers

# About 4 characters per token
def get_tokens(messages: list[str]):
    total_tokens = len(json.dumps(messages))
    return total_tokens // 4

def validate_tool(arguments, tool_schema):
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError as e:
            return f"Tool arguments not valid JSON: {e}"
    
    if not isinstance(arguments, dict):
        return "Tool arguments must be a dict or valid JSON string"
    
    required = tool_schema.get("required", [])
    properties = tool_schema.get("properties", {})
    
    # Check required fields
    for field in required:
        if field not in arguments:
            return f"Missing required field: {field}"
    
    # Validate field types and enums
    for field, value in arguments.items():
        if field not in properties:
            return f"Unknown field: {field}"
        
        prop_schema = properties[field]
        expected_type = prop_schema.get("type")
        
        if expected_type == "array":
            if not isinstance(value, list):
                return f"Field '{field}' must be array, got {type(value).__name__}"
            enum = prop_schema.get("items", {}).get("enum")
            if enum and any(v not in enum for v in value):
                return f"Field '{field}' contains invalid enum values. Valid: {enum}"
        
        elif expected_type == "string":
            if not isinstance(value, str):
                return f"Field '{field}' must be string, got {type(value).__name__}"
            enum = prop_schema.get("enum")
            if enum and value not in enum:
                return f"Field '{field}' must be one of {enum}"
    
    return None

def terminate(status: str, answer: str, confidence: str, sources = None):
    if sources is None:
        sources = []
    log_termination(status, answer)
    return json.loads(final_answer(status,answer,confidence,sources))

def apply_retention():
    results = episodic_collection.get(
        where={"user_id": ENCORE_USER_ID},
        include=["metadatas"]
    )
    delete_ids = []
    now = current_time()

    summaries_by_user = []
    for memory_id, meta in zip(results["ids"], results["metadatas"]):
        timestamp = datetime.fromisoformat(meta["timestamp"])
        age_days = (now - timestamp).days
        retention_days = 90
        max_summaries = 50
        if age_days > retention_days:
            delete_ids.append(memory_id)
        summaries_by_user.append((memory_id, timestamp))
    sorted_summaries = sorted(summaries_by_user, key=lambda x: x[1], reverse=True)
    if len(sorted_summaries) > max_summaries:
        for memory_id, _ in sorted_summaries[max_summaries:]:
            delete_ids.append(memory_id)
    if delete_ids:
        episodic_collection.delete(ids=delete_ids)
        log_memory("delete", "episodic", {
                "deleted_count": len(delete_ids),
                "reason": "retention_policy"
        })

    return len(delete_ids)

def memory_is_stale(meta):
    timestamp = datetime.fromisoformat(meta["timestamp"])
    days = (current_time() - timestamp).days

    return days >= STALENESS_THRESHOLD

def current_time():
    return (datetime.now(timezone.utc) + timedelta(days=TIME_OFFSET))

def run_session(mode: str) -> None:
    print(f"DEBUG: LB_USER = {LB_USER}")
    print(f"DEBUG: LB_TOKEN = {LB_TOKEN[:10] if LB_TOKEN else 'NOT SET'}...")
    print(f"DEBUG: OPENROUTER_API_KEY = {OPENROUTER_API_KEY[:10] if OPENROUTER_API_KEY else 'NOT SET'}...")
    # TODO 3: session start -- load procedural memory (prompts/system_v1.md) and hot state,
    #         retrieve episodic memories for the first message, and front-load them (Part 5)
    global SESSION_ID # Had to make global to avoid excessive amount of parameters
    SESSION_ID = str(uuid.uuid4())
    user_input = input("Welcome to Encore! What would you like to do this session: ")
    
    with open("prompts/system_v1.md", encoding="utf8") as file:
        system_prompt = file.read()
    load_semantic_docs()
    hot_state_text = load_hot_state()
    log_memory(
        "read", 
        "working", 
        {"preferences": len(hot_state_text.get("preferences",[]))}
    )
 
    memory_context = retrieve_memories(user_input)
    log_memory("read", "episodic", {"query": user_input})
    print("===Loaded Memory===")
    print(type(system_prompt))

#! Routing Tools
    domains = route_tools(user_input, mode)
    tools = []
    tool_schemas = []
    for domain in domains:
        tool_names = TOOLSETS[domain]
        for tool in tool_names:
            tools.append(tool)
            tool_schemas.append(TOOL_SCHEMAS[tool])
    tools.append("final_answer")
    tool_schemas.append(TOOL_SCHEMAS["final_answer"])
    
    print(f"DEBUG: domains={domains}")
    print(f"DEBUG: tool_schemas count={len(tool_schemas)}")
    print(f"DEBUG: tool_schemas names={[s['function']['name'] for s in tool_schemas[:5]]}")
    # TODO 4: bounded ReAct loop -- route tools (Lab 4), reinject constraints every call
    #         (Lab 3), retry transient errors with backoff_delays(), gate writes behind
    #         approval, compress working memory past its budget, log every step (Part 6)
    load_dotenv()
    model, url, headers = model_for("cheap",mode)
    
    all_steps = [{
        "role": "system",
        "content": f"Past Memories: \n\n{memory_context}"
    }]
    rolling_summary = {"role": "system", "content": "(no steps summarized yet)"}
    all_steps.append({"role": "user", "content": user_input})
    for step in range(1, MAX_STEPS + 1):
        print(f"\n--- STEP {step} ---")

        windowed, rolling_summary = build_windowed_context(all_steps, rolling_summary, mode)
        print("\n=== CONTEXT BUILDING ===")
        print("Memory Context:")
        print(memory_context)
        print("\nHot State:")
        print(json.dumps(hot_state_text, indent=2))
        print("\nWindow Size:")
        print(len(windowed))
        messages = build_messages(windowed, hot_state_text, memory_context, system_prompt)
        if messages[-1]["role"] != "assistant" or "tool_calls" not in messages[-1]:
            print("WARNING: Last message is not an assistant message with tool_calls")
        payload = {
            "model": model,
            "messages": messages,
            "tools": tool_schemas,
            "tool_choice": "auto",
            "parallel_tool_calls": False,
            "temperature": 0
        }
        
        response = requests.post(
            url,
            json=payload,
            headers=headers,
            timeout=60
        )
        if response.status_code != 200:
            print(response.text)
            break
        message = response.json()["choices"][0]["message"]
        print("\n=== MODEL RESPONSE ===")
        print(json.dumps(message, indent=2))
        all_steps.append(message)
        tool_calls = message.get("tool_calls", [])

        if not tool_calls:
            try:
                model, url, headers = model_for("expensive", mode)
                print(f"DEBUG: model_for returned: model={model}, url={url is not None}, headers={headers is not None}")
                print(f"ERROR: model_for('expensive', '{mode}') returned None")
                if not model:
                    result = terminate(
                        status="stuck",
                        answer="Model failed drift",
                        confidence="low"
                    )    
                    print(json.dumps(result, indent=2))
                finalize_payload = {
                    "model": model,
                    "messages": messages,
                    "tools": tool_schemas,
                    "tool_choice": {
                        "type": "function",
                        "function": {
                            "name": "final_answer"
                        }
                    }
                }
                print(f"DEBUG: Calling expensive model {model}")
                final_response = requests.post(
                    url,
                    json=finalize_payload,
                    headers=headers,
                    timeout=60
                )
                final_response.raise_for_status()
        
                response_json = final_response.json()
                print(f"DEBUG: Response keys: {response_json.keys()}")  # <- add this
                
                if "choices" not in response_json:
                    print(f"ERROR: Unexpected response structure: {response_json}")
                    result = terminate(
                        status="stuck",
                        answer="Model returned unexpected response format.",
                        confidence="low"
                    )
                    print(json.dumps(result, indent=2))
                    return
                final_message = final_response.json()[
                    "choices"
                ][0]["message"]
                final_calls = final_message.get("tool_calls", [])
                if final_calls:
                    for call in final_calls:
                        if call["function"]["name"] == "final_answer":
                            args = json.loads(
                                call["function"]["arguments"]
                            )
                            result = final_answer(**args)
                            print(json.dumps(result,indent=2))
                            return
                else:
                    result = terminate(
                        status="stuck",
                        answer="Model failed to provide a final answer.",
                        confidence="low"
                    )
        
                    print(json.dumps(result, indent=2))
                    return
            except requests.RequestException as e:
                print(f"ERROR: Request failed: {e}")
                result = terminate(
                    status="stuck",
                    answer=f"API request failed: {str(e)}",
                    confidence="low"
                )
                print(json.dumps(result, indent=2))
                return
            except (KeyError, json.JSONDecodeError) as e:
                print(f"ERROR: Response parsing failed: {e}")
                result = terminate(
                    status="stuck",
                    answer="Failed to parse model response.",
                    confidence="low"
                )
                print(json.dumps(result, indent=2))
                return

        for call in tool_calls:
            tool_name = call["function"]["name"]
            tool_arguments = json.loads(
                call["function"]["arguments"]
            )
            start = time.perf_counter()
            not_validated = validate_tool(tool_arguments, TOOL_SCHEMAS[tool_name]["function"]["parameters"])
            if not_validated is not None:
                all_steps.append({
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": not_validated
                })
                continue
            if tool_name in WRITE_TOOLS:
                approval = input(f"Do you want to allow '{tool_name}'? (y/n):")
                if approval.lower() != "y":
                    log_approval(step, tool_name, approval, False)
                    tool_result = fail("resource", f"User declined approval for tool '{tool_name}'.").to_json()
                    all_steps.append({
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "content": tool_result
                    })
                    
                    continue
                log_approval(step, tool_name, approval, True)
            if tool_name in TOOLS:
                try:
                    tool_result = run_tool(tool_name, tool_arguments)
                except Exception as e:
                    tool_result = json.dumps({
                        "error": type(e).__name__,
                        "message": str(e)
                        })
                all_steps.append({
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": tool_result
                })
            else:
                tool_result = json.dumps({
                    "error": "UNKNOWN_TOOL",
                    "content": f"Invalid tool '{tool_name}'. Tool must be in 'TOOLS'"
                })
            
            latency_ms = (
                time.perf_counter() - start
            ) * 1000
            log_tool_call(
                step=step,
                tool_name=tool_name,
                arguments=tool_arguments,
                result=tool_result,
                latency_ms=latency_ms
            )
            all_steps.append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": tool_result
                }
            )
        
    else:
        result = terminate(
            "budget_exceeded", 
            "Maximum reasoning step budget reached before task completion.",
            "low")
        print(json.dumps(result, indent=2))
    session_summary = write_session_summary(all_steps, mode)
    #@FIXTHIS
    if not memory_exists(json.dumps(session_summary)): 
        quality_info = quality_check(session_summary, all_steps, mode)
        with open(f"traces/{SESSION_ID}.jsonl", "a") as file:
            file.write(json.dumps({
                "event": "quality_check",
                "quality": quality_info["quality"],
                "issues": quality_info["issues"]
            }) + "\n"
        )
        store_session_summary(session_summary, quality_info)
        apply_retention()
        new_hot_state = {
            "active_playlist": None,
            "last_recording": None,
            "last_artist": None,
            "preferences": [],
            "pending_approval": None
        }
        save_hot_state(new_hot_state)
    # TODO 5: end the loop only through final_answer or a termination status (Part 4)
    # TODO 6: session end -- structured summary, quality check, dedup, store (Part 5)


def forget_user(user_id: str) -> None:
    # TODO 7: delete every stored item for user_id from EVERY store and print the counts
    episodic_results = episodic_collection.get(where={"user_id": user_id})
    episodic_deleted = len(episodic_results["ids"])
    episodic_collection.delete(where={"user_id": user_id})
    remaining = episodic_collection.get(where={"user_id": user_id})
    episodic_remaining = len(remaining["ids"])
    working_deleted = 0
    if os.path.exists(HOT_STATE_FILE):
        os.remove(HOT_STATE_FILE)
        working_deleted = 1
    print(
    f"""
        Forget completed for '{user_id}'

        Episodic deleted: {episodic_deleted}
        Episodic remaining; {episodic_remaining}
        Working memory deleted: {working_deleted}
        Working memory remaining: {0}
    """)

def main(args):
    global TIME_OFFSET
    mode = (args[0]) if args else "online"
    if "--forget" in args:
        forget_user(ENCORE_USER_ID)
    if "--now" in args:
        index = args.index("--now")
        TIME_OFFSET = int(args[index+1])
    run_session(mode)

if __name__ == "__main__":
    main(sys.argv[1:])