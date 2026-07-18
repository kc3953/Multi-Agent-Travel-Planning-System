"""AI Travel Itinerary Optimizer — FastAPI server and CLI entry point."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sys
import traceback
import uuid
from collections import defaultdict
from typing import Any

import typer
import uvicorn
from agents import InputGuardrailTripwireTriggered, MaxTurnsExceeded, Runner
from agents.extensions.models.litellm_model import LitellmModel
from agents.items import TResponseInputItem
from agents.memory.session import SessionABC
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Suppress noisy third-party loggers that obscure agent workflow visibility.
# "LiteLLM completion() model=..." comes from the LiteLLM logger.
# "OPENAI_API_KEY is not set, skipping trace export" comes from openai.agents.
logging.getLogger("LiteLLM").setLevel(logging.ERROR)
logging.getLogger("litellm").setLevel(logging.ERROR)
logging.getLogger("openai.agents").setLevel(logging.ERROR)

# ── Per-session progress queues ──────────────────────────────────────────────
# Maps session_id → asyncio.Queue of status-message strings.
# SSE stream reads from these; the agent run pushes into them.
_progress_queues: dict[str, asyncio.Queue] = defaultdict(asyncio.Queue)

# Human-readable labels for each sub-agent tool call
_TOOL_LABELS: dict[str, str] = {
    "intake_agent":        "Understanding your trip request…",
    "lodging_agent":       "Searching for hotels & lodging…",
    "activity_agent":      "Finding attractions & activities…",
    "dining_agent":        "Discovering restaurants & dining…",
    "transport_agent":     "Planning transportation routes…",
    "solver_agent":        "Assembling your day-by-day itinerary…",
    "replanner_agent":     "Re-optimizing after disruption…",
    "conversation_agent":  "Analysing your itinerary…",
}


class _ProgressHandler(logging.Handler):
    """Captures 'Invoking tool <name>' DEBUG lines from openai.agents and pushes status events."""

    # openai.agents logs: "Invoking tool intake_agent" or "Invoking tool intake_agent with input ..."
    _TOOL_RE = re.compile(r"Invoking tool (\w+)", re.IGNORECASE)

    def __init__(self, session_id: str) -> None:
        super().__init__(level=logging.DEBUG)
        self.session_id = session_id

    def emit(self, record: logging.LogRecord) -> None:
        msg = record.getMessage()
        m = self._TOOL_RE.search(msg)
        if m:
            tool = m.group(1)
            label = _TOOL_LABELS.get(tool, f"Running {tool}…")
            try:
                _progress_queues[self.session_id].put_nowait({"type": "status", "text": label})
            except Exception:
                pass

# Global LiteLLM retry on 429/503 — fires before the error reaches the SDK.
# 3 retries with exponential backoff: waits 5s, 10s, 20s before each retry.
import litellm  # noqa: E402
litellm.num_retries = 3
litellm.retry_after = 5
litellm.suppress_debug_info = True
litellm.request_timeout = 90   # kill silent Vertex AI hangs; LiteLLM retries automatically

_orchestrator_model_str = os.environ.get("ORCHESTRATOR_MODEL") or os.environ.get("MODEL")
_specialist_model_str = os.environ.get("SPECIALIST_MODEL") or os.environ.get("MODEL")
_replanner_model_str = os.environ.get("REPLANNER_MODEL") or _orchestrator_model_str
_solver_model_str = os.environ.get("SOLVER_MODEL") or _orchestrator_model_str

if not _orchestrator_model_str or not _specialist_model_str:
    sys.exit("ERROR: ORCHESTRATOR_MODEL and SPECIALIST_MODEL (or MODEL) must be set in .env")


# Session & State --------------------------------------------------------------

_sessions_store: dict[str, list[TResponseInputItem]] = {}
_context_store: dict[str, "AppContext"] = {}


class InMemorySession(SessionABC):
    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        if self.session_id not in _sessions_store:
            _sessions_store[self.session_id] = []

    async def get_items(self, limit: int | None = None) -> list[TResponseInputItem]:
        items = _sessions_store[self.session_id]
        return items[-limit:] if limit is not None else list(items)

    async def add_items(self, items: list[TResponseInputItem]) -> None:
        _sessions_store[self.session_id].extend(items)

    async def pop_item(self) -> TResponseInputItem | None:
        if _sessions_store[self.session_id]:
            return _sessions_store[self.session_id].pop()
        return None

    async def clear_session(self) -> None:
        _sessions_store[self.session_id] = []


class AppContext(BaseModel):
    """Serializable session state passed to the agent runner."""
    session_id: str
    advisor_notes: str = ""
    itinerary_json: str = ""    # JSON-serialized current Itinerary (avoids circular imports)
    locked_slots: list[str] = []   # ["day1_morning", "day2_evening", ...]
    disruption_count: int = 0
    pending_delta: str = ""     # JSON-serialized ItineraryDelta set by store_delta tool
    weather_data: str = ""      # JSON-serialized list of per-day weather from get_weather_forecast
    # Keys: "lodging" | "activities" | "dining" → list of PlaceResult-like dicts
    candidate_pool: dict[str, list[dict]] = Field(
        default_factory=lambda: {"lodging": [], "activities": [], "dining": []}
    )
    # "CityA→CityB" → cost_per_person_usd from transport tool
    transport_costs: dict[str, int] = Field(default_factory=dict)
    # hotel_name_lower → nightly_rate_usd (populated post-plan via LiteAPI)
    lodging_rates: dict[str, float] = Field(default_factory=dict)
    # Extracted from the last itinerary for use in post-processing
    last_city: str = ""
    last_country_code: str = "US"
    last_checkin: str = ""   # ISO date string or ""
    last_nights: int = 0

    @classmethod
    def get_or_create(cls, session_id: str) -> "AppContext":
        if session_id not in _context_store:
            _context_store[session_id] = cls(session_id=session_id)
        return _context_store[session_id]

    def save(self) -> None:
        _context_store[self.session_id] = self


# Itinerary capture ------------------------------------------------------------

_PLACE_ID_RE = re.compile(r"&query_place_id=[^)\s]+")

def _strip_place_id_params(text: str) -> str:
    """Remove stale query_place_id params from Maps links — they cause wrong-location redirects."""
    return _PLACE_ID_RE.sub("", text)

def _extract_itinerary_text(text: str) -> str:
    """Strip markdown code fences and leading preamble so the parser sees raw itinerary."""
    # Pull content out of ```...``` or ```text...``` blocks
    fence_match = re.search(r"```(?:text|markdown)?\s*\n?([\s\S]+?)```", text)
    if fence_match:
        return fence_match.group(1).strip()
    # No fence — strip any short preamble before the first **Day or **N-Day line
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if re.match(r"\*\*\d+-Day|\*\*Day\s+\d", line.strip()):
            return "\n".join(lines[i:]).strip()
    return text


def _extract_tool_output(result: Any, tool_name: str) -> str | None:
    """Pull a sub-agent tool's raw output directly from the run's items.

    The orchestrator prompt asks the model to copy the solver_agent output
    verbatim into its own final message. Gemini's recitation safety filter
    occasionally suppresses that echo (empty final_output) since it looks
    like large-scale text reproduction. Reading the tool's own output
    straight from the run trace sidesteps that entirely — no second LLM
    call is needed to reproduce text that was already generated once.
    """
    call_ids: set[str] = set()
    for item in result.new_items:
        if getattr(item, "type", None) != "tool_call_item":
            continue
        raw = item.raw_item
        name = raw.get("name") if isinstance(raw, dict) else getattr(raw, "name", None)
        call_id = raw.get("call_id") if isinstance(raw, dict) else getattr(raw, "call_id", None)
        if name == tool_name and call_id:
            call_ids.add(call_id)

    if not call_ids:
        return None

    for item in result.new_items:
        if getattr(item, "type", None) != "tool_call_output_item":
            continue
        raw = item.raw_item
        call_id = raw.get("call_id") if isinstance(raw, dict) else getattr(raw, "call_id", None)
        if call_id in call_ids:
            output = item.output
            return output if isinstance(output, str) else str(output)
    return None


_CITY_COUNTRY: dict[str, str] = {
    "rome": "IT", "florence": "IT", "venice": "IT", "milan": "IT", "naples": "IT",
    "paris": "FR", "lyon": "FR", "nice": "FR", "bordeaux": "FR",
    "barcelona": "ES", "madrid": "ES", "seville": "ES", "granada": "ES", "valencia": "ES",
    "lisbon": "PT", "porto": "PT", "algarve": "PT",
    "london": "GB", "edinburgh": "GB", "manchester": "GB", "bath": "GB",
    "amsterdam": "NL", "rotterdam": "NL",
    "berlin": "DE", "munich": "DE", "hamburg": "DE",
    "tokyo": "JP", "kyoto": "JP", "osaka": "JP", "hiroshima": "JP", "nara": "JP",
    "taipei": "TW", "kaohsiung": "TW", "tainan": "TW",
    "bangkok": "TH", "chiang mai": "TH", "phuket": "TH",
    "hanoi": "VN", "ho chi minh city": "VN", "hoi an": "VN",
    "singapore": "SG",
    "seoul": "KR", "busan": "KR", "jeju": "KR",
    "bali": "ID", "jakarta": "ID", "yogyakarta": "ID",
    "kuala lumpur": "MY", "penang": "MY",
    "athens": "GR", "santorini": "GR", "mykonos": "GR",
    "dubrovnik": "HR", "split": "HR", "zagreb": "HR",
    "istanbul": "TR", "cappadocia": "TR",
    "marrakech": "MA", "fes": "MA", "casablanca": "MA",
    "dublin": "IE", "galway": "IE", "cork": "IE",
    "copenhagen": "DK", "stockholm": "SE", "oslo": "NO", "bergen": "NO",
    "new york": "US", "washington dc": "US", "boston": "US",
    "los angeles": "US", "san francisco": "US", "las vegas": "US",
}

def _country_code(city: str) -> str:
    return _CITY_COUNTRY.get(city.lower(), "US")


def _looks_like_itinerary(text: str) -> bool:
    """Return True if the response text contains a full itinerary (not just a re-plan description)."""
    return bool(
        len(text) > 800 and
        re.search(r"\*\*Day \d", text) and
        re.search(r"(morning|afternoon|evening)", text, re.IGNORECASE)
    )


def _guardrail_message(user_input: str) -> str:
    """Return a friendly rejection message based on what the user sent."""
    lower = user_input.lower()
    if re.search(r"\$\s*\d+", lower):
        return (
            "That budget doesn't look right for a travel plan — "
            "I can help with trips in the $20–$10,000/day range per person. "
            "What's your daily budget?"
        )
    return (
        "I'm a travel planning assistant — I can help you build itineraries, "
        "find places, and re-plan when things go wrong. "
        "What trip are you planning?"
    )


def _capture_itinerary(ctx: "AppContext", response: str) -> None:
    """If the response contains an itinerary, store it and extract trip metadata."""
    if not _looks_like_itinerary(response):
        return
    ctx.itinerary_json = json.dumps({
        "text": response,
        "version": ctx.disruption_count + 1,
    })
    # Extract city, nights, and checkin from the itinerary header for post-processing
    city_m = re.search(r'\b([A-Z][a-z]+)\s+Itinerary\b', response)
    if city_m:
        ctx.last_city = city_m.group(1)
        ctx.last_country_code = _country_code(ctx.last_city)
    nights_m = re.search(r'\*\*(\d+)-Day', response)
    if nights_m:
        ctx.last_nights = int(nights_m.group(1))
    # Try to extract checkin from weather badge dates like "(2026-05-15)"
    date_m = re.search(r'Day 1 \((\d{4}-\d{2}-\d{2})\)', response)
    if date_m:
        ctx.last_checkin = date_m.group(1)
    logger.info("itinerary captured in AppContext (session=%s, city=%s, %d chars)",
                ctx.session_id, ctx.last_city, len(response))


# Models & Agents --------------------------------------------------------------

_orchestrator_model = LitellmModel(model=_orchestrator_model_str, api_key="unused")
_specialist_model = LitellmModel(model=_specialist_model_str, api_key="unused")
_replanner_model = LitellmModel(model=_replanner_model_str, api_key="unused")
_solver_model = LitellmModel(model=_solver_model_str, api_key="unused")

# Import here (after models are defined) to avoid circular issues at module load
from backend.agents.orchestrator import build_orchestrator  # noqa: E402
from backend.guardrails.input_validation import (  # noqa: E402
    budget_sanity_guardrail,
    init_guardrail_agents,
    off_topic_guardrail,
)

init_guardrail_agents(_specialist_model)

agent = build_orchestrator(
    orchestrator_model=_orchestrator_model,
    specialist_model=_specialist_model,
    input_guardrails=[off_topic_guardrail, budget_sanity_guardrail],
    replanner_model=_replanner_model,
    solver_model=_solver_model,
)


# Server -----------------------------------------------------------------------

app = FastAPI(title="Travel Optimizer")


async def _ping_vertex() -> None:
    """Single keep-alive ping to Vertex AI."""
    try:
        await litellm.acompletion(
            model=_orchestrator_model_str,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=5,
        )
        logger.info("Vertex AI keep-alive ping succeeded")
    except Exception as e:
        logger.warning("Vertex AI keep-alive ping failed: %s", e)


async def _keepalive_loop() -> None:
    """Ping Vertex AI every 3 minutes to prevent cold connection hangs."""
    while True:
        await asyncio.sleep(180)
        await _ping_vertex()


@app.on_event("startup")
async def _warmup_vertex() -> None:
    """Warm up Vertex AI at startup and start keep-alive loop."""
    await _ping_vertex()
    asyncio.create_task(_keepalive_loop())


@app.exception_handler(Exception)
async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
    logger.error("Unhandled exception: %s", traceback.format_exc())
    return JSONResponse(status_code=500, content={"detail": traceback.format_exc()})


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "orchestrator_model": _orchestrator_model_str}


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None
    locked_slots: list[str] = []   # ["day2_morning", "day3_evening", ...]


class ChatResponse(BaseModel):
    response: str
    session_id: str
    delta: dict | None = None      # Serialized ItineraryDelta when re-planning occurred
    weather: list | None = None    # Per-day weather data for badge rendering


def _inject_booking_links_from_pool(text: str, ctx: "AppContext") -> str:
    """Tier 1: inject booking links deterministically from ctx.candidate_pool.
    Runs before Tavily — no API calls, instant. Only touches lines missing a booking link."""
    all_candidates: list[dict] = []
    for places in ctx.candidate_pool.values():
        all_candidates.extend(places)
    if not all_candidates:
        return text

    pool_by_name: dict[str, dict] = {}
    for p in all_candidates:
        name = (p.get("name") or "").strip().lower()
        if name:
            pool_by_name[name] = p

    lines = text.splitlines()
    for i, line in enumerate(lines):
        if not line.strip().startswith(("-", "•", "*")):
            continue
        existing = re.findall(r'\[([^\]]+)\]\((https?://[^)]+)\)', line)
        if any('google.com/maps' not in url for _, url in existing):
            continue
        name_m = re.search(r'\*\*([^*]+)\*\*', line)
        if not name_m:
            continue
        venue_lower = name_m.group(1).strip().lower()
        match = pool_by_name.get(venue_lower) or next(
            (p for n, p in pool_by_name.items() if venue_lower in n or n in venue_lower),
            None,
        )
        if not match:
            continue
        website = (match.get("website") or match.get("booking_url") or "").strip()
        if not website:
            continue
        label = ("Book / Official Site" if "🏨" in line
                 else "Book Tickets" if any(e in line for e in ("🌅", "🌇", "🌆"))
                 else "Reserve")
        lines[i] = line.rstrip() + f" [{label}]({website})"
        logger.info("pool link injected: %r → %s", match.get("name"), website)

    return "\n".join(lines)


async def _enrich_itinerary_links(text: str, city: str) -> str:
    """Find missing booking links in itinerary slots via Tavily. Non-blocking, capped at 20s."""
    from backend.tools.tavily_search import find_booking_url

    lines = text.split('\n')
    slot_re = re.compile(
        r'(?:morning|afternoon|evening|lunch|dinner|lodging)[:\s–—]+(.+?)(?:\s*\(|$)',
        re.IGNORECASE
    )

    targets: list[tuple[int, str, str]] = []  # (line_idx, venue_name, category)
    # Regex to extract hotel name from "- 🏨 Hotel Name (all nights…) [link]…" lines
    hotel_re = re.compile(r'^[-•*]\s*🏨\s*(.+?)(?:\s*\(|\s*\[|$)', re.IGNORECASE)

    for i, line in enumerate(lines):
        # Skip if line already has a non-Maps booking link
        existing_links = re.findall(r'\[([^\]]+)\]\((https?://[^)]+)\)', line)
        has_booking_link = any(
            'google.com/maps' not in url and 'maps.google' not in url
            for _, url in existing_links
        )
        if has_booking_link:
            continue
        # Skip non-slot lines — must be a bullet
        if not line.strip().startswith(('-', '•', '*')):
            continue

        # Hotel lines: identified by 🏨 emoji (no period keyword needed)
        if '🏨' in line:
            hm = hotel_re.search(line.strip())
            if not hm:
                continue
            venue = hm.group(1).strip().rstrip('.,;—–').strip()
            venue = re.sub(r'\*\*', '', venue).strip()        # strip bold markers
            venue = re.sub(r'\s*\([^)]*\)', '', venue).strip()
            if len(venue) < 3:
                continue
            targets.append((i, venue, "lodging"))
            continue

        # All other slots — must contain a period keyword
        if not re.search(r'\b(morning|afternoon|evening|lunch|dinner|lodging)\b', line, re.IGNORECASE):
            continue
        nm = slot_re.search(line)
        if not nm:
            continue
        venue = nm.group(1).strip().rstrip('.,;—–').strip()
        venue = re.sub(r'\*\*', '', venue).strip()            # strip bold markers
        venue = re.sub(r'\s*\([^)]*\)', '', venue).strip()  # remove (Xh, $Y) parentheticals
        venue = re.sub(r'^(?:dinner|lunch|breakfast)\s+at\s+', '', venue, flags=re.IGNORECASE).strip()
        if len(venue) < 3:
            continue
        cat = ("restaurant" if re.search(r'🍽|lunch|dinner', line, re.IGNORECASE)
               else "attraction")
        targets.append((i, venue, cat))

    if not targets:
        return text

    loop = asyncio.get_event_loop()

    async def _fetch(venue: str, cat: str) -> str:
        try:
            return await asyncio.wait_for(
                loop.run_in_executor(None, find_booking_url, venue, city, cat),
                timeout=6.0,
            )
        except Exception:
            return ""

    urls = await asyncio.gather(*[_fetch(v, c) for _, v, c in targets])

    for (idx, venue, cat), url in zip(targets, urls):
        if url:
            label = ("Reserve" if cat == "restaurant"
                     else "Book" if cat == "lodging"
                     else "Book Tickets")
            lines[idx] = lines[idx].rstrip() + f' [{label}]({url})'
            logger.info("link enriched: %r → %s", venue, url)

    return '\n'.join(lines)


async def _enrich_lodging_rates(
    text: str,
    ctx: "AppContext",
    city: str,
    country_code: str,
    checkin: "date | None",
    nights: int,
) -> None:
    """Populate ctx.lodging_rates with LiteAPI nightly rates for hotels found in the itinerary.
    Non-blocking — called with a 15s cap. Results stored in AppContext for the solver summary."""
    from backend.tools.hotel_pricing import get_hotel_nightly_rate
    from datetime import date as _date

    if checkin is None:
        checkin = _date.today()

    # Extract hotel names from 🏨 lines
    hotel_lines = [l for l in text.splitlines() if "🏨" in l]
    hotel_names: list[str] = []
    for line in hotel_lines:
        m = re.search(r"🏨\s*(.+?)(?:\s*[\(\[]|$)", line)
        if m:
            name = m.group(1).strip().rstrip(".,—–").strip()
            if name and name not in hotel_names:
                hotel_names.append(name)

    if not hotel_names:
        return

    async def _fetch_rate(name: str) -> tuple[str, float | None]:
        try:
            rate = await asyncio.wait_for(
                get_hotel_nightly_rate(name, city, country_code, checkin, nights),
                timeout=8.0,
            )
            return name, rate
        except Exception:
            return name, None

    results = await asyncio.gather(*[_fetch_rate(n) for n in hotel_names])
    for name, rate in results:
        if rate is not None:
            ctx.lodging_rates[name.lower()] = rate
            logger.info("lodging_rate %r → $%.0f/night", name, rate)
    ctx.save()


def _looks_like_replan(text: str) -> bool:
    """Return True if the response describes a re-plan rather than a full itinerary."""
    return bool(
        re.search(
            r"\b(replac|remov|adjust|swap|updat|substitut|sick day|disruption|re.?plan|changed|closed|itinerary has been)\b",
            text, re.IGNORECASE
        ) and
        re.search(r"\bDay\s+\d+\b", text, re.IGNORECASE) and
        not _looks_like_itinerary(text)
    )


def _parse_prose_to_delta(response: str) -> str | None:
    """Parse a prose re-plan response into an ItineraryDelta JSON string without an LLM call."""

    # Extract day number
    day_m = re.search(r'\bDay\s+(\d+)\b', response, re.IGNORECASE)
    if not day_m:
        return None
    day_num = int(day_m.group(1))

    def _clean_name(s: str) -> str:
        return re.sub(r'^(the|a|an)\s+', '', s.strip().rstrip(',.'), flags=re.IGNORECASE).strip()

    def _find_period(text: str, pos: int) -> str:
        window = text[max(0, pos - 250):pos]
        pm = re.search(r'\b(morning|afternoon|evening|lunch|dinner)\b', window, re.IGNORECASE)
        if not pm:
            return "morning"
        raw = pm.group(1).lower()
        return "afternoon" if raw in ("lunch",) else "evening" if raw == "dinner" else raw

    removed, changed = [], []

    # Pattern: "X has been removed" / "X, has been removed" / "removing X"
    for m in re.finditer(
        r'([\w][\w\s&\'\-\/,]+?),?\s+(?:has been|have been|is being|are being)\s+removed',
        response, re.IGNORECASE
    ):
        name = _clean_name(m.group(1))
        if len(name) > 3 and not re.search(
            r'\b(morning|afternoon|evening|day|slot|activity|schedule|visit|entire)\b', name, re.IGNORECASE
        ):
            removed.append({
                "day_number": day_num, "period": _find_period(response, m.start()),
                "place_name": name, "cost_usd": 0.0, "notes": "",
                "place_id": "", "category": "activity", "address": "", "booking_url": "", "duration_minutes": 90,
            })

    # Pattern: "X replaced with Y"
    for m in re.finditer(
        r'([\w][\w\s&\'\-\/,]+?)\s+(?:\([^)]*\)\s*)?(?:has been |have been |is |are )?replaced with\s+'
        r'(?:a\s+)?(?:shorter[^,\.]*?visit to\s+)?([\w][\w\s&\'\-\/,]+?)(?:\s*[\(\.,]|$)',
        response, re.IGNORECASE
    ):
        old_name = _clean_name(m.group(1))
        new_name = _clean_name(m.group(2))
        period = _find_period(response, m.start())
        if len(old_name) > 3:
            removed.append({
                "day_number": day_num, "period": period, "place_name": old_name,
                "cost_usd": 0.0, "notes": "", "place_id": "", "category": "activity",
                "address": "", "booking_url": "", "duration_minutes": 90,
            })
        if len(new_name) > 3:
            changed.append({
                "day_number": day_num, "period": period, "place_name": new_name,
                "cost_usd": 0.0, "notes": "replacement", "place_id": "", "category": "activity",
                "address": "", "booking_url": "", "duration_minutes": 90,
            })

    if not removed and not changed:
        return None

    # Extract cost
    cost_m = re.search(r'Day\s*\d+\s*(?:estimated\s+)?cost[:\s]+~?\$\s*([\d,]+)', response, re.IGNORECASE)
    cost_usd = float(cost_m.group(1).replace(',', '')) if cost_m else 0.0

    reasoning = re.sub(r'[*•]\s*', '', response).replace('\n', ' ').strip()[:400]

    delta = {
        "disruption": {"day_number": day_num, "period": "", "description": reasoning[:100], "disruption_type": ""},
        "affected_days": [day_num],
        "changed_slots": changed,
        "removed_slots": removed,
        "reasoning": reasoning,
        "new_daily_costs": [{"day": day_num, "cost_usd": cost_usd}] if cost_usd else [],
    }
    result = json.dumps(delta)
    logger.info("delta parsed from prose: %d removed, %d changed", len(removed), len(changed))
    return result


def _enrich_delta_from_pool(ctx: "AppContext", delta: dict) -> dict:
    """Fill missing address / booking_url in changed_slots from the session candidate pool."""
    all_candidates: list[dict] = []
    for places in ctx.candidate_pool.values():
        all_candidates.extend(places)
    if not all_candidates:
        return delta

    for slot in delta.get("changed_slots", []):
        has_address = bool(slot.get("address"))
        has_booking = bool(slot.get("booking_url"))
        if has_address and has_booking:
            continue

        place_id = slot.get("place_id", "")
        place_name = (slot.get("place_name") or "").strip().lower()

        match = None
        if place_id:
            match = next((p for p in all_candidates if p.get("place_id") == place_id), None)
        if not match and place_name:
            match = next(
                (p for p in all_candidates
                 if place_name in (p.get("name") or "").lower()
                 or (p.get("name") or "").lower() in place_name),
                None,
            )

        if match:
            if not has_address:
                slot["address"] = match.get("address") or ""
            if not has_booking:
                slot["booking_url"] = match.get("booking_url") or match.get("website") or ""

    return delta


def _validate_delta(ctx: "AppContext") -> dict | None:
    """Parse and validate the pending ItineraryDelta, dropping any locked slots."""
    if not ctx.pending_delta:
        return None

    from backend.models.disruption import ItineraryDelta
    try:
        delta = ItineraryDelta.model_validate_json(ctx.pending_delta)
    except Exception as e:
        logger.warning("ItineraryDelta validation failed (%s) — using raw delta dict", e)
        # Don't drop the delta entirely; pass the raw dict so the frontend still gets it
        try:
            raw = json.loads(ctx.pending_delta)
            raw.setdefault("changed_slots", [])
            raw.setdefault("removed_slots", [])
            raw.setdefault("affected_days", [])
            raw.setdefault("reasoning", "")
            raw.setdefault("new_daily_costs", [])
            return raw
        except Exception:
            return None

    locked = set(ctx.locked_slots)
    if not locked:
        return delta.model_dump()

    violations: list[str] = []

    def _is_locked(slot) -> bool:
        key = f"day{slot.day_number}_{slot.period}"
        return slot.day_number > 0 and key in locked

    orig_changed = delta.changed_slots
    orig_removed = delta.removed_slots
    delta.changed_slots = [s for s in orig_changed if not _is_locked(s)]
    delta.removed_slots = [s for s in orig_removed if not _is_locked(s)]

    dropped = (len(orig_changed) - len(delta.changed_slots)) + (len(orig_removed) - len(delta.removed_slots))
    if dropped:
        violations = [
            f"day{s.day_number}_{s.period.value}"
            for s in orig_changed + orig_removed
            if _is_locked(s)
        ]
        delta.reasoning += (
            f" Note: {dropped} locked slot(s) were protected and excluded from re-planning "
            f"({', '.join(violations)})."
        )
        logger.info("validator dropped %d locked slot(s): %s", dropped, violations)

    return delta.model_dump()


def _apply_delta_to_stored_itinerary(ctx: "AppContext", delta: dict) -> None:
    """Patch ctx.itinerary_json with the validated delta so the next orchestrator turn
    sees the updated itinerary text rather than the stale pre-replan version."""
    if not ctx.itinerary_json or not delta:
        return
    try:
        itin = json.loads(ctx.itinerary_json)
        text: str = itin.get("text", "")
        if not text:
            return

        def _norm_period(p: str) -> str:
            p = (p or "").lower()
            if p in ("lunch", "breakfast"):
                return "afternoon"
            if p == "dinner":
                return "evening"
            if p == "hotel":
                return "lodging"
            return p

        # Use lists so multiple changes on the same (day, period) don't overwrite each other.
        removed_slots_list: list[dict] = delta.get("removed_slots", [])
        changed_slots_list: list[dict] = delta.get("changed_slots", [])

        # Matches solver format: "- [emoji] PeriodKeyword: ..."
        # [^\w]* skips any emoji/non-word characters between the bullet and the keyword.
        _SLOT_KW = re.compile(
            r"^[-•*]\s*[^\w]*(morning|afternoon|evening|lunch|dinner|breakfast|lodging|hotel)"
            r"\s*[:\s–—]+",
            re.IGNORECASE,
        )

        def _slot_name(line: str) -> str:
            """Strip bullet, emoji, period keyword, links, and parentheticals → bare place name."""
            s = re.sub(r"^[-•*]\s*", "", line.strip())       # bullet
            s = re.sub(r"^[^\w]*", "", s)                    # leading emoji / non-word chars
            s = re.sub(                                        # period keyword + separator
                r"^(morning|afternoon|evening|lunch|dinner|breakfast|lodging|hotel)\s*[:\s–—]+",
                "", s, flags=re.IGNORECASE,
            )
            s = re.sub(r"\*\*", "", s)                        # bold markers
            s = re.sub(r"\[[^\]]*\]\([^)]*\)", "", s)         # markdown links [text](url)
            s = re.sub(r"\s*[\[(].+$", "", s)                 # parentheticals + everything after
            s = re.sub(r"\s*[—–|].*$", "", s)                 # notes after dash / pipe
            s = re.sub(r"~?\$[\d,.]+\S*", "", s)              # inline costs like ~$25/pp
            return s.strip().lower()

        lines = text.splitlines()
        current_day = 0
        current_period = ""
        new_lines = []
        used_keys: set[tuple] = set()

        for line in lines:
            # Day header
            day_m = re.match(r"\*\*Day\s+(\d+)", line)
            if day_m:
                current_day = int(day_m.group(1))
                current_period = ""
                new_lines.append(line)
                continue

            # Bold period header: **Morning** / **Afternoon** / **Evening**
            bold_m = re.match(r"\s*[-•*]?\s*\*\*(Morning|Afternoon|Evening)\*\*", line, re.IGNORECASE)
            if bold_m:
                current_period = _norm_period(bold_m.group(1))
                new_lines.append(line)
                continue

            # Inline period keyword: "- 🌅 Morning: ..." or "- 🍽️ Lunch: ..."
            kw_m = _SLOT_KW.match(line)
            if kw_m:
                current_period = _norm_period(kw_m.group(1))

            # Process slot lines
            if current_day and line.strip().startswith(("- ", "• ", "* ")):
                name = _slot_name(line)

                # Find a removed_slot matching this line by day + period + fuzzy name
                removed = next(
                    (s for s in removed_slots_list
                     if s.get("day_number") == current_day
                     and _norm_period(s.get("period", "")) == current_period
                     and (rn := (s.get("place_name") or "").strip().lower())
                     and (rn in name or name in rn)
                     and id(s) not in used_keys),
                    None,
                )
                if removed:
                    used_keys.add(id(removed))
                    # Find the matching changed_slot for the same day + period
                    changed = next(
                        (c for c in changed_slots_list
                         if c.get("day_number") == current_day
                         and _norm_period(c.get("period", "")) == current_period
                         and id(c) not in used_keys),
                        None,
                    )
                    if changed:
                        used_keys.add(id(changed))
                        cost_str = f" ~${changed['cost_usd']:.0f}/pp" if changed.get("cost_usd") else ""
                        note_str = f" | {changed['notes']}" if changed.get("notes") else ""
                        maps_str = ""
                        if changed.get("place_name"):
                            from urllib.parse import quote_plus
                            query = changed["place_name"]
                            if ctx.last_city:
                                query += f" {ctx.last_city}"
                            enc = quote_plus(query)
                            maps_str = f" [📍 Maps](https://www.google.com/maps/search/?api=1&query={enc})"
                        book_str = f" [Book]({changed['booking_url']})" if changed.get("booking_url") else ""
                        new_lines.append(f"- **{changed['place_name']}**{cost_str}{note_str}{maps_str}{book_str}")
                    # else: pure removal — drop line without replacement
                    continue

            new_lines.append(line)

        # Pure insertions: changed slots with no matched removal
        for cs in changed_slots_list:
            if id(cs) not in used_keys:
                day_n = cs.get("day_number", 0)
                period_n = _norm_period(cs.get("period", ""))
                cost_str = f" ~${cs['cost_usd']:.0f}/pp" if cs.get("cost_usd") else ""
                note_str = f" | {cs['notes']}" if cs.get("notes") else ""
                new_lines.append(f"  - **{cs['place_name']}**{cost_str}{note_str}  ← inserted Day {day_n} {period_n}")

        updated_text = "\n".join(new_lines)
        ctx.itinerary_json = json.dumps({
            "text": updated_text,
            "version": itin.get("version", 1) + 1,
        })
        logger.info("itinerary text patched with delta (%d removed, %d changed)",
                    len(removed_slots_list), len(changed_slots_list))
    except Exception as e:
        logger.warning("delta patch failed — keeping original itinerary text: %s", e)


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest) -> ChatResponse:
    session_id = request.session_id or str(uuid.uuid4())
    session = InMemorySession(session_id)
    ctx = AppContext.get_or_create(session_id)

    # Sync locked slots sent by the frontend on every request
    if request.locked_slots is not None:
        ctx.locked_slots = request.locked_slots
    ctx.pending_delta = ""   # clear any previous delta before this run

    logger.info("chat session=%s message=%r locked=%s", session_id, request.message[:80], ctx.locked_slots)

    try:
        result = await asyncio.wait_for(
            Runner.run(
                agent,
                request.message,
                session=session,
                context=ctx,
                max_turns=15,
            ),
            timeout=480,  # 8 minutes max per request
        )
    except asyncio.TimeoutError:
        logger.warning("chat timeout session=%s", session_id)
        return ChatResponse(
            response="The request took too long — please try again.",
            session_id=session_id,
        )
    except InputGuardrailTripwireTriggered as e:
        guardrail_name = type(e).__name__
        logger.info("guardrail triggered session=%s guardrail=%s", session_id, guardrail_name)
        msg = _guardrail_message(request.message)
        return ChatResponse(response=msg, session_id=session_id)
    except MaxTurnsExceeded:
        logger.warning("chat max_turns exceeded session=%s", session_id)
        return ChatResponse(
            response=(
                "I wasn't able to figure out exactly which slot to change from that description. "
                "Could you be a bit more specific? For example: \"replace the museum on Day 2 afternoon with a night market\" "
                "or \"swap Day 3 morning to something food-related\"."
            ),
            session_id=session_id,
        )
    except Exception as e:
        err = str(e)
        if any(k in err.lower() for k in ("timeout", "timed out", "read timeout", "connect")):
            logger.warning("LLM call timeout/connection error session=%s: %s", session_id, err[:120])
            return ChatResponse(
                response="A model call timed out — Vertex AI is slow to respond. Please try again.",
                session_id=session_id,
            )
        raise

    # Prefer the solver_agent's own tool output over the orchestrator's echo of it —
    # Gemini's recitation filter occasionally suppresses the echo (see _extract_tool_output).
    raw_output = _extract_tool_output(result, "solver_agent") or result.final_output
    final_output = _strip_place_id_params(_extract_itinerary_text(raw_output))

    # Guard: orchestrator occasionally returns empty when conversation_agent result isn't echoed
    if not final_output.strip():
        logger.warning("chat empty final_output session=%s — asking user to clarify", session_id)
        return ChatResponse(
            response=(
                "I didn't quite catch that. Could you rephrase? "
                "For example: \"replace the museum on Day 2 afternoon with a night market\" "
                "or \"which day has too much Italian food?\""
            ),
            session_id=session_id,
        )

    # Post-process: add missing booking links + fetch hotel nightly rates (non-blocking)
    if _looks_like_itinerary(final_output):
        city_m = re.search(r'\b([A-Z][a-z]+)\s+Itinerary\b', final_output)
        city = city_m.group(1) if city_m else ""
        # Tier 1: deterministic pool lookup (no API call)
        final_output = _inject_booking_links_from_pool(final_output, ctx)
        # Tier 2+3: Tavily for anything still missing
        try:
            final_output = await asyncio.wait_for(
                _enrich_itinerary_links(final_output, city), timeout=20.0
            )
        except asyncio.TimeoutError:
            logger.warning("link enrichment timed out — returning itinerary without enrichment")

        # Fetch real nightly rates from LiteAPI (best-effort, 15s cap)
        if city or ctx.last_city:
            from datetime import date as _date
            checkin_date = None
            if ctx.last_checkin:
                try:
                    checkin_date = _date.fromisoformat(ctx.last_checkin)
                except ValueError:
                    pass
            try:
                await asyncio.wait_for(
                    _enrich_lodging_rates(
                        final_output, ctx,
                        city=city or ctx.last_city,
                        country_code=_country_code(city or ctx.last_city),
                        checkin=checkin_date,
                        nights=ctx.last_nights or 5,
                    ),
                    timeout=15.0,
                )
            except asyncio.TimeoutError:
                logger.warning("lodging rate enrichment timed out")

    _capture_itinerary(ctx, final_output)

    # If store_delta was not called by the replanner, parse prose directly (no extra LLM call)
    if not ctx.pending_delta and ctx.itinerary_json and _looks_like_replan(final_output):
        ctx.pending_delta = _parse_prose_to_delta(final_output) or ""

    validated_delta = _validate_delta(ctx)
    if validated_delta:
        validated_delta = _enrich_delta_from_pool(ctx, validated_delta)
        _apply_delta_to_stored_itinerary(ctx, validated_delta)
    ctx.pending_delta = ""   # consumed
    ctx.save()
    weather = None
    if ctx.weather_data:
        try:
            weather = json.loads(ctx.weather_data)
        except Exception:
            pass

    return ChatResponse(response=final_output, session_id=session_id, delta=validated_delta, weather=weather)


@app.post("/chat/stream")
async def chat_stream_endpoint(request: ChatRequest) -> StreamingResponse:
    """SSE endpoint: emits status events while the agent runs, then a final 'done' event."""
    session_id = request.session_id or str(uuid.uuid4())

    async def event_generator():
        # Fresh queue for this request
        queue: asyncio.Queue = asyncio.Queue()
        _progress_queues[session_id] = queue

        # Attach handler to the openai.agents logger at DEBUG so we capture tool invocations.
        # We temporarily lower its level from ERROR (set above) to DEBUG just for this handler.
        agents_logger = logging.getLogger("openai.agents")
        handler = _ProgressHandler(session_id)
        agents_logger.addHandler(handler)
        prev_level = agents_logger.level
        agents_logger.setLevel(logging.DEBUG)

        yield f"data: {json.dumps({'type': 'status', 'text': 'Connecting to Voyager AI…'})}\n\n"

        async def _run_agent():
            session = InMemorySession(session_id)
            ctx = AppContext.get_or_create(session_id)

            # Sync locked slots sent by the frontend on every request
            if request.locked_slots is not None:
                ctx.locked_slots = request.locked_slots
            ctx.pending_delta = ""   # clear any previous delta before this run

            logger.info("chat/stream session=%s message=%r locked=%s", session_id, request.message[:80], ctx.locked_slots)
            try:
                result = await Runner.run(
                    agent,
                    request.message,
                    session=session,
                    context=ctx,
                    max_turns=15,
                )
            except InputGuardrailTripwireTriggered:
                msg = _guardrail_message(request.message)
                queue.put_nowait({"type": "done", "response": msg, "session_id": session_id})
                return
            except MaxTurnsExceeded:
                logger.warning("chat/stream max_turns exceeded session=%s", session_id)
                queue.put_nowait({
                    "type": "done",
                    "response": (
                        "I wasn't able to figure out exactly which slot to change from that description. "
                        "Could you be a bit more specific? For example: \"replace the museum on Day 2 afternoon with a night market\" "
                        "or \"swap Day 3 morning to something food-related\"."
                    ),
                    "session_id": session_id,
                })
                return
            except Exception as exc:
                logger.error("chat/stream error: %s", traceback.format_exc())
                queue.put_nowait({"type": "error", "detail": str(exc), "session_id": session_id})
                return

            # Prefer the solver_agent's own tool output over the orchestrator's echo of it —
            # Gemini's recitation filter occasionally suppresses the echo (see _extract_tool_output).
            raw_output = _extract_tool_output(result, "solver_agent") or result.final_output
            final_output = _strip_place_id_params(_extract_itinerary_text(raw_output))

            # Guard: orchestrator occasionally returns empty when conversation_agent result isn't echoed
            if not final_output.strip():
                logger.warning("chat/stream empty final_output session=%s — asking user to clarify", session_id)
                queue.put_nowait({
                    "type": "done",
                    "response": (
                        "I didn't quite catch that. Could you rephrase? "
                        "For example: \"replace the museum on Day 2 afternoon with a night market\" "
                        "or \"which day has too much Italian food?\""
                    ),
                    "session_id": session_id,
                })
                return

            logger.info("chat/stream final_output len=%d looks_like_itin=%s first100=%r",
                        len(final_output), _looks_like_itinerary(final_output), final_output[:100])

            # Extract city for post-processing — both enrichments run after done is sent
            city_m = re.search(r'\b([A-Z][a-z]+)\s+Itinerary\b', final_output)
            city = city_m.group(1) if city_m else ""

            _capture_itinerary(ctx, final_output)

            # If store_delta was not called by the replanner, parse prose directly
            if not ctx.pending_delta and ctx.itinerary_json and _looks_like_replan(final_output):
                ctx.pending_delta = _parse_prose_to_delta(final_output) or ""

            validated_delta = _validate_delta(ctx)
            if validated_delta:
                _apply_delta_to_stored_itinerary(ctx, validated_delta)
            ctx.pending_delta = ""   # consumed
            ctx.save()

            weather = None
            if ctx.weather_data:
                try:
                    weather = json.loads(ctx.weather_data)
                except Exception:
                    pass

            queue.put_nowait({
                "type": "done",
                "response": final_output,
                "session_id": session_id,
                "delta": validated_delta,
                "weather": weather,
            })

            # Both enrichments run in background — never block the done event
            if _looks_like_itinerary(final_output):
                async def _bg_enrich(text: str, c: str, context: "AppContext") -> None:
                    from datetime import date as _date
                    # Booking links: Tier 1 (pool, sync) then Tier 2+3 (Tavily, async)
                    enriched = _inject_booking_links_from_pool(text, context)
                    if c:
                        try:
                            enriched = await asyncio.wait_for(
                                _enrich_itinerary_links(enriched, c), timeout=25.0
                            )
                        except Exception:
                            pass
                    if enriched != text and context.itinerary_json:
                        try:
                            itin = json.loads(context.itinerary_json)
                            itin["text"] = enriched
                            context.itinerary_json = json.dumps(itin)
                            context.save()
                        except Exception:
                            pass
                    # Hotel rates
                    eff_city = c or context.last_city
                    if eff_city:
                        checkin_date = None
                        if context.last_checkin:
                            try:
                                checkin_date = _date.fromisoformat(context.last_checkin)
                            except ValueError:
                                pass
                        try:
                            await asyncio.wait_for(
                                _enrich_lodging_rates(
                                    text, context,
                                    city=eff_city,
                                    country_code=_country_code(eff_city),
                                    checkin=checkin_date,
                                    nights=context.last_nights or 5,
                                ),
                                timeout=15.0,
                            )
                        except Exception:
                            pass
                asyncio.create_task(_bg_enrich(final_output, city, ctx))

        task = asyncio.create_task(_run_agent())

        try:
            while True:
                try:
                    msg = await asyncio.wait_for(queue.get(), timeout=0.4)
                    yield f"data: {json.dumps(msg)}\n\n"
                    if msg.get("type") in ("done", "error"):
                        break
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
                    if task.done():
                        break
        finally:
            agents_logger.removeHandler(handler)
            agents_logger.setLevel(prev_level)
            _progress_queues.pop(session_id, None)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/debug/last-run")
async def debug_last_run() -> dict:
    """Return context state for the most recent session — useful during demos."""
    if not _context_store:
        return {"error": "no sessions yet"}
    last_session_id = list(_context_store.keys())[-1]
    ctx = _context_store[last_session_id]
    return {"session_id": last_session_id, "context": ctx.model_dump()}


@app.get("/qr")
async def qr_endpoint(url: str) -> JSONResponse:
    """Generate a QR code PNG for *url* and return it as a base64 data URI."""
    from backend.tools.maps_links import qr_code_base64
    data_uri = qr_code_base64(url)
    if not data_uri:
        return JSONResponse(status_code=500, content={"error": "qr generation failed"})
    return JSONResponse(content={"data_uri": data_uri})


app.mount("/", StaticFiles(directory="static", html=True), name="static")


# CLI --------------------------------------------------------------------------

cli = typer.Typer(help="Travel Optimizer CLI")


@cli.command()
def ask(query: str, session_id: str = "cli_session") -> None:
    """Run a single query against the agent from the CLI."""
    async def _run() -> None:
        typer.echo(f"Session: {session_id}\n" + "-" * 60)
        session = InMemorySession(session_id)
        ctx = AppContext.get_or_create(session_id)

        try:
            result = await Runner.run(agent, query, session=session, context=ctx, max_turns=15)
        except InputGuardrailTripwireTriggered:
            typer.echo(_guardrail_message(query))
            return

        final_output = _extract_tool_output(result, "solver_agent") or result.final_output
        _capture_itinerary(ctx, final_output)
        ctx.save()
        typer.echo(f"Agent:\n{final_output}\n" + "-" * 60)

    asyncio.run(_run())


@cli.command()
def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Start the FastAPI server."""
    typer.echo(f"Starting Travel Optimizer on http://{host}:{port} ...")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    cli()
