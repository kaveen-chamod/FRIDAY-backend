from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import os
import sys
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# .env ගොනුව නිවැරදිව load කරගැනීම
load_dotenv(PROJECT_ROOT / ".env", override=True)

# Response සහ CORSMiddleware නිවැරදිව import කිරීම
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, Request, Response, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from google.genai import types
from pydantic import BaseModel, Field, StrictBool, StrictInt, StrictStr
from starlette.responses import JSONResponse

from app import access_control
from app.agent import MAX_AUDIO_BYTES, FridayAgent
from app.web_bridge import WebPlayerBridge
from app.digital_twin import DigitalTwinEngine, bind_default_engine
from app.ambient_observer import (
    AmbientObserver,
    ObservationConsentRequired,
    ObservationRateLimited,
)
from app.presence import PresenceEngine
from app.predictive_engine import (
    SimulationCapacityError,
    SimulationInputError,
    SimulationJobManager,
    SimulationServiceClosed,
)
from app.self_reflection import (
    ReflectionInputError,
    ReflectionStorageError,
    SelfReflectionLedger,
)
from app.defense.asset_registry import (
    AssetRegistryError,
    AuthorizedAssetRegistry,
)
from app.defense.forensics import ForensicsError, IncidentLedger
from app.defense.orchestrator import (
    DefenseActionError,
    DefenseOrchestrator,
    bind_default_orchestrator,
)
from app.defense.threat_matrix import ThreatEvaluationError, ThreatMatrix
from app.robotics.collision_guard import CollisionGuard, CollisionGuardError
from app.robotics.drone import DroneVehicle
from app.robotics.rc_car import RCCarVehicle
from app.robotics.spatial_map import SpatialMap, SpatialMapError, WaypointRegistry
from app.robotics.twin_adapter import RoboticsTwinAdapter
from core import confirm as confirm_gate
from core import security

_LOG = logging.getLogger("friday.server")
_API_TOKEN = os.environ.get("FRIDAY_API_TOKEN", "my-super-secret-friday-token-1234567890").strip()


async def _retry_telegram_start(service: Any) -> None:
    delay = 5.0
    while True:
        await asyncio.sleep(delay)
        try:
            await service.start()
        except asyncio.CancelledError:
            raise
        except Exception:
            _LOG.warning(
                "Telegram background service retry failed; next retry in %.0f seconds.",
                min(delay * 2, 300.0),
                exc_info=True,
            )
            delay = min(delay * 2, 300.0)
        else:
            _LOG.info("Telegram background service recovered.")
            return


async def _start_telegram_or_schedule_retry(
    service: Any,
) -> asyncio.Task | None:
    try:
        await service.start()
    except Exception as exc:
        _LOG.warning(
            "Telegram background service failed to start; the API will remain "
            "available and webhook registration will retry: %s",
            exc,
            exc_info=True,
        )
        return asyncio.create_task(
            _retry_telegram_start(service),
            name="friday-telegram-startup-retry",
        )
    return None


class ChatTextRequest(BaseModel):
    message: str = Field(min_length=1, max_length=16_000)
    session_id: str = Field(default="default", min_length=1, max_length=128)
    speaker_token: str | None = Field(default=None, max_length=2048)
    master_passphrase: str | None = Field(default=None, max_length=1024)


class ConfirmationRequest(BaseModel):
    accepted: bool


class ToolProposalRequest(BaseModel):
    capability: str = Field(min_length=1, max_length=500)
    master_passphrase: str = Field(min_length=1, max_length=1024)


class MasterPassphraseRequest(BaseModel):
    passphrase: str = Field(min_length=1, max_length=1024)


class SuccessionHeartbeatRequest(BaseModel):
    speaker_token: str = Field(min_length=1, max_length=2048)
    master_passphrase: str | None = Field(default=None, max_length=1024)


class SuccessionEnrollRequest(BaseModel):
    target_speaker_token: str = Field(min_length=1, max_length=2048)
    emergency_passphrase: str = Field(min_length=16, max_length=1024)
    master_passphrase: str = Field(min_length=1, max_length=1024)


class SuccessionClaimRequest(BaseModel):
    speaker_token: str = Field(min_length=1, max_length=2048)
    emergency_passphrase: str = Field(min_length=1, max_length=1024)
    master_passphrase: str = Field(min_length=1, max_length=1024)
    manual_emergency_trigger: bool = False
    master_speaker_token: str | None = Field(default=None, max_length=2048)


class SuccessionGuardianApprovalRequest(BaseModel):
    claim_id: str = Field(min_length=1, max_length=128)
    guardian_speaker_token: str = Field(min_length=1, max_length=2048)
    master_passphrase: str = Field(min_length=1, max_length=1024)


class SuccessionReclaimRequest(BaseModel):
    speaker_token: str = Field(min_length=1, max_length=2048)
    master_passphrase: str = Field(min_length=1, max_length=1024)


class SimulationPredictRequest(BaseModel):
    action_spec: dict[str, Any]
    profile: str = Field(default="STANDARD", min_length=1, max_length=16)
    seed: int | None = Field(default=None, ge=0, lt=2**63)
    idempotency_key: str | None = Field(default=None, min_length=1, max_length=128)


class TwinSimulateRequest(BaseModel):
    device_id: str = Field(min_length=1, max_length=80)
    command: dict[str, Any]
    profile: str = Field(default="STANDARD", min_length=1, max_length=16)
    seed: int | None = Field(default=None, ge=0, lt=2**63)


class AmbientConsentRequest(BaseModel):
    enabled: StrictBool


class PresenceTelemetryRequest(BaseModel):
    pir_active: StrictBool
    person_count: StrictInt = Field(ge=0)
    light_lux: float = Field(default=0.0, ge=0.0, allow_inf_nan=False)


class PresenceWakeRequest(BaseModel):
    source: StrictStr = Field(default="home_mic", min_length=1, max_length=64)


class AmbientObservationRequest(BaseModel):
    event_id: StrictStr = Field(min_length=1, max_length=128)
    source: StrictStr = Field(min_length=1, max_length=64)
    content: StrictStr = Field(min_length=1, max_length=16_384)


class ReflectionFeedbackRequest(BaseModel):
    prediction_id: StrictStr = Field(min_length=1, max_length=128)
    actual_success: StrictBool


class DefenseAssetRegistrationRequest(BaseModel):
    asset_id: StrictStr = Field(min_length=1, max_length=80)
    asset_type: StrictStr = Field(min_length=1, max_length=32)
    allowed_telemetry_sources: list[StrictStr] = Field(default_factory=list, max_length=32)
    permitted_defense_actions: list[StrictStr] = Field(default_factory=list, max_length=5)
    criticality: StrictStr = Field(default="MEDIUM", min_length=1, max_length=32)
    master_passphrase: StrictStr = Field(min_length=1, max_length=1024)


class DefenseMitigationRequest(BaseModel):
    target_asset: StrictStr = Field(min_length=1, max_length=80)
    action: StrictStr = Field(min_length=1, max_length=64)
    indicator: StrictStr = Field(min_length=1, max_length=512)
    ttl_seconds: int | None = Field(default=None, ge=1, le=3_600)
    master_passphrase: StrictStr = Field(min_length=1, max_length=1024)


class DefenseThreatEventRequest(BaseModel):
    target_asset: StrictStr = Field(min_length=1, max_length=80)
    category: StrictStr = Field(min_length=1, max_length=64)
    severity: StrictStr = Field(min_length=1, max_length=16)
    signal_id: StrictStr = Field(min_length=1, max_length=128)
    indicator: StrictStr | None = Field(default=None, max_length=512)
    evidence: list[StrictStr] = Field(default_factory=list, max_length=32)
    false_positive: StrictBool = False


class RoboticsDeviceRegistrationRequest(BaseModel):
    device_id: StrictStr = Field(min_length=1, max_length=80)
    vehicle_type: StrictStr = Field(min_length=1, max_length=16)
    specification: dict[str, Any]
    telemetry_state: dict[str, Any]
    master_passphrase: StrictStr = Field(min_length=1, max_length=1024)


class RoboticsTelemetryRequest(BaseModel):
    device_id: StrictStr = Field(min_length=1, max_length=80)
    telemetry_at: StrictStr = Field(min_length=1, max_length=64)
    state: dict[str, Any]


class RoboticsMapRequest(BaseModel):
    width: int = Field(ge=1, le=250_000)
    height: int = Field(ge=1, le=250_000)
    depth: int = Field(default=1, ge=1, le=250_000)
    obstacles: list[list[int]] = Field(default_factory=list, max_length=100_000)


class RoboticsWaypointRequest(BaseModel):
    name: StrictStr = Field(min_length=1, max_length=64)
    coordinate: list[int] = Field(min_length=2, max_length=3)


class RoboticsPlanRequest(BaseModel):
    start: list[int] = Field(min_length=2, max_length=3)
    goal: list[int] = Field(min_length=2, max_length=3)


class RoboticsPreflightRequest(BaseModel):
    device_id: StrictStr = Field(min_length=1, max_length=80)
    action: dict[str, Any]
    path: list[list[int]] | None = Field(default=None, max_length=10_000)


class RoboticsTelemetryIngestRequest(BaseModel):
    device_id: StrictStr = Field(min_length=1, max_length=80)
    telemetry_at: StrictStr = Field(min_length=1, max_length=64)
    distance_cm: float
    state: dict[str, Any]


_SIMULATION_RATE_LOCK = threading.Lock()
_SIMULATION_SUBMISSIONS: dict[str, deque[float]] = {}
_SIMULATION_RATE_LIMIT = 30
_SIMULATION_RATE_WINDOW_SECONDS = 60.0
_SIMULATION_MAX_REQUEST_BYTES = 64 * 1024
_COGNITIVE_MAX_REQUEST_BYTES = 64 * 1024
_REFLECTION_RATE_LOCK = threading.Lock()
_REFLECTION_SUBMISSIONS: dict[str, deque[float]] = {}
_REFLECTION_RATE_LIMIT = 30
_DEFENSE_RATE_LOCK = threading.Lock()
_DEFENSE_SUBMISSIONS: dict[str, deque[float]] = {}
_DEFENSE_RATE_LIMIT = 60


def _enforce_reflection_rate_limit() -> None:
    now = time.monotonic()
    identity = hashlib.sha256(_API_TOKEN.encode("utf-8")).hexdigest()
    with _REFLECTION_RATE_LOCK:
        submissions = _REFLECTION_SUBMISSIONS.setdefault(identity, deque())
        while submissions and now - submissions[0] >= _SIMULATION_RATE_WINDOW_SECONDS:
            submissions.popleft()
        if len(submissions) >= _REFLECTION_RATE_LIMIT:
            raise HTTPException(
                status_code=429,
                detail="Reflection request rate limit exceeded.",
            )
        submissions.append(now)


def _enforce_defense_rate_limit() -> None:
    now = time.monotonic()
    identity = hashlib.sha256(_API_TOKEN.encode("utf-8")).hexdigest()
    with _DEFENSE_RATE_LOCK:
        submissions = _DEFENSE_SUBMISSIONS.setdefault(identity, deque())
        while submissions and now - submissions[0] >= _SIMULATION_RATE_WINDOW_SECONDS:
            submissions.popleft()
        if len(submissions) >= _DEFENSE_RATE_LIMIT:
            raise HTTPException(
                status_code=429,
                detail="Defense API request rate limit exceeded.",
            )
        submissions.append(now)


def _enforce_simulation_rate_limit() -> None:
    now = time.monotonic()
    identity = hashlib.sha256(_API_TOKEN.encode("utf-8")).hexdigest()
    with _SIMULATION_RATE_LOCK:
        submissions = _SIMULATION_SUBMISSIONS.setdefault(identity, deque())
        while (
            submissions
            and now - submissions[0] >= _SIMULATION_RATE_WINDOW_SECONDS
        ):
            submissions.popleft()
        if len(submissions) >= _SIMULATION_RATE_LIMIT:
            raise HTTPException(
                status_code=429,
                detail="Simulation request rate limit exceeded.",
            )
        submissions.append(now)


@asynccontextmanager
async def lifespan(application: FastAPI):
    if len(_API_TOKEN) < 32:
        raise RuntimeError(
            "Set FRIDAY_API_TOKEN to a random secret of at least 32 characters before startup."
        )

    event_loop = asyncio.get_running_loop()
    bridge = WebPlayerBridge()
    bridge.set_loop(event_loop)
    telegram = None
    telegram_retry_task: asyncio.Task | None = None
    simulation_service: SimulationJobManager | None = None
    digital_twin: DigitalTwinEngine | None = None
    defense_orchestrator: DefenseOrchestrator | None = None
    robotics_collision_guard: CollisionGuard | None = None
    try:
        reflection_ledger = SelfReflectionLedger()
        simulation_service = SimulationJobManager(
            on_completed=reflection_ledger.record_simulation_prediction
        )
        await simulation_service.start()
        digital_twin = DigitalTwinEngine(
            predictive_service=simulation_service
        )
        robotics_adapter = RoboticsTwinAdapter()
        digital_twin.register_adapter(robotics_adapter)
        digital_twin.bind_loop(event_loop)
        bind_default_engine(digital_twin)
        agent = FridayAgent(bridge)
        application.state.bridge = bridge
        application.state.agent = agent
        application.state.api_token = _API_TOKEN
        application.state.succession = agent.succession
        application.state.simulation_service = simulation_service
        application.state.digital_twin = digital_twin
        application.state.ambient_observer = AmbientObserver()
        application.state.presence_engine = PresenceEngine()
        application.state.reflection_ledger = reflection_ledger
        application.state.cognitive_io_slots = asyncio.Semaphore(2)

        # ── Auto-Bootstrap Master Passphrase from .env on First Boot ───────
        initial_passphrase = os.environ.get("FRIDAY_MASTER_PASSPHRASE", "FRIDAY@ACCESSAPPROVED").strip()
        if initial_passphrase and len(initial_passphrase) >= 16:
            try:
                if not access_control.master_passphrase_configured():
                    access_control.configure_master_passphrase(initial_passphrase)
                    _LOG.info("🔐 [MasterAuth] Master Passphrase successfully auto-configured from .env.")
            except Exception as exc:
                _LOG.debug("Master auth auto-bootstrap status: %s", exc)
        # ───────────────────────────────────────────────────────────────────

        def defense_alert(message: str) -> None:
            async def deliver() -> None:
                try:
                    await agent.send_telegram_alert(message)
                except Exception:
                    _LOG.exception("Could not deliver a defense Telegram alert.")

            def schedule_alert() -> None:
                try:
                    asyncio.create_task(deliver())
                except Exception:
                    _LOG.exception("Could not schedule a defense Telegram alert.")

            event_loop.call_soon_threadsafe(schedule_alert)

        asset_registry = AuthorizedAssetRegistry()
        incident_ledger = IncidentLedger()
        defense_orchestrator = DefenseOrchestrator(
            asset_registry,
            ThreatMatrix(asset_registry),
            incident_ledger,
            alert_callback=defense_alert,
        )
        await defense_orchestrator.start()
        bind_default_orchestrator(defense_orchestrator)
        application.state.defense_assets = asset_registry
        application.state.defense_incidents = incident_ledger
        application.state.defense_orchestrator = defense_orchestrator
        application.state.robotics_map = None
        application.state.robotics_waypoints = None
        application.state.robotics_vehicles = {}
        application.state.robotics_specs = {}
        for twin_device in digital_twin.status()["devices"]:
            if twin_device["domain"] != robotics_adapter.domain:
                continue
            try:
                config = digital_twin.device_configuration(twin_device["device_id"])
                spec = config["specification"]
                state = config["state"]
                if not robotics_adapter.validate_device(spec, state):
                    _LOG.error(
                        "Persisted robotics twin %s failed adapter validation; "
                        "vehicle remains disabled.",
                        twin_device["device_id"],
                    )
                    continue
                vehicle_id = twin_device["device_id"]
                if (
                    spec["vehicle_type"] != "rc_car"
                    or not RCCarVehicle.valid_device_id(vehicle_id)
                ):
                    _LOG.error(
                        "Persisted robotics twin %s has no enabled vehicle protocol.",
                        vehicle_id,
                    )
                    continue
                application.state.robotics_vehicles[vehicle_id] = RCCarVehicle(
                    vehicle_id
                )
                application.state.robotics_specs[vehicle_id] = spec
            except (KeyError, TypeError, ValueError) as exc:
                _LOG.error(
                    "Persisted robotics twin %s could not be activated: %s",
                    twin_device["device_id"],
                    exc,
                )

        def emergency_robot_stop(vehicle_id: str) -> str:
            vehicle = application.state.robotics_vehicles.get(vehicle_id)
            if vehicle is None:
                return "Emergency stop not delivered: vehicle is not enrolled."
            return vehicle.emergency_stop()

        robotics_collision_guard = CollisionGuard(
            emergency_stop=emergency_robot_stop
        )
        application.state.robotics_collision_guard = robotics_collision_guard

        def security_log(message: str) -> None:
            bridge.write_log(message)
            if not any(
                marker in message
                for marker in ("BLOCKED", "RATE-LIMITED", "TAMPER")
            ):
                return
            if not (
                os.environ.get("FRIDAY_TELEGRAM_BOT_TOKEN")
                or os.environ.get("TELEGRAM_BOT_TOKEN")
            ) or not os.environ.get("TELEGRAM_ALERT_CHAT_IDS"):
                return

            async def send_alert() -> None:
                try:
                    from app.telegram_bot import send_telegram_alert

                    await send_telegram_alert(
                        f"F.R.I.D.A.Y. security alert: {message}"
                    )
                except Exception:
                    _LOG.exception("Could not deliver a Telegram security alert.")

            event_loop.call_soon_threadsafe(
                lambda: asyncio.create_task(send_alert())
            )

        security.bind(logger=security_log)
        confirm_gate.bind(
            show=bridge.show_confirmation,
            hide=bridge.hide_confirmation,
            log=bridge.write_log,
        )
        from app.telegram_bot import TelegramBotService

        telegram = TelegramBotService(agent)
        telegram_retry_task = await _start_telegram_or_schedule_retry(telegram)
        application.state.telegram = telegram
        yield
    finally:
        try:
            if telegram_retry_task is not None:
                telegram_retry_task.cancel()
                await asyncio.gather(
                    telegram_retry_task, return_exceptions=True
                )
            if telegram is not None:
                await telegram.stop()
        finally:
            try:
                bind_default_orchestrator(None)
                if defense_orchestrator is not None:
                    await defense_orchestrator.stop()
                bind_default_engine(None)
                if digital_twin is not None:
                    digital_twin.bind_loop(None)
                if simulation_service is not None:
                    await simulation_service.shutdown()
            finally:
                bridge.set_loop(None)


app = FastAPI(
    title="F.R.I.D.A.Y. Assistant API",
    version="1.0.0",
    lifespan=lifespan,
)

# ── 1. CORS Middleware (Browser Preflight requests සාර්ථක කර ගැනීමට) ────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_origin_regex=r"https?://.*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


# ── 2. Global Preflight OPTIONS Handler (405 Method Not Allowed වළක්වයි) ───────
@app.options("/{full_path:path}")
async def options_handler(request: Request, full_path: str):
    response = Response(status_code=200)
    origin = request.headers.get("origin", "*")
    response.headers["Access-Control-Allow-Origin"] = origin
    response.headers["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS, PUT, DELETE, PATCH"
    response.headers["Access-Control-Allow-Headers"] = "*"
    response.headers["Access-Control-Allow-Credentials"] = "true"
    return response


@app.middleware("http")
async def sanctuary_lockdown_gate(request: Request, call_next):
    # Preflight OPTIONS requests කිසි විටෙකත් අවහිර නොකර ඉදිරියට යවයි
    if request.method == "OPTIONS":
        return await call_next(request)

    if request.url.path in {
        "/api/simulation/predict",
        "/api/twin/simulate",
        "/api/ambient/consent",
        "/api/ambient/observe",
        "/api/reflection/outcomes",
    } or request.url.path.startswith(
        ("/api/reflection/", "/api/defense/", "/api/robotics/")
    ):
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                request_size = int(content_length)
            except ValueError:
                return JSONResponse(
                    status_code=400,
                    content={"detail": "Invalid Content-Length header."},
                )
            maximum_size = (
                _SIMULATION_MAX_REQUEST_BYTES
                if request.url.path in {
                    "/api/simulation/predict",
                    "/api/twin/simulate",
                }
                else _COGNITIVE_MAX_REQUEST_BYTES
            )
            if request_size > maximum_size:
                return JSONResponse(
                    status_code=413,
                    content={"detail": "Request exceeds the 64 KiB limit."},
                )
        buffered_messages: list[dict[str, Any]] = []
        request_size = 0
        while True:
            message = await request.receive()
            if message["type"] == "http.disconnect":
                return JSONResponse(
                    status_code=400,
                    content={"detail": "Request body was interrupted."},
                )
            if message["type"] != "http.request":
                continue
            request_size += len(message.get("body", b""))
            maximum_size = (
                _SIMULATION_MAX_REQUEST_BYTES
                if request.url.path in {
                    "/api/simulation/predict",
                    "/api/twin/simulate",
                }
                else _COGNITIVE_MAX_REQUEST_BYTES
            )
            if request_size > maximum_size:
                return JSONResponse(
                    status_code=413,
                    content={"detail": "Request exceeds the 64 KiB limit."},
                )
            buffered_messages.append(message)
            if not message.get("more_body", False):
                break

        async def replay_request_body() -> dict[str, Any]:
            if buffered_messages:
                return buffered_messages.pop(0)
            return {"type": "http.request", "body": b"", "more_body": False}

        request._receive = replay_request_body
    defense = getattr(request.app.state, "defense_orchestrator", None)
    if defense is not None and request.url.path not in {
        "/health",
        "/api/defense/rollback",
    } and not request.url.path.startswith("/api/defense/rollback/") and request.method != "HEAD":
        client_ip = request.client.host if request.client is not None else None
        restriction = defense.check_request(client_ip)
        if restriction == "blocked":
            return JSONResponse(
                status_code=403,
                content={"detail": "Request source is temporarily blocked."},
            )
        if restriction == "rate_limited":
            return JSONResponse(
                status_code=429,
                content={"detail": "Request source is temporarily rate-limited."},
                headers={"Retry-After": "60"},
            )
    manager = getattr(request.app.state, "succession", None)
    if manager is not None and manager.is_locked_down() and (
        request.method,
        request.url.path,
    ) not in {
        ("GET", "/health"),
        ("POST", "/api/voice/verify"),
        ("GET", "/api/succession/status"),
        ("POST", "/api/succession/reclaim"),
    }:
        return JSONResponse(
            status_code=423,
            content={"detail": "F.R.I.D.A.Y. is in SANCTUARY_LOCKDOWN."},
        )
    return await call_next(request)


def _token_matches(value: str | None) -> bool:
    token = _API_TOKEN
    supplied = (value or "").strip()
    if supplied.lower().startswith("bearer "):
        supplied = supplied[7:].strip()
    return bool(supplied) and hmac.compare_digest(supplied, token)


async def require_api_token(request: Request) -> None:
    if not _token_matches(request.headers.get("authorization")):
        raise HTTPException(
            status_code=401,
            detail="Valid API credentials are required.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    manager = getattr(request.app.state, "succession", None)
    if manager is not None and manager.is_locked_down():
        if request.url.path not in {
            "/health", "/api/voice/verify", "/api/succession/status",
            "/api/succession/reclaim",
        }:
            raise HTTPException(
                status_code=423,
                detail="F.R.I.D.A.Y. is in SANCTUARY_LOCKDOWN.",
            )


def _agent(request: Request) -> FridayAgent:
    return request.app.state.agent


@app.get("/health")
async def health(request: Request) -> dict[str, str]:
    manager = getattr(request.app.state, "succession", None)
    if manager is not None and manager.is_locked_down():
        return {"status": "SANCTUARY_LOCKDOWN", "assistant": "F.R.I.D.A.Y."}
    return {"status": "ok", "assistant": "F.R.I.D.A.Y."}


@app.post("/api/telegram-webhook")
async def telegram_webhook(request: Request) -> dict[str, str]:
    telegram = getattr(request.app.state, "telegram", None)
    if telegram is None or not telegram.running:
        raise HTTPException(status_code=503, detail="Telegram webhook service is unavailable.")
    if telegram.mode != "webhook":
        raise HTTPException(
            status_code=409,
            detail="Telegram webhooks are disabled while polling mode is active.",
        )
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > 1_048_576:
            raise HTTPException(status_code=413, detail="Telegram update exceeds 1 MB.")
    try:
        processed = await telegram.process_webhook(
            bytes(body),
            request.headers.get("X-Telegram-Bot-Api-Secret-Token"),
        )
    except PermissionError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        _LOG.exception("Telegram webhook is unavailable.")
        raise HTTPException(status_code=503, detail="Telegram update could not be processed.") from exc
    except Exception as exc:
        _LOG.exception("Telegram webhook update failed.")
        raise HTTPException(status_code=503, detail="Telegram update could not be processed.") from exc
    return {"status": "processed" if processed else "ignored_or_duplicate"}


@app.post("/api/chat-text", dependencies=[Depends(require_api_token)])
async def chat_text(payload: ChatTextRequest, request: Request) -> dict[str, str]:
    try:
        response = await _agent(request).chat_text(
            payload.message, payload.session_id, payload.speaker_token,
            payload.master_passphrase,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        _LOG.exception("Text interaction failed.")
        raise HTTPException(status_code=502, detail=f"Assistant request failed: {exc}") from exc
    return {"response": response, "session_id": payload.session_id}


@app.post("/api/chat-voice", dependencies=[Depends(require_api_token)])
async def chat_voice(
    request: Request,
    file: UploadFile = File(...),
    session_id: str = Form(default="default", min_length=1, max_length=128),
    master_passphrase: str | None = Form(default=None, max_length=1024),
    speaker_token: str | None = Form(default=None, max_length=2048),
) -> dict[str, str]:
    mime_type = file.content_type or ""
    if not mime_type.startswith("audio/"):
        raise HTTPException(status_code=415, detail="Upload an audio file.")
    audio = await file.read(MAX_AUDIO_BYTES + 1)
    if len(audio) > MAX_AUDIO_BYTES:
        raise HTTPException(status_code=413, detail="Audio upload exceeds the 25 MB limit.")
    try:
        result = await _agent(request).chat_audio(
            audio, mime_type, session_id, master_passphrase, speaker_token
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        _LOG.exception("Voice interaction failed.")
        raise HTTPException(status_code=502, detail=f"Voice request failed: {exc}") from exc
    return {
        "transcript": result["transcript"],
        "response": result["response"],
        "audio_base64": base64.b64encode(result["audio"]).decode("ascii"),
        "audio_mime_type": result["audio_mime_type"],
        "session_id": session_id,
        "speaker_name": result.get("speaker_name", "Guest"),
        "speaker_role": result.get("speaker_role", "GUEST"),
    }


@app.post(
    "/api/simulation/predict",
    status_code=202,
    dependencies=[Depends(require_api_token)],
)
async def submit_simulation(
    payload: SimulationPredictRequest, request: Request
) -> dict[str, Any]:
    _enforce_simulation_rate_limit()
    try:
        encoded_spec = json.dumps(
            payload.action_spec, ensure_ascii=False, allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=422, detail="action_spec must contain finite JSON values."
        ) from exc
    if len(encoded_spec) > 32_768:
        raise HTTPException(
            status_code=413, detail="action_spec exceeds the 32 KiB limit."
        )
    service = getattr(request.app.state, "simulation_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Simulation service is unavailable.")
    try:
        return await service.submit(
            payload.action_spec,
            profile=payload.profile,
            seed=payload.seed,
            idempotency_key=payload.idempotency_key,
        )
    except SimulationInputError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except SimulationCapacityError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except SimulationServiceClosed as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get(
    "/api/simulation/jobs/{job_id}",
    dependencies=[Depends(require_api_token)],
)
async def get_simulation_job(job_id: str, request: Request) -> dict[str, Any]:
    if len(job_id) != 32 or any(char not in "0123456789abcdef" for char in job_id):
        raise HTTPException(status_code=404, detail="Simulation job was not found.")
    service = getattr(request.app.state, "simulation_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Simulation service is unavailable.")
    job = await service.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Simulation job was not found.")
    return job


@app.delete(
    "/api/simulation/jobs/{job_id}",
    dependencies=[Depends(require_api_token)],
)
async def cancel_simulation_job(job_id: str, request: Request) -> dict[str, Any]:
    if len(job_id) != 32 or any(char not in "0123456789abcdef" for char in job_id):
        raise HTTPException(status_code=404, detail="Simulation job was not found.")
    service = getattr(request.app.state, "simulation_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Simulation service is unavailable.")
    job = await service.cancel(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Simulation job was not found.")
    return job


@app.get("/api/simulation/status", dependencies=[Depends(require_api_token)])
async def simulation_status(request: Request) -> dict[str, Any]:
    service = getattr(request.app.state, "simulation_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Simulation service is unavailable.")
    return await service.resource_snapshot()


@app.post(
    "/api/presence/telemetry",
    dependencies=[Depends(require_api_token)],
)
async def ingest_presence_telemetry(
    payload: PresenceTelemetryRequest, request: Request
) -> dict[str, Any]:
    engine = getattr(request.app.state, "presence_engine", None)
    if engine is None:
        raise HTTPException(status_code=503, detail="Presence engine is unavailable.")
    try:
        return engine.ingest_sensor_telemetry(
            payload.pir_active, payload.person_count, payload.light_lux
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post(
    "/api/presence/wake-word",
    dependencies=[Depends(require_api_token)],
)
async def trigger_presence_wake(
    payload: PresenceWakeRequest, request: Request
) -> dict[str, Any]:
    engine = getattr(request.app.state, "presence_engine", None)
    if engine is None:
        raise HTTPException(status_code=503, detail="Presence engine is unavailable.")
    try:
        return engine.trigger_wake_word(payload.source)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get(
    "/api/presence/state",
    dependencies=[Depends(require_api_token)],
)
async def presence_state(request: Request) -> dict[str, Any]:
    engine = getattr(request.app.state, "presence_engine", None)
    if engine is None:
        raise HTTPException(status_code=503, detail="Presence engine is unavailable.")
    return engine.get_presence_state()


@app.get("/api/ambient/consent", dependencies=[Depends(require_api_token)])
async def ambient_consent_status(
    request: Request,
    speaker_token: str | None = Header(
        default=None, alias="X-Friday-Speaker-Token"
    ),
) -> dict[str, bool]:
    observer = getattr(request.app.state, "ambient_observer", None)
    if observer is None:
        raise HTTPException(status_code=503, detail="Ambient observer is unavailable.")
    profile_id = _ambient_profile_id(request, speaker_token)
    return {"enabled": observer.consent_status(profile_id)}


@app.post("/api/ambient/consent", dependencies=[Depends(require_api_token)])
async def set_ambient_consent(
    payload: AmbientConsentRequest,
    request: Request,
    speaker_token: str | None = Header(
        default=None, alias="X-Friday-Speaker-Token"
    ),
) -> dict[str, bool]:
    observer = getattr(request.app.state, "ambient_observer", None)
    if observer is None:
        raise HTTPException(status_code=503, detail="Ambient observer is unavailable.")
    profile_id = _ambient_profile_id(request, speaker_token)
    try:
        enabled = observer.set_consent(profile_id, payload.enabled)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"enabled": enabled}


@app.post("/api/ambient/observe", dependencies=[Depends(require_api_token)])
async def observe_ambient_event(
    payload: AmbientObservationRequest,
    request: Request,
    speaker_token: str | None = Header(
        default=None, alias="X-Friday-Speaker-Token"
    ),
) -> dict[str, Any]:
    observer = getattr(request.app.state, "ambient_observer", None)
    if observer is None:
        raise HTTPException(status_code=503, detail="Ambient observer is unavailable.")
    profile_id = _ambient_profile_id(request, speaker_token)
    if not observer.consent_status(profile_id):
        raise HTTPException(
            status_code=403, detail="Explicit observer consent is required."
        )
    if payload.source != "python_code_buffer":
        raise HTTPException(
            status_code=422,
            detail=(
                "Only explicitly submitted Python code-buffer events are supported; "
                "conversation transcript processing is NOT_SUPPORTED."
            ),
        )
    try:
        result = await _run_cognitive_io(
            request,
            observer.observe_code,
            profile_id,
            payload.event_id,
            payload.source,
            payload.content,
        )
    except ObservationConsentRequired as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ObservationRateLimited as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        _LOG.exception("Explicit ambient observation failed.")
        raise HTTPException(
            status_code=503, detail="Ambient observation is temporarily unavailable."
        ) from exc
    return result.as_dict()


@app.post("/api/reflection/outcomes", dependencies=[Depends(require_api_token)])
async def record_reflection_feedback(
    payload: ReflectionFeedbackRequest, request: Request
) -> dict[str, Any]:
    _enforce_reflection_rate_limit()
    ledger = getattr(request.app.state, "reflection_ledger", None)
    if ledger is None:
        raise HTTPException(status_code=503, detail="Reflection service is unavailable.")
    try:
        return await _run_cognitive_io(
            request,
            ledger.record_user_feedback,
            payload.prediction_id,
            payload.actual_success,
        )
    except ReflectionInputError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ReflectionStorageError as exc:
        _LOG.exception("Reflection feedback could not be persisted.")
        raise HTTPException(
            status_code=503, detail="Reflection storage is unavailable."
        ) from exc
    except HTTPException:
        raise


@app.get("/api/reflection/report", dependencies=[Depends(require_api_token)])
async def reflection_report(request: Request) -> dict[str, Any]:
    _enforce_reflection_rate_limit()
    ledger = getattr(request.app.state, "reflection_ledger", None)
    if ledger is None:
        raise HTTPException(status_code=503, detail="Reflection service is unavailable.")
    try:
        return await _run_cognitive_io(request, ledger.report)
    except ReflectionStorageError as exc:
        _LOG.exception("Reflection report could not be read.")
        raise HTTPException(
            status_code=503, detail="Reflection storage is unavailable."
        ) from exc
    except HTTPException:
        raise


def _defense_services(request: Request) -> tuple[AuthorizedAssetRegistry, DefenseOrchestrator, IncidentLedger]:
    registry = getattr(request.app.state, "defense_assets", None)
    orchestrator = getattr(request.app.state, "defense_orchestrator", None)
    incidents = getattr(request.app.state, "defense_incidents", None)
    if registry is None or orchestrator is None or incidents is None:
        raise HTTPException(status_code=503, detail="Defense services are unavailable.")
    return registry, orchestrator, incidents


def _require_defense_master(
    request: Request,
    speaker_token: str | None,
    passphrase: str | None,
) -> None:
    _require_actual_master(request, speaker_token)
    _require_master(passphrase)
    _audit_master_event(
        "defense_admin_operation",
        "authorized",
        f"{request.method} {request.url.path}",
    )


@app.post("/api/defense/assets/register", dependencies=[Depends(require_api_token)])
async def register_defense_asset(
    payload: DefenseAssetRegistrationRequest,
    request: Request,
    speaker_token: str | None = Header(
        default=None, alias="X-Friday-Speaker-Token"
    ),
) -> dict[str, Any]:
    _enforce_defense_rate_limit()
    await _run_cognitive_io(
        request,
        _require_defense_master,
        request,
        speaker_token,
        payload.master_passphrase,
    )
    registry, _, _ = _defense_services(request)
    try:
        return await _run_cognitive_io(
            request,
            registry.register,
            {
                "asset_id": payload.asset_id,
                "asset_type": payload.asset_type,
                "allowed_telemetry_sources": payload.allowed_telemetry_sources,
                "permitted_defense_actions": payload.permitted_defense_actions,
                "criticality": payload.criticality,
            },
            ownership_verified=True,
        )
    except AssetRegistryError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


# ── READ-ONLY DEFENSE ROUTES (API Token එකෙන් බැලීමට ඉඩ දී 403 දෝෂය වළක්වයි) ───

@app.get("/api/defense/assets", dependencies=[Depends(require_api_token)])
async def list_defense_assets(
    request: Request,
) -> dict[str, Any]:
    _enforce_defense_rate_limit()
    registry, _, _ = _defense_services(request)
    return {"assets": await _run_cognitive_io(request, registry.list_assets)}


@app.get("/api/defense/threats/active", dependencies=[Depends(require_api_token)])
async def active_defense_threats(
    request: Request,
) -> dict[str, Any]:
    _enforce_defense_rate_limit()
    _, orchestrator, _ = _defense_services(request)
    return await _run_cognitive_io(request, orchestrator.active_status)


@app.get("/api/defense/incidents", dependencies=[Depends(require_api_token)])
async def list_defense_incidents(
    request: Request,
    offset: int = Query(default=0, ge=0, le=1_000_000),
    limit: int = Query(default=50, ge=1, le=100),
) -> dict[str, Any]:
    _enforce_defense_rate_limit()
    _, _, incidents = _defense_services(request)
    try:
        return await _run_cognitive_io(
            request,
            incidents.list_incidents,
            offset=offset,
            limit=limit,
        )
    except ForensicsError as exc:
        _LOG.exception("Defense incident ledger failed verification.")
        raise HTTPException(status_code=503, detail="Incident ledger is unavailable.") from exc

# ─────────────────────────────────────────────────────────────────────────────


@app.post(
    "/api/defense/threats/evaluate",
    status_code=202,
    dependencies=[Depends(require_api_token)],
)
async def evaluate_defense_telemetry(
    payload: DefenseThreatEventRequest,
    request: Request,
) -> dict[str, Any]:
    _enforce_defense_rate_limit()
    _, orchestrator, _ = _defense_services(request)
    source_ip = request.client.host if request.client is not None else ""
    try:
        return await _run_cognitive_io(
            request,
            orchestrator.evaluate_event,
            {
                "target_asset": payload.target_asset,
                "category": payload.category,
                "severity": payload.severity,
                "signal_id": payload.signal_id,
                "indicator": payload.indicator,
                "evidence": payload.evidence,
                "false_positive": payload.false_positive,
                "source": source_ip,
            },
        )
    except (ThreatEvaluationError, AssetRegistryError) as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except DefenseActionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ForensicsError as exc:
        _LOG.exception("Defense telemetry could not be recorded.")
        raise HTTPException(
            status_code=503, detail="Incident ledger is unavailable; no mitigation was applied."
        ) from exc


@app.post("/api/defense/mitigate", dependencies=[Depends(require_api_token)])
async def manual_defense_mitigation(
    payload: DefenseMitigationRequest,
    request: Request,
    speaker_token: str | None = Header(
        default=None, alias="X-Friday-Speaker-Token"
    ),
) -> dict[str, Any]:
    _enforce_defense_rate_limit()
    await _run_cognitive_io(
        request,
        _require_defense_master,
        request,
        speaker_token,
        payload.master_passphrase,
    )
    _, orchestrator, _ = _defense_services(request)
    try:
        return await _run_cognitive_io(
            request,
            orchestrator.manual_mitigate,
            payload.target_asset,
            payload.action,
            payload.indicator,
            ttl_seconds=payload.ttl_seconds,
            master_authorized=True,
        )
    except (DefenseActionError, AssetRegistryError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ForensicsError as exc:
        _LOG.exception("Manual defense operation could not be recorded.")
        raise HTTPException(
            status_code=503, detail="Incident ledger is unavailable; no change was made."
        ) from exc


@app.post(
    "/api/defense/rollback/{action_id}",
    dependencies=[Depends(require_api_token)],
)
async def rollback_defense_action(
    action_id: str,
    request: Request,
    master_passphrase: str | None = Header(
        default=None, alias="X-Friday-Master-Passphrase"
    ),
    speaker_token: str | None = Header(
        default=None, alias="X-Friday-Speaker-Token"
    ),
) -> dict[str, Any]:
    _enforce_defense_rate_limit()
    await _run_cognitive_io(
        request,
        _require_defense_master,
        request,
        speaker_token,
        master_passphrase,
    )
    _, orchestrator, _ = _defense_services(request)
    try:
        return await _run_cognitive_io(request, orchestrator.rollback, action_id)
    except DefenseActionError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/twin/status", dependencies=[Depends(require_api_token)])
async def twin_status(request: Request) -> dict[str, Any]:
    twin = getattr(request.app.state, "digital_twin", None)
    if twin is None:
        raise HTTPException(status_code=503, detail="Digital-twin service is unavailable.")
    try:
        return twin.status()
    except (OSError, ValueError, TypeError) as exc:
        _LOG.exception("Digital-twin status could not be read.")
        raise HTTPException(status_code=503, detail="Digital-twin state is unavailable.") from exc


def _robotics_services(request: Request):
    twin = getattr(request.app.state, "digital_twin", None)
    collision_guard = getattr(request.app.state, "robotics_collision_guard", None)
    vehicles = getattr(request.app.state, "robotics_vehicles", None)
    specs = getattr(request.app.state, "robotics_specs", None)
    if twin is None or collision_guard is None or vehicles is None or specs is None:
        raise HTTPException(status_code=503, detail="Robotics services are unavailable.")
    return twin, collision_guard, vehicles, specs


def _require_robotics_master(
    request: Request,
    speaker_token: str | None,
    passphrase: str | None,
) -> None:
    _require_actual_master(request, speaker_token)
    _require_master(passphrase)
    _audit_master_event(
        "robotics_device_registration",
        "authorized",
        f"{request.method} {request.url.path}",
    )


@app.get("/api/robotics/status", dependencies=[Depends(require_api_token)])
async def robotics_status(request: Request) -> dict[str, Any]:
    twin, _, _, _ = _robotics_services(request)
    spatial_map = getattr(request.app.state, "robotics_map", None)
    waypoints = getattr(request.app.state, "robotics_waypoints", None)
    try:
        twin_status = await _run_cognitive_io(request, twin.status)
        return {
            "status": "ok",
            "digital_twin": twin_status,
            "map": spatial_map.status() if spatial_map is not None else None,
            "waypoints": waypoints.list() if waypoints is not None else [],
            "actuation": (
                "Only the existing confirmed iot_control action dispatches normal movement."
            ),
            "drone_protocol": "disabled",
        }
    except (OSError, ValueError, TypeError) as exc:
        _LOG.exception("Robotics status could not be read.")
        raise HTTPException(status_code=503, detail="Robotics status is unavailable.") from exc


@app.post(
    "/api/robotics/devices/register",
    dependencies=[Depends(require_api_token)],
)
async def register_robotics_device(
    payload: RoboticsDeviceRegistrationRequest,
    request: Request,
    speaker_token: str | None = Header(
        default=None, alias="X-Friday-Speaker-Token"
    ),
) -> dict[str, Any]:
    _enforce_simulation_rate_limit()
    await _run_cognitive_io(
        request,
        _require_robotics_master,
        request,
        speaker_token,
        payload.master_passphrase,
    )
    twin, _, vehicles, specs = _robotics_services(request)
    if payload.vehicle_type == "rc_car":
        if not RCCarVehicle.valid_device_id(payload.device_id):
            raise HTTPException(status_code=422, detail="RC car device_id is invalid.")
        vehicle = RCCarVehicle(payload.device_id)
    elif payload.vehicle_type == "drone":
        raise HTTPException(
            status_code=409,
            detail=(
                "Drone registration and actuation are disabled until a specific "
                "validated flight protocol is configured."
            ),
        )
    else:
        raise HTTPException(
            status_code=422,
            detail="vehicle_type must be rc_car or drone; drone actuation remains disabled.",
        )
    specification = dict(payload.specification)
    specification["vehicle_type"] = payload.vehicle_type
    try:
        await _run_cognitive_io(
            request,
            twin.register_device,
            payload.device_id,
            "robotics",
            specification,
            payload.telemetry_state,
            telemetry_at=datetime.now(timezone.utc).isoformat(),
            telemetry_source="Master-authenticated registration",
        )
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    vehicles[payload.device_id] = vehicle
    specs[payload.device_id] = specification
    return {
        "status": "registered",
        "device_id": payload.device_id,
        "vehicle_type": payload.vehicle_type,
        "protocol_enabled": payload.vehicle_type == "rc_car",
        "detail": (
            "Normal actuation remains available only through the existing confirmed "
            "iot_control path."
            if payload.vehicle_type == "rc_car"
            else "Drone protocol is disabled; no flight packets can be emitted."
        ),
    }


@app.post("/api/robotics/telemetry", dependencies=[Depends(require_api_token)])
async def ingest_robotics_telemetry(
    payload: RoboticsTelemetryIngestRequest,
    request: Request,
) -> dict[str, Any]:
    _enforce_simulation_rate_limit()
    twin, collision_guard, vehicles, specs = _robotics_services(request)
    vehicle = vehicles.get(payload.device_id)
    specification = specs.get(payload.device_id)
    if vehicle is None or specification is None:
        raise HTTPException(status_code=404, detail="Robotics device is not enrolled.")

    try:
        decision = collision_guard.observe_telemetry(
            payload.device_id,
            distance_cm=payload.distance_cm,
            current_a=payload.state.get("current_a"),
            maximum_current_a=specification.get("maximum_current_a"),
        )
    except CollisionGuardError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    state = dict(payload.state)
    state_distance = state.get("distance_cm")
    if (
        isinstance(state_distance, bool)
        or not isinstance(state_distance, (int, float))
        or float(state_distance) != float(payload.distance_cm)
    ):
        raise HTTPException(
            status_code=422,
            detail="state.distance_cm must exactly match the emergency-checked distance_cm.",
        )
    try:
        await _run_cognitive_io(
            request,
            twin.synchronize_telemetry,
            payload.device_id,
            state,
            telemetry_at=payload.telemetry_at,
            source_authenticated=True,
        )
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
        "status": "synchronized",
        "device_id": payload.device_id,
        "collision_guard": decision.as_dict(),
    }


@app.put("/api/robotics/map", dependencies=[Depends(require_api_token)])
async def configure_robotics_map(
    payload: RoboticsMapRequest,
    request: Request,
) -> dict[str, Any]:
    _enforce_simulation_rate_limit()
    try:
        spatial_map = SpatialMap(
            payload.width,
            payload.height,
            payload.depth,
            payload.obstacles,
        )
    except SpatialMapError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    request.app.state.robotics_map = spatial_map
    request.app.state.robotics_waypoints = WaypointRegistry(spatial_map)
    return {"status": "configured", "map": spatial_map.status()}


@app.get("/api/robotics/waypoints", dependencies=[Depends(require_api_token)])
async def list_robotics_waypoints(request: Request) -> dict[str, Any]:
    waypoints = getattr(request.app.state, "robotics_waypoints", None)
    if waypoints is None:
        return {"waypoints": []}
    return {"waypoints": waypoints.list()}


@app.post("/api/robotics/waypoints", dependencies=[Depends(require_api_token)])
async def add_robotics_waypoint(
    payload: RoboticsWaypointRequest,
    request: Request,
) -> dict[str, Any]:
    _enforce_simulation_rate_limit()
    waypoints = getattr(request.app.state, "robotics_waypoints", None)
    if waypoints is None:
        raise HTTPException(status_code=409, detail="Configure the spatial map first.")
    try:
        return waypoints.add(payload.name, payload.coordinate)
    except SpatialMapError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/robotics/plan", dependencies=[Depends(require_api_token)])
async def plan_robotics_path(
    payload: RoboticsPlanRequest,
    request: Request,
) -> dict[str, Any]:
    _enforce_simulation_rate_limit()
    spatial_map = getattr(request.app.state, "robotics_map", None)
    if spatial_map is None:
        raise HTTPException(status_code=409, detail="Configure the spatial map first.")
    try:
        path = spatial_map.find_path(payload.start, payload.goal)
    except SpatialMapError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"status": "planned", "path": path, "physical_action": "not_dispatched"}


@app.post("/api/robotics/preflight", dependencies=[Depends(require_api_token)])
async def preflight_robotics_action(
    payload: RoboticsPreflightRequest,
    request: Request,
) -> dict[str, Any]:
    _enforce_simulation_rate_limit()
    twin, collision_guard, vehicles, specs = _robotics_services(request)
    vehicle = vehicles.get(payload.device_id)
    specification = specs.get(payload.device_id)
    if vehicle is None or specification is None:
        raise HTTPException(status_code=404, detail="Robotics device is not enrolled.")
    if isinstance(vehicle, DroneVehicle):
        return {
            "status": "NOT_SUPPORTED",
            "dispatch_allowed": False,
            "detail": "Drone actuation is disabled because no protocol is configured.",
        }
    if payload.action.get("direction") == "STOP":
        return {
            "status": "STOP_NO_TWIN_REQUIRED",
            "dispatch_allowed": False,
            "detail": (
                "This endpoint never dispatches. Emergency STOP remains available "
                "through the existing iot_control action."
            ),
        }
    try:
        command = vehicle.build_action(payload.action)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    twin_status = await _run_cognitive_io(request, twin.status)
    twin_device = next(
        (
            item for item in twin_status["devices"]
            if item.get("device_id") == payload.device_id
        ),
        None,
    )
    if twin_device is None:
        raise HTTPException(status_code=404, detail="Digital-twin device is unavailable.")
    state = twin_device["state"]
    spatial_map = getattr(request.app.state, "robotics_map", None)
    try:
        return await collision_guard.preflight(
            twin,
            payload.device_id,
            command,
            distance_cm=state.get("distance_cm"),
            current_a=state.get("current_a"),
            maximum_current_a=specification.get("maximum_current_a"),
            spatial_map=spatial_map if payload.path is not None else None,
            path=payload.path,
        )
    except CollisionGuardError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post(
    "/api/twin/simulate",
    status_code=202,
    dependencies=[Depends(require_api_token)],
)
async def simulate_twin_command(
    payload: TwinSimulateRequest, request: Request
) -> dict[str, Any]:
    _enforce_simulation_rate_limit()
    try:
        encoded_command = json.dumps(
            payload.command, ensure_ascii=False, allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=422, detail="command must contain finite JSON values."
        ) from exc
    if len(encoded_command) > 32_768:
        raise HTTPException(status_code=413, detail="command exceeds the 32 KiB limit.")
    twin = getattr(request.app.state, "digital_twin", None)
    if twin is None:
        raise HTTPException(status_code=503, detail="Digital-twin service is unavailable.")
    try:
        return await twin.simulate_command(
            payload.device_id,
            payload.command,
            profile=payload.profile,
            seed=payload.seed,
        )
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/reload-modules", dependencies=[Depends(require_api_token)])
async def reload_modules(
    request: Request,
    master_passphrase: str | None = Header(
        default=None, alias="X-Friday-Master-Passphrase"
    ),
    speaker_token: str | None = Header(
        default=None, alias="X-Friday-Speaker-Token"
    ),
) -> dict:
    _require_master_speaker(request, speaker_token)
    await asyncio.to_thread(_require_master, master_passphrase)
    await asyncio.to_thread(
        _audit_master_event, "module_reload", "authorized",
        "Master-authorized module reload.",
    )
    try:
        registries = await asyncio.to_thread(_agent(request).reload_modules)
    except Exception as exc:
        _LOG.exception("Module reload failed.")
        raise HTTPException(status_code=500, detail=f"Module reload failed: {exc}") from exc
    await request.app.state.bridge.broadcast("modules_reloaded", registries)
    return registries


@app.post("/api/tool-maker/stage", dependencies=[Depends(require_api_token)])
async def stage_tool_proposal(
    payload: ToolProposalRequest,
    request: Request,
    speaker_token: str | None = Header(
        default=None, alias="X-Friday-Speaker-Token"
    ),
) -> dict[str, str]:
    _require_master_speaker(request, speaker_token)
    await asyncio.to_thread(_require_master, payload.master_passphrase)
    await asyncio.to_thread(
        _audit_master_event, "tool_proposal", "authorized",
        "Master-authorized tool proposal.",
    )
    try:
        path = await asyncio.to_thread(
            _agent(request).stage_tool_proposal, payload.capability
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        _LOG.exception("Tool proposal generation failed.")
        raise HTTPException(status_code=502, detail=f"Tool proposal failed: {exc}") from exc
    return {
        "status": "staged_for_manual_review",
        "path": path,
        "execution": "not_run",
        "activation": "not_loaded",
    }


def _require_master(passphrase: str | None) -> None:
    try:
        access_control.require_master_passphrase(passphrase)
    except access_control.MasterAuthError as exc:
        _audit_master_event(
            "master_authorization", "denied", "Master Passphrase missing or invalid."
        )
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except RuntimeError as exc:
        _LOG.exception("Master authorization configuration is unavailable.")
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def _require_master_speaker(request: Request, speaker_token: str | None) -> None:
    speaker = _agent(request)._verified_speaker(speaker_token)
    if speaker and speaker.get("enrolled_role", speaker.get("role")) == "MASTER":
        return
    _audit_master_event(
        "master_speaker_authorization", "denied",
        "A verified MASTER speaker token is required.",
    )
    raise HTTPException(
        status_code=403, detail="A verified MASTER speaker is required."
    )


def _audit_master_event(action: str, decision: str, detail: str) -> None:
    try:
        security.audit(
            "master_gate", action, security.TIER_DANGEROUS, decision, detail
        )
    except Exception as exc:
        _LOG.exception("Could not write Master authorization audit event.")
        raise HTTPException(
            status_code=503, detail="Security audit is unavailable; operation denied."
        ) from exc


def _verified_family_speaker(
    request: Request, speaker_token: str
) -> dict[str, str]:
    speaker = _agent(request)._verified_speaker(speaker_token)
    if not speaker:
        raise HTTPException(status_code=403, detail="A valid enrolled speaker token is required.")
    return speaker


def _ambient_profile_id(request: Request, speaker_token: str | None) -> str:
    speaker = _verified_family_speaker(request, speaker_token or "")
    if speaker.get("role") == "GUEST":
        raise HTTPException(
            status_code=403,
            detail="An enrolled family speaker is required for observer consent.",
        )
    return speaker["profile_id"]


async def _run_cognitive_io(
    request: Request, function: Any, *args: Any, **kwargs: Any
) -> Any:
    semaphore = getattr(request.app.state, "cognitive_io_slots", None)
    if semaphore is None:
        raise HTTPException(status_code=503, detail="Cognitive service is unavailable.")
    try:
        await asyncio.wait_for(semaphore.acquire(), timeout=0.05)
    except asyncio.TimeoutError as exc:
        raise HTTPException(
            status_code=429,
            detail="Cognitive service is busy; retry shortly.",
        ) from exc

    task = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
    release_here = True
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        release_here = False

        def release_after_completion(future: asyncio.Task) -> None:
            try:
                future.exception()
            except asyncio.CancelledError:
                pass
            semaphore.release()

        task.add_done_callback(release_after_completion)
        raise
    finally:
        if release_here:
            semaphore.release()


def _require_actual_master(request: Request, speaker_token: str | None) -> dict[str, str]:
    speaker = _verified_family_speaker(request, speaker_token or "")
    if speaker.get("enrolled_role") != "MASTER":
        _audit_master_event(
            "succession_master_check", "denied",
            "An enrolled root MASTER speaker is required.",
        )
        raise HTTPException(status_code=403, detail="An enrolled Master speaker is required.")
    return speaker


@app.post("/api/succession/heartbeat", dependencies=[Depends(require_api_token)])
async def succession_heartbeat(
    payload: SuccessionHeartbeatRequest, request: Request
) -> dict[str, str]:
    speaker = _verified_family_speaker(request, payload.speaker_token)
    role = speaker.get("enrolled_role", speaker["role"])
    if role == "MASTER":
        await asyncio.to_thread(_require_master, payload.master_passphrase)
    try:
        await asyncio.to_thread(
            request.app.state.succession.heartbeat,
            role,
            speaker["profile_id"],
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        _LOG.exception("Could not persist succession heartbeat.")
        raise HTTPException(status_code=503, detail="Succession state is unavailable.") from exc
    return {"status": "recorded", "role": role}


@app.post("/api/succession/enroll", dependencies=[Depends(require_api_token)])
async def succession_enroll_emergency_credential(
    payload: SuccessionEnrollRequest,
    request: Request,
    master_speaker_token: str | None = Header(
        default=None, alias="X-Friday-Speaker-Token"
    ),
) -> dict[str, str]:
    _require_actual_master(request, master_speaker_token)
    await asyncio.to_thread(_require_master, payload.master_passphrase)
    candidate = _verified_family_speaker(request, payload.target_speaker_token)
    role = candidate.get("enrolled_role", candidate["role"])
    if role not in {"BROTHER", "SISTER"}:
        raise HTTPException(
            status_code=422, detail="Emergency credentials are only for Brother or Sister."
        )
    try:
        await asyncio.to_thread(
            request.app.state.succession.enroll_emergency_credential,
            role,
            candidate["profile_id"],
            payload.emergency_passphrase,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        _LOG.exception("Could not persist emergency credential.")
        raise HTTPException(status_code=503, detail="Succession state is unavailable.") from exc
    await asyncio.to_thread(
        _audit_master_event, "succession_credential", "configured",
        f"role={role}",
    )
    return {"status": "configured", "role": role}


@app.post("/api/succession/claim", dependencies=[Depends(require_api_token)])
async def succession_claim(
    payload: SuccessionClaimRequest, request: Request
) -> dict[str, str]:
    claimant = _verified_family_speaker(request, payload.speaker_token)
    role = claimant.get("enrolled_role", claimant["role"])
    if role not in {"BROTHER", "SISTER"}:
        raise HTTPException(status_code=403, detail="Only Brother or Sister may claim succession.")
    await asyncio.to_thread(_require_master, payload.master_passphrase)
    master_profile_id = None
    if payload.manual_emergency_trigger:
        master = _require_actual_master(request, payload.master_speaker_token)
        master_profile_id = master["profile_id"]
    try:
        result = await asyncio.to_thread(
            request.app.state.succession.claim,
            role,
            claimant["profile_id"],
            payload.emergency_passphrase,
            manual_emergency=payload.manual_emergency_trigger,
            master_profile_id=master_profile_id,
        )
    except PermissionError as exc:
        await asyncio.to_thread(
            _audit_master_event, "succession_claim", "denied", f"role={role}"
        )
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        _LOG.exception("Succession claim could not be persisted.")
        raise HTTPException(status_code=503, detail="Succession state is unavailable.") from exc
    return result


@app.post(
    "/api/succession/guardian-approve",
    dependencies=[Depends(require_api_token)],
)
async def succession_guardian_approve(
    payload: SuccessionGuardianApprovalRequest, request: Request
) -> dict[str, str]:
    guardian = _verified_family_speaker(request, payload.guardian_speaker_token)
    role = guardian.get("enrolled_role", guardian["role"])
    if role not in {"FATHER", "MOTHER"}:
        raise HTTPException(
            status_code=403, detail="Approval requires an enrolled Father or Mother."
        )
    await asyncio.to_thread(_require_master, payload.master_passphrase)
    try:
        result = await asyncio.to_thread(
            request.app.state.succession.approve_claim,
            payload.claim_id,
            role,
            guardian["profile_id"],
        )
    except PermissionError as exc:
        await asyncio.to_thread(
            _audit_master_event, "succession_guardian_approval", "denied",
            f"guardian={role}",
        )
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except RuntimeError as exc:
        _LOG.exception("Guardian approval could not be persisted.")
        raise HTTPException(status_code=503, detail="Succession state is unavailable.") from exc
    return result


@app.post("/api/succession/reclaim", dependencies=[Depends(require_api_token)])
async def succession_reclaim(
    payload: SuccessionReclaimRequest, request: Request
) -> dict[str, str]:
    master = _require_actual_master(request, payload.speaker_token)
    await asyncio.to_thread(_require_master, payload.master_passphrase)
    try:
        await asyncio.to_thread(
            request.app.state.succession.reclaim, master["profile_id"]
        )
    except (ValueError, RuntimeError) as exc:
        _LOG.exception("Master reclaim could not be persisted.")
        raise HTTPException(status_code=503, detail="Master reclaim failed.") from exc
    return {"status": "ACTIVE", "succession_revoked": "true"}


@app.get("/api/succession/status", dependencies=[Depends(require_api_token)])
async def succession_status(
    request: Request,
    master_passphrase: str | None = Header(
        default=None, alias="X-Friday-Master-Passphrase"
    ),
    speaker_token: str | None = Header(
        default=None, alias="X-Friday-Speaker-Token"
    ),
) -> dict:
    _require_actual_master(request, speaker_token)
    await asyncio.to_thread(_require_master, master_passphrase)
    try:
        return await asyncio.to_thread(request.app.state.succession.status)
    except RuntimeError as exc:
        _LOG.exception("Succession status is unreadable.")
        raise HTTPException(status_code=503, detail="Succession state is unavailable.") from exc


@app.post("/api/master/setup", dependencies=[Depends(require_api_token)])
async def setup_master_passphrase(payload: MasterPassphraseRequest) -> dict[str, str]:
    await asyncio.to_thread(
        _audit_master_event, "master_passphrase_setup", "attempted",
        "Initial Master Passphrase setup requested.",
    )
    try:
        await asyncio.to_thread(
            access_control.configure_master_passphrase, payload.passphrase
        )
    except access_control.MasterAuthError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except OSError as exc:
        _LOG.exception("Could not persist Master Passphrase configuration.")
        raise HTTPException(status_code=500, detail="Could not save Master Passphrase.") from exc
    await asyncio.to_thread(
        _audit_master_event, "master_passphrase_setup", "configured",
        "Initial Master Passphrase configured.",
    )
    return {"status": "configured"}


@app.post("/api/master/verify", dependencies=[Depends(require_api_token)])
async def verify_master_passphrase(payload: MasterPassphraseRequest) -> dict[str, Any]:
    try:
        verified = await asyncio.to_thread(
            access_control.verify_master_passphrase, payload.passphrase
        )
    except RuntimeError as exc:
        _LOG.exception("Master authorization configuration is unavailable.")
        raise HTTPException(status_code=503, detail="Master authorization is unavailable.") from exc
    if not verified:
        await asyncio.to_thread(
            _audit_master_event, "master_passphrase_verify", "denied",
            "Master Passphrase verification failed.",
        )
        raise HTTPException(status_code=403, detail="Master Passphrase verification failed.")
    await asyncio.to_thread(
        _audit_master_event, "master_passphrase_verify", "verified",
        "Master Passphrase challenge succeeded.",
    )

    # ── Verify වූ සැනින් Master Speaker Token නිකුත් කිරීම (Frontend එක Master බවට පත්වේ) ──
    master_token = access_control.issue_speaker_token(
        "master_owner",
        "MASTER",
        access_control.load_speaker_signing_secret(),
    )
    return {
        "verified": True,
        "role": "MASTER",
        "name": "Sir",
        "speaker_token": master_token,
    }


@app.post("/api/voice/enroll", dependencies=[Depends(require_api_token)])
async def enroll_voice_profile(
    request: Request,
    name: str = Form(..., min_length=1, max_length=80),
    role: str = Form(..., min_length=1, max_length=16),
    master_passphrase: str = Form(..., min_length=1, max_length=1024),
    speaker_token: str | None = Form(default=None, max_length=2048),
    samples: list[UploadFile] = File(...),
) -> dict[str, str]:
    requested_role = role.upper()
    try:
        existing_profiles = await asyncio.to_thread(
            _agent(request).voice_profiles.list_metadata
        )
    except (RuntimeError, ImportError) as exc:
        _LOG.exception("Could not read voice profiles before enrollment.")
        raise HTTPException(status_code=503, detail="Voice profile store is unavailable.") from exc
    has_master_profile = any(
        profile["role"] == "MASTER" for profile in existing_profiles
    )
    if requested_role != "MASTER" or has_master_profile:
        _require_master_speaker(request, speaker_token)
    await asyncio.to_thread(_require_master, master_passphrase)
    await asyncio.to_thread(
        _audit_master_event, "voice_profile_enroll", "authorized",
        f"role={role.upper()}",
    )
    if not 2 <= len(samples) <= 8:
        raise HTTPException(status_code=422, detail="Enroll with 2 to 8 separate WAV samples.")
    payloads: list[bytes] = []
    for sample in samples:
        payload = await sample.read(10 * 1024 * 1024 + 1)
        if len(payload) > 10 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="A voice sample exceeds the 10 MB limit.")
        payloads.append(payload)
    try:
        result = await asyncio.to_thread(
            _agent(request).voice_profiles.enroll, name, role, payloads
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (RuntimeError, ImportError) as exc:
        _LOG.exception("Voice profile enrollment is unavailable.")
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    await asyncio.to_thread(
        _audit_master_event, "voice_profile_enroll", "completed",
        f"role={role.upper()}",
    )
    return result


@app.get("/api/voice/profiles", dependencies=[Depends(require_api_token)])
async def list_voice_profiles(
    request: Request,
    master_passphrase: str | None = Header(
        default=None, alias="X-Friday-Master-Passphrase"
    ),
    speaker_token: str | None = Header(
        default=None, alias="X-Friday-Speaker-Token"
    ),
) -> dict[str, list[dict[str, str]]]:
    _require_master_speaker(request, speaker_token)
    await asyncio.to_thread(_require_master, master_passphrase)
    await asyncio.to_thread(
        _audit_master_event, "voice_profile_list", "authorized",
        "Voice profile metadata requested.",
    )
    try:
        profiles = await asyncio.to_thread(
            _agent(request).voice_profiles.list_metadata
        )
    except (RuntimeError, ImportError) as exc:
        _LOG.exception("Voice profile metadata is unavailable.")
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"profiles": profiles}


@app.delete(
    "/api/voice/profiles/{profile_id}",
    dependencies=[Depends(require_api_token)],
)
async def delete_voice_profile(
    profile_id: str,
    request: Request,
    master_passphrase: str | None = Header(
        default=None, alias="X-Friday-Master-Passphrase"
    ),
    speaker_token: str | None = Header(
        default=None, alias="X-Friday-Speaker-Token"
    ),
) -> dict[str, str]:
    _require_master_speaker(request, speaker_token)
    await asyncio.to_thread(_require_master, master_passphrase)
    await asyncio.to_thread(
        _audit_master_event, "voice_profile_delete", "authorized",
        f"profile_id={profile_id[:32]}",
    )
    try:
        deleted = await asyncio.to_thread(
            _agent(request).voice_profiles.delete, profile_id
        )
    except (RuntimeError, ImportError) as exc:
        _LOG.exception("Voice profile deletion is unavailable.")
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if not deleted:
        raise HTTPException(status_code=404, detail="Voice profile not found.")
    await asyncio.to_thread(
        _audit_master_event, "voice_profile_delete", "completed",
        f"profile_id={profile_id[:32]}",
    )
    return {"status": "deleted"}


@app.post("/api/voice/verify", dependencies=[Depends(require_api_token)])
async def verify_voice_profile(
    request: Request, file: UploadFile = File(...)
) -> dict[str, str | bool]:
    if not (file.content_type or "").startswith("audio/"):
        raise HTTPException(status_code=415, detail="Upload a WAV audio sample.")
    payload = await file.read(10 * 1024 * 1024 + 1)
    if len(payload) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Voice sample exceeds the 10 MB limit.")
    try:
        from app.voice import decode_to_pcm_wav

        normalized = await asyncio.to_thread(decode_to_pcm_wav, payload)
        speaker = await asyncio.to_thread(
            _agent(request).voice_profiles.identify, normalized
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (RuntimeError, ImportError, OSError) as exc:
        _LOG.exception("Voice verification is unavailable.")
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if speaker["role"] == "GUEST":
        return {"verified": False, "name": "Guest", "role": "GUEST"}
    token = access_control.issue_speaker_token(
        speaker["profile_id"], speaker["role"],
        access_control.load_speaker_signing_secret(),
    )
    return {
        "verified": True,
        "name": speaker["name"],
        "role": speaker["role"],
        "speaker_token": token,
    }


@app.post("/api/confirm", dependencies=[Depends(require_api_token)])
async def resolve_confirmation(payload: ConfirmationRequest) -> dict[str, str]:
    if not confirm_gate.pending_title():
        raise HTTPException(status_code=409, detail="No confirmation is currently pending.")
    confirm_gate.resolve(payload.accepted, source="authenticated API")
    return {"status": "accepted" if payload.accepted else "cancelled"}


@app.get("/api/confirmation", dependencies=[Depends(require_api_token)])
async def pending_confirmation(request: Request) -> dict:
    title = confirm_gate.pending_title()
    bridge: WebPlayerBridge = request.app.state.bridge
    pending = bridge.pending_confirmation
    if not title or not pending or pending["title"] != title:
        if pending:
            bridge.hide_confirmation()
        return {"pending": False}
    return {"pending": True, **pending}


async def _authenticate_websocket(websocket: WebSocket) -> bool:
    header = websocket.headers.get("authorization")
    token = websocket.query_params.get("token")
    authenticated = _token_matches(header) or _token_matches(token)
    manager = getattr(websocket.app.state, "succession", None)
    return authenticated and not (manager and manager.is_locked_down())


@app.websocket("/ws/telemetry")
async def telemetry_socket(websocket: WebSocket) -> None:
    if not await _authenticate_websocket(websocket):
        await websocket.close(code=4401, reason="Authentication required")
        return
    bridge: WebPlayerBridge = websocket.app.state.bridge
    await bridge.connect(websocket)
    title = confirm_gate.pending_title()
    pending = bridge.pending_confirmation
    if title and pending and pending["title"] == title:
        await bridge.replay_confirmation(websocket)
    elif pending:
        bridge.hide_confirmation()
    try:
        while True:
            message = await websocket.receive_text()
            if message == "ping":
                await websocket.send_text('{"type":"pong"}')
    except WebSocketDisconnect:
        bridge.disconnect(websocket)


@app.websocket("/ws/audio")
async def audio_socket(websocket: WebSocket) -> None:
    if not await _authenticate_websocket(websocket):
        await websocket.close(code=4401, reason="Authentication required")
        return
    await websocket.accept()
    audio = bytearray()
    mime_type = "audio/ogg"
    session_id = "default"
    master_passphrase: str | None = None
    speaker_token: str | None = None
    try:
        while True:
            message = await websocket.receive()
            if message.get("bytes") is not None:
                audio.extend(message["bytes"])
                if len(audio) > MAX_AUDIO_BYTES:
                    await websocket.send_json({"type": "error", "detail": "Audio exceeds the 25 MB limit."})
                    await websocket.close(code=1009, reason="Audio too large")
                    return
                continue
            text = message.get("text")
            if not text:
                continue
            try:
                command = json.loads(text)
            except json.JSONDecodeError:
                await websocket.send_json({"type": "error", "detail": "Expected a JSON control message."})
                continue
            kind = command.get("type")
            if kind == "audio_start":
                audio.clear()
                mime_type = str(command.get("mime_type") or "audio/ogg")
                session_id = str(command.get("session_id") or "default")[:128]
                master_passphrase = str(command.get("master_passphrase") or "")[:1024] or None
                speaker_token = str(command.get("speaker_token") or "")[:2048] or None
                if not mime_type.startswith("audio/"):
                    await websocket.send_json({"type": "error", "detail": "Audio MIME type required."})
                    continue
                await websocket.send_json({"type": "audio_ready"})
            elif kind == "audio_chunk":
                try:
                    chunk = base64.b64decode(command.get("data", ""), validate=True)
                except (ValueError, TypeError) as exc:
                    await websocket.send_json({"type": "error", "detail": f"Invalid audio chunk: {exc}"})
                    continue
                audio.extend(chunk)
                if len(audio) > MAX_AUDIO_BYTES:
                    await websocket.send_json({"type": "error", "detail": "Audio exceeds the 25 MB limit."})
                    audio.clear()
            elif kind == "audio_end":
                try:
                    result = await websocket.app.state.agent.chat_audio(
                        bytes(audio), mime_type, session_id, master_passphrase,
                        speaker_token,
                    )
                except Exception as exc:
                    _LOG.exception("WebSocket voice interaction failed.")
                    await websocket.send_json({"type": "error", "detail": str(exc)})
                else:
                    await websocket.send_json({
                        "type": "voice_response",
                        "transcript": result["transcript"],
                        "response": result["response"],
                        "audio_base64": base64.b64encode(result["audio"]).decode("ascii"),
                        "audio_mime_type": result["audio_mime_type"],
                        "session_id": session_id,
                        "speaker_name": result.get("speaker_name", "Guest"),
                        "speaker_role": result.get("speaker_role", "GUEST"),
                    })
                finally:
                    audio.clear()
            else:
                await websocket.send_json({"type": "error", "detail": "Unsupported audio control message."})
    except WebSocketDisconnect:
        return


@app.websocket("/ws/live")
async def live_audio_socket(websocket: WebSocket) -> None:
    if not await _authenticate_websocket(websocket):
        await websocket.close(code=4401, reason="Authentication required")
        return
    await websocket.accept()
    try:
        first_message = await websocket.receive()
        if first_message.get("type") == "websocket.disconnect":
            return
        raw_handshake = first_message.get("text")
        if not isinstance(raw_handshake, str):
            await websocket.close(code=4400, reason="JSON audio handshake required")
            return
        if len(raw_handshake) > 4096:
            await websocket.close(code=4400, reason="Audio handshake is too large")
            return
        try:
            handshake = json.loads(raw_handshake)
        except json.JSONDecodeError:
            await websocket.close(code=4400, reason="Invalid JSON audio handshake")
            return
        channels = handshake.get("channels", 1) if isinstance(handshake, dict) else None
        audio_format = (
            handshake.get("format", "s16le")
            if isinstance(handshake, dict)
            else None
        )
        if (
            not isinstance(handshake, dict)
            or not isinstance(handshake.get("client"), str)
            or handshake["client"] not in {"web", "esp32_satellite"}
            or type(handshake.get("sample_rate")) is not int
            or handshake["sample_rate"] != 16_000
            or type(channels) is not int
            or channels != 1
            or not isinstance(audio_format, str)
            or audio_format != "s16le"
        ):
            await websocket.close(
                code=4400,
                reason="Expected mono 16 kHz signed 16-bit PCM handshake",
            )
            return

        agent = websocket.app.state.agent
        is_satellite = handshake["client"] == "esp32_satellite"
        presence_engine = getattr(
            websocket.app.state, "presence_engine", None
        )
        presence_snapshot = (
            presence_engine.get_presence_state()
            if is_satellite and presence_engine is not None
            else None
        )
        last_presence_state = (
            presence_snapshot["state"] if presence_snapshot is not None else None
        )
        speaker_token = handshake.get("speaker_token")
        master_passphrase = handshake.get("master_passphrase")
        if (
            speaker_token is not None
            and (not isinstance(speaker_token, str) or len(speaker_token) > 2048)
        ) or (
            master_passphrase is not None
            and (
                not isinstance(master_passphrase, str)
                or len(master_passphrase) > 1024
            )
        ):
            await websocket.close(code=4400, reason="Invalid authentication context")
            return

        async with agent.live_session(
            speaker_token=speaker_token,
            master_passphrase=master_passphrase,
        ) as (session, speaker):
            await websocket.send_json({
                "type": "ready",
                "client": handshake["client"],
                "input_audio": "audio/pcm;rate=16000",
                "output_audio": "audio/pcm;rate=24000",
                "speaker_role": speaker.get("role", "GUEST") if speaker else "GUEST",
                "presence_state": (
                    presence_snapshot["state"]
                    if presence_snapshot is not None
                    else None
                ),
                "mic_active": (
                    presence_snapshot["mic_active"]
                    if presence_snapshot is not None
                    else True
                ),
            })
            total_audio_bytes = 0
            tool_calls_in_turn = 0

            async def receive_live_responses() -> None:
                nonlocal tool_calls_in_turn
                async for live_response in session.receive():
                    content = getattr(live_response, "server_content", None)
                    if content is not None:
                        input_transcription = getattr(
                            content, "input_transcription", None
                        )
                        if getattr(input_transcription, "text", None):
                            await websocket.send_json({
                                "type": "input_transcription",
                                "text": input_transcription.text,
                            })
                        output_transcription = getattr(
                            content, "output_transcription", None
                        )
                        if getattr(output_transcription, "text", None):
                            await websocket.send_json({
                                "type": "output_transcription",
                                "text": output_transcription.text,
                            })
                        model_turn = getattr(content, "model_turn", None)
                        for part in getattr(model_turn, "parts", None) or []:
                            inline_data = getattr(part, "inline_data", None)
                            audio_data = getattr(inline_data, "data", None)
                            if audio_data:
                                await websocket.send_json({
                                    "type": "audio_chunk",
                                    "mime_type": "audio/pcm;rate=24000",
                                })
                                await websocket.send_bytes(bytes(audio_data))
                            if getattr(part, "text", None):
                                await websocket.send_json({
                                    "type": "text",
                                    "text": part.text,
                                })
                        if getattr(content, "interrupted", False):
                            await websocket.send_json({"type": "interrupted"})
                        if getattr(content, "turn_complete", False):
                            tool_calls_in_turn = 0
                            await websocket.send_json({"type": "turn_complete"})

                    tool_call = getattr(live_response, "tool_call", None)
                    function_calls = getattr(tool_call, "function_calls", None) or []
                    if function_calls:
                        responses = []
                        for function_call in function_calls:
                            name = str(getattr(function_call, "name", "") or "")
                            call_id = getattr(function_call, "id", None)
                            if tool_calls_in_turn >= 8:
                                result = (
                                    "Tool call limit reached for this turn; "
                                    "no additional tool was executed."
                                )
                            else:
                                arguments = getattr(function_call, "args", None)
                                if not isinstance(arguments, dict):
                                    result = "Tool arguments must be a JSON object."
                                else:
                                    tool_calls_in_turn += 1
                                    try:
                                        result = await agent.dispatch_live_tool(
                                            name,
                                            arguments,
                                            speaker_token=speaker_token,
                                            master_passphrase=master_passphrase,
                                        )
                                    except Exception as exc:
                                        _LOG.exception(
                                            "Live voice tool execution failed: %s",
                                            name,
                                        )
                                        result = (
                                            "Tool execution failed safely: "
                                            f"{type(exc).__name__}."
                                        )
                            responses.append(types.FunctionResponse(
                                name=name,
                                id=call_id,
                                response={"result": result},
                            ))
                        await session.send_tool_response(
                            function_responses=responses
                        )

            response_task = asyncio.create_task(receive_live_responses())
            input_task: asyncio.Task | None = None
            try:
                while True:
                    if input_task is None:
                        input_task = asyncio.create_task(websocket.receive())
                    done, _ = await asyncio.wait(
                        (input_task, response_task),
                        timeout=1.0,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    if not done:
                        if is_satellite and presence_engine is not None:
                            presence_snapshot = (
                                presence_engine.get_presence_state()
                            )
                            if presence_snapshot["state"] != last_presence_state:
                                last_presence_state = presence_snapshot["state"]
                                await websocket.send_json({
                                    "type": "presence_state",
                                    **presence_snapshot,
                                })
                        continue
                    if response_task in done:
                        input_task.cancel()
                        await asyncio.gather(input_task, return_exceptions=True)
                        input_task = None
                        await response_task
                        break
                    incoming = input_task.result()
                    input_task = None
                    if incoming.get("type") == "websocket.disconnect":
                        break
                    audio_chunk = incoming.get("bytes")
                    if audio_chunk is not None:
                        if (
                            is_satellite
                            and presence_engine is not None
                            and not presence_engine.is_mic_active()
                        ):
                            presence_snapshot = (
                                presence_engine.get_presence_state()
                            )
                            last_presence_state = presence_snapshot["state"]
                            await websocket.send_json({
                                "type": "audio_discarded",
                                "reason": "satellite_microphone_muted",
                                "presence_state": last_presence_state,
                            })
                            continue
                        if not audio_chunk or len(audio_chunk) % 2:
                            await websocket.send_json({
                                "type": "error",
                                "detail": "PCM chunks must contain complete 16-bit samples.",
                            })
                            continue
                        if len(audio_chunk) > 64 * 1024:
                            await websocket.close(
                                code=1009, reason="PCM frame exceeds 64 KiB"
                            )
                            break
                        total_audio_bytes += len(audio_chunk)
                        if total_audio_bytes > MAX_AUDIO_BYTES:
                            await websocket.close(
                                code=1009, reason="Live audio limit exceeded"
                            )
                            break
                        await session.send_realtime_input(
                            audio=types.Blob(
                                data=audio_chunk,
                                mime_type="audio/pcm;rate=16000",
                            )
                        )
                        continue

                    text = incoming.get("text")
                    if not text:
                        continue
                    if len(text) > 20_000:
                        await websocket.send_json({
                            "type": "error",
                            "detail": "Control message exceeds the size limit.",
                        })
                        continue
                    try:
                        command = json.loads(text)
                    except json.JSONDecodeError:
                        await websocket.send_json({
                            "type": "error",
                            "detail": "Expected a JSON control message.",
                        })
                        continue
                    if not isinstance(command, dict):
                        await websocket.send_json({
                            "type": "error",
                            "detail": "Control message must be a JSON object.",
                        })
                    elif command.get("type") == "telemetry" and is_satellite:
                        if presence_engine is None:
                            await websocket.send_json({
                                "type": "error",
                                "detail": "Presence engine is unavailable.",
                            })
                            continue
                        prior = presence_engine.get_presence_state()
                        try:
                            presence_snapshot = (
                                presence_engine.ingest_sensor_telemetry(
                                    command.get("pir_active", prior["pir_active"]),
                                    command.get("person_count", prior["person_count"]),
                                    command.get("light_lux", prior["light_lux"]),
                                )
                            )
                        except ValueError as exc:
                            await websocket.send_json({
                                "type": "error",
                                "detail": str(exc),
                            })
                            continue
                        last_presence_state = presence_snapshot["state"]
                        await websocket.send_json({
                            "type": "telemetry_ack",
                            **presence_snapshot,
                        })
                    elif command.get("type") == "wake_word" and is_satellite:
                        if presence_engine is None:
                            await websocket.send_json({
                                "type": "error",
                                "detail": "Presence engine is unavailable.",
                            })
                            continue
                        try:
                            presence_snapshot = presence_engine.trigger_wake_word(
                                command.get("source", "esp32_satellite")
                            )
                        except ValueError as exc:
                            await websocket.send_json({
                                "type": "error",
                                "detail": str(exc),
                            })
                            continue
                        last_presence_state = presence_snapshot["state"]
                        await websocket.send_json({
                            "type": "presence_state",
                            **presence_snapshot,
                        })
                    elif command.get("type") == "audio_end":
                        if (
                            is_satellite
                            and presence_engine is not None
                            and not presence_engine.is_mic_active()
                        ):
                            total_audio_bytes = 0
                            await websocket.send_json({
                                "type": "audio_discarded",
                                "reason": "satellite_microphone_muted",
                                "presence_state": (
                                    presence_engine.get_presence_state()["state"]
                                ),
                            })
                            continue
                        await session.send_realtime_input(audio_stream_end=True)
                        total_audio_bytes = 0
                    elif command.get("type") == "text":
                        message_text = command.get("text")
                        if not isinstance(message_text, str) or not message_text.strip():
                            await websocket.send_json({
                                "type": "error",
                                "detail": "Text input must not be empty.",
                            })
                        elif len(message_text) > 16_000:
                            await websocket.send_json({
                                "type": "error",
                                "detail": "Text input exceeds the 16,000 character limit.",
                            })
                        else:
                            await session.send_realtime_input(text=message_text)
                    elif command.get("type") == "ping":
                        await websocket.send_json({"type": "pong"})
                    else:
                        await websocket.send_json({
                            "type": "error",
                            "detail": "Unsupported live control message.",
                        })
            finally:
                if input_task is not None and not input_task.done():
                    input_task.cancel()
                response_task.cancel()
                pending_tasks = [response_task]
                if input_task is not None:
                    pending_tasks.append(input_task)
                await asyncio.gather(*pending_tasks, return_exceptions=True)
    except WebSocketDisconnect:
        return
    except Exception:
        _LOG.exception("Live WebSocket session failed.")
        try:
            await websocket.send_json({
                "type": "error",
                "detail": "Live voice session ended unexpectedly.",
            })
            await websocket.close(code=1011, reason="Live session failed")
        except Exception:
            _LOG.debug("Live WebSocket was already closed during error handling.")


def run() -> None:
    import uvicorn

    hosted_port = os.environ.get("PORT")
    port = int(hosted_port or os.environ.get("FRIDAY_PORT", "8000"))
    host = (
        "0.0.0.0"
        if hosted_port
        else os.environ.get("FRIDAY_HOST", "127.0.0.1")
    )
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=False,
    )


if __name__ == "__main__":
    run()