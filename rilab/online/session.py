"""Single-owner online state machine with concurrent receive and cancellable inference."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import Enum
import json
import math
import time

from rilab.decision import AdapterError
from rilab.mapping.action_mapping import resolve_selection


class SessionState(str, Enum):
    Connecting = "Connecting"
    Playing = "Playing"
    Finishing = "Finishing"
    Closed = "Closed"
    Disconnected = "Disconnected"


@dataclass(frozen=True)
class Budget:
    target: float
    deadline: float

    @classmethod
    def read(cls, fields, received, *, inbound, reserve):
        values = [fields[k] for k in ("deadline_ms", "grace_ms", "bank_ms")]
        if any(
            type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in values
        ):
            raise AdapterError("invalid online time budget")
        deadline, grace, bank = (v / 1000 for v in values)
        # Take the stricter limit if a server advertises inconsistent fields.
        hard = received + min(deadline, grace + bank) - inbound - reserve
        return cls(min(hard, received + grace - inbound - reserve), hard)


@dataclass
class Request:
    request_id: int
    budget: Budget
    prepared: object = None
    sent_index: int | None = None
    sent: bool = False
    ended: bool = False
    inference: object = None
    timer: object = None
    acknowledged: bool = False
    received: float = 0.0


class OnlineSession:
    def __init__(
        self,
        policy,
        backend,
        *,
        validation=True,
        send_reserve_seconds=0.1,
        inbound_estimate_seconds=0.1,
        finish_timeout_seconds=5.0,
        queue_limit=256,
        log_limit=100000,
        log_byte_limit=64 * 1024 * 1024,
        secrets=(),
        start_timeout_seconds=None,
    ):
        for value in (
            send_reserve_seconds,
            inbound_estimate_seconds,
            finish_timeout_seconds,
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError("invalid session timing configuration")
        if (
            finish_timeout_seconds == 0
            or queue_limit <= 0
            or log_limit <= 0
            or log_byte_limit <= 0
        ):
            raise ValueError("session limits must be positive")
        self.policy, self.backend = policy, backend
        self.validation = validation
        self.reserve, self.inbound = send_reserve_seconds, inbound_estimate_seconds
        self.finish_timeout = finish_timeout_seconds
        self.queue = asyncio.Queue(maxsize=queue_limit)
        self.log_limit, self.secrets = log_limit, tuple(s for s in secrets if s)
        self.log_bytes, self.log_byte_limit = 0, log_byte_limit
        self.state = SessionState.Connecting
        self.requests = {}
        self.tasks = set()
        self.records = []
        self.faults = []
        self.validation_result = None
        self.completed = False
        self.end_seen = False
        self.finish_deadline = None
        if start_timeout_seconds is not None and (
            not math.isfinite(start_timeout_seconds) or start_timeout_seconds <= 0
        ):
            raise ValueError("start timeout must be positive")
        self.start_timeout = start_timeout_seconds

    def redact(self, value):
        if isinstance(value, dict):
            return {
                k: (
                    "[REDACTED]"
                    if k.lower() in {"token", "api_key", "authorization", "x-api-key"}
                    else self.redact(v)
                )
                for k, v in value.items()
            }
        if isinstance(value, list):
            return [self.redact(v) for v in value]
        if isinstance(value, str):
            for secret in self.secrets:
                value = value.replace(secret, "[REDACTED]")
        return value

    def record(self, kind, **fields):
        record = self.redact({"kind": kind, **fields})
        size = len(json.dumps(record, ensure_ascii=False).encode("utf-8"))
        if (
            len(self.records) >= self.log_limit
            or self.log_bytes + size > self.log_byte_limit
        ):
            raise AdapterError("online log capacity exceeded")
        self.log_bytes += size
        self.records.append(record)

    def fault(self, reason):
        reason = self.redact(reason)
        self.faults.append(reason)
        self.policy.clear()
        try:
            self.record("fault", reason=reason)
        except AdapterError:
            pass  # Keep the summary even if the bounded trace is already full.

    def spawn(self, coroutine):
        task = asyncio.create_task(coroutine)
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
        return task

    def end_request(self, request):
        request.ended = True
        for task in (request.inference, request.timer):
            if task and not task.done():
                task.cancel()

    async def receive(self, ws):
        disconnect_reason = None
        try:
            async for raw in ws:
                received = time.perf_counter()
                if not isinstance(raw, str):
                    continue  # Protocol ignores binary frames.
                try:
                    message = json.loads(raw)
                    if not isinstance(message, dict) or not isinstance(
                        message.get("type"), str
                    ):
                        raise ValueError("invalid envelope")
                except (ValueError, TypeError):
                    self.queue.put_nowait(("broken", "invalid_message", received))
                    return
                self.queue.put_nowait(("message", message, received))
        except asyncio.QueueFull:
            # Receive must not wait behind slow processing; fail closed on overload.
            self.state = SessionState.Disconnected
            self.receive_error = "receive_queue_capacity_exceeded"
            return
        except Exception as exc:
            disconnect_reason = type(exc).__name__
        finally:
            # Queue may be full. Owner also observes receiver completion.
            if not self.queue.full():
                self.queue.put_nowait(
                    ("disconnect", disconnect_reason, time.perf_counter())
                )

    async def infer(self, request):
        p = request.prepared
        try:
            result = await self.backend.predict(
                p.task, json.loads(p.state_json), json.loads(p.candidates_json)
            )
            binding = resolve_selection(result["selected_action"], list(p.bindings))
            index = next(i for i, b in enumerate(p.bindings) if b is binding)
            await self.queue.put(
                ("prediction", (request.request_id, index, result), time.perf_counter())
            )
        except asyncio.CancelledError:
            return
        except Exception as exc:
            await self.queue.put(
                (
                    "prediction_error",
                    (request.request_id, type(exc).__name__),
                    time.perf_counter(),
                )
            )

    async def timer(self, request):
        await asyncio.sleep(max(0, request.budget.target - time.perf_counter()))
        await self.queue.put(("timeout", request.request_id, time.perf_counter()))

    async def reply(self, ws, request, index, *, fallback=False, reason=None):
        if request.ended or request.sent or self.state != SessionState.Playing:
            return
        if index is None or time.perf_counter() >= request.budget.deadline:
            self.end_request(request)
            self.policy.clear()
            if index is None:
                self.fault("no_provable_legal_fallback")
            self.record(
                "server_default_required",
                request_id=request.request_id,
                reason=reason or "no_legal_reply_budget",
            )
            return
        action = dict(request.prepared.wires[index], request_id=request.request_id)
        # Mark before awaiting I/O: a partial/failed send is never retried.
        request.sent, request.sent_index = True, index
        remaining = request.budget.deadline - time.perf_counter()
        async with asyncio.timeout(remaining):
            await ws.send(json.dumps(action))
        self.end_request(request)
        self.record(
            "reply",
            request_id=request.request_id,
            action=action,
            fallback=fallback,
            reason=reason,
            elapsed_seconds=time.perf_counter() - request.received,
        )

    async def request(self, ws, message, received):
        rid = message.get("request_id")
        if type(rid) is not int or rid < 0:
            raise AdapterError("invalid request_id")
        if rid in self.requests:
            self.record("duplicate_request", request_id=rid)
            return
        if self.requests and rid <= max(self.requests):
            raise AdapterError("non-monotonic request_id")
        if self.state != SessionState.Playing or self.faults:
            self.record("ignored_request", request_id=rid)
            return
        if any(not r.acknowledged for r in self.requests.values()):
            raise AdapterError("new request before previous action confirmation")
        request = Request(
            rid,
            Budget.read(
                message["time"], received, inbound=self.inbound, reserve=self.reserve
            ),
        )
        request.received = received
        self.requests[rid] = request  # Claim identity before touching Observation.
        request.prepared = self.policy.prepare(message)
        p = request.prepared
        self.record(
            "prepared",
            request_id=rid,
            task=p.task,
            hand_id=p.hand_id,
            new_events=json.loads(p.events_json),
            state=json.loads(p.state_json),
            candidates=json.loads(p.candidates_json),
            bindings=[
                {
                    "candidate": b.candidate,
                    "engine_action": b.action.to_dict(),
                    "riichi_tile": b.riichi_tile,
                }
                for b in p.bindings
            ],
            forced_riichi_discard=p.forced_index is not None,
            budget=message["time"],
        )
        if p.forced_index is not None:
            await self.reply(ws, request, p.forced_index)
        elif time.perf_counter() >= request.budget.target:
            await self.reply(
                ws,
                request,
                p.fallback_index,
                fallback=True,
                reason="preparation_budget",
            )
        else:
            request.inference = self.spawn(self.infer(request))
            request.timer = self.spawn(self.timer(request))

    async def message(self, ws, message, received):
        kind = message["type"]
        self.record("received", message=message)
        if kind == "start_game":
            if self.state != SessionState.Connecting or message.get("id") not in range(
                4
            ):
                raise AdapterError("invalid start_game")
            self.state = SessionState.Playing
            self.policy.player_id = message["id"]
        elif kind == "request_action":
            await self.request(ws, message, received)
        elif kind == "action_ack":
            rid, status = message.get("request_id"), message.get("status")
            if status not in {
                "accepted",
                "rejected",
                "unparseable",
                "stale",
                "defaulted",
            }:
                raise AdapterError("unknown action_ack status")
            r = self.requests.get(rid)
            if r is None or r.acknowledged:
                self.record("ignored_ack", request_id=rid)
                return
            self.end_request(r)
            r.acknowledged = True
            if r.prepared:
                self.policy.confirm(
                    r.prepared, r.sent_index, status, message.get("action")
                )
            self.record(
                "execution",
                request_id=rid,
                status=status,
                actual_action=(
                    r.prepared.wires[r.sent_index]
                    if status == "accepted" and r.sent_index is not None
                    else message.get("action")
                ),
            )
            if status in {"rejected", "unparseable", "stale"}:
                self.fault("action_" + status)
        elif kind == "end_game":
            if self.end_seen:
                return  # Repeated terminal events cannot extend the finish timeout.
            self.end_seen = True
            self.state = SessionState.Finishing
            self.policy.clear()
            for r in self.requests.values():
                self.end_request(r)
            self.finish_deadline = time.perf_counter() + self.finish_timeout
            if not self.validation:
                self.completed = True
                self.state = SessionState.Closed
        elif kind == "validation_result":
            if not self.end_seen or type(message.get("passed")) is not bool:
                raise AdapterError("unexpected validation_result")
            self.validation_result = self.redact(message)
            self.completed = True
            self.state = SessionState.Closed
        elif kind == "end_kyoku":
            self.policy.clear()
            for r in self.requests.values():
                self.end_request(r)
        elif kind in {
            "start_kyoku",
            "tsumo",
            "dahai",
            "chi",
            "pon",
            "daiminkan",
            "ankan",
            "kakan",
            "reach",
            "reach_accepted",
            "dora",
            "hora",
            "ryukyoku",
        }:
            # Transport trace only. Observation.new_events is the sole tracker input.
            pass
        elif message.get("affects_state") or any(
            k in message for k in ("observation", "events", "scores", "tehais")
        ):
            raise AdapterError("unknown state-affecting message")
        else:
            self.record("unknown_message", type=kind)

    async def run(self, ws):
        receiver = self.spawn(self.receive(ws))
        start_deadline = (
            None
            if self.start_timeout is None
            else time.perf_counter() + self.start_timeout
        )
        try:
            while self.state not in (SessionState.Closed, SessionState.Disconnected):
                timeout = (
                    None
                    if self.finish_deadline is None
                    else max(0, self.finish_deadline - time.perf_counter())
                )
                if self.state == SessionState.Connecting and start_deadline is not None:
                    timeout = max(0, start_deadline - time.perf_counter())
                try:
                    kind, payload, received = await asyncio.wait_for(
                        self.queue.get(), timeout
                    )
                except TimeoutError:
                    waiting_start = self.state == SessionState.Connecting
                    self.fault(
                        "matchmaking_timeout"
                        if waiting_start
                        else "validation_result_timeout"
                    )
                    self.state = (
                        SessionState.Disconnected
                        if waiting_start
                        else SessionState.Closed
                    )
                    break
                if kind == "disconnect":
                    if self.end_seen and payload is None:
                        self.state = (
                            SessionState.Closed
                        )  # Explicit close after end_game.
                        self.completed = True
                    else:
                        self.state = SessionState.Disconnected
                        self.fault("disconnected_unfinished")
                    break
                if kind == "broken":
                    raise AdapterError(payload)
                if kind == "message":
                    await self.message(ws, payload, received)
                elif kind in ("prediction", "prediction_error", "timeout"):
                    rid = payload if kind == "timeout" else payload[0]
                    r = self.requests[rid]
                    if kind == "prediction_error":
                        self.record(
                            "inference_failure", request_id=rid, error_type=payload[1]
                        )
                    if r.ended or r.sent or self.faults:
                        self.record("late_result_ignored", request_id=rid)
                        continue
                    if kind == "prediction" and time.perf_counter() < r.budget.target:
                        self.record("prediction", request_id=rid, result=payload[2])
                        await self.reply(ws, r, payload[1])
                    else:
                        await self.reply(
                            ws, r, r.prepared.fallback_index, fallback=True, reason=kind
                        )
                if (
                    receiver.done()
                    and self.queue.empty()
                    and self.state
                    not in (SessionState.Closed, SessionState.Disconnected)
                ):
                    self.state = SessionState.Disconnected
                    self.fault("receive_stopped")
        except asyncio.CancelledError:
            self.state = SessionState.Disconnected
            self.policy.clear()
            raise
        except Exception as exc:
            self.fault(
                str(exc) if isinstance(exc, AdapterError) else type(exc).__name__
            )
            self.state = SessionState.Disconnected
        finally:
            if getattr(self, "receive_error", None):
                self.fault(self.receive_error)
            for r in self.requests.values():
                self.end_request(r)
            self.policy.clear()
            tasks = list(self.tasks)
            for task in tasks:
                task.cancel()
            if tasks:
                await asyncio.wait(tasks, timeout=0.1)
        return self.summary()

    def summary(self):
        return {
            "state": self.state.value,
            "completed": self.completed,
            "faults": self.faults,
            "validation_result": self.validation_result,
            "requests": len(self.requests),
            "replies": sum(r.sent for r in self.requests.values()),
        }
