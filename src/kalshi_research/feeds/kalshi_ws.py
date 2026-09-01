from __future__ import annotations

import base64
import json
import time
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding


class SequenceGap(RuntimeError):
    pass


def create_auth_headers(api_key_id: str, private_key_path: Path, method: str, path: str) -> dict[str, str]:
    """Create Kalshi RSA-PSS auth headers; query parameters are excluded from signed path."""
    private_key = serialization.load_pem_private_key(private_key_path.read_bytes(), password=None)
    timestamp = str(int(time.time() * 1000))
    signed_path = path.split("?", 1)[0]
    message = f"{timestamp}{method.upper()}{signed_path}".encode()
    signature = private_key.sign(
        message,
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
        hashes.SHA256(),
    )
    return {
        "KALSHI-ACCESS-KEY": api_key_id,
        "KALSHI-ACCESS-TIMESTAMP": timestamp,
        "KALSHI-ACCESS-SIGNATURE": base64.b64encode(signature).decode(),
    }


@dataclass(slots=True)
class BinaryOrderBook:
    ticker: str
    yes: dict[Decimal, Decimal] = field(default_factory=dict)
    no: dict[Decimal, Decimal] = field(default_factory=dict)
    last_seq: int | None = None

    def apply_snapshot(self, seq: int, yes_levels: list[list[str]], no_levels: list[list[str]]) -> None:
        self.yes = {Decimal(p): Decimal(q) for p, q in yes_levels if Decimal(q) > 0}
        self.no = {Decimal(p): Decimal(q) for p, q in no_levels if Decimal(q) > 0}
        self.last_seq = seq

    def apply_delta(self, seq: int, side: str, price: str, delta: str) -> None:
        if self.last_seq is None:
            raise SequenceGap("delta received before snapshot")
        if seq != self.last_seq + 1:
            raise SequenceGap(f"expected seq {self.last_seq + 1}, got {seq}")
        levels = self.yes if side == "yes" else self.no
        px = Decimal(price)
        new_size = levels.get(px, Decimal("0")) + Decimal(delta)
        if new_size < 0:
            raise ValueError("delta would make level negative")
        if new_size == 0:
            levels.pop(px, None)
        else:
            levels[px] = new_size
        self.last_seq = seq

    @property
    def best_yes_bid(self) -> Decimal | None:
        return max(self.yes, default=None)

    @property
    def best_no_bid(self) -> Decimal | None:
        return max(self.no, default=None)

    @property
    def implied_yes_ask(self) -> Decimal | None:
        return None if self.best_no_bid is None else Decimal("1") - self.best_no_bid

    @property
    def implied_no_ask(self) -> Decimal | None:
        return None if self.best_yes_bid is None else Decimal("1") - self.best_yes_bid

    def executable_yes_spread(self) -> Decimal | None:
        bid, ask = self.best_yes_bid, self.implied_yes_ask
        return None if bid is None or ask is None else ask - bid


def parse_ws_message(raw: str) -> dict[str, Any]:
    payload = json.loads(raw)
    if not isinstance(payload, dict) or "type" not in payload:
        raise ValueError("invalid Kalshi websocket message")
    return payload
