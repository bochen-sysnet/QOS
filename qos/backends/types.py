"""
Minimal placeholder for QPU type expected by qos.backends.database and helpers.
"""
from typing import Any, Dict, List


class QPU:
    def __init__(self) -> None:
        self.id: int | None = None
        self.name: str = ""
        self.alias: str = ""
        self.provider: str = ""
        self.args: Dict[str, Any] = {}
        self.local_queue: List[Any] = []
