"""Frontend WebSocket Client Service.

Provides helper methods to test, inspect, and connect to real-time market streams.
"""

from typing import Any, Dict, Optional
import json
import websockets
import asyncio


async def test_ws_connection(ws_url: str) -> Dict[str, Any]:
    """Test connecting to the backend WebSocket stream and fetch initial handshake."""
    try:
        async with websockets.connect(ws_url, open_timeout=2.0) as ws:
            raw_ack = await asyncio.wait_for(ws.recv(), timeout=2.0)
            ack = json.loads(raw_ack)
            return {"connected": True, "ack": ack}
    except Exception as exc:
        return {"connected": False, "error": str(exc)}


def check_ws_connection_sync(ws_url: Optional[str] = None, timeout: float = 1.5) -> bool:
    """Synchronously test WebSocket connectivity with short timeout."""
    if not ws_url:
        import streamlit as st
        from frontend.utils.state import get_backend_url
        base = st.session_state.get("backend_url") if "backend_url" in st.session_state else get_backend_url()
        ws_url = base.replace("http://", "ws://").replace("https://", "wss://") + "/ws"
    try:
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(lambda: asyncio.run(test_ws_connection(ws_url)))
            res = future.result(timeout=timeout)
            return bool(res.get("connected", False))
    except Exception:
        return False
