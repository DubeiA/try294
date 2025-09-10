from typing import Dict, Any
import time
import requests
from eva_env_base import log

class ComfyClient:
    """Enhanced ComfyUI API client with better error handling"""

    def __init__(self, api_base: str = "http://127.0.0.1:8188"):
        self.api_base = api_base.rstrip("/")

    def queue(self, workflow: Dict[str, Any]) -> str:
        """Queue a workflow for generation"""
        url = f"{self.api_base}/prompt"
        payload = {"prompt": workflow}
        try:
            r = requests.post(url, json=payload, timeout=120)
            r.raise_for_status()
            data = r.json()
            pid = data.get("prompt_id") or data.get("id")
            if not pid:
                raise ValueError("ComfyUI queue: missing prompt_id in response")
            return pid
        except Exception as e:
            log.error(f"Failed to queue workflow: {e}")
            raise

    def wait(self, prompt_id: str, timeout_s: int = 3600, poll_s: int = 2) -> Dict[str, Any]:
        """Wait for workflow completion"""
        url = f"{self.api_base}/history/{prompt_id}"
        t0 = time.time()

        while True:
            try:
                r = requests.get(url, timeout=60)
                if r.status_code == 200:
                    hist = r.json()
                    entry = hist[prompt_id] if isinstance(hist, dict) and prompt_id in hist else hist
                    if (isinstance(entry, dict) and entry.get("status", {}).get("completed")) or (isinstance(entry, dict) and entry.get("outputs")):
                        return entry
            except Exception as e:
                log.warning(f"Polling error: {e}")

            if time.time() - t0 > timeout_s:
                raise TimeoutError(f"Job {prompt_id} timed out after {timeout_s}s")
            time.sleep(poll_s)

    def object_info(self) -> Dict[str, Any]:
        """Get ComfyUI object info"""
        url = f"{self.api_base}/object_info"
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        return r.json()
