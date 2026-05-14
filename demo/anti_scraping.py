"""
anti_scraping.py
════════════════
pipeline.py 依赖的网络/反爬工具模块。
提供:代理池管理、速率限制、UA 轮换、yt-dlp 封装搜索、JSON 重试请求。
"""

import os
import time
import random
import json
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError
from typing import Optional

# ── 代理池 ────────────────────────────────────────────────────────────────────
_PROXY_LIST: list[str] = []

def set_proxy_list(proxies: list[str]):
    global _PROXY_LIST
    _PROXY_LIST = [p for p in proxies if p]

def load_proxies_from_file(path: str):
    global _PROXY_LIST
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
        _PROXY_LIST = [l.strip() for l in lines if l.strip()]
    except Exception:
        pass

def load_proxies_from_env():
    global _PROXY_LIST
    raw = os.environ.get("SCRAPER_PROXIES", "")
    if raw:
        _PROXY_LIST = [p.strip() for p in raw.split(",") if p.strip()]

def _get_proxy() -> Optional[str]:
    return random.choice(_PROXY_LIST) if _PROXY_LIST else None


# ── 速率限制 ──────────────────────────────────────────────────────────────────
_GLOBAL_MIN_DELAY: float = 0.0
_HOST_DELAYS: dict[str, float] = {}
_LAST_REQUEST_TIME: dict[str, float] = {}

def set_global_rate_limit(delay: float):
    global _GLOBAL_MIN_DELAY
    _GLOBAL_MIN_DELAY = max(0.0, delay)

def set_host_rate_limit(host: str, delay: float):
    _HOST_DELAYS[host] = max(0.0, delay)

def _rate_wait(host: str = ""):
    delay = _HOST_DELAYS.get(host, _GLOBAL_MIN_DELAY)
    if delay <= 0:
        return
    last = _LAST_REQUEST_TIME.get(host, 0.0)
    elapsed = time.time() - last
    if elapsed < delay:
        time.sleep(delay - elapsed)
    _LAST_REQUEST_TIME[host] = time.time()


# ── User-Agent 轮换 ───────────────────────────────────────────────────────────
_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.4.1 Safari/605.1.15",
]

def _random_ua() -> str:
    return random.choice(_USER_AGENTS)


# ── 平台专属 Headers ─────────────────────────────────────────────────────────
_PLATFORM_HEADERS: dict[str, dict] = {
    "bilisearch": {
        "User-Agent"      : _random_ua(),
        "Referer"         : "https://www.bilibili.com/",
        "Accept-Language" : "zh-CN,zh;q=0.9",
    },
    "niconico": {
        "User-Agent"      : _random_ua(),
        "Accept-Language" : "ja,en;q=0.9",
    },
    "odysee": {
        "User-Agent": _random_ua(),
    },
    "peertube": {
        "User-Agent": _random_ua(),
    },
}

def get_platform_headers(platform: str) -> dict:
    """返回指定平台的 HTTP headers，未配置的平台返回通用 UA。"""
    return _PLATFORM_HEADERS.get(platform, {"User-Agent": _random_ua()})


# ── JSON 重试请求 ─────────────────────────────────────────────────────────────
def fetch_json_with_retry(url: str, retries: int = 3, timeout: int = 10) -> Optional[dict]:
    """带重试的 JSON 请求，支持代理和 UA 轮换。"""
    from urllib.parse import urlparse
    host = urlparse(url).netloc
    for attempt in range(retries):
        try:
            _rate_wait(host)
            headers = {"User-Agent": _random_ua()}
            req = Request(url, headers=headers)
            proxy = _get_proxy()
            if proxy:
                from urllib.request import ProxyHandler, build_opener
                opener = build_opener(ProxyHandler({"http": proxy, "https": proxy}))
                resp = opener.open(req, timeout=timeout)
            else:
                resp = urlopen(req, timeout=timeout)
            data = json.loads(resp.read().decode("utf-8"))
            _LAST_REQUEST_TIME[host] = time.time()
            return data
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise e
    return None


# ── yt-dlp 平台搜索封装 ───────────────────────────────────────────────────────
def yt_dlp_search_for_platform(
    platform: str,
    n: int,
    keyword: str,
    http_headers: Optional[dict] = None,
) -> list[dict]:
    """
    用 yt-dlp 对指定平台做关键词搜索，返回 entry 字典列表。
    支持传入自定义 http_headers（用于规避反爬）。
    """
    try:
        import yt_dlp
    except ImportError:
        return []

    search_query = f"{platform}{n}:{keyword}"
    opts: dict = {
        "quiet"        : True,
        "no_warnings"  : True,
        "extract_flat" : True,
        "skip_download": True,
    }
    if http_headers:
        opts["http_headers"] = http_headers

    proxy = _get_proxy()
    if proxy:
        opts["proxy"] = proxy

    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(search_query, download=False)
            entries = [e for e in (info or {}).get("entries", []) if e]
            return entries
    except Exception:
        return []
