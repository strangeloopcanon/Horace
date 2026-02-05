from __future__ import annotations

import ipaddress
import socket
from typing import Iterable
from urllib.parse import urljoin, urlsplit


_ALLOWED_SCHEMES = {"http", "https"}
_BLOCKED_HOSTS = {"localhost", "localhost.localdomain"}
IPAddress = ipaddress.IPv4Address | ipaddress.IPv6Address


def _is_public_ip(ip: IPAddress) -> bool:
    return not (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def _iter_resolved_ips(hostname: str, *, port: int) -> Iterable[IPAddress]:
    infos = socket.getaddrinfo(hostname, port, proto=socket.IPPROTO_TCP)
    seen: set[str] = set()
    for info in infos:
        sockaddr = info[4]
        if not sockaddr:
            continue
        raw_ip = str(sockaddr[0]).strip()
        if not raw_ip or raw_ip in seen:
            continue
        seen.add(raw_ip)
        try:
            yield ipaddress.ip_address(raw_ip)
        except ValueError:
            continue


def validate_public_http_url(url: str, *, resolve_dns: bool = True) -> str:
    """Validate an external URL and reject local/private targets.

    So what: URL scoring fetches remote pages; this blocks local file and internal-network
    targets that could leak host data or hit private services.
    """
    raw = str(url or "").strip()
    if not raw:
        raise ValueError("url is empty")

    parts = urlsplit(raw)
    scheme = str(parts.scheme or "").lower()
    if scheme not in _ALLOWED_SCHEMES:
        raise ValueError(f"unsupported URL scheme: {scheme or '(missing)'}")
    if parts.username or parts.password:
        raise ValueError("URL userinfo is not allowed")

    host = str(parts.hostname or "").strip().lower()
    if not host:
        raise ValueError("URL host is missing")
    if host in _BLOCKED_HOSTS or host.endswith(".localhost"):
        raise ValueError("local hostnames are not allowed")

    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        ip = None

    if ip is not None:
        if not _is_public_ip(ip):
            raise ValueError(f"non-public target is not allowed: {ip}")
        return raw

    if bool(resolve_dns):
        port = int(parts.port) if parts.port is not None else (443 if scheme == "https" else 80)
        try:
            resolved_ips = list(_iter_resolved_ips(host, port=port))
        except socket.gaierror as e:
            raise ValueError(f"unable to resolve host: {host}") from e
        if not resolved_ips:
            raise ValueError(f"unable to resolve host: {host}")
        if any(not _is_public_ip(ip_addr) for ip_addr in resolved_ips):
            raise ValueError(f"non-public target is not allowed: {host}")

    return raw


def validate_public_redirect_target(current_url: str, location: str, *, resolve_dns: bool = True) -> str:
    """Resolve and validate a redirect Location header target.

    So what: SSRF checks must apply to every redirect hop, not just the first URL.
    """
    base = str(current_url or "").strip()
    if not base:
        raise ValueError("current_url is empty")
    loc = str(location or "").strip()
    if not loc:
        raise ValueError("redirect location is empty")
    target = urljoin(base, loc)
    return validate_public_http_url(target, resolve_dns=bool(resolve_dns))
