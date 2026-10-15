#!/usr/bin/env python3
"""Validate local assets, page links and anchors in a built MkDocs site."""
from __future__ import annotations

import argparse
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urljoin, urlsplit


class AssetParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.urls: list[str] = []
        self.anchors: set[str] = set()

    def handle_starttag(self, tag: str,
                        attrs: list[tuple[str, str | None]]) -> None:
        for name, value in attrs:
            if name in ("src", "href") and value:
                self.urls.append(value)
            if value and (name == "id" or (tag == "a" and name == "name")):
                self.anchors.add(value)


def page_base(site_dir: Path, page: Path, prefix: str) -> str:
    relative = page.relative_to(site_dir)
    base = "/" + prefix.strip("/") + "/" if prefix.strip("/") else "/"
    if relative.name != "index.html":
        return base + relative.as_posix()
    parent = relative.parent.as_posix()
    return base if parent == "." else f"{base}{parent}/"


def site_path(url: str, prefix: str) -> str:
    path = urlsplit(url).path
    base = "/" + prefix.strip("/") + "/" if prefix.strip("/") else "/"
    if base != "/" and path.startswith(base):
        path = path[len(base):]
    return unquote(path.lstrip("/"))


def check_site(site_dir: Path, prefix: str = "") -> tuple[int, list[str]]:
    pages: dict[Path, AssetParser] = {}
    for page in sorted(site_dir.rglob("*.html")):
        parser = AssetParser()
        parser.feed(page.read_text(encoding="utf-8"))
        pages[page] = parser

    checked = 0
    failures: list[str] = []
    base = "/" + prefix.strip("/") + "/" if prefix.strip("/") else "/"
    for page, parser in pages.items():
        for url in sorted(set(parser.urls)):
            parsed = urlsplit(url)
            if parsed.scheme or parsed.netloc:
                continue
            checked += 1
            resolved = urljoin(page_base(site_dir, page, prefix), url)
            label = f"{page.relative_to(site_dir)}: {url}"
            if not urlsplit(resolved).path.startswith(base):
                failures.append(f"outside site prefix: {label} -> {resolved}")
                continue
            target = site_dir / site_path(resolved, prefix)
            if target.is_dir():
                target = target / "index.html"
            if not target.is_file():
                failures.append(
                    f"missing target: {label} -> {target.relative_to(site_dir)}")
                continue
            fragment = unquote(urlsplit(resolved).fragment)
            if fragment and target.suffix.lower() == ".html":
                target_parser = pages.get(target)
                if target_parser is None or fragment not in target_parser.anchors:
                    failures.append(
                        f"missing anchor: {label} -> "
                        f"{target.relative_to(site_dir)}#{fragment}")
    return checked, failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("site_dir", type=Path)
    parser.add_argument("--site-prefix", default="")
    args = parser.parse_args()
    site_dir = args.site_dir.resolve()
    if not site_dir.is_dir():
        parser.error(f"site directory does not exist: {site_dir}")
    checked, failures = check_site(site_dir, args.site_prefix)
    if failures:
        print("\n".join(failures))
        return 1
    print(f"checked {checked} local asset/page/anchor links; all targets exist")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
