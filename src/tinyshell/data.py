"""Build the TinyShell dataset from tldr-pages.

Pipeline:
1. Download/sparse-checkout tldr-pages (CC-BY-4.0).
2. Parse Markdown examples into (description, command template) pairs.
3. Filter noisy/complex examples.
4. Split by command (page) so unseen-command evaluation is honest.
5. Fill placeholders with realistic values and render natural-language requests.
6. Write train/val/test JSONL files plus a manifest and statistics.
"""

from __future__ import annotations

import argparse
import hashlib
import random
import re
import shutil
import statistics
import subprocess
import time
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from .config import Config
from .text import assess_risk, is_valid_bash, lower_first
from .utils import ensure_dir, human_time, write_json, write_jsonl

EXAMPLE_RE = re.compile(r"^- (.+?):\s*$")
CODE_RE = re.compile(r"^`(.+)`$")
PLACEHOLDER_RE = re.compile(r"\{\{(.*?)\}\}")
FILL_RE = re.compile(r"\{\{(.*?)\}\}(\.[A-Za-z0-9]+)?")
SHELL_KEYWORDS = {"case", "for", "while", "until", "if", "function", "select", "then", "fi"}

# Prompt styles. 0-7 are seen during training; 8-9 are held out for the
# seen-command paraphrase split; 10-11 are held out for the hardest split.
REQUEST_STYLES = [
    "How do I {desc}?",
    "I want to {desc}.",
    "Can you give me a command to {desc}?",
    "Please show me how to {desc}.",
    "Write a shell command to {desc}.",
    "Which Linux command should I use to {desc}?",
    "I need to {desc}.",
    "Command to {desc}:",
    "Give me a one-liner for this: {desc}.",
    "What's the best way to {desc}?",
    "Help me {desc}.",
    "Show me the command for this: {desc}.",
]
TRAIN_STYLES = list(range(8))
VAL_STYLE = 8
TEST_SEEN_STYLE = 9
PARAPHRASE_STYLES = [10, 11]

INLINE_PATTERNS = {
    "path": re.compile(r"\b(?:a|the|given|specified|current)\s+path\b", re.IGNORECASE),
    "directory": re.compile(
        r"\b(?:a|the|given|specified|current)\s+(?:directory|folder)\b", re.IGNORECASE
    ),
    "file": re.compile(r"\b(?:a|the|given|specified)\s+(?:file|filename)\b", re.IGNORECASE),
    "pattern": re.compile(
        r"\b(?:a|the|given|specified)\s+(?:pattern|glob|regex|regular expression)\b",
        re.IGNORECASE,
    ),
    "host": re.compile(
        r"\b(?:a|the|given|specified)\s+(?:host|hostname|domain|server)\b", re.IGNORECASE
    ),
    "user": re.compile(r"\b(?:a|the|given|specified)\s+(?:user|username|owner)\b", re.IGNORECASE),
    "command": re.compile(r"\b(?:a|the|given|specified)\s+command\b", re.IGNORECASE),
    "package": re.compile(r"\b(?:a|the|given|specified)\s+(?:package|module)\b", re.IGNORECASE),
    "port": re.compile(r"\b(?:a|the|given|specified)\s+port\b", re.IGNORECASE),
    "url": re.compile(r"\b(?:a|the|given|specified)\s+(?:URL|url|link)\b", re.IGNORECASE),
    "name": re.compile(r"\b(?:a|the|given|specified)\s+name\b", re.IGNORECASE),
    "number": re.compile(
        r"\b(?:a|the|given|specified)\s+(?:number|count|amount)\b", re.IGNORECASE
    ),
}


@dataclass(frozen=True)
class Example:
    page: str
    platform: str
    description: str
    command_template: str
    source: str


@dataclass(frozen=True)
class SlotValue:
    value: str
    label: str
    category: str
    inline_key: str | None
    is_value: bool


@dataclass
class FilledCommand:
    command: str
    values: list[SlotValue] = field(default_factory=list)
    template: str = ""


def _hash_fraction(text: str) -> float:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16) / float(1 << 64)


def _variant_seed(base: int, key: str, index: int) -> int:
    digest = hashlib.sha256(f"{base}:{key}:{index}".encode()).hexdigest()[:16]
    return int(digest, 16)


def parse_page(path: Path) -> list[Example]:
    """Parse a tldr Markdown page into examples. Returns [] for malformed pages."""
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    if not lines:
        return []
    page = lines[0].lstrip("#").strip()
    if not page:
        return []
    platform = path.parent.name
    examples: list[Example] = []
    i = 0
    while i < len(lines):
        match = EXAMPLE_RE.match(lines[i])
        if not match:
            i += 1
            continue
        description = match.group(1).strip()
        j = i + 1
        while j < len(lines) and not lines[j].strip():
            j += 1
        if j < len(lines):
            code_match = CODE_RE.match(lines[j].strip())
            if code_match:
                command = code_match.group(1).strip()
                if description and command:
                    examples.append(
                        Example(
                            page=page,
                            platform=platform,
                            description=description,
                            command_template=command,
                            source=str(path),
                        )
                    )
                i = j + 1
                continue
        i += 1
    return examples


def _filter_reason(example: Example, cfg: Config) -> str | None:
    desc = example.description.strip()
    command = example.command_template.strip()
    if len(desc) < cfg.data.min_description_chars:
        return "description_too_short"
    if len(desc) > cfg.data.max_description_chars:
        return "description_too_long"
    if not command or "\n" in command:
        return "empty_or_multiline"
    if len(command) > cfg.data.max_command_chars:
        return "command_too_long"
    stripped = PLACEHOLDER_RE.sub("", command)
    if "{{" in stripped or "}}" in stripped:
        return "malformed_placeholder"
    if "{{$" in command or "$(" in command or "${" in command:
        return "shell_substitution_or_variable"
    if "!!" in command or re.search(r"![$^]", command):
        return "history_expansion"
    if example.page in {"!", "[[", "]]"}:
        return "shell_snippet"
    if command.startswith("<") or re.search(r"<[A-Za-z][^>]*>", command):
        return "interactive_keybinding"
    if command.startswith(":"):
        return "editor_command"
    first = command.split(maxsplit=1)[0].lower()
    if first in SHELL_KEYWORDS or first in {"{", "(", "[["}:
        return "shell_snippet"
    if command.startswith("#") or " #" in command:
        return "comment"
    if ";" in command:
        return "multiple_statements"
    return None


def load_examples(cfg: Config) -> tuple[dict[str, list[Example]], dict[str, Any]]:
    """Load, filter, and group examples by page title."""
    source = Path(cfg.data.source_dir)
    page_paths = sorted(source.glob("pages/common/*.md")) + sorted(source.glob("pages/linux/*.md"))
    if not page_paths:
        raise FileNotFoundError(
            f"No tldr pages under {source}. Run with --redownload or check the source path."
        )

    raw_pages = 0
    raw_examples = 0
    reasons: Counter[str] = Counter()
    grouped: dict[str, list[Example]] = {}
    seen: set[tuple[str, str]] = set()

    for path in page_paths:
        raw_pages += 1
        for example in parse_page(path):
            raw_examples += 1
            reason = _filter_reason(example, cfg)
            if reason:
                reasons[reason] += 1
                continue
            key = (example.description, example.command_template)
            if key in seen:
                reasons["duplicate"] += 1
                continue
            seen.add(key)
            grouped.setdefault(example.page, []).append(example)

    if not grouped:
        raise RuntimeError("No usable examples after filtering")

    stats = {
        "raw_pages": raw_pages,
        "raw_examples": raw_examples,
        "usable_pages": len(grouped),
        "usable_examples": sum(len(v) for v in grouped.values()),
        "filter_reasons": dict(reasons.most_common()),
    }
    return grouped, stats


def split_pages(
    grouped: dict[str, list[Example]], cfg: Config
) -> dict[str, list[str]]:
    """Deterministically split page titles into train/val/test."""
    train, val, test = [], [], []
    for page in sorted(grouped):
        h = _hash_fraction(page)
        if h < cfg.data.train_ratio:
            train.append(page)
        elif h < cfg.data.train_ratio + cfg.data.val_ratio:
            val.append(page)
        else:
            test.append(page)
    if not train or not val or not test:
        raise RuntimeError(f"Empty split: train={len(train)} val={len(val)} test={len(test)}")
    return {"train": train, "val": val, "test": test}


class SlotFiller:
    """Turn tldr ``{{placeholders}}`` into realistic, consistent values."""

    PATHS = [
        "/tmp/project",
        "/var/log",
        "/home/alice/data",
        "/etc/nginx",
        "src",
        "docs",
        "/opt/app",
        "~/backups",
        "/srv/www",
        "data/raw",
    ]
    DIRS = [
        "/tmp/project",
        "/var/log",
        "/home/alice/data",
        "/etc/nginx",
        "src",
        "docs",
        "public",
        "build",
    ]
    FILES = [
        "/tmp/notes.txt",
        "/tmp/report.csv",
        "/var/log/syslog",
        "/etc/hosts",
        "README.md",
        "src/main.py",
        "package.json",
        "config.yaml",
    ]
    NAMES = ["my-project", "alice", "web", "api", "backup", "report", "demo", "staging"]
    USERS = ["alice", "bob", "deploy", "www-data", "root", "ubuntu"]
    PACKAGES = ["requests", "numpy", "flake8", "nginx", "git", "curl", "ripgrep"]
    HOSTS = ["example.com", "api.example.com", "192.0.2.10", "localhost", "db.internal"]
    URLS = ["https://example.com", "https://api.example.com/v1", "http://localhost:8080"]
    PORTS = ["8080", "443", "3000", "5432", "22"]
    PATTERNS = ["*.py", "*.log", "test_*", "*.txt", "error*", "*.json"]
    REGEXES = [r"^test_", r"\d{4}-\d{2}-\d{2}", r"error|fail", r"TODO"]
    COMMANDS = ["ls -la", "git status", "curl -I https://example.com", "python3 -m pytest"]
    PASSWORDS = ["correct-horse", "s3cret-pass", "changeme"]
    VERSIONS = ["1.2.3", "2.0.0", "latest", "v3.1"]
    DATES = ["2026-01-15", "2025-12-01", "today"]
    NUMBERS = ["5", "10", "20", "3", "100"]
    TEXTS = ["hello", "error", "example", "needle", "sample"]
    DEVICES = ["/dev/sda", "/dev/nvme0n1", "/dev/ttyUSB0"]
    IMAGES = ["nginx:latest", "alpine:3.19", "python:3.12", "postgres:16"]
    INTERFACES = ["eth0", "wlan0", "docker0", "lo"]
    UNITS = ["nginx.service", "ssh.service", "postgresql.service"]
    SERVICES = ["nginx", "postgresql", "redis", "docker"]
    REPOS = ["my-project", "backend", "infra", "api"]
    BRANCHES = ["main", "develop", "feature/login", "release/1.2"]
    TAGS = ["v1.0.0", "latest", "stable", "v2.1.3"]
    REVISIONS = ["HEAD~1", "main", "abc1234", "v1.0.0"]

    EXT_FILES = {
        ".txt": ["/tmp/notes.txt", "/tmp/output.txt"],
        ".log": ["/var/log/syslog", "/tmp/app.log"],
        ".csv": ["/tmp/report.csv", "/tmp/data.csv"],
        ".json": ["/tmp/data.json", "/tmp/config.json"],
        ".yaml": ["/tmp/config.yaml"],
        ".yml": ["/tmp/config.yml"],
        ".py": ["src/main.py", "/tmp/script.py"],
        ".sh": ["/tmp/backup.sh"],
        ".md": ["README.md", "/tmp/notes.md"],
        ".png": ["/tmp/screenshot.png"],
        ".jpg": ["/tmp/photo.jpg"],
        ".jpeg": ["/tmp/photo.jpeg"],
        ".pdf": ["/tmp/document.pdf"],
        ".html": ["/tmp/index.html"],
        ".xml": ["/tmp/data.xml"],
        ".zip": ["/tmp/archive.zip"],
        ".tar": ["/tmp/archive.tar"],
        ".gz": ["/tmp/archive.tar.gz"],
        ".tiff": ["/tmp/input.tiff"],
        ".tif": ["/tmp/input.tif"],
        ".pnm": ["/tmp/output.pnm"],
        ".stl": ["/tmp/model.stl"],
        ".obj": ["/tmp/model.obj"],
        ".o": ["/tmp/module.o"],
        ".php": ["/tmp/config.php"],
        ".yml.bak": ["/tmp/config.yml"],
        ".conf": ["/etc/nginx/nginx.conf"],
        ".ini": ["/tmp/config.ini"],
        ".toml": ["pyproject.toml"],
        ".db": ["/tmp/app.db"],
        ".sql": ["/tmp/schema.sql"],
        ".mp4": ["/tmp/video.mp4"],
        ".mkv": ["/tmp/video.mkv"],
        ".mp3": ["/tmp/audio.mp3"],
        ".wav": ["/tmp/audio.wav"],
        ".iso": ["/tmp/image.iso"],
        ".img": ["/tmp/disk.img"],
    }

    def __init__(self, rng: random.Random):
        self.rng = rng
        self.used: set[str] = set()

    def _file_for_ext(self, ext: str) -> str:
        ext = ext.lower()
        if ext in self.EXT_FILES:
            return self._pick(self.EXT_FILES[ext])
        stem = self.rng.choice(["input", "output", "example", "data", "backup"])
        return self._pick([f"/tmp/{stem}{ext}"])

    def _pick(self, pool: list[str]) -> str:
        """Prefer a value not already used in this command, to avoid no-op examples."""
        for _ in range(min(6, len(pool))):
            value = self.rng.choice(pool)
            if value not in self.used:
                self.used.add(value)
                return value
        value = self.rng.choice(pool)
        self.used.add(value)
        return value

    @staticmethod
    def _label(raw: str, category: str) -> str:
        label = raw.strip()
        label = re.sub(r"^path/to/", "", label)
        label = re.sub(r"^path/to", "", label)
        label = re.sub(r"^the\s+", "", label, flags=re.IGNORECASE)
        label = re.sub(r"^a\s+", "", label, flags=re.IGNORECASE)
        label = re.sub(r"\.[A-Za-z0-9]+$", "", label)
        label = label.replace("_", " ").replace("/", " ")
        label = re.sub(r"\d+$", "", label).strip()
        label = re.sub(r"\s+", " ", label)
        if not label or len(label) == 1:
            return category
        return label

    def _categorize(self, raw: str) -> str:
        c = raw.lower()
        ext = Path(raw).suffix.lower()
        if c.startswith(("http://", "https://")):
            return "url"
        if re.fullmatch(r"(?:[a-z0-9-]+\.)+(?:com|org|net|io|dev|ai|co|edu|gov)", c):
            return "host"
        if re.fullmatch(r"\d+\.\d+(?:\.\d+)?", c):
            return "version"
        if c.startswith("/dev/"):
            return "device"
        if c in {"eth0", "ethx", "wlan0", "docker0", "lo"}:
            return "literal"
        if "password" in c or "passwd" in c or c in {"pass", "secret", "token"}:
            return "password"
        if any(k in c for k in ("dir", "folder")):
            return "directory"
        if "file" in c or ext:
            return "file"
        if "path" in c or "/" in c:
            return "path"
        if any(k in c for k in ("username", "user", "owner", "group")):
            return "user"
        if any(k in c for k in ("package", "pkg", "module")):
            return "package"
        if any(k in c for k in ("hostname", "host", "domain", "server", "ip_address", "address")):
            return "host"
        if any(k in c for k in ("url", "link")):
            return "url"
        if "port" in c:
            return "port"
        if any(k in c for k in ("pattern", "glob")):
            return "pattern"
        if any(k in c for k in ("regex", "expression")):
            return "regex"
        if any(k in c for k in ("command", "cmd")):
            return "command"
        if "version" in c:
            return "version"
        if any(k in c for k in ("date", "time")):
            return "date"
        if c.endswith("_id") or c in {"id", "pid", "uid", "gid", "process_id", "task_id"}:
            return "number"
        if any(
            k in c
            for k in (
                "number",
                "count",
                "length",
                "amount",
                "width",
                "height",
                "second",
                "minute",
                "hour",
                "level",
                "size",
                "limit",
                "offset",
                "timeout",
                "duration",
                "interval",
                "priority",
                "weight",
                "ratio",
                "percent",
                "total",
                "index",
            )
        ) or c in {"n", "100"}:
            return "number"
        if any(
            k in c
            for k in (
                "key",
                "value",
                "keyword",
                "query",
                "search",
                "message",
                "title",
                "body",
                "description",
                "string",
                "content",
                "subject",
                "text",
                "email",
                "argument",
            )
        ):
            return "text"
        if "image" in c and (":" in c or "tag" in c):
            return "image"
        if "interface" in c or c in {"ethx"}:
            return "interface"
        if "unit" in c:
            return "unit"
        if "service" in c:
            return "service"
        if "branch" in c:
            return "branch"
        if c in {"tag", "tag_name"}:
            return "tag"
        if "revision" in c or c in {"commit", "revset", "revsets"}:
            return "revision"
        if "repository" in c or c in {"repo", "repository_name"}:
            return "repository"
        if "image" in c:
            return "image"
        if "container" in c:
            return "name"
        if any(
            k in c
            for k in (
                "target",
                "source",
                "sink",
                "project",
                "unit",
                "namespace",
                "cluster",
                "pod",
                "vm",
                "snapshot",
                "node",
                "remote",
                "database",
                "table",
                "app",
                "program",
                "team",
                "organization",
                "pipeline",
                "session",
                "bucket",
                "option",
                "flag",
            )
        ):
            return "name"
        if "name" in c:
            return "name"
        return "generic"

    def _value_for(self, category: str, raw: str) -> str:
        ext = Path(raw).suffix
        if category == "file":
            return self._file_for_ext(ext) if ext else self._pick(self.FILES)
        pools = {
            "path": self.PATHS,
            "directory": self.DIRS,
            "name": self.NAMES,
            "user": self.USERS,
            "package": self.PACKAGES,
            "host": self.HOSTS,
            "url": self.URLS,
            "port": self.PORTS,
            "pattern": self.PATTERNS,
            "regex": self.REGEXES,
            "command": self.COMMANDS,
            "password": self.PASSWORDS,
            "version": self.VERSIONS,
            "date": self.DATES,
            "number": self.NUMBERS,
            "text": self.TEXTS,
            "device": self.DEVICES,
            "image": self.IMAGES,
            "interface": self.INTERFACES,
            "unit": self.UNITS,
            "service": self.SERVICES,
            "repository": self.REPOS,
            "branch": self.BRANCHES,
            "tag": self.TAGS,
            "revision": self.REVISIONS,
        }
        pool = pools.get(category)
        if pool:
            return self._pick(pool)
        return raw

    def resolve(self, content: str) -> SlotValue:
        raw = content.strip()
        if not raw:
            return SlotValue("", "value", "literal", None, False)

        # tldr optional/alternative syntax: {{[-a|--all]}}, {{[clou|cloudinit]}}.
        # Prefer the longest/last alternative: it is the unabbreviated, valid form.
        if raw.startswith("[") and raw.endswith("]"):
            choices = [c.strip() for c in raw[1:-1].split("|") if c.strip() and c.strip() != "..."]
            return SlotValue(choices[-1] if choices else "", "", "literal", None, False)
        if "|" in raw:
            choices = [c.strip() for c in raw.split("|") if c.strip() and c.strip() != "..."]
            return SlotValue(choices[-1] if choices else "", "", "literal", None, False)

        # Literal values that tldr uses to make examples concrete.
        if re.fullmatch(r"\d+%?", raw):
            return SlotValue(raw, "", "literal", None, False)
        if raw.startswith("-") or raw.startswith("$") or ".." in raw:
            return SlotValue(raw, "", "literal", None, False)
        if raw == "/dev/sdX":
            return SlotValue(raw, "", "literal", None, False)

        # Multiple values in one placeholder: {{path/to/file1 path/to/file2 ...}}
        if " " in raw or "..." in raw:
            tokens = [t for t in raw.split() if t != "..."]
            values = [self.resolve(t) for t in tokens]
            joined = " ".join(v.value for v in values if v.value)
            first = values[0] if values else SlotValue("", "", "generic", None, False)
            label = first.label if len(values) == 1 else first.label + "s"
            return SlotValue(joined, label, first.category, first.inline_key, True)

        category = self._categorize(raw)
        inline_key = {
            "path": "path",
            "directory": "directory",
            "file": "file",
            "pattern": "pattern",
            "regex": "pattern",
            "host": "host",
            "url": "url",
            "user": "user",
            "command": "command",
            "package": "package",
            "port": "port",
            "name": "name",
            "number": "number",
            "version": None,
            "password": None,
            "date": None,
            "text": None,
            "device": None,
            "image": None,
            "interface": None,
            "unit": None,
            "service": None,
            "repository": None,
            "branch": None,
            "tag": None,
            "revision": None,
            "generic": None,
        }.get(category)
        label = self._label(raw, category)
        if category == "host":
            if "ip" in raw.lower():
                label = "IP address"
            elif "domain" in raw.lower() or re.fullmatch(
                r"(?:[a-z0-9-]+\.)+(?:com|org|net|io|dev|ai|co|edu|gov)", raw.lower()
            ):
                label = "domain"
            elif "server" in raw.lower():
                label = "server"
            else:
                label = "host"
        elif category == "url":
            label = "URL"
        value = self._value_for(category, raw)
        # Unknown placeholders are usually literal mode/format names (ascii, man,
        # target); keep the word itself and do not add it to the request context.
        is_value = category not in {"literal", "generic"}
        return SlotValue(value, label, category, inline_key, is_value)

    def fill(self, template: str) -> FilledCommand:
        values: list[SlotValue] = []
        pieces: list[str] = []
        last = 0
        for match in FILL_RE.finditer(template):
            pieces.append(template[last : match.start()])
            slot = self.resolve(match.group(1))
            suffix = match.group(2) or ""
            # tldr often writes {{path/to/input}}.avif; include the suffix in the
            # concrete value so the request and command describe the same file.
            if (
                suffix
                and slot.is_value
                and slot.category in {"file", "path"}
                and not slot.value.endswith(suffix)
            ):
                slot = replace(slot, value=slot.value + suffix)
            values.append(slot)
            pieces.append(slot.value)
            last = match.end()
        pieces.append(template[last:])
        command = "".join(pieces)
        command = re.sub(r"[ \t]+", " ", command).strip()
        return FilledCommand(command=command, values=values, template=template)


def _context_sentence(values: list[SlotValue]) -> str:
    groups: dict[str, list[str]] = {}
    for value in values:
        if not value.is_value or not value.value:
            continue
        groups.setdefault(value.label or value.category, []).append(value.value)
    if not groups:
        return ""
    parts = []
    for label, vals in groups.items():
        # A multi-value placeholder is stored as one space-joined string.
        if len(vals) == 1 and " " in vals[0] and label.endswith("s"):
            vals = vals[0].split()
        quoted = ", ".join(f"`{v}`" for v in vals)
        plural = len(vals) > 1 or label.endswith("s")
        if label.startswith(("number of", "amount of")):
            plural = False
        verb = "are" if plural else "is"
        noun = label
        if len(vals) > 1 and not label.endswith("s"):
            noun = label + "s"
        parts.append(f"the {noun} {verb} {quoted}")
    if len(parts) == 1:
        return f"Assume {parts[0]}."
    return "Assume " + ", ".join(parts[:-1]) + f", and {parts[-1]}."


def build_request(description: str, filled: FilledCommand, style: int, rng: random.Random) -> str:
    """Render a natural-language request that contains the slot values."""
    desc = description.strip().rstrip(".:")
    desc = lower_first(desc)

    inline_ids: set[int] = set()
    used_keys: set[str] = set()
    for value in filled.values:
        if not value.is_value or not value.inline_key or value.inline_key in used_keys:
            continue
        pattern = INLINE_PATTERNS.get(value.inline_key)
        if pattern and pattern.search(desc):
            desc = pattern.sub(lambda _m, v=value.value: v, desc, count=1)
            inline_ids.add(id(value))
            used_keys.add(value.inline_key)

    remaining = [v for v in filled.values if id(v) not in inline_ids]
    context = _context_sentence(remaining)
    negative = desc.startswith(("don't ", "do not ", "never ", "avoid "))
    # "How do I don't ...?" is ungrammatical; use a neutral framing for negatives.
    core = (
        f"Command for this task: {desc}"
        if negative
        else REQUEST_STYLES[style].format(desc=desc)
    )
    request = f"{core} {context}".strip() if context else core

    # Small stylistic variation in how the context is attached.
    if context and rng.random() < 0.25:
        request = f"{context} {core}"
    return re.sub(r"\s+", " ", request).strip()


def make_variant(
    example: Example,
    style: int,
    seed: int,
    cfg: Config,
) -> dict[str, Any] | None:
    rng = random.Random(seed)
    filled = SlotFiller(rng).fill(example.command_template)
    if not filled.command or "{{" in filled.command or "}}" in filled.command:
        return None
    if len(filled.command) > cfg.data.max_command_chars:
        return None
    request = build_request(example.description, filled, style, rng)
    if not request or len(request) > 400:
        return None
    if not filled.command.strip():
        return None
    digest = hashlib.sha1(
        f"{example.page}|{example.description}|{filled.command}|{style}|{seed}".encode()
    ).hexdigest()[:16]
    return {
        "id": digest,
        "page": example.page,
        "platform": example.platform,
        "description": example.description,
        "request": request,
        "command": filled.command,
        "style": style,
        "template": example.command_template,
        "slots": [
            {"value": v.value, "label": v.label, "category": v.category}
            for v in filled.values
            if v.is_value
        ],
        "source": example.source,
    }


def _sample(items: list[Any], n: int, rng: random.Random) -> list[Any]:
    if len(items) <= n:
        return list(items)
    return rng.sample(items, n)


def _dedupe(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str]] = set()
    out = []
    for row in rows:
        key = (row["request"], row["command"])
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def _percentiles(values: list[int]) -> dict[str, float]:
    if not values:
        return {}
    values = sorted(values)
    def pct(p: float) -> float:
        idx = min(len(values) - 1, max(0, int(round(p * (len(values) - 1)))))
        return float(values[idx])
    return {
        "mean": round(statistics.mean(values), 2),
        "p50": pct(0.50),
        "p90": pct(0.90),
        "p99": pct(0.99),
        "max": float(values[-1]),
    }


def download_tldr(dest: Path, redownload: bool = False) -> None:
    if (dest / "pages").exists() and not redownload:
        return
    if dest.exists():
        if any(dest.iterdir()):
            shutil.rmtree(dest)
        else:
            dest.rmdir()
    ensure_dir(dest.parent)
    print(f"Cloning tldr-pages into {dest} ...")
    subprocess.run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "--filter=blob:none",
            "--sparse",
            "https://github.com/tldr-pages/tldr.git",
            str(dest),
        ],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(dest), "sparse-checkout", "set", "pages/common", "pages/linux"],
        check=True,
    )


def prepare_dataset(cfg: Config, redownload: bool = False) -> dict[str, Any]:
    started = time.perf_counter()
    source = Path(cfg.data.source_dir)
    download_tldr(source, redownload=redownload)
    grouped, load_stats = load_examples(cfg)
    page_splits = split_pages(grouped, cfg)

    out_dir = ensure_dir(cfg.data.processed_dir)

    train_pages = page_splits["train"]
    val_pages = page_splits["val"]
    test_pages = page_splits["test"]

    train_examples = [ex for page in train_pages for ex in grouped[page]]
    val_examples = [ex for page in val_pages for ex in grouped[page]]
    test_examples = [ex for page in test_pages for ex in grouped[page]]

    # Training variants: seen pages, seen styles, varied slots.
    train_rows: list[dict[str, Any]] = []
    for ex in train_examples:
        for k in range(cfg.data.variants_per_example):
            style = TRAIN_STYLES[k % len(TRAIN_STYLES)]
            seed = _variant_seed(cfg.data.seed, f"{ex.page}|{ex.description}|train", k)
            row = make_variant(ex, style, seed, cfg)
            if row:
                train_rows.append(row)
    train_rows = _dedupe(train_rows)

    # Evaluation sets are sampled from base examples, then one variant each.
    eval_n = cfg.data.eval_examples

    def make_eval_rows(
        examples: list[Example], styles: list[int], n: int, key: str
    ) -> list[dict[str, Any]]:
        chosen = _sample(examples, n, random.Random(f"{cfg.data.seed}:{key}:sample"))
        rows = []
        for i, ex in enumerate(chosen):
            style = styles[i % len(styles)]
            seed = _variant_seed(cfg.data.seed, f"{ex.page}|{ex.description}|{key}", i)
            row = make_variant(ex, style, seed, cfg)
            if row:
                rows.append(row)
        return _dedupe(rows)

    val_rows = make_eval_rows(train_examples, [VAL_STYLE], eval_n, "val")
    test_seen_rows = make_eval_rows(train_examples, [TEST_SEEN_STYLE], eval_n, "test_seen")
    test_paraphrase_rows = make_eval_rows(train_examples, PARAPHRASE_STYLES, eval_n, "test_paraphrase")
    val_unseen_rows = make_eval_rows(val_examples, TRAIN_STYLES, eval_n, "val_unseen")
    test_unseen_rows = make_eval_rows(test_examples, TRAIN_STYLES, eval_n, "test_unseen")
    test_unseen_para_rows = make_eval_rows(
        test_examples, PARAPHRASE_STYLES, eval_n, "test_unseen_paraphrase"
    )

    splits = {
        "train": train_rows,
        "val": val_rows,
        "val_unseen": val_unseen_rows,
        "test_seen": test_seen_rows,
        "test_paraphrase": test_paraphrase_rows,
        "test_unseen": test_unseen_rows,
        "test_unseen_paraphrase": test_unseen_para_rows,
    }
    for name, rows in splits.items():
        for row in rows:
            row["split"] = name
        write_jsonl(out_dir / f"{name}.jsonl", rows)

    sample_rows = _sample(test_seen_rows, 20, random.Random(cfg.data.seed + 7))
    write_jsonl(out_dir / "samples.jsonl", sample_rows)

    all_commands = [row["command"] for rows in splits.values() for row in rows]
    all_requests = [row["request"] for rows in splits.values() for row in rows]
    risk_counts = Counter(assess_risk(cmd).level for cmd in all_commands)
    slot_categories = Counter(
        slot["category"] for rows in splits.values() for row in rows for slot in row["slots"]
    )

    validity_sample = _sample(all_commands, min(2000, len(all_commands)), random.Random(cfg.data.seed))
    valid_count = sum(1 for cmd in validity_sample if is_valid_bash(cmd))

    manifest = {
        "source": "tldr-pages (https://github.com/tldr-pages/tldr)",
        "license": "CC-BY-4.0",
        "source_commit": _git_commit(source),
        "config": cfg.to_dict()["data"],
        **load_stats,
        "pages": {
            "train": len(train_pages),
            "val": len(val_pages),
            "test": len(test_pages),
        },
        "splits": {name: len(rows) for name, rows in splits.items()},
        "styles": {
            "train": TRAIN_STYLES,
            "val": [VAL_STYLE],
            "test_seen": [TEST_SEEN_STYLE],
            "paraphrase": PARAPHRASE_STYLES,
        },
        "request_chars": _percentiles([len(r) for r in all_requests]),
        "command_chars": _percentiles([len(c) for c in all_commands]),
        "risk": dict(risk_counts),
        "slot_categories": dict(slot_categories.most_common()),
        "bash_valid_sample": {
            "n": len(validity_sample),
            "valid": valid_count,
            "rate": round(valid_count / max(1, len(validity_sample)), 4),
        },
        "prep_seconds": round(time.perf_counter() - started, 2),
    }
    write_json(out_dir / "dataset_manifest.json", manifest)

    readme = out_dir / "README.md"
    readme.write_text(
        "# TinyShell processed data\n\n"
        "Generated by `tinyshell-prepare`. Do not edit by hand.\n\n"
        "| Split | Examples |\n|---|---:|\n"
        + "\n".join(f"| {name} | {len(rows):,} |" for name, rows in splits.items())
        + "\n\n"
        "Splits:\n"
        "- `train`: seen commands, seen prompt styles, varied slot values.\n"
        "- `val`: seen commands, held-out prompt style (used for checkpoint selection).\n"
        "- `val_unseen`: unseen commands, seen styles.\n"
        "- `test_seen`: seen commands, held-out style, new slots.\n"
        "- `test_paraphrase`: seen commands, unseen prompt styles.\n"
        "- `test_unseen`: unseen commands, seen styles.\n"
        "- `test_unseen_paraphrase`: unseen commands + unseen styles (hardest).\n\n"
        "See `dataset_manifest.json` for counts, filtering, and source commit.\n",
        encoding="utf-8",
    )

    print("Prepared TinyShell data")
    for name, rows in splits.items():
        print(f"  {name:24s} {len(rows):>8,}")
    print(f"  request chars p50/p95: {manifest['request_chars'].get('p50')}/"
          f"{manifest['request_chars'].get('p99')}")
    print(f"  bash -n valid sample: {manifest['bash_valid_sample']['rate']:.1%}")
    print(f"  elapsed: {human_time(manifest['prep_seconds'])}")
    return manifest


def _git_commit(path: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return out.stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return None


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Prepare the TinyShell dataset")
    parser.add_argument("--config", default="configs/tinyshell.yaml")
    parser.add_argument("--redownload", action="store_true", help="Re-clone tldr-pages")
    parser.add_argument("--force", action="store_true", help="Rebuild processed data")
    args = parser.parse_args(argv)

    cfg = Config.from_yaml(args.config)
    out = Path(cfg.data.processed_dir) / "train.jsonl"
    if out.exists() and not args.force:
        print(f"{out} already exists; use --force to rebuild.")
        return
    prepare_dataset(cfg, redownload=args.redownload)


if __name__ == "__main__":
    main()
