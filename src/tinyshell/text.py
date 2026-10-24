"""Prompt formatting, output normalization, command parsing, and safety checks."""

from __future__ import annotations

import re
import shlex
import subprocess
from collections import Counter
from dataclasses import dataclass

SPECIAL_TOKENS = ["<|pad|>", "<|eos|>", "<|system|>", "<|user|>", "<|assistant|>"]
PAD_TOKEN = "<|pad|>"
EOS_TOKEN = "<|eos|>"
SYSTEM_TOKEN = "<|system|>"
USER_TOKEN = "<|user|>"
ASSISTANT_TOKEN = "<|assistant|>"

SYSTEM_PROMPT = (
    "You are a Linux shell command assistant. Convert the request into exactly one "
    "command. Reply with the command only, no explanation."
)


def render_prompt(request: str) -> str:
    """Render the exact prompt used for training and inference."""
    return (
        f"{SYSTEM_TOKEN}\n{SYSTEM_PROMPT}\n"
        f"{USER_TOKEN}\n{request.strip()}\n"
        f"{ASSISTANT_TOKEN}\n"
    )


def render_training_text(request: str, command: str) -> str:
    """Prompt + completion + EOS, used for tokenizer training and corpus inspection."""
    return render_prompt(request) + command.strip() + EOS_TOKEN


_SPECIAL_RE = re.compile(r"<\|[^|]*\|>")


def strip_special_tokens(text: str) -> str:
    return _SPECIAL_RE.sub("", text)


def clean_generation(text: str) -> str:
    """Turn a raw model completion into a single candidate command."""
    text = strip_special_tokens(text).strip()
    text = re.sub(r"^```[a-zA-Z0-9_+-]*\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    line = lines[0]
    line = re.sub(r"^(?:\$|>)\s+", "", line)
    line = re.sub(r"^(?:command|assistant|output)\s*:\s*", "", line, flags=re.IGNORECASE)
    return line.strip().strip("`").strip()


def normalize_command(command: str) -> str:
    """Canonical form for exact-match scoring (does not change semantics)."""
    command = clean_generation(command)
    command = re.sub(r"\s+", " ", command).strip()
    command = command.rstrip(";").strip()
    return command


def command_tokens(command: str) -> list[str]:
    try:
        return shlex.split(command)
    except ValueError:
        return command.split()


def command_name(command: str) -> str:
    """Best-effort first executable token, skipping common wrappers."""
    tokens = command_tokens(normalize_command(command))
    wrappers = {"sudo", "env", "time", "nohup", "exec", "command", "builtin", "nice", "ionice"}
    for token in tokens:
        if token in wrappers:
            continue
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", token):
            continue
        return token
    return ""


def token_f1(prediction: str, reference: str, include_name: bool = False) -> float:
    """Multiset token F1 over command arguments."""
    pred = command_tokens(normalize_command(prediction))
    ref = command_tokens(normalize_command(reference))
    if not include_name:
        pred = pred[1:]
        ref = ref[1:]
    if not pred and not ref:
        return 1.0
    if not pred or not ref:
        return 0.0
    common = Counter(pred) & Counter(ref)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred)
    recall = overlap / len(ref)
    return 2 * precision * recall / (precision + recall)


def is_valid_bash(command: str, timeout: float = 3.0) -> bool:
    """Parse a command with ``bash -n``; never executes it."""
    command = normalize_command(command)
    if not command:
        return False
    try:
        result = subprocess.run(
            ["bash", "-n"],
            input=command,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        return result.returncode == 0
    except (subprocess.SubprocessError, OSError):
        return False


@dataclass(frozen=True)
class RiskAssessment:
    level: str  # safe | caution | destructive
    reasons: tuple[str, ...]

    @property
    def is_safe(self) -> bool:
        return self.level == "safe"


_DESTRUCTIVE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("recursive force delete", re.compile(r"\brm\s+(-[A-Za-z]*[rf][A-Za-z]*\s+)+")),
    ("disk overwrite", re.compile(r"\bdd\b.*\bof=/dev/")),
    ("filesystem creation", re.compile(r"\bmkfs(?:\.\w+)?\b")),
    ("system shutdown", re.compile(r"\b(?:shutdown|reboot|poweroff|halt)\b")),
    ("partition editing", re.compile(r"\b(?:wipefs|fdisk|parted|sfdisk)\b")),
    ("raw device write", re.compile(r">\s*/dev/(?:sd|nvme|hd|vd)")),
    ("unsafe recursive permissions", re.compile(r"\bchmod\s+-R\s+777\s+/")),
    ("fork bomb", re.compile(r":\(\)\s*\{")),
    ("truncate file", re.compile(r"\btruncate\s+-s\s*0\b")),
    ("secure delete", re.compile(r"\bshred\b")),
]

_CAUTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("privilege escalation", re.compile(r"\bsudo\b")),
    ("file deletion", re.compile(r"\brm\b")),
    ("file move/overwrite", re.compile(r"\b(?:mv|cp)\b")),
    ("permission change", re.compile(r"\b(?:chmod|chown)\b")),
    ("process termination", re.compile(r"\b(?:kill|killall|pkill)\b")),
    ("service change", re.compile(r"\bsystemctl\s+(?:stop|disable|mask)\b")),
    ("firewall change", re.compile(r"\b(?:iptables|nft|ufw)\b")),
    ("git history rewrite", re.compile(r"\bgit\s+(?:reset\s+--hard|clean\s+-[A-Za-z]*f)")),
    ("pipe to shell", re.compile(r"\b(?:curl|wget)\b.*\|\s*(?:ba|z|fi)?sh\b")),
    ("find delete", re.compile(r"\bfind\b.*\s-delete\b")),
    ("xargs delete", re.compile(r"\bxargs\b.*\brm\b")),
    ("output redirection", re.compile(r"(?<![0-9])>\s*[^&|]")),
]


def assess_risk(command: str) -> RiskAssessment:
    """Classify a command for warning purposes. This is not a security boundary."""
    command = normalize_command(command)
    reasons: list[str] = []
    for label, pattern in _DESTRUCTIVE_PATTERNS:
        if pattern.search(command):
            reasons.append(label)
    if reasons:
        return RiskAssessment("destructive", tuple(reasons))
    for label, pattern in _CAUTION_PATTERNS:
        if pattern.search(command):
            reasons.append(label)
    if reasons:
        return RiskAssessment("caution", tuple(reasons))
    return RiskAssessment("safe", ())


def lower_first(text: str) -> str:
    """Lowercase the first character unless it starts an acronym/proper token."""
    text = text.strip()
    if not text:
        return text
    first = text.split()[0]
    if len(first) > 1 and first.isupper():
        return text
    return text[0].lower() + text[1:]
