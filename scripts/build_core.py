#!/usr/bin/env python3
"""Build the focused ``data/core`` subset used by the TinyShell Core experiment.

The broad dataset contains thousands of command families, which is far more
than a 2M-parameter model can learn from ~9M tokens.  This script keeps pages
whose first command token is in a curated list of common Linux/DevOps commands,
then packs the result with the same document-aligned layout as the broad run.

Run from the repository root after ``make data`` and ``make tokenizer``:

    python scripts/build_core.py
    python -m tinyshell.train --config configs/core.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tinyshell.dataset import pack_split
from tinyshell.text import EOS_TOKEN, PAD_TOKEN
from tinyshell.tokenizer import load_tokenizer, special_id
from tinyshell.utils import ensure_dir, write_json

CORE_COMMANDS = {
    "ls", "find", "grep", "sed", "awk", "tar", "git", "docker", "systemctl",
    "ps", "kill", "df", "du", "chmod", "chown", "curl", "wget", "ssh", "scp",
    "rsync", "apt", "npm", "pip", "python", "node", "make", "cargo", "go",
    "java", "mvn", "gradle", "kubectl", "helm", "terraform", "aws", "gcloud",
    "az", "ffmpeg", "pdftoppm", "unzip", "zip", "mount", "umount", "dd", "ln",
    "cp", "mv", "rm", "mkdir", "rmdir", "touch", "cat", "head", "tail", "less",
    "more", "sort", "uniq", "wc", "cut", "tr", "diff", "patch", "xargs", "tee",
    "watch", "crontab", "journalctl", "service", "iptables", "ufw", "netstat",
    "ss", "lsof", "ping", "traceroute", "dig", "nslookup", "host", "ip",
    "ifconfig", "nmcli", "openssl", "gpg", "ssh-keygen", "useradd", "usermod",
    "passwd", "groupadd", "visudo", "su", "sudo", "env", "alias", "history",
    "which", "whereis", "locate", "man", "info", "help", "echo", "printf",
    "date", "sleep", "screen", "tmux", "vim", "emacs",
}

SPLITS = ["train", "val", "test_seen", "test_paraphrase"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--output-dir", default="data/core")
    parser.add_argument("--tokenizer", default="artifacts/tokenizer")
    parser.add_argument("--seq-len", type=int, default=256)
    args = parser.parse_args()

    src = Path(args.processed_dir)
    out = ensure_dir(Path(args.output_dir))

    for split in SPLITS:
        rows = []
        for line in (src / f"{split}.jsonl").read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row["page"].split()[0] in CORE_COMMANDS:
                rows.append(row)
        (out / f"{split}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        print(f"{split}: {len(rows)} examples")

    # train.py expects every configured split to exist.  The core experiment
    # only evaluates seen pages, so alias the unseen files to their seen twins.
    (out / "val_unseen.jsonl").write_text(
        (out / "val.jsonl").read_text(encoding="utf-8"), encoding="utf-8"
    )
    (out / "test_unseen.jsonl").write_text(
        (out / "test_seen.jsonl").read_text(encoding="utf-8"), encoding="utf-8"
    )
    (out / "test_unseen_paraphrase.jsonl").write_text(
        (out / "test_paraphrase.jsonl").read_text(encoding="utf-8"), encoding="utf-8"
    )

    tokenizer = load_tokenizer(args.tokenizer)
    eos_id = special_id(tokenizer, EOS_TOKEN)
    pad_id = special_id(tokenizer, PAD_TOKEN)
    packed = ensure_dir(out / "packed")
    manifest = {"seq_len": args.seq_len, "packing": "document_aligned", "splits": {}}
    for split in ["train", "val", "val_unseen", "test_seen", "test_paraphrase"]:
        stats = pack_split(
            out / f"{split}.jsonl", packed, tokenizer, args.seq_len, eos_id, pad_id
        )
        manifest["splits"][split] = stats
        print(f"{split}: {stats}")
    write_json(packed / "manifest.json", manifest)


if __name__ == "__main__":
    main()
