"""Training loop for TinyShell."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from .config import Config
from .dataset import PackedDataset, pack_all_splits
from .model import TinyLM
from .text import (
    EOS_TOKEN,
    assess_risk,
    clean_generation,
    is_valid_bash,
    render_prompt,
)
from .tokenizer import decode, encode, load_tokenizer, special_id
from .utils import (
    ensure_dir,
    environment_info,
    get_git_commit,
    human_time,
    read_jsonl,
    save_torch,
    set_seed,
    setup_logging,
    write_json,
    write_yaml,
)


class Trainer:
    def __init__(self, cfg: Config, run_dir: Path, resume: Path | None = None):
        self.cfg = cfg
        self.run_dir = ensure_dir(run_dir)
        self.ckpt_dir = ensure_dir(run_dir / "checkpoints")
        self.logger = setup_logging(run_dir / "train.log")
        self.device = torch.device(cfg.train.device)
        torch.set_num_threads(cfg.train.num_threads)

        self.tokenizer = load_tokenizer(cfg.data.tokenizer_dir)
        self.eos_id = special_id(self.tokenizer, EOS_TOKEN)

        self.model = TinyLM(cfg.model).to(self.device)
        self.optimizer = self.model.configure_optimizer(
            lr=cfg.train.lr,
            weight_decay=cfg.train.weight_decay,
            betas=(cfg.train.beta1, cfg.train.beta2),
        )

        packed = Path(cfg.data.processed_dir) / "packed"
        self.train_ds = PackedDataset(
            packed / "train_ids.npy",
            packed / "train_labels.npy",
            packed / "train_doc_ids.npy",
            packed / "train_position_ids.npy",
        )
        self.val_ds = PackedDataset(
            packed / "val_ids.npy",
            packed / "val_labels.npy",
            packed / "val_doc_ids.npy",
            packed / "val_position_ids.npy",
        )
        self.val_unseen_ds = PackedDataset(
            packed / "val_unseen_ids.npy",
            packed / "val_unseen_labels.npy",
            packed / "val_unseen_doc_ids.npy",
            packed / "val_unseen_position_ids.npy",
        )

        self.micro_per_epoch = len(self.train_ds) // cfg.train.batch_size
        if self.micro_per_epoch == 0:
            raise ValueError("Training set is smaller than one batch")
        self.steps_per_epoch = max(1, self.micro_per_epoch // cfg.train.grad_accum_steps)
        self.total_steps = self.steps_per_epoch * cfg.train.epochs
        if cfg.train.max_steps is not None:
            self.total_steps = min(self.total_steps, cfg.train.max_steps)
        self.total_steps = max(1, self.total_steps)

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=self._lr_lambda
        )

        self.step = 0
        self.epoch = 0
        self.batch_in_epoch = 0
        self.best_val = float("inf")
        self.evals_since_improve = 0
        self._log_state: dict[str, float] = {}

        if resume is not None:
            self.load_checkpoint(resume)

        self._write_run_metadata()
        self.logger.info(
            "model: %.2fM params (%.2fM non-embedding), vocab=%d, seq_len=%d",
            self.model.num_parameters / 1e6,
            self.model.num_parameters_non_embedding / 1e6,
            cfg.model.vocab_size,
            cfg.model.max_seq_len,
        )
        self.logger.info(
            "data: %d train chunks, %d val chunks, %d optimizer steps/epoch, %d total steps",
            len(self.train_ds),
            len(self.val_ds),
            self.steps_per_epoch,
            self.total_steps,
        )

    def _lr_lambda(self, step: int) -> float:
        warmup = max(1, min(self.cfg.train.warmup_steps, self.total_steps // 5))
        if step < warmup:
            return step / warmup
        ratio = self.cfg.train.min_lr / self.cfg.train.lr
        progress = (step - warmup) / max(1, self.total_steps - warmup)
        progress = min(1.0, max(0.0, progress))
        return ratio + (1.0 - ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    def _write_run_metadata(self) -> None:
        if not (self.run_dir / "config.yaml").exists():
            write_yaml(self.run_dir / "config.yaml", self.cfg.to_dict())
        else:
            write_yaml(self.run_dir / "config_resume.yaml", self.cfg.to_dict())
        write_json(self.run_dir / "env.json", environment_info())
        write_json(
            self.run_dir / "model_summary.json",
            {
                "parameters": self.model.num_parameters,
                "non_embedding_parameters": self.model.num_parameters_non_embedding,
                "config": self.cfg.model.__dict__,
            },
        )
        commit = get_git_commit()
        if commit:
            (self.run_dir / "git_commit.txt").write_text(commit + "\n", encoding="utf-8")
        tokenizer_out = ensure_dir(self.run_dir / "tokenizer")
        for name in ("tokenizer.json", "tokenizer_config.json", "stats.json"):
            src = Path(self.cfg.data.tokenizer_dir) / name
            if src.exists():
                shutil.copy2(src, tokenizer_out / name)
        readme = self.run_dir / "README.md"
        if not readme.exists():
            readme.write_text(
                f"# Run: {self.run_dir.name}\n\n"
                f"- Model: {self.model.num_parameters:,} parameters\n"
                f"- Context: {self.cfg.model.max_seq_len}\n"
                f"- Train chunks: {len(self.train_ds):,}\n"
                f"- Val chunks: {len(self.val_ds):,}\n"
                f"- Optimizer steps: {self.total_steps:,}\n\n"
                "Files: `config.yaml`, `env.json`, `train.log`, `metrics.jsonl`, "
                "`samples.jsonl`, `checkpoints/`, `tokenizer/`, `plots/`.\n",
                encoding="utf-8",
            )

    def _append_metrics(self, row: dict[str, Any]) -> None:
        with open(self.run_dir / "metrics.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    def _append_sample(self, row: dict[str, Any]) -> None:
        with open(self.run_dir / "samples.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    def save_checkpoint(self, name: str) -> Path:
        path = self.ckpt_dir / f"{name}.pt"
        save_torch(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "step": self.step,
                "epoch": self.epoch,
                "batch_in_epoch": self.batch_in_epoch,
                "best_val": self.best_val,
                "config": self.cfg.to_dict(),
            },
            path,
        )
        return path

    def load_checkpoint(self, path: str | Path) -> None:
        path = Path(path)
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        self.step = int(ckpt.get("step", 0))
        self.epoch = int(ckpt.get("epoch", 0))
        self.batch_in_epoch = int(ckpt.get("batch_in_epoch", 0))
        self.best_val = float(ckpt.get("best_val", float("inf")))
        self.logger.info("resumed %s at step=%d epoch=%d", path, self.step, self.epoch)

    @torch.inference_mode()
    def evaluate_loss(self, dataset: PackedDataset) -> float:
        self.model.eval()
        loader = DataLoader(dataset, batch_size=self.cfg.train.batch_size, shuffle=False)
        total_loss = 0.0
        total_tokens = 0
        max_len = self.cfg.model.max_seq_len
        for x, y, doc_ids, position_ids in loader:
            x = x[:, :max_len][:, :-1].contiguous().to(self.device)
            y = y[:, 1:max_len].contiguous().to(self.device)
            doc_ids = doc_ids[:, :max_len][:, :-1].contiguous().to(self.device)
            position_ids = position_ids[:, :max_len][:, :-1].contiguous().to(self.device)
            _, loss = self.model(x, y, doc_ids=doc_ids, position_ids=position_ids)
            if loss is None or not torch.isfinite(loss):
                continue
            active = int((y != -100).sum().item())
            total_loss += float(loss.item()) * active
            total_tokens += active
        self.model.train()
        if total_tokens == 0:
            return float("nan")
        return total_loss / total_tokens

    def generate_samples(self) -> None:
        sample_path = Path(self.cfg.data.processed_dir) / "samples.jsonl"
        if not sample_path.exists():
            return
        rows = list(read_jsonl(sample_path))[:5]
        self.model.eval()
        for row in rows:
            prompt = render_prompt(row["request"])
            ids = encode(self.tokenizer, prompt)
            input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
            output = self.model.generate(
                input_ids,
                max_new_tokens=32,
                temperature=0.0,
                eos_id=self.eos_id,
            )
            new_ids = output[0, len(ids) :].tolist()
            text = decode(self.tokenizer, new_ids)
            command = clean_generation(text)
            risk = assess_risk(command)
            self._append_sample(
                {
                    "step": self.step,
                    "page": row.get("page"),
                    "request": row["request"],
                    "reference": row["command"],
                    "generated": command,
                    "raw": text,
                    "valid_bash": is_valid_bash(command),
                    "risk": risk.level,
                }
            )
        self.model.train()

    def _log_progress(self, now: float, train_loss: float, grad_norm: float) -> None:
        elapsed = now - self._log_state.get("last_time", now)
        tokens = self._log_state.get("tokens", 0.0)
        tok_s = tokens / elapsed if elapsed > 0 else 0.0
        lr = self.scheduler.get_last_lr()[0]
        self.logger.info(
            "step %6d | loss %.4f | lr %.2e | grad %.2f | %6.0f tok/s | %s",
            self.step,
            train_loss,
            lr,
            grad_norm,
            tok_s,
            human_time(now - self._log_state.get("start_time", now)),
        )
        self._append_metrics(
            {
                "step": self.step,
                "epoch": self.epoch,
                "split": "train",
                "loss": train_loss,
                "lr": lr,
                "grad_norm": grad_norm,
                "tokens_per_sec": tok_s,
                "elapsed": now - self._log_state.get("start_time", now),
            }
        )
        self._log_state["last_time"] = now
        self._log_state["tokens"] = 0.0
        self._log_state["loss_sum"] = 0.0
        self._log_state["loss_count"] = 0

    def train(self) -> dict[str, Any]:
        self.model.train()
        loader = DataLoader(
            self.train_ds,
            batch_size=self.cfg.train.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.cfg.train.num_workers,
        )
        self._log_state = {
            "last_time": time.perf_counter(),
            "start_time": time.perf_counter(),
            "tokens": 0.0,
            "loss_sum": 0.0,
            "loss_count": 0,
        }
        grad_accum = self.cfg.train.grad_accum_steps
        stopped_early = False
        start_epoch = self.epoch
        start_batch = self.batch_in_epoch

        for epoch in range(start_epoch, self.cfg.train.epochs):
            self.epoch = epoch
            for batch_idx, (x, y, doc_ids, position_ids) in enumerate(loader):
                if self.cfg.train.max_steps is not None and self.step >= self.cfg.train.max_steps:
                    break
                if epoch == start_epoch and batch_idx < start_batch:
                    continue
                max_len = self.cfg.model.max_seq_len
                x = x[:, :max_len][:, :-1].contiguous().to(self.device)
                y = y[:, 1:max_len].contiguous().to(self.device)
                doc_ids = doc_ids[:, :max_len][:, :-1].contiguous().to(self.device)
                position_ids = position_ids[:, :max_len][:, :-1].contiguous().to(self.device)
                _, loss = self.model(
                    x,
                    y,
                    doc_ids=doc_ids,
                    position_ids=position_ids,
                    first_token_weight=self.cfg.train.first_token_weight,
                )
                if loss is None or not torch.isfinite(loss):
                    self.logger.warning("non-finite loss at step %d; skipping batch", self.step)
                    self.optimizer.zero_grad(set_to_none=True)
                    continue
                (loss / grad_accum).backward()
                self._log_state["loss_sum"] += float(loss.item())
                self._log_state["loss_count"] += 1
                self._log_state["tokens"] += x.numel()

                if (batch_idx + 1) % grad_accum == 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.cfg.train.grad_clip
                    )
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.step += 1
                    self.batch_in_epoch = batch_idx + 1

                    now = time.perf_counter()
                    if self.step % self.cfg.train.log_interval == 0:
                        count = max(1, self._log_state["loss_count"])
                        self._log_progress(now, self._log_state["loss_sum"] / count, float(grad_norm))

                    if self.step % self.cfg.train.eval_interval == 0:
                        val_loss = self.evaluate_loss(self.val_ds)
                        val_ppl = math.exp(min(val_loss, 20)) if math.isfinite(val_loss) else float("nan")
                        improved = math.isfinite(val_loss) and val_loss < self.best_val
                        if improved:
                            self.best_val = val_loss
                            self.evals_since_improve = 0
                            self.save_checkpoint("best")
                        else:
                            self.evals_since_improve += 1
                        self.logger.info(
                            "eval step %d | val_loss %.4f | ppl %.2f | best %.4f%s",
                            self.step,
                            val_loss,
                            val_ppl,
                            self.best_val,
                            " *" if improved else "",
                        )
                        self._append_metrics(
                            {
                                "step": self.step,
                                "epoch": self.epoch,
                                "split": "val",
                                "loss": val_loss,
                                "perplexity": val_ppl,
                                "best_val": self.best_val,
                            }
                        )
                        if (
                            self.cfg.train.early_stopping_patience > 0
                            and self.evals_since_improve >= self.cfg.train.early_stopping_patience
                        ):
                            self.logger.info("early stopping at step %d", self.step)
                            stopped_early = True
                            break

                    if self.step % self.cfg.train.sample_interval == 0:
                        self.generate_samples()

                    if self.step % self.cfg.train.save_interval == 0:
                        self.save_checkpoint("last")

                if stopped_early:
                    break
            self.batch_in_epoch = 0
            if stopped_early:
                break
            if self.cfg.train.max_steps is not None and self.step >= self.cfg.train.max_steps:
                break

        self.save_checkpoint("last")
        self.save_checkpoint("final")
        if not (self.ckpt_dir / "best.pt").exists():
            self.save_checkpoint("best")
        final_metrics = {
            "step": self.step,
            "epoch": self.epoch,
            "best_val": self.best_val,
            "val_loss": self.evaluate_loss(self.val_ds),
            "val_unseen_loss": self.evaluate_loss(self.val_unseen_ds),
            "stopped_early": stopped_early,
            "train_chunks": len(self.train_ds),
            "val_chunks": len(self.val_ds),
            "parameters": self.model.num_parameters,
            "elapsed_seconds": time.perf_counter() - self._log_state.get("start_time", time.perf_counter()),
        }
        write_json(self.run_dir / "final_metrics.json", final_metrics)
        self.logger.info("finished: %s", json.dumps(final_metrics))
        return final_metrics


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train TinyShell")
    parser.add_argument("--config", default="configs/tinyshell.yaml")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    cfg = Config.from_yaml(args.config)
    if args.max_steps is not None:
        cfg.train.max_steps = args.max_steps
    if args.device is not None:
        cfg.train.device = args.device
    set_seed(cfg.seed)
    torch.set_num_threads(cfg.train.num_threads)

    packed = Path(cfg.data.processed_dir) / "packed" / "train_ids.npy"
    if not packed.exists():
        pack_all_splits(cfg)

    if args.resume:
        run_dir = Path(args.resume).resolve().parent.parent
    else:
        run_id = args.run_id or f"{time.strftime('%Y%m%d-%H%M%S')}-{cfg.run_name}"
        run_dir = Path(cfg.train.output_dir) / run_id
    trainer = Trainer(cfg, run_dir, resume=Path(args.resume) if args.resume else None)
    trainer.train()


if __name__ == "__main__":
    main()
