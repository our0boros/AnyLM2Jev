"""Frozen-LM loading and the option-label readout.

Two readouts come out of a single forward pass over a choice-form prompt:

* ``last_label_logits`` -- logits at the final token, restricted to the label
  letters (``A``/``B``/``C``...).  This is the stock ``SemIf``-style baseline:
  what the model would emit right after ``Answer:``.
* ``option_hidden`` -- the hidden state at each option's *label* token, i.e. a
  contextualised representation of "A. <option text>".  This is what the trained
  head consumes, and it generalises to any option count.

The model must not be fine-tuned here; this module only reads.
"""

from __future__ import annotations

import contextlib
from typing import Optional, Sequence

import numpy as np
import torch

from .prompts import LABELS

try:  # progress is nice-to-have, never required
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None


def _find_token(offsets, char_off: int) -> Optional[int]:
    """Index of the token whose character span contains ``char_off``."""
    for t, span in enumerate(offsets):
        if span is None:
            continue
        start, end = int(span[0]), int(span[1])
        if start <= char_off < end:
            return t
    return None


def _hidden_size_of(config) -> Optional[int]:
    for obj in (config, getattr(config, "text_config", None)):
        if obj is None:
            continue
        size = getattr(obj, "hidden_size", None)
        if size:
            return int(size)
    return None


def load_lm(
    path: str,
    dtype: str = "bfloat16",
    device: str = "cuda",
    trust_remote_code: bool = False,
):
    """Load a frozen causal LM, trying several Auto classes for exotic archs.

    Qwen3.5 checkpoints are ``*ForConditionalGeneration`` with a hybrid
    linear-attention text tower, so ``AutoModelForCausalLM`` does not always
    accept them.  We try a small ladder of loaders and report every failure.
    """
    import transformers
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=trust_remote_code)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or tok.unk_token
    tok.padding_side = "left"  # so the final position is the real final token

    torch_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype]

    attempts: list[str] = []
    tried: set[tuple[str, str]] = set()
    loaders = ("AutoModelForCausalLM", "AutoModelForImageTextToText", "AutoModel")
    for cls_name in loaders:
        cls = getattr(transformers, cls_name, None)
        if cls is None:
            continue
        for kw in ("dtype", "torch_dtype"):
            if (cls_name, kw) in tried:
                continue
            tried.add((cls_name, kw))
            try:
                model = cls.from_pretrained(
                    path, trust_remote_code=trust_remote_code, **{kw: torch_dtype}
                )
            except TypeError as exc:  # kwarg name not supported in this version
                attempts.append(f"{cls_name}({kw}): TypeError {exc}")
                continue
            except Exception as exc:  # architecture not handled by this loader
                attempts.append(f"{cls_name}({kw}): {type(exc).__name__} {str(exc)[:200]}")
                break

            model.eval()
            for p in model.parameters():
                p.requires_grad_(False)
            model.to(device)
            return model, tok

    raise RuntimeError("could not load model {!r}:\n  ".format(path) + "\n  ".join(attempts))


class OptionReader:
    """Reads label logits and per-option hidden states from a frozen LM."""

    def __init__(self, model, tokenizer, device: str = "cuda", labels=None) -> None:
        self.model = model
        self.tok = tokenizer
        self.device = device
        self.hidden_size = _hidden_size_of(model.config)
        self.labels = list(labels) if labels is not None else list(LABELS)
        self.label_token_ids = [
            tokenizer.encode(label, add_special_tokens=False)[0] for label in self.labels
        ]
        self._use_logits_keep = True

    def _forward(self, prompts: Sequence[str], max_length: int, requires_grad: bool = False):
        enc = self.tok(
            list(prompts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            return_offsets_mapping=True,
            add_special_tokens=True,
        )
        offsets = enc.pop("offset_mapping")
        enc = {k: v.to(self.device) for k, v in enc.items()}

        ctx = contextlib.nullcontext() if requires_grad else torch.no_grad()
        with ctx:
            if self._use_logits_keep:
                try:
                    out = self.model(**enc, output_hidden_states=True, logits_to_keep=1)
                except TypeError:
                    # this transformers/model pair does not support the kwarg
                    self._use_logits_keep = False
                    out = self.model(**enc, output_hidden_states=True)
            else:
                out = self.model(**enc, output_hidden_states=True)

        if getattr(out, "logits", None) is None:
            raise RuntimeError(
                "model produced no logits; this checkpoint is not a causal LM and "
                "cannot be read with OptionReader"
            )
        # left padding => the final position is the real final token of every row
        last_logits = out.logits[:, -1]  # [B, V]
        hidden = out.hidden_states[-1]  # [B, S, H]
        return last_logits, hidden, offsets

    @torch.no_grad()
    def read(
        self,
        prompts: Sequence[str],
        label_offsets: Sequence[Sequence[int]],
        hidden_offsets: Sequence[Sequence[int]] | None = None,
        batch_size: int = 8,
        max_length: int = 512,
        keep_hidden: bool = True,
    ) -> dict:
        """Return label logits and (optionally) per-option hidden states.

        ``label_offsets[i]`` holds the character offsets of the option labels in
        ``prompts[i]``, in display order; the logit baseline is read at the final
        token over those label ids.  ``hidden_offsets[i]`` defaults to
        ``label_offsets[i]`` but should be the *end of each option's text* so that
        the captured hidden state has actually consumed the option.

        Returns padded tensors: ``last_label_logits`` [N, Kmax], ``mask``
        [N, Kmax] and, when requested, ``option_hidden`` [N, Kmax, H] float16.
        """
        n = len(prompts)
        kmax = max(len(o) for o in label_offsets)
        if kmax > len(self.label_token_ids):
            raise ValueError(
                f"{kmax} options but only {len(self.label_token_ids)} labels were "
                "configured on this OptionReader"
            )
        label_ids = torch.tensor(self.label_token_ids, device=self.device)

        last_rows: list[np.ndarray] = []
        mask_rows: list[np.ndarray] = []
        hidden_rows: list[np.ndarray] = []

        starts = list(range(0, n, batch_size))
        if tqdm is not None and len(starts) > 4:
            starts = tqdm(starts, desc="read", unit="batch")

        for start in starts:
            chunk = list(prompts[start : start + batch_size])
            chunk_offsets = list(label_offsets[start : start + batch_size])
            if hidden_offsets is None:
                chunk_hidden_offsets = chunk_offsets
            else:
                chunk_hidden_offsets = list(hidden_offsets[start : start + batch_size])
            last_logits, hidden, offsets = self._forward(chunk, max_length)
            b = len(chunk)

            if keep_hidden and self.hidden_size is None:
                self.hidden_size = int(hidden.shape[-1])

            last = np.full((b, kmax), np.nan, dtype=np.float32)
            m = np.zeros((b, kmax), dtype=bool)
            hid = (
                np.zeros((b, kmax, self.hidden_size), dtype=np.float16)
                if keep_hidden
                else None
            )

            for bi in range(b):
                for k in range(len(chunk_offsets[bi])):
                    last[bi, k] = float(last_logits[bi, label_ids[k]].float())
                    m[bi, k] = True
                    if hid is not None:
                        h_off = chunk_hidden_offsets[bi][k]
                        tok_idx = _find_token(offsets[bi], h_off)
                        if tok_idx is None:
                            raise ValueError(
                                f"hidden offset {h_off} not found in tokenised prompt; "
                                "the prompt was probably truncated (raise --max-length)"
                            )
                        hid[bi, k] = hidden[bi, tok_idx].to(torch.float16).cpu().numpy()

            last_rows.append(last)
            mask_rows.append(m)
            if hid is not None:
                hidden_rows.append(hid)

        result = {
            "last_label_logits": torch.from_numpy(np.concatenate(last_rows, axis=0)),
            "mask": torch.from_numpy(np.concatenate(mask_rows, axis=0)),
        }
        if keep_hidden:
            result["option_hidden"] = torch.from_numpy(np.concatenate(hidden_rows, axis=0))
        return result

    def read_grad(
        self,
        prompts: Sequence[str],
        hidden_offsets: Sequence[Sequence[int]],
        max_length: int = 512,
    ) -> torch.Tensor:
        """Differentiable readout of per-option hidden states.

        Same gathering as :meth:`read` but keeps the autograd graph, which is what
        LoRA training needs.  Assumes a constant option count across the batch.
        Returns ``option_hidden`` [B, K, H] on the model device.
        """
        _last_logits, hidden, offsets = self._forward(prompts, max_length, requires_grad=True)
        rows = []
        for bi, offs in enumerate(hidden_offsets):
            tokens = []
            for char_off in offs:
                tok_idx = _find_token(offsets[bi], char_off)
                if tok_idx is None:
                    raise ValueError(
                        f"hidden offset {char_off} not found; raise --max-length"
                    )
                tokens.append(hidden[bi, tok_idx])
            rows.append(torch.stack(tokens, dim=0))
        return torch.stack(rows, dim=0)

    def _yesno_ids(self) -> tuple[int, int]:
        if not hasattr(self, "_yesno"):
            yes = self.tok.encode("Yes", add_special_tokens=False)[0]
            no = self.tok.encode("No", add_special_tokens=False)[0]
            self._yesno = (yes, no)
        return self._yesno

    @torch.no_grad()
    def read_noul(
        self,
        prompts: Sequence[str],
        batch_size: int = 8,
        max_length: int = 512,
        keep_hidden: bool = True,
    ) -> dict:
        """Option-conditioned yes/no readout.

        One prompt per option; returns ``p_yes`` [N] (sigmoid of the Yes-No logit
        contrast at the final token) and, when requested, the final-token hidden
        state [N, H] which the head consumes.
        """
        yes_id, no_id = self._yesno_ids()
        n = len(prompts)
        p_yes = np.zeros(n, dtype=np.float32)
        hidden_out = None

        starts = list(range(0, n, batch_size))
        if tqdm is not None and len(starts) > 4:
            starts = tqdm(starts, desc="noul", unit="batch")

        for start in starts:
            chunk = list(prompts[start : start + batch_size])
            last_logits, hidden, _offsets = self._forward(chunk, max_length)
            if keep_hidden and self.hidden_size is None:
                self.hidden_size = int(hidden.shape[-1])
            if keep_hidden and hidden_out is None:
                hidden_out = np.zeros((n, self.hidden_size), dtype=np.float16)
            for bi in range(len(chunk)):
                p_yes[start + bi] = torch.sigmoid(
                    (last_logits[bi, yes_id] - last_logits[bi, no_id]).float()
                ).item()
                if hidden_out is not None:
                    hidden_out[start + bi] = (
                        hidden[bi, -1].to(torch.float16).cpu().numpy()
                    )

        result = {"p_yes": torch.from_numpy(p_yes)}
        if hidden_out is not None:
            result["option_hidden"] = torch.from_numpy(hidden_out)
        return result
