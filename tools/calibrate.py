"""Tune the engine's hardware-dependent settings on this PC (setup's --calibrate).

Three settings depend on the machine more than on the model, and the defaults are right for the PC they were
measured on (a Ryzen 5 7600 + RTX 5070 on PCIe 5):
  --pcie-frac     the share of the experts missing from VRAM that are copied over PCIe and run on the GPU instead of
                  on the CPU.  A fast PCIe link and a slow CPU want more; a laptop's x8 link or a fast CPU want less.
  --spec-min-p    how sure the draft layer must be to extend a verify window by another guess.  A slower CPU pays more
                  per extra window row (more experts per window), so it wants a higher floor.
  --pool-workers  the CPU threads that compute experts.  Every physical core is not always best: on hybrid CPUs the
                  efficiency cores can make the whole window wait for them.
  --spec          how many guesses the draft layer makes per check (setup's 4).  A card that finishes a check in
                  about the same time whatever its width (a big one at a high hit rate) gains from more guesses;
                  a slower one pays for the extra rows.
The first two are measured through one engine (per-request `strata_tune` keys); the worker count and the draft
depth need a restart per value.  Decode speed only: the prompt path streams every expert whatever these settings say.

A setting is kept only when it beats the default by more than MIN_GAIN in an interleaved re-measurement - the
adaptive expert tier and the OS make single measurements noisy by a few percent.

    python tools/calibrate.py strata-q2_0.json        # measure and print; setup.py --calibrate also saves it
"""
from __future__ import annotations

import json
import statistics
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))

MIN_GAIN = 0.03                    # a setting must beat the default by this much to be kept
SWEEP_PASSES = 2                   # each sweep setting is measured this many times, the settings interleaved
PCIE_FRACS = (0.0, 0.2, 0.35, 0.55, 0.75)
SPEC_MIN_PS = (0.3, 0.5, 0.7)
SPECS = (5, 6)                     # draft depths tried beyond setup's --spec 4 (the engine caps the window at 8)
MAX_NEW = 128
PROMPTS = (
    "Write a Python function that merges two sorted lists into one sorted list, with a docstring and two tests.",
    "Explain in two paragraphs how a refrigerator moves heat from inside to outside.",
    "List twelve European capitals with one sentence about each.",
)


def chat_ids(tok, text: str) -> list[int]:
    return tok.encode(f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
                      parse_special=True)


def arg_value(args: list[str], flag: str) -> str | None:
    return args[args.index(flag) + 1] if flag in args and args.index(flag) + 1 < len(args) else None


def with_arg(args: list[str], flag: str, value: str | None) -> list[str]:
    """`args` with `flag value` set (replaced if present), or removed when value is None."""
    out = list(args)
    if flag in out:
        i = out.index(flag)
        del out[i:i + 2]
    if value is not None:
        out += [flag, value]
    return out


def worker_candidates(default: int) -> list[int]:
    """The engine's own count, and fewer: two thirds and a half (at least 2), without repeats."""
    c = [default]
    for w in (round(default * 2 / 3), round(default / 2)):
        if w >= 2 and w not in c:
            c.append(w)
    return c


def pick(measured: dict, default_key, min_gain: float = MIN_GAIN):
    """The key with the best median tok/s, or `default_key` unless the best beats it by more than min_gain."""
    med = {k: statistics.median(v) for k, v in measured.items() if v}
    if not med or default_key not in med:
        return default_key
    best = max(med, key=med.get)
    return best if med[best] > med[default_key] * (1.0 + min_gain) else default_key


class Session:
    """One running engine: measure decode tok/s for a setting (the median of the prompts' rates)."""

    def __init__(self, engine, ids_list, sampling: dict | None = None):
        self.engine = engine
        self.ids_list = ids_list
        # the config's own sampling block when it has one: a setting tuned for greedy decoding (every draft the
        # model agrees with is taken) is not the one for sampled decoding, where fewer drafts survive
        self.sampling = {**sampling, "seed": 1} if sampling else {"temperature": 0}

    def rate(self, tune: dict | None = None) -> float:
        rates = []
        for ids in self.ids_list:
            sampling = dict(self.sampling)
            if tune:
                sampling["strata_tune"] = tune
            n = sum(1 for t in self.engine.generate(ids, MAX_NEW, sampling, threading.Event()) if t is not None)
            ms = (self.engine.last or {}).get("decode_ms") or 0.0
            if n > 8 and ms > 0:
                rates.append(n / (ms / 1000.0))
        return statistics.median(rates) if rates else 0.0

    def warm_up(self, rounds: int = 2):
        for _ in range(rounds):
            self.rate()


def run(cfg: dict, say=print, start_engine=None) -> dict:
    """Measure on the engine `cfg` describes; returns {"settings": {flag: value}, "report": {...}}.
    `start_engine(args)` returns a started engine (serve.server.StrataEngine or a stand-in in tests)."""
    if start_engine is None:
        from serve.server import StrataEngine, child_env

        def start_engine(args):
            return StrataEngine(cfg["exe"], args, cwd=cfg.get("cwd"), log=cfg.get("log"), env=child_env(cfg))
    import strata_tokenizer as ST
    tpath = Path(cfg["tokenizer"])
    vocab = json.loads((tpath / "vocab.json").read_text(encoding="utf-8"))
    toks = [None] * len(vocab)
    for t, i in vocab.items():
        toks[i] = t
    tok = ST.Tokenizer(toks, (tpath / "merges.txt").read_text(encoding="utf-8").split("\n"),
                       json.loads((tpath / "token_type.json").read_text()))
    ids_list = [chat_ids(tok, p) for p in PROMPTS]
    sampling = {k: v for k, v in (cfg.get("sampling") or {}).items() if k in ("temperature", "top_p", "top_k", "min_p")}
    if sampling:
        say("  Measured with the config's sampling (" + ", ".join(f"{k} {v}" for k, v in sampling.items()) + ")")
    return measure(cfg["args"], ids_list, start_engine, say, sampling or None)


def measure(base_args: list[str], ids_list, start_engine, say=print, sampling: dict | None = None) -> dict:
    t0 = time.time()
    report: dict = {}
    say("  Loading the model for the measurements ...")
    base_args = apply(base_args, {})                   # the product defaults: what the measurements must beat
    eng = start_engine(base_args)
    try:
        info = dict(getattr(eng, "info", {}) or {})
        d_pcie = float(info.get("pcie_frac", 0.55))
        d_minp = float(info.get("spec_min_p", 0.5))
        d_workers = int(info.get("pool_workers", 0)) or None
        s = Session(eng, ids_list, sampling)
        s.warm_up(3)
        # The speed drifts over a session by more than the settings differ (the adaptive expert tier keeps moving
        # experts, the card's clocks and the OS wander), so a sweep is never one measurement per setting in a
        # row: every setting is measured SWEEP_PASSES times with the settings interleaved, and the median counts.

        def sweep(keys, tune_of, label):
            got = {k: [] for k in keys}
            for _ in range(SWEEP_PASSES):
                for k in keys:
                    got[k].append(s.rate(tune_of(k)))
            for k in keys:
                say(f"    {label} {k:.2f}: {statistics.median(got[k]):.1f} tok/s  ({', '.join(f'{r:.1f}' for r in got[k])})")
            return got

        # 1. the PCIe share, at the default draft floor
        by_pcie = sweep(sorted(set(PCIE_FRACS) | {round(d_pcie, 2)}),
                        lambda f: {"pcie_frac": f, "spec_min_p": d_minp}, "PCIe share")
        best_pcie = max(by_pcie, key=lambda k: statistics.median(by_pcie[k]))
        # 2. the draft floor, at that share
        by_minp = sweep(sorted(set(SPEC_MIN_PS) | {round(d_minp, 2)}),
                        lambda p: {"pcie_frac": best_pcie, "spec_min_p": p}, "draft floor")
        best_minp = max(by_minp, key=lambda k: statistics.median(by_minp[k]))
        # 3. the winner against the default, interleaved, three times each - the decision, so it is shown
        dflt, cand = (round(d_pcie, 2), round(d_minp, 2)), (best_pcie, best_minp)
        confirm = {dflt: [], cand: []}
        if cand != dflt:
            for _ in range(3):
                for k in (dflt, cand):
                    confirm[k].append(s.rate({"pcie_frac": k[0], "spec_min_p": k[1]}))
            for k, name in ((dflt, "the defaults"), (cand, "the candidate")):
                say(f"    {name} (PCIe share {k[0]:.2f}, draft floor {k[1]:.2f}): {statistics.median(confirm[k]):.1f} tok/s  "
                    f"({', '.join(f'{r:.1f}' for r in confirm[k])})")
        # the candidate must also win most of its interleaved pairs: a median 3% ahead on three pairs of which it
        # lost two is one lucky measurement, not a setting
        wins = sum(c > d for c, d in zip(confirm[cand], confirm[dflt])) if cand != dflt else 0
        chosen = pick(confirm, dflt) if cand != dflt and wins * 2 > len(confirm[dflt]) else dflt
        report.update(default={"pcie_frac": dflt[0], "spec_min_p": dflt[1], "pool_workers": d_workers},
                      pcie_sweep={str(k): v for k, v in by_pcie.items()},
                      min_p_sweep={str(k): v for k, v in by_minp.items()},
                      confirm={f"{k[0]}/{k[1]}": v for k, v in confirm.items()})
    finally:
        close(eng)
    settings = {}
    if chosen != dflt:
        settings["--pcie-frac"] = f"{chosen[0]:.2f}"
        settings["--spec-min-p"] = f"{chosen[1]:.2f}"
    base_rate = statistics.median(confirm[chosen]) if confirm.get(chosen) else None
    # 4. fewer CPU workers (a restart each), with the chosen settings
    if d_workers and len(worker_candidates(d_workers)) > 1:
        tuned = with_arg(with_arg(base_args, "--pcie-frac", f"{chosen[0]:.2f}"), "--spec-min-p", f"{chosen[1]:.2f}")
        by_workers = {}
        for w in worker_candidates(d_workers):
            say(f"  Measuring with {w} CPU workers (restarts the engine) ...")
            e = start_engine(with_arg(tuned, "--pool-workers", None if w == d_workers else str(w)))
            try:
                sw = Session(e, ids_list, sampling)
                sw.warm_up(1)
                by_workers[w] = [sw.rate(), sw.rate()]
                say(f"    {w} workers: {statistics.median(by_workers[w]):.1f} tok/s")
            finally:
                close(e)
        w_best = pick(by_workers, d_workers)
        report["workers"] = {str(k): v for k, v in by_workers.items()}
        if w_best != d_workers:
            settings["--pool-workers"] = str(w_best)
            base_rate = statistics.median(by_workers[w_best])
        elif by_workers.get(d_workers):
            base_rate = statistics.median(by_workers[d_workers])
    # 5. the draft depth, with everything chosen so far (a restart each).  The default's figure is the worker
    # step's when that ran (the same procedure), else measured here.
    d_spec = int(arg_value(base_args, "--spec") or DEFAULTS["--spec"])
    tuned = with_arg(with_arg(with_arg(base_args, "--pcie-frac", f"{chosen[0]:.2f}"), "--spec-min-p", f"{chosen[1]:.2f}"),
                     "--pool-workers", settings.get("--pool-workers"))

    def restart_rate(args, label) -> list:
        say(f"  Measuring with {label} (restarts the engine) ...")
        e = start_engine(args)
        try:
            se = Session(e, ids_list, sampling)
            se.warm_up(1)
            return [se.rate(), se.rate()]
        finally:
            close(e)

    by_spec = {}
    for sp in (d_spec, *SPECS):
        if sp in by_spec or sp < d_spec:
            continue
        if sp == d_spec and base_rate is not None and d_workers:
            by_spec[sp] = [base_rate]                   # the worker step measured the default depth already
            continue
        by_spec[sp] = restart_rate(with_arg(tuned, "--spec", str(sp)), f"{sp} draft guesses per check")
        say(f"    --spec {sp}: {statistics.median(by_spec[sp]):.1f} tok/s")
    s_best = pick(by_spec, d_spec)
    report["spec"] = {str(k): v for k, v in by_spec.items()}
    if s_best != d_spec:
        settings["--spec"] = str(s_best)
        base_rate = statistics.median(by_spec[s_best])
    elif by_spec.get(d_spec):
        base_rate = statistics.median(by_spec[d_spec])
    report["seconds"] = round(time.time() - t0)
    report["tok_s"] = round(base_rate, 1) if base_rate else None
    return {"settings": settings, "report": report}


def close(eng):
    proc = getattr(eng, "proc", None)
    if proc is None:
        return
    try:
        proc.stdin.write("QUIT\n")
        proc.stdin.flush()
        proc.stdin.close()
        proc.wait(60)
    except Exception:
        proc.kill()


DEFAULTS = {"--pcie-frac": None, "--spec-min-p": "0.5", "--pool-workers": None, "--spec": "4"}   # None: the engine's own choice


def apply(args: list[str], settings: dict) -> list[str]:
    """`args` with the calibrated settings; a setting the calibration did not change goes back to the product
    default (setup's --spec-min-p 0.5, the engine's own PCIe share and worker count), so an older calibration's
    values never linger."""
    out = list(args)
    for flag, default in DEFAULTS.items():
        out = with_arg(out, flag, settings.get(flag, default))
    return out


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: calibrate.py <strata-*.json>")
    res = run(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8-sig")))
    print(json.dumps(res, indent=1))
