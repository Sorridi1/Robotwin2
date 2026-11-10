"""Reserve evaluation destinations without overwriting earlier attempts."""

from pathlib import Path


def result_directory(run_dir, epoch, mode):
    if mode not in ("expert", "rl", "expert_fail", "baseline_fail"):
        raise ValueError(f"Unsupported evaluation scene mode: {mode}")
    directory = Path(run_dir) / "eval_results" / f"epoch_{int(epoch)}"
    return directory if mode == "expert" else directory / f"scene_{mode}"


def reserve_results(run_dir, epoch, mode):
    directory = result_directory(run_dir, epoch, mode)
    # Parent epoch folders can already exist because a different mode ran there.
    # Reserve a separate marker for expert mode instead of claiming the whole parent.
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory / "_fpopp_protocol.json"
    if any(item.is_file() for item in directory.iterdir()):
        raise FileExistsError(f"Evaluation results already exist: {directory}. "
                              "Use --eval_output_dir with a new directory to preserve this attempt.")
    try:
        # Exclusive creation also rejects concurrent/repeated attempts, including aborted ones.
        with marker.open("x", encoding="utf-8") as handle:
            handle.write('{"status": "initializing"}\n')
    except FileExistsError as exc:
        raise FileExistsError(f"An evaluation attempt already owns {directory}. "
                              "Use --eval_output_dir with a new directory.") from exc
    return marker
