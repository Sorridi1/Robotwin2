#!/usr/bin/env python3
"""Read-only dependency/source checks; simulator imports and CUDA are opt-in."""

import argparse
import importlib
import importlib.metadata as metadata
import importlib.util
import json
from pathlib import Path
import sys


PACKAGES = {
    "torch": "torch",
    "torchvision": "torchvision",
    "numpy": "numpy",
    "sapien": "sapien",
    "mplib": "mplib",
    "pytorch3d": "pytorch3d",
    "nvidia-curobo": "curobo",
    "hydra-core": "hydra",
    "omegaconf": "omegaconf",
    "gymnasium": "gymnasium",
    "diffusers": "diffusers",
    "timm": "timm",
    "zarr": "zarr",
    "maniflow": "maniflow",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--imports", action="store_true", help="Import core modules without creating a scene")
    parser.add_argument("--cuda", action="store_true", help="Also query torch CUDA availability")
    parser.add_argument("--local-source", action="store_true", help="Use this checkout exactly as the FPO++ entry points do")
    args = parser.parse_args()
    if args.local_source:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from fpopp.runtime import configure_paths
        configure_paths()

    robotwin_root = Path(__file__).resolve().parents[3]
    expected_maniflow = robotwin_root / "policy/ManiFlow/ManiFlow/maniflow"
    report = {
        "executable": sys.executable,
        "python": sys.version.split()[0],
        "robotwin_root": str(robotwin_root),
        "expected_maniflow_source": str(expected_maniflow),
        "packages": {},
        "imports": {},
        "issues": [],
    }
    issues = report["issues"]
    if sys.version_info[:2] != (3, 10):
        issues.append("Expected the existing RoboTwin2 Python 3.10 stack; activate the target environment.")

    for package, module in PACKAGES.items():
        entry = {}
        report["packages"][package] = entry
        try:
            dist = metadata.distribution(package)
            entry["version"] = dist.version
            direct_url = dist.read_text("direct_url.json")
            if direct_url:
                entry["direct_url"] = json.loads(direct_url)
        except metadata.PackageNotFoundError:
            issues.append(f"Missing distribution: {package}")
        except (ValueError, OSError) as exc:
            issues.append(f"Cannot read {package} metadata: {type(exc).__name__}: {exc}")

        try:
            spec = importlib.util.find_spec(module)
            entry["module_origin"] = spec.origin if spec else None
            locations = list(spec.submodule_search_locations or ()) if spec else []
            entry["module_search_locations"] = locations
            if spec is None:
                issues.append(f"Module not found: {module}; check the environment and editable source path.")
            elif module == "maniflow":
                # ManiFlow is a namespace package in this checkout: origin is None.
                sources = locations or ([spec.origin] if spec.origin else [])
                if not sources or any(
                    not Path(source).resolve().is_relative_to(expected_maniflow.resolve())
                    for source in sources
                ):
                    issues.append(f"maniflow resolves outside this checkout: {sources}")
        except (ImportError, AttributeError, ValueError, OSError) as exc:
            issues.append(f"Cannot locate {module}: {type(exc).__name__}: {exc}")

    if args.imports:
        for module in ("torch", "torchvision", "sapien.core", "mplib", "pytorch3d", "curobo", "maniflow"):
            try:
                imported = importlib.import_module(module)
                report["imports"][module] = {"ok": True, "file": getattr(imported, "__file__", None)}
            except Exception as exc:
                report["imports"][module] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                issues.append(f"Import failed: {module}")

    if args.cuda:
        try:
            torch = importlib.import_module("torch")
            available = torch.cuda.is_available()
            report["cuda"] = {
                "available": available,
                "torch_cuda_version": torch.version.cuda,
                "device_count": torch.cuda.device_count(),
            }
            if not available:
                issues.append("PyTorch CUDA is unavailable for this invocation.")
        except Exception as exc:
            issues.append(f"CUDA check failed: {type(exc).__name__}: {exc}")

    report["ok"] = not issues
    report["scope"] = "Package/source checks only; no simulator scene or rollout was created."
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
