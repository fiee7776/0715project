from __future__ import annotations

import argparse
import hashlib
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


def copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")

    model_dir = output / "model"
    mesh_dir = model_dir / "meshes"
    policy_dir = output / "policy"
    mesh_dir.mkdir(parents=True)
    policy_dir.mkdir(parents=True)

    tree = ET.parse(args.xml)
    compiler = tree.getroot().find("compiler")
    source_mesh_dir = args.xml.parent / (compiler.get("meshdir", "") if compiler is not None else "")
    mesh_files = sorted({element.get("file") for element in tree.findall(".//mesh") if element.get("file")})
    for filename in mesh_files:
        copy_file(source_mesh_dir / filename, mesh_dir / filename)

    copy_file(args.xml, model_dir / "g1_29dof.xml")
    copy_file(args.policy, policy_dir / "policy.onnx")
    copy_file(args.config, policy_dir / "deploy.yaml")
    copy_file(args.motion, output / "input5_v2_50hz.csv")
    copy_file(args.runner, output / "run_input5_mujoco.py")

    readme = """Input5 Windows MuJoCo package

Run from PowerShell in this directory:

  .\\.venv\\Scripts\\python.exe run_input5_mujoco.py `
    --model model\\g1_29dof.xml `
    --policy policy\\policy.onnx `
    --config policy\\deploy.yaml `
    --motion input5_v2_50hz.csv

Use --headless for a non-visual verification run.
Policy rate: 50 Hz. Physics rate: 200 Hz.
"""
    (output / "README.txt").write_text(readme, encoding="utf-8")

    manifest_paths = sorted(path for path in output.rglob("*") if path.is_file())
    manifest = "\n".join(f"{sha256(path)}  {path.relative_to(output).as_posix()}" for path in manifest_paths)
    (output / "SHA256SUMS.txt").write_text(manifest + "\n", encoding="ascii")

    archive = shutil.make_archive(str(output), "zip", root_dir=output.parent, base_dir=output.name)
    print(f"Copied {len(mesh_files)} referenced meshes")
    print(f"Created {archive}")


if __name__ == "__main__":
    main()
