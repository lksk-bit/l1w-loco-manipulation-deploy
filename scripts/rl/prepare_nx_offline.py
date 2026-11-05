#!/usr/bin/env python3
"""Download a Python 3.10/aarch64 base bundle on an online development PC.

Uses only the standard library. --cpu adds official CPU torch 2.8. --jetson-cu126
adds NVIDIA staff-provided torch and cuSPARSELt for JetPack 6.2/CUDA 12.6.
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import tarfile
import urllib.request
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import parse_qs, unquote, urljoin, urlsplit, urlunsplit

PACKAGES = (
    ("pip", "25.2", "py3-none-any.whl"),
    ("setuptools", "80.9.0", "py3-none-any.whl"),
    ("wheel", "0.45.1", "py3-none-any.whl"),
    ("numpy", "1.26.4", "cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl"),
)
TORCH_PACKAGES = (
    ("filelock", "3.19.1", "py3-none-any.whl"),
    ("typing-extensions", "4.15.0", "py3-none-any.whl"),
    ("sympy", "1.14.0", "py3-none-any.whl"),
    ("mpmath", "1.3.0", "py3-none-any.whl"),
    ("networkx", "3.4.2", "py3-none-any.whl"),
    ("jinja2", "3.1.6", "py3-none-any.whl"),
    (
        "MarkupSafe",
        "3.0.2",
        "cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl",
    ),
    ("fsspec", "2025.9.0", "py3-none-any.whl"),
)
CPU_TORCH_NAME = "torch-2.8.0+cpu-cp310-cp310-manylinux_2_28_aarch64.whl"
CPU_INDEX = "https://download.pytorch.org/whl/cpu/torch/"
TORCH_NAME = "torch-2.8.0-cp310-cp310-linux_aarch64.whl"
TORCH_URL = f"https://media.githubusercontent.com/media/davidl-nv/torch/refs/heads/main/torch-2.8/{TORCH_NAME}"
TORCH_SHA256 = "9619cfb78383f3765b626a048cf75b967b07c8e6d3f5c93a2ddb43c768f3d331"
REPO_NAME = "cusparselt-local-tegra-repo-ubuntu2204-0.8.1_0.8.1-1_arm64.deb"
REPO_URL = f"https://developer.download.nvidia.com/compute/cusparselt/0.8.1/local_installers/{REPO_NAME}"
REPO_SHA256 = "c3445239a57331eedcd57cc760d3220983e0fbe7458cc12811bb2e8fa7fb60ad"
RUNTIME_PACKAGES = (
    "libcusparselt0-cuda-12_0.8.1.1-1_arm64.deb",
    "libcusparselt0-dev-cuda-12_0.8.1.1-1_arm64.deb",
    "cusparselt-cuda-12_0.8.1.1-1_arm64.deb",
)


def fetch(url):
    request = urllib.request.Request(url, headers={"User-Agent": "l1w-offline-setup/1"})
    return urllib.request.urlopen(request, timeout=60)


def download(url, destination, expected=None):
    if (
        destination.is_file()
        and expected
        and hashlib.sha256(destination.read_bytes()).hexdigest() == expected
    ):
        print(f"Cached: {destination.name}", flush=True)
        return expected
    print(f"Downloading: {destination.name}", flush=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    with fetch(url) as response, temporary.open("wb") as stream:
        shutil.copyfileobj(response, stream)
    digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
    if expected and digest != expected:
        raise ValueError(f"SHA-256 mismatch: {destination.name}")
    temporary.replace(destination)
    return digest


class WheelLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self.links.extend(value for key, value in attrs if key == "href" and value)


def add_cpu_files(output, manifest):
    with fetch(CPU_INDEX) as response:
        parser = WheelLinks()
        parser.feed(response.read().decode())
    matches = [
        link
        for link in parser.links
        if unquote(urlsplit(link).path).endswith("/" + CPU_TORCH_NAME)
    ]
    if len(matches) != 1:
        raise ValueError(
            "Expected exactly one official torch 2.8 CPU cp310/aarch64 wheel"
        )
    parts = urlsplit(urljoin(CPU_INDEX, matches[0]))
    if parts.scheme != "https" or parts.hostname not in {
        "download.pytorch.org",
        "download-r2.pytorch.org",
    }:
        raise ValueError("Unexpected official CPU wheel origin")
    expected = parse_qs(parts.fragment)["sha256"][0]
    url = urlunsplit(parts._replace(fragment=""))
    digest = download(url, output / "wheels" / CPU_TORCH_NAME, expected)
    manifest["files"][f"wheels/{CPU_TORCH_NAME}"] = {"sha256": digest, "url": url}
    manifest["torch_included"] = True
    manifest["torch_device"] = "cpu"
    manifest["torch_requirements"] = [
        "torch==2.8.0+cpu",
        "numpy==1.26.4",
        *(f"{name}=={version}" for name, version, _ in TORCH_PACKAGES),
    ]


def add_gpu_files(output, manifest):
    if not shutil.which("dpkg-deb"):
        raise ValueError(
            "GPU bundle preparation requires dpkg-deb on the development PC"
        )
    digest = download(TORCH_URL, output / "wheels" / TORCH_NAME, TORCH_SHA256)
    manifest["files"][f"wheels/{TORCH_NAME}"] = {"sha256": digest, "url": TORCH_URL}
    vendor = output / "nvidia"
    vendor.mkdir(exist_ok=True)
    repo = vendor / REPO_NAME
    download(REPO_URL, repo, REPO_SHA256)
    extracted = output / "vendor-repo"
    subprocess.run(["dpkg-deb", "--extract", str(repo), str(extracted)], check=True)
    manifest["system_debs"] = []
    for name in RUNTIME_PACKAGES:
        source = next(extracted.rglob(name))
        destination = vendor / name
        shutil.copyfile(source, destination)
        relative = str(destination.relative_to(output))
        manifest["system_debs"].append(relative)
        manifest["files"][relative] = {
            "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
            "url": REPO_URL,
            "repo_sha256": REPO_SHA256,
        }
    manifest["torch_included"] = True
    manifest["torch_device"] = "cuda"
    manifest["cuda"] = "12.6"
    manifest["torch_requirements"] = [
        "torch==2.8.0",
        "numpy==1.26.4",
        *(f"{name}=={version}" for name, version, _ in TORCH_PACKAGES),
    ]
    runtime_script = Path(__file__).with_name("install_nx_gpu_runtime.sh")
    shutil.copyfile(runtime_script, output / runtime_script.name)
    manifest["files"][runtime_script.name] = {
        "sha256": hashlib.sha256(runtime_script.read_bytes()).hexdigest(),
        "source": "scripts/rl/install_nx_gpu_runtime.sh",
    }


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    variant = parser.add_mutually_exclusive_group()
    variant.add_argument(
        "--cpu",
        action="store_true",
        help="Add official ARM64 CPU torch; no CUDA system packages",
    )
    variant.add_argument(
        "--jetson-cu126",
        action="store_true",
        help="Add Jetson GPU torch and its offline dependencies",
    )
    args = parser.parse_args()
    if args.cpu:
        name = "l1w-nx-torch28-cpu"
    else:
        name = "l1w-nx-torch28-cu126" if args.jetson_cu126 else "l1w-nx-python310-base"
    output = (args.output or root / "build" / name).expanduser().resolve()
    wheels = output / "wheels"
    wheels.mkdir(parents=True, exist_ok=True)
    manifest = {
        "python": "3.10",
        "machine": "aarch64",
        "torch_included": False,
        "files": {},
    }
    packages = PACKAGES + TORCH_PACKAGES if args.cpu or args.jetson_cu126 else PACKAGES
    for name, version, suffix in packages:
        with fetch(f"https://pypi.org/pypi/{name}/{version}/json") as response:
            metadata = json.load(response)
        matches = [
            item for item in metadata["urls"] if item["filename"].endswith(suffix)
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one wheel for {name} {version} {suffix}"
            )
        wheel = matches[0]
        destination = wheels / wheel["filename"]
        digest = download(wheel["url"], destination, wheel["digests"]["sha256"])
        manifest["files"][str(destination.relative_to(output))] = {
            "sha256": digest,
            "url": wheel["url"],
        }
    url = "https://bootstrap.pypa.io/get-pip.py"
    manifest["files"]["get-pip.py"] = {
        "sha256": download(url, output / "get-pip.py"),
        "url": url,
    }
    if args.cpu:
        add_cpu_files(output, manifest)
    elif args.jetson_cu126:
        add_gpu_files(output, manifest)
    shutil.copyfile(
        Path(__file__).with_name("install_nx_offline.py"), output / "install.py"
    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    archive = output.with_suffix(".tar.gz")
    with tarfile.open(archive, "w:gz") as stream:
        # Include only the completed bundle, never stale/partial downloads.
        for name in [*manifest["files"], "manifest.json", "install.py"]:
            stream.add(output / name, arcname=str(Path(output.name) / name))
    print(f"Ready: {archive} ({archive.stat().st_size / 1e6:.1f} MB)")
    if args.cpu:
        print(
            "CPU torch pack. On NX, run install.py; no CUDA packages or sudo required."
        )
    elif args.jetson_cu126:
        print(
            "Jetson CUDA 12.6 pack. Install the system GPU runtime, then run install.py."
        )
    else:
        print(
            "Base environment only: pip, setuptools, wheel, NumPy. Torch is not included."
        )


if __name__ == "__main__":
    main()
