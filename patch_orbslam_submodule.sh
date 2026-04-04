#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
orb_root="${repo_root}/thirdParty/ORB_SLAM3"

python - <<'PY' "${orb_root}"
from pathlib import Path
import sys

orb_root = Path(sys.argv[1])

top_cmake = orb_root / "CMakeLists.txt"
python_cmake = orb_root / "python" / "CMakeLists.txt"

top_text = top_cmake.read_text()
needle = "find_package(Eigen3 3.3.7 REQUIRED NO_MODULE)\nfind_package(Pangolin REQUIRED)"
replacement = "find_package(Eigen3 3.3.7 REQUIRED NO_MODULE)\nfind_package(OpenGL REQUIRED COMPONENTS OpenGL EGL)\nfind_package(Pangolin REQUIRED)"
if "find_package(OpenGL REQUIRED COMPONENTS OpenGL EGL)" not in top_text:
    if needle not in top_text:
        raise RuntimeError(f"Did not find expected Pangolin block in {top_cmake}")
    top_text = top_text.replace(needle, replacement, 1)
    top_cmake.write_text(top_text)

py_text = python_cmake.read_text()
replacements = {
    "find_package(NumPy REQUIRED)\nfind_package(PythonLibs 3.11 REQUIRED)\nfind_package(Boost 1.80.0 REQUIRED COMPONENTS python311)": "find_package(PythonInterp REQUIRED)\nfind_package(NumPy REQUIRED)\nfind_package(PythonLibs ${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR} REQUIRED)\nfind_package(Boost 1.80.0 REQUIRED COMPONENTS python${PYTHON_VERSION_MAJOR}${PYTHON_VERSION_MINOR})",
    "DESTINATION lib/python3.11/site-packages": "DESTINATION lib/python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}/site-packages",
}
for old, new in replacements.items():
    if old in py_text:
        py_text = py_text.replace(old, new)
if "find_package(PythonLibs ${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR} REQUIRED)" not in py_text:
    raise RuntimeError(f"Python binding patch did not apply cleanly to {python_cmake}")
python_cmake.write_text(py_text)
PY
