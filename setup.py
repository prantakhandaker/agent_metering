"""Build helpers: install agent_metering.pth into site-packages (wheel root)."""

from __future__ import annotations

import shutil
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py as _build_py

_ROOT = Path(__file__).resolve().parent
_PTH_NAME = "agent_metering.pth"


class build_py(_build_py):
    def run(self) -> None:
        super().run()
        src = _ROOT / _PTH_NAME
        if not src.is_file():
            return
        dest_dir = Path(self.build_lib)
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest_dir / _PTH_NAME)


setup(cmdclass={"build_py": build_py})
