# Copyright (C) 2026 Panagiotis Tirchas
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import os
import subprocess
import sys
from pathlib import Path


def run_variant(use_retrieval: bool, extra_args: list[str]) -> None:
    root = Path(__file__).resolve().parent
    main_path = root / "main.py"

    env = os.environ.copy()
    env["USE_RETRIEVAL"] = "true" if use_retrieval else "false"

    label = "WITH retrieval" if use_retrieval else "WITHOUT retrieval"
    cmd = [sys.executable, str(main_path), *extra_args]

    print("\n" + "=" * 70)
    print(f"Starting run {label}")
    print(f"USE_RETRIEVAL={env['USE_RETRIEVAL']}")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 70 + "\n")

    subprocess.run(cmd, check=True, env=env, cwd=root)


def main() -> None:
    extra_args = sys.argv[1:]

    run_variant(use_retrieval=False, extra_args=extra_args)
    run_variant(use_retrieval=True, extra_args=extra_args)


if __name__ == "__main__":
    main()
