from __future__ import annotations

from .common import run_cli


def main() -> int:
    return run_cli("beihang_paper", "Run beihang_paper_sim scenarios.")


if __name__ == "__main__":
    raise SystemExit(main())
