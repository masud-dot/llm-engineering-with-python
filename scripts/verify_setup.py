"""Confirm the environment is ready. Makes no network calls."""

import importlib
import sys

from llmapp.config import Settings

REQUIRED = ("pydantic", "pydantic_settings", "pytest")
MIN_PYTHON = (3, 12)


def check_python() -> bool:
    ok = sys.version_info >= MIN_PYTHON
    got = ".".join(str(n) for n in sys.version_info[:3])
    want = ".".join(str(n) for n in MIN_PYTHON)
    print(f"[{'ok' if ok else 'XX'}] python {got} (need {want}+)")
    return ok


def check_packages() -> bool:
    ok = True
    for name in REQUIRED:
        try:
            mod = importlib.import_module(name)
        except ImportError:
            print(f"[XX] {name} is not installed")
            ok = False
            continue
        version = getattr(mod, "__version__", "?")
        print(f"[ok] {name} {version}")
    return ok


def check_settings() -> bool:
    try:
        settings = Settings()
    except Exception as exc:  # configuration is fatal
        print(f"[XX] settings did not load: {exc}")
        return False
    print(f"[ok] environment  {settings.environment}")
    print(f"[ok] provider     {settings.provider}")
    print(f"[ok] api key set  {settings.api_key is not None}")
    print(f"[ok] cost ceiling ${settings.daily_cost_limit_usd}")
    return True


def main() -> int:
    results = [check_python(), check_packages(), check_settings()]
    if all(results):
        print("\nEnvironment ready.")
        return 0
    print("\nFix the items marked XX above.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
