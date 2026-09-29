"""Architecture rules from ARCH.md, enforced so a violation fails CI.

- app/api/ is HTTP only: no SQLAlchemy, Redis, ORM, repository or infra imports.
- ORM models (app.db.models) are imported only by repositories (and app.db itself).
- Repositories never raise HTTP errors or touch the cache.
- No secret values are hard-coded anywhere in app/.
"""

import ast
import re
from pathlib import Path

APP = Path(__file__).resolve().parents[2] / "app"


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def _py(sub: str) -> list[Path]:
    return sorted((APP / sub).rglob("*.py")) if sub else sorted(APP.rglob("*.py"))


def _matches(module: str, forbidden: tuple[str, ...]) -> bool:
    """True if module is one of the forbidden packages or inside one."""
    return any(module == f or module.startswith(f + ".") for f in forbidden)


def _violations(files, forbidden):
    found = {
        str(f.relative_to(APP)): sorted(m for m in _imports(f) if _matches(m, forbidden))
        for f in files
    }
    return {k: v for k, v in found.items() if v}


def test_api_layer_is_http_only():
    forbidden = ("sqlalchemy", "redis", "app.db", "app.infra", "app.repositories")
    assert _violations(_py("api"), forbidden) == {}


def test_orm_models_only_used_by_repositories():
    allowed = {"repositories", "db"}
    users = [f for f in _py("") if f.relative_to(APP).parts[0] not in allowed]
    assert _violations(users, ("app.db.models",)) == {}


def test_repositories_do_not_raise_http_errors_or_touch_cache():
    assert _violations(_py("repositories"), ("fastapi", "fastapi_cache", "app.infra.cache")) == {}


def test_no_hard_coded_secrets():
    literal = re.compile(
        # name (optionally annotated) assigned a quoted literal: x_secret = "...", pw: str = "..."
        r"""(password|passwd|secret|token|api_key)(?!url)\w*\s*(?::[^=\n]*)?[:=]\s*["'][^"'\s]{4,}["']""",
        re.IGNORECASE,
    )
    hits = [
        f"{f.relative_to(APP)}:{n}: {line.strip()}"
        for f in _py("")
        for n, line in enumerate(f.read_text().splitlines(), 1)
        if literal.search(line)
    ]
    assert hits == []
