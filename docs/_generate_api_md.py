from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

from bs4 import BeautifulSoup
from markdownify import markdownify
from pdoc import doc as pdoc_doc
from pdoc import render


ROOT = Path(__file__).resolve().parents[1]
PKG_ROOT = ROOT / "rcpsp_cf_ivfth"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PACKAGE = "rcpsp_cf_ivfth"
OUTPUT = Path(__file__).with_name("api.md")


def ensure_pyomo_stub() -> None:
    if "pyomo.environ" in sys.modules:
        return

    pyomo_pkg = types.ModuleType("pyomo")
    environ = types.ModuleType("pyomo.environ")

    class _DummyConstraintList(list):
        def add(self, expr) -> None:
            self.append(expr)

    class _DummySolver:
        def solve(self, *_, **__):
            return {"status": "ok"}

        def terminate(self):
            return None

    def _solver_factory(*_, **__):
        return _DummySolver()

    environ.ConcreteModel = type("ConcreteModel", (), {})
    environ.Set = type("Set", (), {})
    environ.Var = type("Var", (), {})
    environ.Param = type("Param", (), {})
    environ.NonNegativeReals = object()
    environ.Binary = object()
    environ.Integers = object()
    environ.Reals = object()
    environ.Objective = type("Objective", (), {})
    environ.Constraint = type("Constraint", (), {})
    environ.ConstraintList = _DummyConstraintList
    environ.SolverFactory = _solver_factory
    environ.value = staticmethod(lambda x: x)
    environ.maximize = "maximize"

    pyomo_pkg.environ = environ  # type: ignore[attr-defined]
    sys.modules["pyomo"] = pyomo_pkg
    sys.modules["pyomo.environ"] = environ


def load_sanitized_model() -> None:
    module_name = f"{PACKAGE}.model"
    if module_name in sys.modules:
        return

    ensure_pyomo_stub()
    source_path = PKG_ROOT / "model.py"
    source = source_path.read_text(encoding="utf-8")
    sanitized = source.replace("ConstraintList := []", "ConstraintList = []")

    spec = importlib.util.spec_from_loader(module_name, loader=None)
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    module.__file__ = str(source_path)
    module.RCPSP_CF_IVFTH = type("RCPSP_CF_IVFTH", (), {})  # placeholder for import cycle
    sys.modules[module_name] = module
    exec(compile(sanitized, str(source_path), "exec"), module.__dict__)


def collect_modules(root: str) -> dict[str, pdoc_doc.Module]:
    modules: dict[str, pdoc_doc.Module] = {}
    stack = [pdoc_doc.Module.from_name(root)]
    while stack:
        module = stack.pop()
        if module.modulename in modules:
            continue
        modules[module.modulename] = module
        stack.extend(module.submodules)
    return modules


def build_markdown(root_module: pdoc_doc.Module, all_modules: dict[str, pdoc_doc.Module]) -> str:
    render.configure(docformat="markdown", search=False, show_source=False)
    html = render.html_module(root_module, all_modules)

    soup = BeautifulSoup(html, "html.parser")
    main = soup.find("main", {"class": "pdoc"})
    if main is None:
        raise RuntimeError("Failed to locate pdoc main content.")

    markdown_body = markdownify(str(main), heading_style="ATX")
    header = "# API Reference\n\n> _Generated automatically with pdoc._\n\n"
    return header + markdown_body.strip() + "\n"


def main() -> None:
    load_sanitized_model()
    modules = collect_modules(PACKAGE)
    root = modules[PACKAGE]
    markdown = build_markdown(root, modules)
    OUTPUT.write_text(markdown, encoding="utf-8")


if __name__ == "__main__":
    main()
