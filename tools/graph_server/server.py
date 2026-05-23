from __future__ import annotations

import argparse
import importlib.util
import re
from pathlib import Path
from types import ModuleType
from typing import Any
from urllib.parse import urlencode

from flask import Flask, jsonify, redirect, request

GRAPH_DIR = Path(__file__).resolve().parent / "graphs"
GRAPH_NAME_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]*$")


def create_server() -> Flask:
    server = Flask(__name__)

    @server.get("/")
    def index():
        return redirect("/graphs/initial_positions_los_3d")

    @server.get("/api/graphs")
    def list_graphs():
        return jsonify({"graphs": [path.stem for path in _graph_files()]})

    return server


def create_dash_app(server: Flask):
    try:
        from dash import Dash, Input, Output, dcc, html
    except ImportError as exc:
        raise RuntimeError(
            "The graph server requires dash and plotly. "
            "Install them with `python -m pip install dash plotly`."
        ) from exc

    app = Dash(
        __name__,
        server=server,
        routes_pathname_prefix="/_dash/",
        requests_pathname_prefix="/_dash/",
        suppress_callback_exceptions=True,
    )
    app.layout = html.Div(
        [
            dcc.Location(id="location"),
            html.Div(id="page"),
        ],
        style={"height": "100vh", "width": "100vw", "margin": "0"},
    )

    @server.get("/graphs/<path:filename>")
    def graph_page(filename: str):
        name = _normalize_graph_name(filename)
        query = {"graph": name}
        if request.args.get("data"):
            query["data"] = request.args["data"]
        if request.args.get("records"):
            query["records"] = request.args["records"]
        return f"""<!doctype html>
<html>
  <head>
    <title>{name}</title>
    <style>
      html, body, #dash-root, iframe {{
        width: 100%;
        height: 100%;
        margin: 0;
        border: 0;
      }}
    </style>
  </head>
  <body>
    <iframe src="/_dash/?{urlencode(query)}"></iframe>
  </body>
</html>
"""

    @app.callback(Output("page", "children"), Input("location", "search"))
    def render_graph(search: str):
        from dash import dcc, html
        from urllib.parse import parse_qs

        query = parse_qs((search or "").lstrip("?"))
        name = _normalize_graph_name(query.get("graph", ["initial_positions_los_3d"])[0])
        data = query.get("data", [None])[0]
        records = query.get("records", [None])[0]
        try:
            module = _load_graph_module(name)
            figure = module.create_figure(data=data, records=records)
        except Exception as exc:
            return html.Pre(
                f"Failed to render graph {name!r}: {exc}",
                style={"padding": "24px", "whiteSpace": "pre-wrap", "fontFamily": "monospace"},
            )
        return dcc.Graph(figure=figure, style={"height": "100vh", "width": "100vw"})

    return app


def _graph_files() -> list[Path]:
    return sorted(
        path
        for path in GRAPH_DIR.glob("*.py")
        if path.name != "__init__.py" and path.is_file()
    )


def _normalize_graph_name(filename: str) -> str:
    value = str(filename).strip()
    if value.endswith(".py"):
        value = value[:-3]
    if "/" in value or "\\" in value or not GRAPH_NAME_RE.match(value):
        raise ValueError(
            "Graph filename must be a Python filename from tools/graph_server/graphs, "
            "for example initial_positions_los_3d or initial_positions_los_3d.py"
        )
    return value


def _load_graph_module(name: str) -> ModuleType:
    graph_path = GRAPH_DIR / f"{name}.py"
    if not graph_path.exists():
        known = ", ".join(path.stem for path in _graph_files())
        raise FileNotFoundError(f"Unknown graph {name!r}. Available graphs: {known}")
    spec = importlib.util.spec_from_file_location(f"tools.graph_server.graphs.{name}", graph_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import graph module {graph_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    create_figure = getattr(module, "create_figure", None)
    if not callable(create_figure):
        raise TypeError(f"{graph_path} must define create_figure(**kwargs)")
    return module


def _main() -> None:
    parser = argparse.ArgumentParser(description="Serve local Plotly/Dash graph modules.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    server = create_server()
    create_dash_app(server)
    server.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    _main()
