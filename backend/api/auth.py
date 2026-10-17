"""Autenticación del panel (RQNF19).

Secreto compartido leído de la variable de entorno `PANEL_TOKEN` y comparado
en tiempo constante (`hmac.compare_digest`). Nunca se escribe en el
repositorio. Sin la variable, el panel arranca abierto (modo laboratorio) y
lo indica en `/api/health`.
"""
from __future__ import annotations

import hmac

from flask import (Blueprint, jsonify, redirect, render_template, request)

from .context import runtime

bp = Blueprint("auth", __name__)
TOKEN_TTL = 8 * 3600
PUBLIC_PATHS = {"/", "/login", "/api/health"}


def _ok(supplied: str) -> bool:
    token = runtime().panel_token
    return bool(supplied) and hmac.compare_digest(supplied, token)


@bp.before_app_request
def guard():
    if not runtime().panel_token:
        return None
    if request.path in PUBLIC_PATHS or request.path.startswith("/static/"):
        return None
    if request.path == "/api/stream" and request.remote_addr in ("127.0.0.1", "::1"):
        return None
    supplied = (request.headers.get("X-Panel-Token", "")
                or request.cookies.get("panel", ""))
    if _ok(supplied):
        return None
    if request.path.startswith("/api/"):
        return jsonify({"error": "token de panel requerido (RQNF19)"}), 401
    return redirect("/login")


@bp.route("/login", methods=["GET", "POST"])
def login():
    token = runtime().panel_token
    if not token:
        return redirect("/")
    if request.method == "POST":
        if _ok(request.form.get("token", "")):
            resp = redirect("/")
            resp.set_cookie("panel", token, httponly=True, max_age=TOKEN_TTL,
                            samesite="Lax")
            return resp
        return render_template("login.html", error="Token incorrecto"), 401
    return render_template("login.html", error=None)
