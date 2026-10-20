"""Karachi House Price Predictor: Flask, MongoDB and the supplied ML pipeline."""
from __future__ import annotations

import csv
import io
import json
import math
import os
import re
from datetime import datetime, timedelta, timezone
from functools import wraps
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from bson import ObjectId
from dotenv import load_dotenv
from flask import (Flask, Response, abort, flash, make_response, redirect,
                   render_template, request, session, url_for)
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_wtf.csrf import CSRFProtect
from pymongo import ASCENDING, DESCENDING, MongoClient
from pymongo.errors import PyMongoError, ServerSelectionTimeoutError
from werkzeug.security import check_password_hash, generate_password_hash

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")
# The starter project supplies this local credentials file; it is ignored by Git.
load_dotenv(ROOT / "atlas-credentials.env")
app = Flask(__name__)
app.config.update(
    SECRET_KEY=os.getenv("SECRET_KEY", "local-development-key-change-before-deploy"),
    PERMANENT_SESSION_LIFETIME=timedelta(minutes=30),
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=os.getenv("COOKIE_SECURE", "false").lower() == "true",
    MAX_CONTENT_LENGTH=2 * 1024 * 1024,
)
csrf = CSRFProtect(app)
limiter = Limiter(get_remote_address, app=app, default_limits=[])

# Load the supplied trained model and metadata once at startup.
model = joblib.load(ROOT / "house_price_model.pkl")
meta = json.loads((ROOT / "model_meta.json").read_text(encoding="utf-8"))
LOCATIONS = meta["locations"]
TYPES = meta["property_types"]
LOC_STATS = meta["location_stats"]
db = users = predictions = None
client = None
MONGO_URI = os.getenv("MONGODB_URI") or os.getenv("MONGO_URI")
if MONGO_URI:
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        db = client[os.getenv("MONGO_DB", "house_price_db")]
        users, predictions = db.users, db.predictions
        users.create_index([("email", ASCENDING)], unique=True)
        predictions.create_index([("user_id", ASCENDING), ("created_at", DESCENDING)])
    except (PyMongoError, ServerSelectionTimeoutError) as exc:
        app.logger.warning("MongoDB is unavailable during startup: %s", exc.__class__.__name__)
        db = users = predictions = None


def now_utc():
    return datetime.now(timezone.utc)


def database_ready():
    if users is None or predictions is None:
        flash("Database connect nahi ho saka. MONGODB_URI check karein.", "danger")
        return False
    return True


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("user_id"):
            flash("Pehle login karo.", "warning")
            return redirect(url_for("login"))
        return view(*args, **kwargs)
    return wrapped


def admin_required(view):
    @wraps(view)
    @login_required
    def wrapped(*args, **kwargs):
        user_id = session.get("user_id", "")
        if session.get("role") != "admin" or users is None or not valid_object_id(user_id):
            abort(403)
        current_user = users.find_one({"_id": ObjectId(user_id)}, {"role": 1})
        if not current_user or current_user.get("role") != "admin":
            session["role"] = "user"
            abort(403)
        return view(*args, **kwargs)
    return wrapped


@app.before_request
def maintain_session():
    if session.get("user_id"):
        last = session.get("last_seen")
        if last:
            try:
                if (now_utc() - datetime.fromtimestamp(float(last), timezone.utc)) > timedelta(minutes=30):
                    session.clear()
                    flash("Session expire ho gaya. Dobara login karein.", "info")
                    return redirect(url_for("login"))
            except (ValueError, TypeError):
                session.clear()
        session.permanent = True
        session["last_seen"] = now_utc().timestamp()


@app.template_filter("pkr")
def pkr(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "—"
    if value >= 10_000_000:
        return f"₨ {value / 10_000_000:,.2f} Crore"
    if value >= 100_000:
        return f"₨ {value / 100_000:,.2f} Lakh"
    return f"₨ {value:,.0f}"


@app.template_filter("number")
def number(value):
    try:
        return f"{float(value):,.0f}"
    except (TypeError, ValueError):
        return "—"


@app.context_processor
def shared_context():
    return {"year": now_utc().year}


def valid_object_id(value):
    return ObjectId.is_valid(value)


def parse_property(form):
    location = form.get("location", "").strip()
    prop_type = form.get("property_type", "").strip()
    unit = form.get("unit", "Sq. Ft")
    if location not in LOCATIONS or prop_type not in TYPES:
        raise ValueError("Location ya property type select karein.")
    if unit not in {"Sq. Ft", "Sq. Yard", "Marla", "Kanal"}:
        raise ValueError("Area unit durust select karein.")
    try:
        beds, baths, area = int(form.get("bedrooms", "")), int(form.get("baths", "")), float(form.get("area", ""))
        listed_raw = form.get("listed_price", "").replace(",", "").strip()
        listed = float(listed_raw) if listed_raw else None
    except (TypeError, ValueError):
        raise ValueError("Bedrooms, bathrooms, area aur listed price mein valid numbers dein.")
    if not 0 <= beds <= 10 or not 0 <= baths <= 10:
        raise ValueError("Bedrooms aur bathrooms 0 se 10 ke darmiyan hone chahiye.")
    if not math.isfinite(area) or area <= 0:
        raise ValueError("Area zero se zyada hona chahiye.")
    factors = {"Sq. Ft": 1, "Sq. Yard": 9, "Marla": 272.25, "Kanal": 5445}
    area_sqft = area * factors[unit]
    area_min, area_max = float(meta["area_min"]), float(meta["area_max"])
    if area_sqft < area_min or area_sqft > area_max:
        raise ValueError(f"Converted area model ke training range ({area_min:,.0f}–{area_max:,.0f} sq ft) ke andar dein.")
    if listed is not None and (not math.isfinite(listed) or listed <= 0):
        raise ValueError("Listed price zero se zyada hona chahiye.")
    row = pd.DataFrame([{
        "location": location,
        "property_type": prop_type,
        "bedrooms": beds,
        "baths": baths,
        "area_sqft": area_sqft,
    }], columns=["location", "property_type", "bedrooms", "baths", "area_sqft"])
    estimated = max(0.0, float(np.expm1(model.predict(row)[0])))
    verdict = None
    if listed is not None:
        gap = (listed - estimated) / estimated * 100 if estimated else 0
        verdict = "Overpriced" if gap > 15 else "Good Deal" if gap < -15 else "Fair Price"
    return {"location": location, "property_type": prop_type, "bedrooms": beds,
            "baths": baths, "area_sqft": round(area_sqft, 2), "predicted_price": estimated,
            "low_price": estimated * 0.85, "high_price": estimated * 1.15,
            "listed_price": listed, "verdict": verdict,
            "location_stat": LOC_STATS.get(location, {})}


@app.get("/")
def index():
    if session.get("user_id"):
        return redirect(url_for("predict"))
    return render_template("landing.html", stats=meta.get("metrics", {}))


@app.route("/register", methods=["GET", "POST"])
def register():
    if session.get("user_id"):
        return redirect(url_for("predict"))
    if request.method == "POST":
        if not database_ready():
            return render_template("register.html")
        name, email, password = request.form.get("name", "").strip(), request.form.get("email", "").strip().lower(), request.form.get("password", "")
        if not name or not email or not password:
            flash("Tamam fields fill karein.", "danger")
        elif not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", email):
            flash("Valid email address dein.", "danger")
        elif len(password) < 6:
            flash("Password kam az kam 6 characters ka hona chahiye.", "danger")
        else:
            try:
                users.insert_one({"name": name, "email": email, "password_hash": generate_password_hash(password), "role": "user", "created_at": now_utc()})
                flash("Account created. Ab login karein.", "success")
                return redirect(url_for("login"))
            except PyMongoError as exc:
                flash("Yeh email pehle se registered hai." if exc.__class__.__name__ == "DuplicateKeyError" else "Account save nahi ho saka.", "danger")
    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
@limiter.limit("5 per minute", methods=["POST"])
def login():
    if session.get("user_id"):
        return redirect(url_for("predict"))
    if request.method == "POST":
        if not database_ready():
            return render_template("login.html")
        email, password = request.form.get("email", "").strip().lower(), request.form.get("password", "")
        user = users.find_one({"email": email})
        if user and check_password_hash(user.get("password_hash", ""), password):
            session.clear()
            session.permanent = True
            session.update(user_id=str(user["_id"]), name=user["name"], role=user.get("role", "user"), last_seen=now_utc().timestamp())
            flash("Khush aamdeed!", "success")
            return redirect(url_for("predict"))
        flash("Email ya password ghalat hai.", "danger")
    return render_template("login.html")


@app.get("/logout")
def logout():
    session.clear()
    flash("Logged out.", "info")
    return redirect(url_for("login"))


@app.route("/predict", methods=["GET", "POST"])
@login_required
def predict():
    result = None
    if request.method == "POST":
        if not database_ready():
            return render_template("predict.html", locations=LOCATIONS, types=TYPES, result=None)
        try:
            result = parse_property(request.form)
            result["user_id"] = session["user_id"]
            result["created_at"] = now_utc()
            predictions.insert_one(result.copy())
            flash("Prediction history mein save ho gayi.", "success")
        except ValueError as exc:
            flash(str(exc), "danger")
    return render_template("predict.html", locations=LOCATIONS, types=TYPES, result=result)


@app.get("/history")
@login_required
def history():
    if not database_ready():
        return render_template("history.html", records=[], page=1, pages=1)
    query = {"user_id": session["user_id"]}
    loc, verdict, kind = request.args.get("q", "").strip(), request.args.get("verdict", ""), request.args.get("type", "")
    if loc: query["location"] = {"$regex": re.escape(loc), "$options": "i"}
    if verdict in {"Good Deal", "Fair Price", "Overpriced"}: query["verdict"] = verdict
    if kind in TYPES: query["property_type"] = kind
    sort = request.args.get("sort", "date")
    sort_key = "predicted_price" if sort == "price" else "created_at"
    direction = ASCENDING if request.args.get("direction") == "asc" else DESCENDING
    page = max(1, request.args.get("page", 1, type=int))
    total = predictions.count_documents(query)
    pages = max(1, math.ceil(total / 10))
    page = min(page, pages)
    records = list(predictions.find(query).sort(sort_key, direction).skip((page - 1) * 10).limit(10))
    return render_template("history.html", records=records, page=page, pages=pages, total=total, types=TYPES,
                           filters={"q": loc, "verdict": verdict, "type": kind, "sort": sort})


@app.post("/history/delete/<record_id>")
@login_required
def delete_prediction(record_id):
    if not valid_object_id(record_id): abort(404)
    if database_ready():
        result = predictions.delete_one({"_id": ObjectId(record_id), "user_id": session["user_id"]})
        flash("Prediction delete kar di." if result.deleted_count else "Record nahi mila.", "success" if result.deleted_count else "warning")
    return redirect(url_for("history"))


@app.get("/history/export")
@login_required
def export_history():
    if not database_ready(): return redirect(url_for("history"))
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Date", "Location", "Property type", "Bedrooms", "Bathrooms", "Area sq ft", "Predicted price PKR", "Listed price PKR", "Verdict"])
    for doc in predictions.find({"user_id": session["user_id"]}).sort("created_at", DESCENDING):
        writer.writerow([doc.get("created_at", ""), doc.get("location", ""), doc.get("property_type", ""), doc.get("bedrooms", ""), doc.get("baths", ""), doc.get("area_sqft", ""), doc.get("predicted_price", ""), doc.get("listed_price", ""), doc.get("verdict", "")])
    return Response("\ufeff" + output.getvalue(), mimetype="text/csv", headers={"Content-Disposition": "attachment; filename=karachi-house-predictions.csv"})


@app.get("/dashboard")
@login_required
def dashboard():
    docs = list(predictions.find({"user_id": session["user_id"]})) if database_ready() else []
    total = len(docs)
    prices = [float(x.get("predicted_price", 0)) for x in docs]
    verdicts = {k: sum(x.get("verdict") == k for x in docs) for k in ("Good Deal", "Fair Price", "Overpriced")}
    by_location, per_day = {}, {}
    for doc in docs:
        loc = doc.get("location", "Other")
        by_location.setdefault(loc, []).append(float(doc.get("predicted_price", 0)))
        created = doc.get("created_at")
        day = created.strftime("%Y-%m-%d") if isinstance(created, datetime) else "Unknown"
        per_day[day] = per_day.get(day, 0) + 1
    loc_names = list(by_location)
    charts = {"verdicts": verdicts, "locations": loc_names, "location_prices": [sum(by_location[k]) / len(by_location[k]) for k in loc_names],
              "days": sorted(per_day), "daily": [per_day[k] for k in sorted(per_day)],
              "market_locations": list(LOC_STATS)[:10], "market_ppsf": [LOC_STATS[k]["avg_ppsf"] for k in list(LOC_STATS)[:10]]}
    summary = {"total": total, "average": sum(prices) / total if total else 0, "highest": max(prices, default=0), "good": verdicts["Good Deal"], "overpriced": verdicts["Overpriced"]}
    return render_template("dashboard.html", summary=summary, charts=charts)


@app.route("/compare", methods=["GET", "POST"])
@login_required
def compare():
    result_a = result_b = None
    if request.method == "POST":
        try:
            form_a = {k.removesuffix("_a"): v for k, v in request.form.items() if k.endswith("_a")}
            form_b = {k.removesuffix("_b"): v for k, v in request.form.items() if k.endswith("_b")}
            result_a, result_b = parse_property(form_a), parse_property(form_b)
            for item in (result_a, result_b):
                item["price_per_sqft"] = item["predicted_price"] / item["area_sqft"]
            result_a["better"] = result_a["price_per_sqft"] <= result_b["price_per_sqft"]
            result_b["better"] = result_b["price_per_sqft"] < result_a["price_per_sqft"]
        except ValueError as exc:
            flash(str(exc), "danger")
    return render_template("compare.html", locations=LOCATIONS, types=TYPES, a=result_a, b=result_b)


@app.get("/map")
@login_required
def map_explorer():
    return render_template("map.html", locations=LOC_STATS)


@app.route("/profile", methods=["GET", "POST"])
@login_required
def profile():
    if not database_ready(): return render_template("profile.html", user={"name": session.get("name"), "email": ""}, count=0)
    if not valid_object_id(session.get("user_id", "")):
        session.clear()
        return redirect(url_for("login"))
    user = users.find_one({"_id": ObjectId(session["user_id"])})
    if not user: session.clear(); return redirect(url_for("login"))
    if request.method == "POST":
        action = request.form.get("action")
        if action == "update":
            name, old, new = request.form.get("name", "").strip(), request.form.get("old_password", ""), request.form.get("new_password", "")
            update = {}
            if not name: flash("Name required hai.", "danger")
            elif new and (not check_password_hash(user["password_hash"], old) or len(new) < 6):
                flash("Password change ke liye old password sahi dein aur naya password 6+ characters ka ho.", "danger")
            else:
                update["name"] = name
                if new: update["password_hash"] = generate_password_hash(new)
                users.update_one({"_id": user["_id"]}, {"$set": update})
                session["name"] = name
                flash("Profile update ho gayi.", "success")
                return redirect(url_for("profile"))
        elif action == "delete":
            if check_password_hash(user["password_hash"], request.form.get("confirm_password", "")):
                predictions.delete_many({"user_id": session["user_id"]})
                users.delete_one({"_id": user["_id"]})
                session.clear()
                flash("Account aur uski predictions delete kar di gayi hain.", "info")
                return redirect(url_for("login"))
            flash("Password ghalat hai; account delete nahi hua.", "danger")
    count = predictions.count_documents({"user_id": session["user_id"]})
    return render_template("profile.html", user=user, count=count)


@app.post("/profile/delete")
@login_required
def profile_delete():
    # Compatibility route for the route table; deletion form is also accepted by /profile.
    return profile()


@app.get("/admin")
@admin_required
def admin():
    if not database_ready(): return render_template("admin.html", members=[], total_users=0, total_predictions=0, chart={"days": [], "counts": []})
    members = list(users.find({}, {"password_hash": 0}).sort("created_at", DESCENDING))
    for member in members:
        member["prediction_count"] = predictions.count_documents({"user_id": str(member["_id"])})
    aggregate = list(predictions.aggregate([{"$group": {"_id": {"$dateToString": {"format": "%Y-%m-%d", "date": "$created_at"}}, "count": {"$sum": 1}}}, {"$sort": {"_id": 1}}]))
    return render_template("admin.html", members=members, total_users=users.count_documents({}), total_predictions=predictions.count_documents({}), chart={"days": [x["_id"] for x in aggregate], "counts": [x["count"] for x in aggregate]})


@app.post("/admin/delete/<user_id>")
@admin_required
def admin_delete(user_id):
    if not valid_object_id(user_id): abort(404)
    if user_id == session["user_id"]:
        flash("Apna admin account yahan se delete nahi kar sakte.", "warning")
    elif database_ready():
        predictions.delete_many({"user_id": user_id})
        users.delete_one({"_id": ObjectId(user_id)})
        flash("User aur uski predictions delete kar di.", "success")
    return redirect(url_for("admin"))


@app.post("/admin/role/<user_id>")
@admin_required
def admin_role(user_id):
    if not valid_object_id(user_id): abort(404)
    role = request.form.get("role")
    if role not in {"user", "admin"}: abort(400)
    if database_ready():
        users.update_one({"_id": ObjectId(user_id)}, {"$set": {"role": role}})
        if user_id == session["user_id"]:
            session["role"] = role
        flash("User role update kar diya.", "success")
    return redirect(url_for("admin"))


@app.errorhandler(403)
def forbidden(_):
    return render_template("error.html", code=403, title="Access denied", message="Is page ke liye admin access zaroori hai."), 403


@app.errorhandler(400)
def bad_request(_):
    return render_template("error.html", code=400, title="Request could not be processed", message="Form expire ho gaya ya request invalid thi. Page refresh karke dobara try karein."), 400


@app.errorhandler(404)
def not_found(_):
    return render_template("error.html", code=404, title="Page not found", message="Yeh page available nahi hai."), 404


@app.errorhandler(500)
def server_error(_):
    return render_template("error.html", code=500, title="Something went wrong", message="Thori dair baad dobara koshish karein."), 500


@app.errorhandler(429)
def too_many(_):
    return render_template("error.html", code=429, title="Please slow down", message="Login attempts ki limit exceed ho gayi. Ek minute baad dobara koshish karein."), 429


if __name__ == "__main__":
    app.run(debug=os.getenv("FLASK_DEBUG", "false").lower() == "true")
