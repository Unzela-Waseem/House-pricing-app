# Karachi House Price Predictor

A responsive Flask app that estimates Karachi home prices with the supplied, already-trained scikit-learn/XGBoost pipeline. It stores per-user estimates in MongoDB Atlas and includes account management, saved history, analytics, property comparison and an area map.

## Features

- Register and sign in with hashed passwords, CSRF protection and login rate limiting.
- Estimate a home's price using the model's exact features: `location`, `property_type`, `bedrooms`, `baths`, `area_sqft`.
- Convert Sq. Ft, Sq. Yard, Marla and Kanal to square feet before prediction. The model output is converted back with `numpy.expm1`.
- Optional asking-price verdicts: Overpriced (>15%), Good Deal (<-15%), or Fair Price (within ±15%).
- Personal prediction history with search, filters, sort, pagination, deletion and CSV export.
- Personal dashboard, side-by-side property comparison, Karachi locality map, account profile and admin user management.

## Run locally

1. Use Python 3.10–3.12. Create and activate a virtual environment.
2. Install the pinned dependencies: `pip install -r requirements.txt`.
3. Copy `.env.example` to `.env`, then set `MONGODB_URI` and a long random `SECRET_KEY`. The app also reads the starter project's ignored `atlas-credentials.env` (if present) and accepts `MONGO_URI` for the connection string.
4. Ensure MongoDB Atlas permits the network connection from your machine. The app creates a unique email index and a user-history index at startup.
5. Run `python app.py`, then open `http://127.0.0.1:5000`.

`COOKIE_SECURE=true` is appropriate behind HTTPS (for example, Render). Set it to `false` for local HTTP development. `FLASK_DEBUG` defaults to `false`.

## Model files and compatibility

Keep `house_price_model.pkl` and `model_meta.json` in the project root. The model is loaded once when the app starts and is not retrained or altered. The pickle records scikit-learn 1.6.1; `requirements.txt` pins that version and XGBoost 2.1.4. If the model loader reports an XGBoost incompatibility, use the exact XGBoost version from the original training environment and update the pin accordingly.

## First admin account

Register an account normally. In MongoDB Atlas, open `house_price_db.users`, locate that account by its email, and set its `role` field to `admin`. The Admin link will appear after the next login. Never promote users through an unprotected public setup route.

## Deploy to Render or Railway

- Push this repository to GitHub.
- Create a Python web service with build command `pip install -r requirements.txt` and start command `gunicorn app:app` (also recorded in `Procfile`).
- Set `MONGODB_URI`, `MONGO_DB` (optional), `SECRET_KEY`, and `COOKIE_SECURE=true` in the host's environment settings. Do not upload `.env` or `atlas-credentials.env`.
- Configure Atlas network access for the host. Prefer restricting access to the host's outbound IPs where available.

## Limitations and data note

The supplied training data is from 2018 and covers Karachi only. The model uses five features; street, age, floor, furnishing, corner plot and construction quality are not represented. Sparse localities were grouped as `Other`. Its reported median error is 17.5%; the displayed ±15% estimate range is only an approximation. Predictions are estimates, not official valuations, and current market prices can differ materially.

## Project files

```text
app.py                     Flask routes and ML/database integration
templates/                 Jinja pages
static/css/style.css       Responsive Everlight-inspired visual theme
house_price_model.pkl      Supplied trained model (unchanged)
model_meta.json            Supplied location and model metadata
model_comparison.csv       Supplied evaluation summary
requirements.txt           Pinned application and model dependencies
Procfile                   Gunicorn start command
```
