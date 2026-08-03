# 💧 AquaTrace — Smart Water Governance System

A full-stack, real-time hostel water monitoring system with:
- **FastAPI** REST backend
- **SQLite / PostgreSQL** database
- **Live sensor simulation** (IoT data stream)
- **Streamlit** dashboard with ML anomaly detection

---

## 📁 Project Structure

```
aquatrace/
├── app.py                  ← Streamlit dashboard (frontend)
├── generate_data.py        ← Standalone CSV data generator (legacy)
├── dashboard.py            ← Dash dashboard (alternate UI)
├── requirements.txt        ← All dependencies
│
└── backend/
    ├── main.py             ← FastAPI application (REST API)
    ├── database.py         ← DB engine & session (SQLite or PostgreSQL)
    ├── models.py           ← SQLAlchemy ORM table definition
    ├── schemas.py          ← Pydantic request/response schemas
    ├── seed_db.py          ← Pre-populate DB with historical data
    └── sensor_sim.py       ← Simulated sensor publisher
```

---

## ⚙️ Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Seed the database with historical data
```bash
cd backend
python seed_db.py
```

### 3. Start the FastAPI backend
```bash
cd backend
uvicorn main:app --reload
```
> API docs auto-available at: **http://127.0.0.1:8000/docs**

### 4. Start the sensor simulator (new terminal)
```bash
cd backend
python sensor_sim.py
```
> This mimics real IoT sensors, POSTing live readings to the API every 2 seconds.

### 5. Start the Streamlit dashboard (new terminal)
```bash
streamlit run app.py
```
> Dashboard available at: **http://localhost:8501**

---

## 🔌 API Endpoints

| Method | Endpoint                  | Description                        |
|--------|---------------------------|------------------------------------|
| GET    | `/api/health`             | API health check                   |
| POST   | `/api/readings`           | Ingest a sensor reading            |
| GET    | `/api/readings`           | Fetch latest readings (filterable) |
| GET    | `/api/readings/summary`   | KPI summary (avg, max, anomalies)  |
| GET    | `/api/anomalies`          | Fetch all flagged anomaly events   |
| GET    | `/api/blocks`             | List blocks and floor counts       |

### Query parameters for `/api/readings`
- `block=MH1` — filter by hostel block
- `floor=3` — filter by floor
- `limit=200` — max rows returned

---

## 🗄️ Database

**Default:** SQLite (`backend/aquatrace.db`) — zero setup, works out of the box.

**Switch to PostgreSQL:**
1. Install driver: `pip install psycopg2-binary`
2. Set environment variable:
```bash
export DB_URL="postgresql://username:password@localhost:5432/aquatrace"
```
3. Re-run `seed_db.py` to create tables and seed data.

---

## 🧠 ML Features

- **Isolation Forest** anomaly detection on `total_liters` + `flow_rate`
- Contamination rate: 5% (flags top 5% as anomalies)
- Labels assigned at ingest time by `sensor_sim.py`
- Dashboard shows real-time anomaly count + timeline

---

## 🏗️ Architecture

```
[sensor_sim.py]
      │  POST /api/readings (every 2s)
      ▼
[FastAPI Backend] ──── SQLAlchemy ORM ────► [SQLite / PostgreSQL]
      │
      │  GET /api/readings (every refresh interval)
      ▼
[Streamlit Dashboard]
  - KPI cards
  - Live trend charts
  - Anomaly detection
  - Hourly heatmap
  - Block comparison
```
