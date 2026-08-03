import pandas as pd
import numpy as np

np.random.seed(42)

rows = []

hostels = ["MH1", "MH2", "MH3", "MH4"]
floors = range(1, 11)

# hostel usage weight
hostel_weight = {
    "MH1": 1.3,
    "MH2": 1.1,
    "MH3": 1.0,
    "MH4": 0.8
}

timestamps = pd.date_range("2025-01-01", periods=1500, freq="H")

for t in timestamps:

    hour = t.hour
    day = t.weekday()  # 0 = Monday

    for hostel in hostels:
        for floor in floors:

            # ---------- BASE TIME PATTERN ----------
            if 6 <= hour <= 9:
                base = np.random.randint(80, 140)
            elif 18 <= hour <= 22:
                base = np.random.randint(60, 110)
            elif 0 <= hour <= 4:
                base = np.random.randint(0, 10)
            else:
                base = np.random.randint(20, 60)

            # ---------- WEEKEND EFFECT ----------
            if day >= 5:  # Sat/Sun
                base *= 1.2

            # ---------- HOSTEL EFFECT ----------
            base *= hostel_weight[hostel]

            # ---------- FLOOR EFFECT ----------
            base *= (1.2 - floor * 0.03)  # higher floor less usage

            # ---------- RANDOM ZERO (EMPTY ROOM) ----------
            if np.random.rand() < 0.08:
                base = 0

            # ---------- RANDOM SPIKES ----------
            if np.random.rand() < 0.04:
                base += np.random.randint(80, 200)

            # ---------- GRADUAL ANOMALY ----------
            if np.random.rand() < 0.02:
                base += (hour * 5)

            # ---------- NOISE ----------
            noise = np.random.normal(0, 5)
            total = max(int(base + noise), 0)

            # ---------- FLOW RATE ----------
            flow = total + np.random.randint(-10, 15)
            flow = max(flow, 0)

            # ---------- LEAK ----------
            leak = 1 if total > 150 else 0

            rows.append({
                "timestamp": t,
                "block": hostel,
                "floor": floor,
                "total_liters": total,
                "flow_rate": flow,
                "leak_detected": leak
            })

df = pd.DataFrame(rows)

df.to_csv("water_data.csv", index=False)

print("🔥 Ultra realistic dataset generated!")