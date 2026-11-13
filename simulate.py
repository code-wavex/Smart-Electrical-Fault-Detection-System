import numpy as np, pandas as pd

CLASSES = ["normal", "short_circuit", "earth_fault", "overload", "open_circuit"]
rng = np.random.default_rng(42)

def sample(label):
    if label == "normal":
        v, i, leak, pf = rng.normal(228, 10), rng.uniform(1.5, 20), rng.uniform(0, 12), rng.uniform(0.6, 0.98)
    elif label == "short_circuit":
        v, i, leak, pf = rng.uniform(60, 200), np.exp(rng.uniform(np.log(30), np.log(250))), rng.uniform(0, 20), rng.uniform(0.2, 0.75)
    elif label == "earth_fault":
        v, i, leak, pf = rng.normal(222, 12), rng.uniform(4, 30), np.exp(rng.uniform(np.log(8), np.log(300))), rng.uniform(0.6, 0.95)
    elif label == "overload":
        v, i, leak, pf = rng.uniform(195, 228), rng.uniform(16, 40), rng.uniform(0, 12), rng.uniform(0.6, 0.9)
    else:  # open_circuit
        v, i, leak, pf = rng.uniform(220, 250), rng.uniform(0, 2), rng.uniform(0, 3), rng.uniform(0, 0.5)
    # sensor measurement noise
    v += rng.normal(0, 3)
    i = max(0, i * (1 + rng.normal(0, 0.05)))
    leak = max(0, leak + rng.normal(0, 1.5))
    pf = float(np.clip(pf + rng.normal(0, 0.03), 0, 1))
    return [float(v), float(i), float(leak), pf]

def make_dataset(n_per_class=1500):
    rows = [sample(c) + [c] for c in CLASSES for _ in range(n_per_class)]
    df = pd.DataFrame(rows, columns=["voltage", "current", "leakage_ma", "power_factor", "label"])
    return df.sample(frac=1, random_state=42).reset_index(drop=True)

if __name__ == "__main__":
    make_dataset().to_csv("fault_data.csv", index=False)
