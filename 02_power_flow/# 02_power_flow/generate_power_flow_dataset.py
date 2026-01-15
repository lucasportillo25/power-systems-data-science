# 02_power_flow/generate_power_flow_dataset.py

import numpy as np
import pandas as pd

from network_data import get_network_data
from power_flow_newton_raphson import power_flow_newton_raphson

def generate_dataset(PQ_variations, slack_bus):
    records = []

    # Base network
    Ybus, P_spec_base, Q_spec_base, _ = get_network_data()

    for p2 in PQ_variations:
        for q2 in PQ_variations:
            # Modify load at bus 2 (example)
            P_spec = P_spec_base.copy()
            Q_spec = Q_spec_base.copy()

            P_spec[1] = -p2
            Q_spec[1] = -q2

            try:
                V, theta = power_flow_newton_raphson(Ybus, P_spec, Q_spec, slack_bus)

                records.append({
                    "P_load_bus2": p2,
                    "Q_load_bus2": q2,
                    "V_bus1": V[0],
                    "V_bus2": V[1],
                    "V_bus3": V[2],
                    "theta_bus1": theta[0],
                    "theta_bus2": theta[1],
                    "theta_bus3": theta[2]
                })
            except Exception as e:
                # skip non-convergent cases
                continue

    return pd.DataFrame(records)

if __name__ == "__main__":
    slack_bus = 0
    PQ_range = np.linspace(0.5, 2.0, 8)  # pu

    df = generate_dataset(PQ_range, slack_bus)

    df.to_csv("power_flow_dataset.csv", index=False)
    print("Dataset generated:", df.shape)
    print(df.head())
