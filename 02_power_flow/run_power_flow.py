# 02_power_flow/run_power_flow.py

from network_data import get_network_data
from power_flow_model import power_flow_newton_raphson

def main():
    # Load network
    Ybus, P_spec, Q_spec, slack_bus = get_network_data()

    # Run power flow
    V, theta = power_flow_newton_raphson(Ybus, P_spec, Q_spec, slack_bus)

    # Display results
    print("\nPower Flow Solution (per unit):")
    print("--------------------------------")
    for i in range(len(V)):
        print(f"Bus {i+1:2d}: Voltage = {V[i]:.4f} pu, Angle = {theta[i]:.4f} rad")

if __name__ == "__main__":
    main()
