import numpy as np

def get_network_data():
    """
    Defines a simple 3-bus power system.
    Bus 1: Slack
    Bus 2: PQ
    Bus 3: PQ
    """

    Ybus = np.array([
        [10 - 30j, -5 + 15j, -5 + 15j],
        [-5 + 15j, 10 - 30j, -5 + 15j],
        [-5 + 15j, -5 + 15j, 10 - 30j]
    ])

    P_spec = np.array([0.0, -1.0, -0.8])  # pu
    Q_spec = np.array([0.0, -0.5, -0.3])  # pu

    slack_bus = 0

    return Ybus, P_spec, Q_spec, slack_bus
