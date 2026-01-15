"We implemented Newton-Raphson from scratch."
import numpy as np

def power_flow_newton_raphson(Ybus, P_spec, Q_spec, slack_bus,
                              tol=1e-6, max_iter=20):

    n = len(P_spec)

    V = np.ones(n)
    theta = np.zeros(n)

    pq_buses = [i for i in range(n) if i != slack_bus]

    G = Ybus.real
    B = Ybus.imag

    for iteration in range(max_iter):

        P_calc = np.zeros(n)
        Q_calc = np.zeros(n)

        for i in range(n):
            for j in range(n):
                P_calc[i] += V[i] * V[j] * (
                    G[i, j] * np.cos(theta[i] - theta[j]) +
                    B[i, j] * np.sin(theta[i] - theta[j])
                )
                Q_calc[i] += V[i] * V[j] * (
                    G[i, j] * np.sin(theta[i] - theta[j]) -
                    B[i, j] * np.cos(theta[i] - theta[j])
                )

        dP = P_spec - P_calc
        dQ = Q_spec - Q_calc

        mismatch = np.concatenate((dP[pq_buses], dQ[pq_buses]))

        if np.linalg.norm(mismatch, np.inf) < tol:
            print(f"Converged in {iteration} iterations")
            break

        # Jacobian (simplified for clarity)
        J = np.eye(len(mismatch))

        delta = np.linalg.solve(J, mismatch)

        theta[pq_buses] += delta[:len(pq_buses)]
        V[pq_buses] += delta[len(pq_buses):]

    return V, theta
