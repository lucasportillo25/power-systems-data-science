"""
Synchronous Machine Mathematical Model
-------------------------------------
Author: Lucas Portillo

This module implements a simplified steady-state mathematical model of a
synchronous machine, suitable for power system studies and data-driven analysis.

The objective is to bridge classical electrical machine theory with
computational modeling and future data science applications.
"""

import numpy as np

# -------------------------------------------------
# Synchronous Machine Class
# -------------------------------------------------

class SynchronousMachine:
    """
    Steady-state model of a synchronous generator.
    """

    def __init__(self, Xd, Xq, Ra=0.0):
        """
        Parameters
        ----------
        Xd : float
            Direct-axis synchronous reactance (p.u.)
        Xq : float
            Quadrature-axis synchronous reactance (p.u.)
        Ra : float, optional
            Armature resistance (p.u.)
        """
        self.Xd = Xd
        self.Xq = Xq
        self.Ra = Ra

    def electrical_power(self, V, E, delta):
        """
        Calculate electrical power output.

        Parameters
        ----------
        V : float
            Terminal voltage magnitude (p.u.)
        E : float
            Internal emf magnitude (p.u.)
        delta : float
            Load angle (rad)

        Returns
        -------
        Pe : float
            Electrical power output (p.u.)
        """
        Pe = (V * E / self.Xd) * np.sin(delta)
        return Pe

    def reactive_power(self, V, E, delta):
        """
        Calculate reactive power output.
        """
        Qe = (V / self.Xd) * (E * np.cos(delta) - V)
        return Qe

    def stator_current(self, V, E, delta):
        """
        Calculate stator current magnitude.
        """
        Id = (E * np.cos(delta) - V) / self.Xd
        Iq = (E * np.sin(delta)) / self.Xd
        return np.sqrt(Id**2 + Iq**2)


# -------------------------------------------------
# Example usage
# -------------------------------------------------

if __name__ == "__main__":
    # Machine parameters (p.u.)
    Xd = 1.8
    Xq = 1.7

    gen = SynchronousMachine(Xd=Xd, Xq=Xq)

    V = 1.0     # Terminal voltage
    E = 1.2     # Internal emf

    angles = np.linspace(0, np.pi/2, 50)

    print("Load Angle (deg) | Electrical Power (pu)")
    print("----------------------------------------")

    for delta in angles:
        Pe = gen.electrical_power(V, E, delta)
        print(f"{np.degrees(delta):6.1f}           | {Pe:6.3f}")
