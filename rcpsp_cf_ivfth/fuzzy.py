"""
Fuzzy number definitions for RCPSP-CF-IVFTH.

This module contains the NIVTF (Normalized Interval-Valued Triangular Fuzzy) 
number implementation and related fuzzy arithmetic operations.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class NIVTF:
    """
    Normalized Interval-Valued Triangular Fuzzy number (NIVTF).
    This stores the **lower** and **upper** triangular fuzzy numbers.

    For a triangular fuzzy number (a_o, a_m, a_p) we follow the paper's normalization:
    - normalized => peak membership = 1 at a_m
    - For NIVTF: lower triangle L = (a_o^L, a_m^L, a_p^L) and upper triangle U = (a_o^U, a_m^U, a_p^U)
      with a_m^L = a_m^U and  a_o^U < a_o^L < a_m^L(=a_m^U) < a_p^L < a_p^U.

    In the paper:
        E1(x) = (a_o + a_m) / 2
        E2(x) = (a_m + a_p) / 2
        EV(x) = (a_o + 2 a_m + a_p) / 4
    applied to lower (L) and upper (U) sides.

    Here we expose:
        E1_L(), E1_U(), E2_L(), E2_U(), EV_L(), EV_U()
    which return floats.

    Input Validation:
        - Ensures the order constraints for NIVTF are satisfied (best-effort checks).
    """
    ao_L: float
    am_L: float
    ap_L: float
    ao_U: float
    am_U: float
    ap_U: float

    def __post_init__(self) -> None:
        # Basic checks for monotonicity and normalization consistency
        if not (self.ao_U < self.ao_L < self.am_L <= self.am_U < self.ap_L < self.ap_U):
            raise ValueError(
                f"NIVTF ordering violated:\n"
                f"Require ao_U < ao_L < am_L(=am_U) <= am_U < ap_L < ap_U\n"
                f"Got: (ao_U, ao_L, am_L, am_U, ap_L, ap_U) = "
                f"({self.ao_U}, {self.ao_L}, {self.am_L}, {self.am_U}, {self.ap_L}, {self.ap_U})"
            )
        if abs(self.am_L - self.am_U) > 1e-12:
            raise ValueError("Normalization requires am_L == am_U for NIVTF.")

    # Expected (Jiménez-style) computations on LOWER side
    def E1_L(self) -> float:
        return 0.5 * (self.ao_L + self.am_L)

    def E2_L(self) -> float:
        return 0.5 * (self.am_L + self.ap_L)

    def EV_L(self) -> float:
        return 0.25 * (self.ao_L + 2.0 * self.am_L + self.ap_L)

    # Expected computations on UPPER side
    def E1_U(self) -> float:
        return 0.5 * (self.ao_U + self.am_U)

    def E2_U(self) -> float:
        return 0.5 * (self.am_U + self.ap_U)

    def EV_U(self) -> float:
        return 0.25 * (self.ao_U + 2.0 * self.am_U + self.ap_U)

    # Convenience: midpoints of expected intervals (optional)
    def E1_mid(self) -> float:
        return 0.5 * (self.E1_L() + self.E1_U())

    def E2_mid(self) -> float:
        return 0.5 * (self.E2_L() + self.E2_U())

    def EV_mid(self) -> float:
        return 0.5 * (self.EV_L() + self.EV_U())


def create_triangle(a0: float, am: float, ap: float, widen: float = 1.0) -> Tuple[float, float, float, float, float, float]:
    """
    Convenience function to create an NIVTF by widening lower/upper supports around am.

    Parameters
    ----------
    a0, am, ap : float
        Base triangle points with a0 < am < ap.
    widen : float
        Controls widening of U and narrowing of L around the same am.

    Returns
    -------
    NIVTF args: (ao_L, am_L, ap_L, ao_U, am_U, ap_U)
        
    Examples
    --------
    >>> args = create_triangle(5, 7, 9, widen=0.5)
    >>> nivtf = NIVTF(*args)
    """
    # Upper triangle further away, lower triangle closer, share am
    ao_U = a0 - 0.5 * widen * (am - a0)
    ap_U = ap + 0.5 * widen * (ap - am)

    ao_L = a0 + 0.25 * widen * (am - a0)
    ap_L = ap - 0.25 * widen * (ap - am)

    am_L = am_U = am
    return (ao_L, am_L, ap_L, ao_U, am_U, ap_U)
