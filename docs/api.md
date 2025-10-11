# API Reference

> _Generated automatically with pdoc._

# rcpsp\_cf\_ivfth

RCPSP-CF-IVFTH: Bi-objective Resource-Constrained Project Scheduling with Cash-Flow under fuzzy uncertainty.

This package implements the model from:
"A New Bi-Objective Model for Resource-Constrained Project Scheduling and Cash Flow Problems
with Financial Constraints under Uncertainty: A Case Study"

Main components:

* Fuzzy numbers (NIVTF) for handling uncertainty
* Data structures for activities, modes, finance, and calendar parameters
* MILP model builder with Extended IVF-TH scalarization
* Solver interface for various MILP solvers

Example usage:
from rcpsp\_cf\_ivfth import RCPSP\_CF\_IVFTH, IVFTHTargets, IVFTHWeights
from rcpsp\_cf\_ivfth.examples import build\_toy\_instance

```
activities, finance, calendar = build_toy_instance()
ivfth = RCPSP_CF_IVFTH(activities, finance, calendar)

targets = IVFTHTargets(alpha_level=0.5, Z1_PIS=10.0, Z1_NIS=60.0, Z2_PIS=30000.0, Z2_NIS=0.0)
weights = IVFTHWeights(theta1=0.5, theta2=0.5, gamma_tradeoff=0.5)

model = ivfth.build_model(targets, weights)
results = ivfth.solve(model, solver_name="glpk")
```

class
RCPSP\_CF\_IVFTH:

@dataclass

class
Activity:

Activity with multiple modes and precedence list.

## Attributes

name : str
Unique identifier for the activity.
predecessors : List[str]
List of activity names that must be completed before this activity can start.
modes : Dict[int, ModeData]
Dictionary mapping mode index to mode data.

Activity( name: str, predecessors: List[str], modes: Dict[int, [ModeData](#ModeData)])

name: str

predecessors: List[str]

modes: Dict[int, [ModeData](#ModeData)]

@dataclass

class
ModeData:

Per-activity, per-mode data.

## Attributes

duration : NIVTF
Uncertain duration (NIVTF).
renewables : Dict[int, NIVTF]
Per renewable resource k, uncertain daily use NIVTF.
nonrenewables : Dict[int, NIVTF]
Per non-renewable resource l, uncertain daily use NIVTF.
payment : float
Payment amount PA\_{i,m} devoted to the period where the activity completes (or next if delayed).

ModeData( duration: [NIVTF](#NIVTF), renewables: Dict[int, [NIVTF](#NIVTF)], nonrenewables: Dict[int, [NIVTF](#NIVTF)], payment: float)

duration: [NIVTF](#NIVTF)

renewables: Dict[int, [NIVTF](#NIVTF)]

nonrenewables: Dict[int, [NIVTF](#NIVTF)]

payment: float

@dataclass

class
FinanceParams:

Financial parameters for the model.

## Attributes

alpha\_excess\_cash : float
Interest on excess cash (per 30-day period), used as (1+alpha)^30 in model equation.
beta\_delayed\_pay : float
Interest applied to delayed payments (per 30-day period).
gamma\_LTL : float
Long-term loan interest (per 30-day period).
delta\_STL : float
Short-term loan interest (per 30-day period).
IC : float
Initial capital (received at period 1).
max\_LTL : float
Upper bound on long-term loan (single, period 1 only).
max\_STL : float
Upper bound on short-term loan per period.
min\_CF : float
Minimum cash-flow per period (credit floor).
CC\_daily\_cap : float
Upper bound on total daily resource cost (constraint (12)).
CR\_k : Dict[int, float]
Cost per unit of renewable resource k per day.
CW\_l : Dict[int, float]
Cost per unit of non-renewable resource l per day.

FinanceParams( alpha\_excess\_cash: float, beta\_delayed\_pay: float, gamma\_LTL: float, delta\_STL: float, IC: float, max\_LTL: float, max\_STL: float, min\_CF: float, CC\_daily\_cap: float, CR\_k: Dict[int, float], CW\_l: Dict[int, float])

alpha\_excess\_cash: float

beta\_delayed\_pay: float

gamma\_LTL: float

delta\_STL: float

IC: float

max\_LTL: float

max\_STL: float

min\_CF: float

CC\_daily\_cap: float

CR\_k: Dict[int, float]

CW\_l: Dict[int, float]

@dataclass

class
CalendarParams:

Time partitioning parameters.

## Attributes

T\_days : int
Total number of short-term daily periods considered (>= project horizon).
Y\_periods : List[Tuple[int, int]]
Monthly (or long) periods y as (a\_y, b\_y) inclusive daily indices (1-based).
TY\_y = [a\_y, b\_y], with y in {1,...,Yn}. Constraint (19) sums BU\_t over t in [a\_y, b\_y].

## Notes

* The model uses days (t) and periods (y). You must set T\_days and the period intervals coherently.

CalendarParams(T\_days: int, Y\_periods: List[Tuple[int, int]])

T\_days: int

Y\_periods: List[Tuple[int, int]]

@dataclass

class
IVFTHTargets:

Alpha-level and PiS/NiS targets for IVF-TH membership functions.

## Attributes

alpha\_level : float
Alpha in [0, 1] used in fuzzy-to-crisp conversion in constraints (30)-(34).
Z1\_PIS : float
Positive ideal solution for Z1 (makespan), i.e., a *small* target (best).
Z1\_NIS : float
Negative ideal solution for Z1 (worst makespan bound).
Z2\_PIS : float
Positive ideal solution for Z2 (final cash-flow), i.e., a *large* target (best).
Z2\_NIS : float
Negative ideal solution for Z2 (worst cash-flow bound).

## Notes

These targets can be computed by separate runs (min Z1, max Z2) or set by domain knowledge.
The membership functions `mu1` and `mu2` are linear in `Z1` and `Z2` using these PiS/NiS anchors.

IVFTHTargets( alpha\_level: float, Z1\_PIS: float, Z1\_NIS: float, Z2\_PIS: float, Z2\_NIS: float)

alpha\_level: float

Z1\_PIS: float

Z1\_NIS: float

Z2\_PIS: float

Z2\_NIS: float

@dataclass

class
IVFTHWeights:

Weights for the Torabi-Hassini scalarization.

## Attributes

theta1 : float
Weight for objective 1 membership (makespan).
theta2 : float
Weight for objective 2 membership (final cash-flow).
gamma\_tradeoff : float
Trade-off parameter in [0, 1].
Objective: maximize `gamma_tradeoff * zeta + (1 - gamma_tradeoff) * (theta1 * mu1 + theta2 * mu2)`.

## Notes

`mu1` and `mu2` are non-negative and satisfy `mu1 + mu2 = 1` (enforced with tolerance).

IVFTHWeights(theta1: float, theta2: float, gamma\_tradeoff: float)

theta1: float

theta2: float

gamma\_tradeoff: float

@dataclass(frozen=True)

class
NIVTF:

Normalized Interval-Valued Triangular Fuzzy number (NIVTF).
This stores the **lower** and **upper** triangular fuzzy numbers.

For a triangular fuzzy number (a\_o, a\_m, a\_p) we follow the paper's normalization:

* normalized => peak membership = 1 at a\_m
* For NIVTF: lower triangle L = (a\_o^L, a\_m^L, a\_p^L) and upper triangle U = (a\_o^U, a\_m^U, a\_p^U)
  with a\_m^L = a\_m^U and a\_o^U < a\_o^L < a\_m^L(=a\_m^U) < a\_p^L < a\_p^U.

In the paper:
E1(x) = (a\_o + a\_m) / 2
E2(x) = (a\_m + a\_p) / 2
EV(x) = (a\_o + 2 a\_m + a\_p) / 4
applied to lower (L) and upper (U) sides.

Here we expose:
E1\_L(), E1\_U(), E2\_L(), E2\_U(), EV\_L(), EV\_U()
which return floats.

Input Validation:
- Ensures the order constraints for NIVTF are satisfied (best-effort checks).

NIVTF( ao\_L: float, am\_L: float, ap\_L: float, ao\_U: float, am\_U: float, ap\_U: float)

ao\_L: float

am\_L: float

ap\_L: float

ao\_U: float

am\_U: float

ap\_U: float

def
E1\_L(self) -> float:

def
E2\_L(self) -> float:

def
EV\_L(self) -> float:

def
E1\_U(self) -> float:

def
E2\_U(self) -> float:

def
EV\_U(self) -> float:

def
E1\_mid(self) -> float:

def
E2\_mid(self) -> float:

def
EV\_mid(self) -> float:

def
create\_triangle( a0: float, am: float, ap: float, widen: float = 1.0) -> Tuple[float, float, float, float, float, float]:

Convenience function to create an NIVTF by widening lower/upper supports around am.

## Parameters

a0, am, ap : float
Base triangle points with a0 < am < ap.
widen : float
Controls widening of U and narrowing of L around the same am.

## Returns

NIVTF args: (ao\_L, am\_L, ap\_L, ao\_U, am\_U, ap\_U)

## Examples

```
>>> args = create_triangle(5, 7, 9, widen=0.5)
>>> nivtf = NIVTF(*args)
```
