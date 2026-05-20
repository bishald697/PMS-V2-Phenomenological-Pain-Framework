"""
PMS_V2.py

===============================================================
PAIN MONITORING AND SIMULATION FRAMEWORK (PMS-V2)
===============================================================

Author:
Bishal Das
Theoretical Physicist
University of Nottingham, United Kingdom

Year:
2026

---------------------------------------------------------------
INTRODUCTION
---------------------------------------------------------------

The Pain Monitoring and Simulation Framework (PMS-V2) is a
phenomenological computational framework developed for the
study of temporally evolving nociceptive dynamics under
complex multimodal pharmacological intervention.

The framework was motivated by longitudinal observational
monitoring of a single-patient advanced recurrent Head and
Neck Squamous Cell Carcinoma (HNSCC) case involving carcinoma
of the tongue with extensive inflammatory and cervical nodal
disease progression.

The primary objective of PMS-V2 is not the construction of a
generalized medical prediction engine, but rather the
development of a dynamical phenomenological framework capable
of:

1. Modeling evolving nociceptive behaviour,
2. Studying perturbative pain dynamics,
3. Investigating inflammatory activation,
4. Examining pharmacodynamic suppression,
5. Exploring sensory-autonomic amplification,
6. Characterizing bounded stabilization regimes,
7. Understanding event-driven perturbative cascades.

The framework models pain as a coupled nonlinear dynamical
system composed of multiple interacting components:

    P(t) =
        w1*Pm(t)
      + w2*Pi(t)
      + w3*Pb(t)
      + w4*Ps(t)

where:

Pm(t) : movement perturbation component
Pi(t) : inflammatory component
Pb(t) : baseline persistent nociception
Ps(t) : sensory-autonomic contribution

The framework further incorporates a coupled
Acid-Reflux–Autonomic Perturbative Cascade:

AR(t) -> B(t) -> N(t) -> G(t) -> P(t) -> Ds(t)

where:

AR(t) : reflux-associated irritation
B(t)  : restlessness-associated discomfort
N(t)  : nausea-associated perturbation
G(t)  : anxiety/autonomic amplification
Ds(t) : dyspnea-like subjective amplification

The observable pain structure is modeled as:

    P_obs(t) = P(t) + lambda*Ds(t)

The framework additionally studies:
- damping dynamics,
- perturbative stabilization,
- metastable plateau states,
- pharmacodynamic suppression,
- regime transitions,
- nonlinear recovery behaviour.

---------------------------------------------------------------
IMPORTANT DISCLAIMER
---------------------------------------------------------------

This program is an exploratory phenomenological research
framework.

It is NOT:
- a medical device,
- a diagnostic system,
- a clinical prediction engine,
- a treatment recommendation system,
- or a substitute for professional medical judgment.

All clinical decisions, pharmacological interventions,
oncological management, palliative care decisions, and
treatment modifications must remain under the supervision
of qualified medical professionals.

The framework is intended solely for theoretical,
computational, and phenomenological research purposes.
"""

# ============================================================
# IMPORTS
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

from dataclasses import dataclass

from scipy.integrate import solve_ivp
from scipy.signal import find_peaks

# ============================================================
# GLOBAL CONFIGURATION
# ============================================================

DT = 0.05
T_MAX = 48

TIME = np.arange(0, T_MAX, DT)

# ============================================================
# SAFE MATPLOTLIB CONFIGURATION
# ============================================================

plt.rcParams['mathtext.default'] = 'regular'

# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class DrugDose:
    time: float
    dose: float


@dataclass
class DrugParameters:
    name: str
    decay_constant: float
    delay: float
    efficacy: float
    hill_n: float
    half_max: float

# ============================================================
# DRUG DEFINITIONS
# ============================================================

MORPHINE = DrugParameters(
    name="Morphine",
    decay_constant=0.22,
    delay=0.7,
    efficacy=1.8,
    hill_n=2.0,
    half_max=18.0
)

PARACETAMOL = DrugParameters(
    name="Paracetamol",
    decay_constant=0.30,
    delay=0.5,
    efficacy=1.0,
    hill_n=1.5,
    half_max=800.0
)

ETORICOXIB = DrugParameters(
    name="Etoricoxib",
    decay_constant=0.06,
    delay=2.0,
    efficacy=2.2,
    hill_n=1.8,
    half_max=60.0
)

RESCUE_MORPHINE = DrugParameters(
    name="RescueMorphine",
    decay_constant=0.35,
    delay=0.3,
    efficacy=2.5,
    hill_n=2.0,
    half_max=10.0
)

# ============================================================
# DOSING SCHEDULES
# ============================================================

morphine_doses = [
    DrugDose(0, 20),
    DrugDose(8, 20),
    DrugDose(16, 20),
    DrugDose(24, 20),
    DrugDose(32, 20),
    DrugDose(40, 20),
]

paracetamol_doses = [
    DrugDose(6, 1000),
    DrugDose(18, 1000),
    DrugDose(30, 1000),
    DrugDose(42, 1000),
]

etoricoxib_doses = [
    DrugDose(9, 60),
    DrugDose(21, 60),
    DrugDose(33, 60),
    DrugDose(45, 60),
]

rescue_doses = [
    DrugDose(17.5, 20),
]

# ============================================================
# EVENT WINDOWS
# ============================================================

movement_events = [
    (4.0, 4.4),
    (22.5, 23.0),
]

sleep_break_events = [
    (19.0, 19.5),
]

inflammatory_events = [
    (10.0, 11.5),
    (35.0, 36.0),
]

acid_reflux_events = [
    (11.0, 12.0),
]

# ============================================================
# PHARMACODYNAMIC FUNCTIONS
# ============================================================

def delayed_concentration(t, doses, params):

    conc = 0

    for d in doses:

        if t > d.time + params.delay:

            dt = t - d.time - params.delay

            conc += d.dose * np.exp(
                -params.decay_constant * dt
            )

    return conc


def hill_suppression(c, efficacy, hill_n, half_max):

    return efficacy * (c**hill_n) / (
        half_max**hill_n + c**hill_n + 1e-8
    )

# ============================================================
# EVENT GENERATOR
# ============================================================

def rectangular_event(
    t,
    windows,
    amplitude=1.0
):

    val = 0

    for a, b in windows:

        if a <= t <= b:
            val += amplitude

    return val

# ============================================================
# COUPLING COEFFICIENTS
# ============================================================

# Pain component weights

w1 = 0.20
w2 = 0.50
w3 = 0.15
w4 = 0.15

# Distress amplification

lambda_ds = 0.65

# Cascade couplings

alpha_B = 0.8
alpha_N = 0.7
alpha_G = 0.9
alpha_D = 1.0

# Damping coefficients

kappa_m = 1.8
kappa_i = 0.4
kappa_b = 0.15
kappa_s = 0.6

kappa_AR = 1.0
kappa_B = 0.8
kappa_N = 0.9
kappa_G = 0.7
kappa_D = 0.6

# ============================================================
# CORE DYNAMICAL SYSTEM
# ============================================================

def pms_dynamics(t, X):

    # --------------------------------------------------------
    # STATE VARIABLES
    # --------------------------------------------------------

    Pm = X[0]
    Pi = X[1]
    Pb = X[2]
    Ps = X[3]

    AR = X[4]
    B = X[5]
    N = X[6]
    G = X[7]
    Ds = X[8]

    # --------------------------------------------------------
    # EFFECTIVE DRUG CONCENTRATIONS
    # --------------------------------------------------------

    M_eff = delayed_concentration(
        t,
        morphine_doses,
        MORPHINE
    )

    P_eff = delayed_concentration(
        t,
        paracetamol_doses,
        PARACETAMOL
    )

    E_eff = delayed_concentration(
        t,
        etoricoxib_doses,
        ETORICOXIB
    )

    RM_eff = delayed_concentration(
        t,
        rescue_doses,
        RESCUE_MORPHINE
    )

    # --------------------------------------------------------
    # NONLINEAR SUPPRESSION
    # --------------------------------------------------------

    S_M = hill_suppression(
        M_eff,
        MORPHINE.efficacy,
        MORPHINE.hill_n,
        MORPHINE.half_max
    )

    S_P = hill_suppression(
        P_eff,
        PARACETAMOL.efficacy,
        PARACETAMOL.hill_n,
        PARACETAMOL.half_max
    )

    S_E = hill_suppression(
        E_eff,
        ETORICOXIB.efficacy,
        ETORICOXIB.hill_n,
        ETORICOXIB.half_max
    )

    S_RM = hill_suppression(
        RM_eff,
        RESCUE_MORPHINE.efficacy,
        RESCUE_MORPHINE.hill_n,
        RESCUE_MORPHINE.half_max
    )

    # --------------------------------------------------------
    # EXTERNAL EVENTS
    # --------------------------------------------------------

    movement_force = rectangular_event(
        t,
        movement_events,
        amplitude=1.4
    )

    inflammatory_force = rectangular_event(
        t,
        inflammatory_events,
        amplitude=2.0
    )

    sleep_force = rectangular_event(
        t,
        sleep_break_events,
        amplitude=0.8
    )

    reflux_force = rectangular_event(
        t,
        acid_reflux_events,
        amplitude=1.2
    )

    # --------------------------------------------------------
    # MOVEMENT DYNAMICS
    # --------------------------------------------------------

    dPm = (
        movement_force
        - kappa_m * Pm
        - 0.45 * S_M
        - 0.15 * S_P
    )

    # --------------------------------------------------------
    # INFLAMMATORY DYNAMICS
    # --------------------------------------------------------

    dPi = (
        0.15
        + inflammatory_force
        - kappa_i * Pi
        - 0.8 * S_E
        - 0.25 * S_M
    )

    # --------------------------------------------------------
    # BASELINE DYNAMICS
    # --------------------------------------------------------

    dPb = (
        0.08
        - kappa_b * Pb
        - 0.5 * S_M
    )

    # --------------------------------------------------------
    # SENSORY-AUTONOMIC DYNAMICS
    # --------------------------------------------------------

    dPs = (
        0.5 * G
        + 0.4 * Ds
        + sleep_force
        - kappa_s * Ps
        - 0.2 * S_M
    )

    # --------------------------------------------------------
    # ACID REFLUX CASCADE
    # --------------------------------------------------------

    dAR = (
        reflux_force
        - kappa_AR * AR
    )

    dB = (
        alpha_B * AR
        - kappa_B * B
    )

    dN = (
        alpha_N * B
        - kappa_N * N
    )

    dG = (
        alpha_G * N
        + 0.2 * Ps
        - kappa_G * G
    )

    dDs = (
        alpha_D * G
        + 0.4 * Pi
        + 0.2 * Ps
        - kappa_D * Ds
    )

    return [
        dPm,
        dPi,
        dPb,
        dPs,
        dAR,
        dB,
        dN,
        dG,
        dDs
    ]

# ============================================================
# INITIAL CONDITIONS
# ============================================================

X0 = [
    0.0,
    0.0,
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
]

# ============================================================
# NUMERICAL SOLUTION
# ============================================================

solution = solve_ivp(
    pms_dynamics,
    [0, T_MAX],
    X0,
    t_eval=TIME,
    method='RK45'
)

# ============================================================
# EXTRACT COMPONENTS
# ============================================================

Pm = solution.y[0]
Pi = solution.y[1]
Pb = solution.y[2]
Ps = solution.y[3]

AR = solution.y[4]
B = solution.y[5]
N = solution.y[6]
G = solution.y[7]
Ds = solution.y[8]

# ============================================================
# TOTAL OBSERVABLE PAIN
# ============================================================

P_total = (
    w1 * Pm +
    w2 * Pi +
    w3 * Pb +
    w4 * Ps
)

P_observable_raw = P_total + lambda_ds * Ds

# Physical clamping

P_observable = np.maximum(
    P_observable_raw,
    0
)

# ============================================================
# REGIME CLASSIFIER
# ============================================================

def classify_regime(signal):

    avg = np.mean(signal)

    peaks, _ = find_peaks(
        signal,
        prominence=0.4
    )

    spike_rate = len(peaks) / (T_MAX + 1e-8)

    if avg > 4.0:

        return (
            "Regime-I: Persistent "
            "Unstable Escalation"
        )

    elif spike_rate > 0.12:

        return (
            "Regime-II: Oscillatory "
            "Partial Stabilization"
        )

    else:

        return (
            "Regime-III: Bounded "
            "Perturbative Stabilization"
        )

regime = classify_regime(P_observable)

# ============================================================
# STABILITY METRICS
# ============================================================

def compute_stability_metrics(signal):

    peaks, _ = find_peaks(
        signal,
        prominence=0.3
    )

    metrics = {

        "Mean Pain":
            np.mean(signal),

        "Variance":
            np.var(signal),

        "Maximum Pain":
            np.max(signal),

        "Spike Count":
            len(peaks),

        "Stability Index":
            1 / (1 + np.var(signal))
    }

    return metrics

metrics = compute_stability_metrics(
    P_observable
)

# ============================================================
# PRINT RESULTS
# ============================================================

print("\n==============================")
print(" PMS-V2 STABILITY ANALYSIS")
print("==============================\n")

print(f"Detected Regime: {regime}\n")

for k, v in metrics.items():

    print(f"{k}: {v:.4f}")

# ============================================================
# VISUALIZATION 1
# ============================================================

plt.figure(figsize=(15, 8))

plt.plot(
    TIME,
    P_observable,
    linewidth=3,
    label='Observable Pain'
)

plt.plot(
    TIME,
    P_total,
    '--',
    linewidth=2,
    label='Composite Pain'
)

plt.xlabel('Time (hours)')
plt.ylabel('Pain Intensity')

plt.title(
    'PMS-V2 Observable Pain Evolution'
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "PMS_Observable_Pain.png",
    dpi=300
)

plt.show()

# ============================================================
# VISUALIZATION 2
# ============================================================

plt.figure(figsize=(16, 9))

plt.plot(
    TIME,
    Pm,
    label='Movement Component Pm(t)'
)

plt.plot(
    TIME,
    Pi,
    label='Inflammatory Component Pi(t)'
)

plt.plot(
    TIME,
    Pb,
    label='Baseline Component Pb(t)'
)

plt.plot(
    TIME,
    Ps,
    label='Sensory-Autonomic Component Ps(t)'
)

plt.xlabel('Time (hours)')
plt.ylabel('Component Strength')

plt.title(
    'Multi-Component Nociceptive Structure'
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "PMS_Component_Decomposition.png",
    dpi=300
)

plt.show()

# ============================================================
# VISUALIZATION 3
# ============================================================

plt.figure(figsize=(16, 9))

plt.plot(
    TIME,
    AR,
    label='Reflux Irritation AR(t)'
)

plt.plot(
    TIME,
    B,
    label='Restlessness B(t)'
)

plt.plot(
    TIME,
    N,
    label='Nausea N(t)'
)

plt.plot(
    TIME,
    G,
    label='Autonomic Amplification G(t)'
)

plt.plot(
    TIME,
    Ds,
    label='Dyspnea-like Distress Ds(t)'
)

plt.xlabel('Time (hours)')
plt.ylabel('Activation Strength')

plt.title(
    'Acid-Reflux–Autonomic Perturbative Cascade'
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "PMS_Autonomic_Cascade.png",
    dpi=300
)

plt.show()

# ============================================================
# VISUALIZATION 4
# ============================================================

M_curve = [
    delayed_concentration(
        t,
        morphine_doses,
        MORPHINE
    )
    for t in TIME
]

P_curve = [
    delayed_concentration(
        t,
        paracetamol_doses,
        PARACETAMOL
    )
    for t in TIME
]

E_curve = [
    delayed_concentration(
        t,
        etoricoxib_doses,
        ETORICOXIB
    )
    for t in TIME
]

RM_curve = [
    delayed_concentration(
        t,
        rescue_doses,
        RESCUE_MORPHINE
    )
    for t in TIME
]

plt.figure(figsize=(16, 9))

plt.plot(
    TIME,
    M_curve,
    label='Morphine'
)

plt.plot(
    TIME,
    P_curve,
    label='Paracetamol'
)

plt.plot(
    TIME,
    E_curve,
    label='Etoricoxib'
)

plt.plot(
    TIME,
    RM_curve,
    label='Rescue Morphine'
)

plt.xlabel('Time (hours)')
plt.ylabel('Effective Concentration')

plt.title(
    'Effective Pharmacodynamic Drug Profiles'
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "PMS_Drug_Profiles.png",
    dpi=300
)

plt.show()

# ============================================================
# END
# ============================================================

print("\nPMS-V2 simulation completed successfully.\n")

print("Generated Figures:")
print("1. PMS_Observable_Pain.png")
print("2. PMS_Component_Decomposition.png")
print("3. PMS_Autonomic_Cascade.png")
print("4. PMS_Drug_Profiles.png")