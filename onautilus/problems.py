from optproblems.zdt import ZDT1, ZDT2, ZDT3
import numpy as np


def zdt1func(x):
    x = np.asarray(x)
    evaluate = ZDT1()
    if x.ndim == 2:
        return np.asarray([evaluate(xi) for xi in x])
    elif x.ndim == 1:
        return evaluate(x)


def zdt2func(x):
    x = np.asarray(x)
    evaluate = ZDT2()
    if x.ndim == 2:
        return np.asarray([evaluate(xi) for xi in x])
    elif x.ndim == 1:
        return evaluate(x)


def zdt3func(x):
    x = np.asarray(x)
    evaluate = ZDT3()
    if x.ndim == 2:
        return np.asarray([evaluate(xi) for xi in x])
    elif x.ndim == 1:
        return evaluate(x)


def riverfunc(x):
    obj1 = 4.07 + 2.27 * x[:, 0]
    obj2 = (
        2.60
        + 0.03 * x[:, 0]
        + 0.02 * x[:, 1]
        + 0.01 / (1.39 - x[:, 0] ** 2)
        + 0.30 / (1.39 - x[:, 1] ** 2)
    )
    obj3 = 8.21 - 0.71 / (1.09 - x[:, 0] ** 2)
    obj4 = -0.96 + 0.96 / (1.09 - x[:, 1] ** 2)
    obj5 = np.max([np.abs(x[:, 0] - 0.65), np.abs(x[:, 1] - 0.65)], axis=0)
    return np.vstack((obj1, obj2, obj3, obj4, obj5)).T


def vehicle_crash_worthiness(x: np.ndarray):
    """Problem RE3-5-4: The vehicle crashworthiness design problem.

    5 variables, all in the range [1, 3]. 3 Objectives: weight(f_1), accleration
    characteristics(f_2), and toe-board instruction(f_3), all to be minimized.

    Parameters
    ----------
    x : np.ndarray
        Two-dimensional array with 5 columns representing 5 decision variables.
    """
    f1 = (
        1640.2823
        + 2.3573285 * x[:, 0]
        + 2.3220035 * x[:, 1]
        + 4.5688768 * x[:, 2]
        + 7.7213633 * x[:, 3]
        + 4.4559504 * x[:, 4]
    )
    f2 = (
        6.5856
        + 1.15 * x[:, 0]
        - 1.0427 * x[:, 1]
        + 0.9738 * x[:, 2]
        + 0.8364 * x[:, 3]
        - 0.3695 * x[:, 0] * x[:, 3]
        + 0.0861 * x[:, 0] * x[:, 4]
        + 0.3628 * x[:, 1] * x[:, 3]
        - 0.1106 * x[:, 0] ** 2
        - 0.3437 * x[:, 2] ** 2
        + 0.1764 * x[:, 3] ** 2
    )

    f3 = (
        -0.0551
        + 0.0181 * x[:, 0]
        + 0.1024 * x[:, 1]
        + 0.0421 * x[:, 2]
        - 0.0073 * x[:, 0] * x[:, 1]
        + 0.0240 * x[:, 1] * x[:, 2]
        - 0.0118 * x[:, 1] * x[:, 3]
        - 0.0204 * x[:, 2] * x[:, 3]
        - 0.0080 * x[:, 2] * x[:, 4]
        - 0.0241 * x[:, 1] ** 2
        + 0.0109 * x[:, 3] ** 2
    )
    return np.vstack((f1, f2, f3)).T


def rocket_injector(x: np.ndarray):
    """The rocket injector problem.

    Four decision variables, all in range [0, 1]. Three objectives: the maximum
    temperature of the injector face (f_1), the distance from the inlet (f_2), and the
    maximum temperature on the post tip (f_3), all to be minimized.

    Parameters
    ----------
    x : np.ndarray
        Two-dimensional array with 4 columns representing 4 decision variables.
    """
    f1 = (
        0.692
        + (0.47700 * x[:, 0])
        - (0.68700 * x[:, 1])
        - (0.08000 * x[:, 2])
        - (0.06500 * x[:, 3])
        - (0.16700 * x[:, 0] * x[:, 0])
        - (0.01290 * x[:, 1] * x[:, 0])
        + (0.07960 * x[:, 1] * x[:, 1])
        - (0.06340 * x[:, 2] * x[:, 0])
        - (0.02570 * x[:, 2] * x[:, 1])
        + (0.08770 * x[:, 2] * x[:, 2])
        - (0.05210 * x[:, 3] * x[:, 0])
        + (0.00156 * x[:, 3] * x[:, 1])
        + (0.00198 * x[:, 3] * x[:, 2])
        + (0.01840 * x[:, 3] * x[:, 3])
    )
    f2 = (
        0.153
        - (0.3220 * x[:, 0])
        + (0.3960 * x[:, 1])
        + (0.4240 * x[:, 2])
        + (0.0226 * x[:, 3])
        + (0.1750 * x[:, 0] * x[:, 0])
        + (0.0185 * x[:, 1] * x[:, 0])
        - (0.0701 * x[:, 1] * x[:, 1])
        - (0.2510 * x[:, 2] * x[:, 0])
        + (0.1790 * x[:, 2] * x[:, 1])
        + (0.0150 * x[:, 2] * x[:, 2])
        + (0.0134 * x[:, 3] * x[:, 0])
        + (0.0296 * x[:, 3] * x[:, 1])
        + (0.0752 * x[:, 3] * x[:, 2])
        + (0.0192 * x[:, 3] * x[:, 3])
    )
    f3 = (
        0.370
        - (0.2050 * x[:, 0])
        + (0.0307 * x[:, 1])
        + (0.1080 * x[:, 2])
        + (1.0190 * x[:, 3])
        - (0.1350 * x[:, 0] * x[:, 0])
        + (0.0141 * x[:, 1] * x[:, 0])
        + (0.0998 * x[:, 1] * x[:, 1])
        + (0.2080 * x[:, 2] * x[:, 0])
        - (0.0301 * x[:, 2] * x[:, 1])
        - (0.2260 * x[:, 2] * x[:, 2])
        + (0.3530 * x[:, 3] * x[:, 0])
        - (0.0497 * x[:, 3] * x[:, 2])
        - (0.4230 * x[:, 3] * x[:, 3])
        + (0.2020 * x[:, 1] * x[:, 0] * x[:, 0])
        - (0.2810 * x[:, 2] * x[:, 0] * x[:, 0])
        - (0.3420 * x[:, 1] * x[:, 1] * x[:, 0])
        - (0.2450 * x[:, 1] * x[:, 1] * x[:, 2])
        + (0.2810 * x[:, 2] * x[:, 2] * x[:, 1])
        - (0.1840 * x[:, 3] * x[:, 3] * x[:, 0])
        - (0.2810 * x[:, 2] * x[:, 0] * x[:, 2])
    )
    return np.vstack((f1, f2, f3)).T
