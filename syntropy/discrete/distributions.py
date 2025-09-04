import itertools as it

GIANT_BIT: dict[tuple[int, int, int], float] = {(0, 0, 0): 1 / 2, (1, 1, 1): 1 / 2}

XOR_DIST: dict[tuple[int, int, int], float] = {
    (0, 0, 0): 1 / 4,
    (0, 1, 1): 1 / 4,
    (1, 0, 1): 1 / 4,
    (1, 1, 0): 1 / 4,
}

AND_DIST: dict[tuple[int, int, int], float] = {
    (0, 0, 0): 1 / 4,
    (0, 1, 0): 1 / 4,
    (1, 0, 0): 1 / 4,
    (1, 1, 1): 1 / 4,
}

OR_DIST: dict[tuple[int, int, int], float] = {
    (0, 0, 0): 1 / 4,
    (0, 1, 1): 1 / 4,
    (1, 0, 1): 1 / 4,
    (1, 1, 1): 1 / 4,
}

ONE_HOT_2_DIST: dict[tuple[int, int], float] = {(0, 1): 1 / 2, (1, 0): 1 / 2}

ONE_HOT_3_DIST: dict[tuple[int, int, int], float] = {
    (0, 0, 1): 1 / 3,
    (0, 1, 0): 1 / 3,
    (1, 0, 0): 1 / 3,
}

ONE_HOT_4_DIST: dict[tuple[int, int, int, int], float] = {
    (0, 0, 0, 1): 1 / 4,
    (0, 0, 1, 0): 1 / 4,
    (0, 1, 0, 0): 1 / 4,
    (1, 0, 0, 0): 1 / 4,
}

JAMES_DYADIC: dict[tuple[int, int, int], float] = {
    (0, 0, 0): 0.125,
    (0, 2, 1): 0.125,
    (1, 0, 2): 0.125,
    (1, 2, 3): 0.125,
    (2, 1, 0): 0.125,
    (2, 3, 1): 0.125,
    (3, 1, 2): 0.125,
    (3, 3, 3): 0.125,
}

JAMES_TRIADIC: dict[tuple[int, int, int], float] = {
    (0, 0, 0): 0.125,
    (0, 2, 2): 0.125,
    (1, 1, 1): 0.125,
    (1, 3, 3): 0.125,
    (2, 0, 2): 0.125,
    (2, 2, 0): 0.125,
    (3, 1, 3): 0.125,
    (3, 3, 1): 0.125,
}

TWO_BIT_COPY: dict[tuple[int, int, int], float] = {
    (0, 0, 0): 1 / 4,
    (0, 1, 1): 1 / 4,
    (1, 0, 2): 1 / 4,
    (1, 1, 3): 1 / 4,
}

SUMMED_DICE: dict[tuple[int, int, int], float] = {
    (i, j, i + j): 1 / 36 for i in range(1, 7) for j in range(1, 7)
}

numbers: tuple[str, ...] = tuple(str(i) for i in range(2, 11)) + ("J", "Q", "K", "A")
suits: tuple[str, str, str, str] = ("C", "D", "H", "S")
N: int = len(numbers) * len(suits)

DECK_OF_CARDS: dict[tuple[str, str], float] = {
    (numbers[i], suits[j]): 1 / N
    for i in range(len(numbers))
    for j in range(len(suits))
}

probs: list[float, ...] = [0.1545648, 0.17202542, 0.31893614, 0.35447365]
states: list[tuple[int, int], ...] = list(it.product((0, 1), repeat=2))

RANDOM_DIST_2: dict[tuple[int, int], float] = {states[i]: probs[i] for i in range(4)}

probs: list[float, ...] = [
    0.16753926,
    0.02883267,
    0.3976242,
    0.14245073,
    0.0096225,
    0.02749607,
    0.21933501,
    0.00709958,
]
states: list[tuple[int, int, int], ...] = list(it.product((0, 1), repeat=3))

RANDOM_DIST_3: dict[tuple[int, int, int], float] = {
    states[i]: probs[i] for i in range(8)
}

probs: list[float, ...] = [
    0.01518969,
    0.00366897,
    0.03148444,
    0.14726786,
    0.04471321,
    0.0551291,
    0.08041194,
    0.06299127,
    0.0430596,
    0.01575873,
    0.03058048,
    0.19003903,
    0.03761112,
    0.0023867,
    0.06482958,
    0.17487828,
]
states: list[tuple[int, int, int, int], ...] = list(it.product((0, 1), repeat=4))

RANDOM_DIST_4: dict[tuple[int, int, int, int], float] = {
    states[i]: probs[i] for i in range(16)
}

MAXENT_DIST_3: dict[tuple[int, int, int], float] = {
    (0, 0, 0): 1 / 8,
    (0, 0, 1): 1 / 8,
    (0, 1, 0): 1 / 8,
    (0, 1, 1): 1 / 8,
    (1, 0, 0): 1 / 8,
    (1, 0, 1): 1 / 8,
    (1, 1, 0): 1 / 8,
    (1, 1, 1): 1 / 8,
}

MAXENT_DIST_4: dict[tuple[int, int, int, int], float] = {
    (0, 0, 0, 0): 1 / 16,
    (0, 0, 0, 1): 1 / 16,
    (0, 0, 1, 0): 1 / 16,
    (0, 0, 1, 1): 1 / 16,
    (0, 1, 0, 0): 1 / 16,
    (0, 1, 0, 1): 1 / 16,
    (0, 1, 1, 0): 1 / 16,
    (0, 1, 1, 1): 1 / 16,
    (1, 0, 0, 0): 1 / 16,
    (1, 0, 0, 1): 1 / 16,
    (1, 0, 1, 0): 1 / 16,
    (1, 0, 1, 1): 1 / 16,
    (1, 1, 0, 0): 1 / 16,
    (1, 1, 0, 1): 1 / 16,
    (1, 1, 1, 0): 1 / 16,
    (1, 1, 1, 1): 1 / 16,
}
