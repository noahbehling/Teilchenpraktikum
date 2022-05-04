import numpy as np
import pandas as pd
import os

forbidden_strings = ["MC", "Weight", "Corsika"]

# Inf, NaN und unnötige Attribute aus Signal entfernen
signal = pd.read_csv(f"{os.getcwd()}/IceCube/data/signal.csv", delimiter=";")

kill_columns = [any(x in y for x in forbidden_strings) for y in signal.columns.values]

signal = signal.drop(columns=signal.columns.values[kill_columns])
signal = signal.replace([np.inf, -np.inf], np.nan)
signal = signal.dropna(axis=1, how="any")

# Inf, NaN und unnötige Attribute aus Hintergrund entfernen
background = pd.read_csv(f"{os.getcwd()}/IceCube/data/background.csv", delimiter=";")

kill_columns = [
    any(x in y for x in forbidden_strings) for y in background.columns.values
]

background = background.drop(columns=background.columns.values[kill_columns])
background = background.replace([np.inf, -np.inf], np.nan)
background = background.dropna(axis=1, how="any")


# Attribute entfernen, die nur in einem df enthalten sind
common_cols = [col for col in set(signal.columns).intersection(background.columns)]
print(len(common_cols))

kill_uniques = [any(x in y for x in common_cols) for y in signal.columns.values]
kill_uniques = np.logical_not(kill_uniques)
signal = signal.drop(columns=signal.columns.values[kill_uniques])
kill_uniques = [any(x in y for x in common_cols) for y in background.columns.values]
kill_uniques = np.logical_not(kill_uniques)
background = background.drop(columns=background.columns.values[kill_uniques])
