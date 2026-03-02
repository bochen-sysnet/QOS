import pandas as pd
import matplotlib.pyplot as plt

# Dataset: gate-based qubit counts over time
data = [
    ("IBM", 2019, 27),
    ("IBM", 2020, 65),
    ("IBM", 2021, 127),
    ("IBM", 2022, 433),
    ("IBM", 2023, 1121),

    ("Rigetti", 2020, 32),
    ("Rigetti", 2022, 80),

    ("IonQ", 2022, 36),
    ("IonQ", 2025, 64),

    ("Quantinuum", 2022, 20),
    ("Quantinuum", 2023, 32),
    ("Quantinuum", 2024, 56)
]

df = pd.DataFrame(data, columns=["Provider", "Year", "Qubits"])

# font size
plt.rcParams.update({'font.size': 20})

# Plot
plt.figure(figsize=(8, 4))
label_offsets = {
    "IBM": (8, -20),
    "Rigetti": (8, -8),
    "IonQ": (8, 30),
    "Quantinuum": (2, -20),
}
value_offsets = {
    "IBM": {"min": (-18, -12), "max": (-18, 10)},
    "Rigetti": {"min": (-12, 12), "max": (8, 10)},
    "IonQ": {"min": (-60, 0), "max": (10, 10)},
    "Quantinuum": {"min": (-22, -8), "max": (-16, 10)},
}

for provider, group in df.groupby("Provider"):
    group = group.sort_values("Year")
    line, = plt.plot(group["Year"], group["Qubits"], marker="o")
    x_last = group["Year"].iloc[-1]
    y_last = group["Qubits"].iloc[-1]
    dx, dy = label_offsets.get(provider, (8, 0))
    plt.annotate(
        provider,
        xy=(x_last, y_last),
        xytext=(dx, dy),
        textcoords="offset points",
        color=line.get_color(),
        ha="left",
        va="center",
    )

    min_row = group.loc[group["Qubits"].idxmin()]
    max_row = group.loc[group["Qubits"].idxmax()]
    min_dx, min_dy = value_offsets.get(provider, {}).get("min", (-12, -10))
    max_dx, max_dy = value_offsets.get(provider, {}).get("max", (8, 10))

    if min_row.name == max_row.name:
        plt.annotate(
            f"min=max {int(min_row['Qubits'])}",
            xy=(min_row["Year"], min_row["Qubits"]),
            xytext=(max_dx, max_dy),
            textcoords="offset points",
            color=line.get_color(),
            fontsize=13,
            ha="left",
            va="center",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.8, ec="none"),
        )
    else:
        plt.annotate(
            f"min {int(min_row['Qubits'])}",
            xy=(min_row["Year"], min_row["Qubits"]),
            xytext=(min_dx, min_dy),
            textcoords="offset points",
            color=line.get_color(),
            fontsize=13,
            ha="left",
            va="center",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.8, ec="none"),
        )
        plt.annotate(
            f"max {int(max_row['Qubits'])}",
            xy=(max_row["Year"], max_row["Qubits"]),
            xytext=(max_dx, max_dy),
            textcoords="offset points",
            color=line.get_color(),
            fontsize=13,
            ha="left",
            va="center",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.8, ec="none"),
        )

plt.yscale("log")
plt.xlabel("Year")
plt.ylabel("Qubit Count")
plt.xlim(df["Year"].min() - 0.2, df["Year"].max() + 1.0)
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.savefig("evaluation/plots/motivation_results/qubit_number_results.pdf")
