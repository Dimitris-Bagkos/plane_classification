import pandas as pd
import matplotlib.pyplot as plt

# Load the data
filepath = r"clap_similarity_scores_new.csv"
df = pd.read_csv(filepath)

# Select only the two columns of interest
x_col = "a plane cruising"
y_col = "a quiet forest"

x = df[x_col]
y = df[y_col]

# Create the scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(x, y)

# Annotate each point with its index or filename if available
for i, (xi, yi) in enumerate(zip(x, y)):
    plt.annotate(str(i), (xi, yi), textcoords="offset points", xytext=(5, 5), ha='left')

plt.title("CLAP Similarities: Plane vs Car")
plt.xlabel(x_col)
plt.ylabel(y_col)
plt.grid(True)
plt.tight_layout()
plt.show()
