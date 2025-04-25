import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

input_csv = r"data\raw\solusdt_5m_2024_2025.csv"
df = pd.read_csv(input_csv, parse_dates=['timestamp'])

df['percent_change'] = (df['close'] - df['open']) / df['open'] * 100

num_classes = 9

# Create quantile-based bins
if num_classes % 2 == 0:
    print("Even class value. Results may be incorrect, use odd value.")
class_midpoint = num_classes // 2
class_labels = [str(i - class_midpoint) for i in range(num_classes)]
df['class'] = pd.qcut(df['percent_change'], q=num_classes, labels=class_labels)

# Display results
print(df[['percent_change', 'class']].head())

# Calculate quantile values
quantile_vals = df['percent_change'].quantile([i/num_classes for i in range(1, num_classes)])
print("\nQuantile Values:")
print(list(quantile_vals))

# # Save quantile values to a text file
# output_txt_file = "quantile_values.txt"
# quantile_vals.to_csv(output_txt_file, header=['Quantile Value'], index=False)
# print(f"Quantile values saved to {output_txt_file}")

# Remove outliers
q1 = df['percent_change'].quantile(0.25)
q3 = df['percent_change'].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

# Filter outliers
df_filtered = df[(df['percent_change'] >= lower_bound) & (df['percent_change'] <= upper_bound)]

# Calculate class frequencies
class_frequencies = df_filtered['class'].value_counts().sort_index()
print("\nClass Frequencies:")
print(class_frequencies)

# Plot distribution
plt.figure(figsize=(14, 8))
df_filtered['percent_change'].hist(bins=200, range=(-1, 1))

plt.title('Distribution of Percentage Changes (Outliers Removed)')
plt.xlabel('Percentage Change (%)')
plt.ylabel('Frequency')
plt.grid(True)

df_filtered['percent_change'].plot(kind='kde', ax=plt.gca(), secondary_y=True, color='red')
plt.ylabel('Density', fontsize=12, rotation=270, labelpad=12)

plt.tight_layout()

# X-axis labels
x_ticks = np.concatenate([[-1], quantile_vals.values, [1]])
x_tick_labels = [f'{x:.2f}%' for x in x_ticks]
plt.xticks(x_ticks, x_tick_labels, rotation=45, ha='right')
plt.show()
