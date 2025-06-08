import pandas as pd

# Define the columns you need
cols = ['frame','x1','y1','x2','y2','class']
df = pd.DataFrame(columns=cols)

# Save an empty template
df.to_csv('ground_truth_main.csv', index=False)
print("Created empty template ground_truth.csv")
