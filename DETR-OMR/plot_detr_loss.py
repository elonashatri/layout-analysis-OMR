import json
import matplotlib.pyplot as plt

# Read the JSON file
with open('val_losses.json', 'r') as file:
    val_losses = json.load(file)

# Plot the values
plt.figure(figsize=(10, 6))
plt.plot(val_losses, marker='o', linestyle='-', color='b', label='valid Loss')
plt.xlabel('Index')
plt.ylabel('Loss Value')
plt.title('valid Loss Over Time')
plt.legend()
plt.grid(True)
plt.show()
