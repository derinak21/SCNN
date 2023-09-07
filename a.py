import matplotlib.pyplot as plt

# Create a figure and axis for the diagram
fig, ax = plt.subplots(figsize=(8, 4))

# Define the number of input channels, output channels, and kernel size
in_channels = 1
out_channels = 32
kernel_size = 3

# Define the positions of layers
input_x = 0.2
conv_x = 1.0
output_x = 1.8

# Define the vertical position for all layers
layer_height = 0.4

# Create rectangles for layers
input_layer = plt.Rectangle((input_x, 0), 0.6, layer_height, fill=False, edgecolor='black', linewidth=2)
conv_layer = plt.Rectangle((conv_x, 0), 0.6, layer_height, fill=False, edgecolor='black', linewidth=2)
output_layer = plt.Rectangle((output_x, 0), 0.6, layer_height, fill=False, edgecolor='black', linewidth=2)

# Add the rectangles to the plot
ax.add_patch(input_layer)
ax.add_patch(conv_layer)
ax.add_patch(output_layer)

# Add text labels for layers
ax.text(input_x + 0.3, layer_height * 0.5, f'Input\n({in_channels} channel)', ha='center', va='center', fontsize=12)
ax.text(conv_x + 0.3, layer_height * 0.5, f'Conv1D\n({out_channels} filters,\n{kernel_size}x1 kernel)', ha='center', va='center', fontsize=12)
ax.text(output_x + 0.3, layer_height * 0.5, 'Output', ha='center', va='center', fontsize=12)

# Create lines to connect layers
line1 = plt.Line2D([input_x + 0.3, conv_x], [layer_height * 0.5, layer_height * 0.5], color='black', linewidth=2)
line2 = plt.Line2D([conv_x + 0.3, output_x], [layer_height * 0.5, layer_height * 0.5], color='black', linewidth=2)

# Add the lines to the plot
ax.add_line(line1)
ax.add_line(line2)

# Set axis limits and labels
ax.set_xlim(0, 2.5)
ax.set_ylim(0, layer_height)
ax.axis('off')

# Show the diagram
plt.show()
