import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# Define the color mapping for super pathways
super_pathway_colors = {
    "Amino Acids": "#d62728",
    "Carbohydrates": "#7f7f7f",
    "Cofactors and Vitamins": "#9467bd",
    "Energy": "#8c564b",
    "Lipids": "#ba8e23",
    "Nucleotides": "#2ca02c",
    "Peptides": "#e377c2",
    "Xenobiotics": "#1f77b4"
}

# Function to create donut plot for super pathways, considering only the specific challenge
def create_filtered_donut(ax, data, challenge_value, title):
    # Filter data based on the challenge value (e.g., Fasting, PAT, OLTT)
    filtered_data = data[data["challenge"] == challenge_value]

    # Count the metabolites by super pathway for the specific challenge
    super_pathway_counts = filtered_data["super_pathway"].value_counts()

    # Sizes and labels for the donut plot
    sizes = super_pathway_counts
    labels = super_pathway_counts.index

    # Assign colors based on super pathway
    colors = [super_pathway_colors[label] for label in labels]

    # Plotting the donut (no labels inside the donut, but displaying counts inside segments)
    wedges, texts, autotexts = ax.pie(
        sizes, labels=None, colors=colors, startangle=90,
        wedgeprops=dict(width=0.4, edgecolor='w'),  # Increase width to make the donut thicker
        autopct='%1.0f%%', pctdistance=0.85  # Adjusted distance for inner labels
    )

    # Format the text in the plot
    for i, autotext in enumerate(autotexts):
        autotext.set_fontsize(14)  # Adjust font size
        autotext.set_verticalalignment('center')  # Align text properly
        autotext.set_text(f'{sizes[i]}')  # Display counts inside each segment

    # Add the total number of significant metabolites in the center
    total_significant = len(filtered_data)
    ax.text(0, 0, f"{title}\ntotal\n{total_significant}", ha='center', va='center', fontsize=16)

# Main function to execute the pipeline
def main():
    # Load the data
    data = pd.read_csv("results/unique_metabolites_per_challenge.csv")

    # Create subplots
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # Generate donut plots for each of the challenges
    create_filtered_donut(axs[0], data, 'Fasting', 'Fasting')
    create_filtered_donut(axs[1], data, 'PAT', 'Physical Activity')
    create_filtered_donut(axs[2], data, 'OLTT', 'OLTT')

    # Create one shared legend for all three plots with square markers (quadrats)
    legend_handles = [mlines.Line2D([], [], color=color, marker='s', linestyle='None', markersize=20)
                      for color in super_pathway_colors.values()]
    legend_labels = list(super_pathway_colors.keys())

    # Place the legend at the top of the figure in one line
    fig.legend(legend_handles, legend_labels, loc="upper center", bbox_to_anchor=(0.5, 1.05),
               fontsize=13, frameon=False, ncol=len(super_pathway_colors))

    # Adjust layout to avoid overlap
    plt.tight_layout()

    # Save the plot to a file (e.g., PNG format)
    plt.savefig("results/plots/unique_metabolites_per_superpathways.png", dpi=300, bbox_inches='tight')

    # Show the plot
    plt.show()

# Call the main function to run the script
if __name__ == "__main__":
    main()
