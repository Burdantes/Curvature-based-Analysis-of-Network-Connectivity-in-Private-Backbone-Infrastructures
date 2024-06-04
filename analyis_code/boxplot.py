import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import pandas as pd
import os
from util import month, year, project_dir, start_date, which_cloud

def plot_ricci_curvature_vs_threshold(month, year, threshold_range, cloud_to_study = None):
    """
    Plot the Ricci Curvature against different threshold values for cloud graphs.

    Parameters:
    month (str): The month of the data.
    year (str): The year of the data.
    cloud_to_study (str): The cloud to study ('aws', 'google', 'azure' (only i).
    """

    plt.rc('font', family='serif')

    # Initialize lists to store Ricci Curvature values and corresponding thresholds
    ricci_curvature_values = []
    thresholds = []
    connected_components_sizes = {}

    for threshold in threshold_range:
        try:
            if cloud_to_study:
                # Read the graphml file for the given threshold
                graph = nx.read_graphml(f'{project_dir}/Datasets/Graph/{start_date}/{cloud_to_study}_graph_ricci_{threshold}.graphml')
            else:
                # Read the graphml file for the given threshold
                graph = nx.read_graphml(f'{project_dir}/Datasets/Graph/{start_date}/graph_ricci_{threshold}.graphml')
        except FileNotFoundError:
            # If file is not found, add zero values and continue
            thresholds.append(threshold)
            ricci_curvature_values.extend([0, 0])
            continue
        print(graph.edges(data=True))
        # Get Ricci Curvature values for the edges
        ricci_curvature_dict = nx.get_edge_attributes(graph, 'ricciCurvature')
        ricci_curvature_values.extend(ricci_curvature_dict.values())

        # Get the top 10 edges with the lowest Ricci Curvature values
        top_ricci_curvatures = sorted(ricci_curvature_dict.items(), key=lambda x: x[1])[:10]

        # Get the cities attribute from the graph
        cities = nx.get_node_attributes(graph, 'city')

        # Print the threshold and corresponding cities for debug purposes
        print(threshold, cities)

        # Store the number of connected components for the given threshold
        connected_components_sizes[threshold] = len(sorted(nx.connected_components(graph), key=len, reverse=True))

        # Store the threshold for each Ricci Curvature value
        thresholds.extend([threshold] * len(ricci_curvature_dict))

    # Create a DataFrame for Ricci Curvature values and thresholds
    df = pd.DataFrame(ricci_curvature_values, columns=['Ricci Curvature'])
    df['Threshold'] = pd.Series(thresholds)

    print(df)
    # Create a figure for the plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot()

    # Create a DataFrame for connected components sizes and normalize the values
    size_df = pd.DataFrame(pd.Series(connected_components_sizes), columns=['# of connected components'])
    size_df = size_df.apply(lambda x: ((3 * x / float(max(size_df['# of connected components']))) - 2))
    size_df = size_df[size_df.index.isin(list(set(df['Threshold'].values)))]

    # Plot the boxplot for Ricci Curvature vs Threshold
    sns.boxplot(y='Ricci Curvature', x='Threshold', data=df, width=0.3, color='grey', whis=[3, 97], ax=ax)

    # Set labels and font sizes for the plot
    plt.ylabel('Ricci Curvature', fontsize=25)
    plt.xlabel('Threshold', fontsize=25)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=20)
    if cloud_to_study:
        os.makedirs(f'{project_dir}/Outputs/Visuals/{cloud_to_study}', exist_ok=True)
        # Save the plot as a PDF file
        plt.savefig(f'{project_dir}/Outputs/Visuals/{cloud_to_study}/boxplot{month}-{year}.pdf')
    else:
        os.makedirs(f'{project_dir}/Outputs/Visuals', exist_ok=True)
        # Save the plot as a PDF file
        plt.savefig(f'{project_dir}/Outputs/Visuals/boxplot{month}-{year}.pdf')

# Example usage
plot_ricci_curvature_vs_threshold(month, year, range(2, 120, 10), which_cloud)