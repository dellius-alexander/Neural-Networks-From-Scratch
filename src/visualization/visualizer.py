import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from abc import ABC, abstractmethod

from src.utils.plot import get_camera_view


class NeuralNetworkVisualizer:
    def __init__(self, layer_nodes, weights, biases):
        """
        Initialize the visualizer with the structure of the neural network.

        :param layer_nodes: List of integers representing the number of nodes in each layer.
        :param weights: List of weight matrices (one matrix per layer).
        :param biases: List of bias vectors (one vector per layer).
        """
        self.layer_nodes = layer_nodes
        self.weights = weights
        self.biases = biases

    def _plot_connections_2d(self, ax, x1, y1, x2, y2, weights):
        """
        Plot connections between two layers in 2D.

        :param ax: The matplotlib axes object.
        :param x1, y1: Coordinates of the first layer nodes.
        :param x2, y2: Coordinates of the second layer nodes.
        :param weights: Weight matrix connecting the layers.
        """
        for i, (x_start, y_start) in enumerate(zip(x1, y1)):
            for j, (x_end, y_end) in enumerate(zip(x2, y2)):
                weight = weights[i][j]
                ax.plot([x_start, x_end], [y_start, y_end], color='black', linewidth=abs(weight), alpha=0.7)

    def visualize_2d(self):
        """
        Visualize the neural network in 2D.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        num_layers = len(self.layer_nodes)

        # Set up coordinates for the layers
        layer_spacing = 2.0
        max_nodes = max(self.layer_nodes)
        x_coords = np.arange(0, num_layers * layer_spacing, layer_spacing)

        for l in range(num_layers):
            y_coords = np.linspace(-max_nodes/2, max_nodes/2, self.layer_nodes[l])
            ax.scatter([x_coords[l]]*self.layer_nodes[l], y_coords, s=100, color='blue', zorder=3)
            if l > 0:
                self._plot_connections_2d(ax, [x_coords[l-1]]*self.layer_nodes[l-1],
                                          np.linspace(-max_nodes/2, max_nodes/2, self.layer_nodes[l-1]),
                                          [x_coords[l]]*self.layer_nodes[l], y_coords, self.weights[l-1])

        ax.set_title("2D Neural Network Visualization")
        ax.set_xlabel("Layers")
        ax.set_ylabel("Neurons")
        plt.show()

    def _plot_connections_3d(self, ax, x1, y1, z1, x2, y2, z2, weights):
        """
        Plot connections between two layers in 3D.

        :param ax: The matplotlib 3D axes object.
        :param x1, y1, z1: Coordinates of the first layer nodes.
        :param x2, y2, z2: Coordinates of the second layer nodes.
        :param weights: Weight matrix connecting the layers.
        """
        for i, (x_start, y_start, z_start) in enumerate(zip(x1, y1, z1)):
            for j, (x_end, y_end, z_end) in enumerate(zip(x2, y2, z2)):
                weight = weights[i][j]
                ax.plot([x_start, x_end], [y_start, y_end], [z_start, z_end], color='black', linewidth=abs(weight), alpha=0.7)

    def visualize_3d(self):
        """
        Visualize the neural network in 3D.
        """
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        num_layers = len(self.layer_nodes)

        # Set up coordinates for the layers
        layer_spacing = 3.0
        max_nodes = max(self.layer_nodes)
        x_coords = np.arange(0, num_layers * layer_spacing, layer_spacing)

        for l in range(num_layers):
            y_coords = np.linspace(-max_nodes/2, max_nodes/2, self.layer_nodes[l])
            z_coords = np.zeros_like(y_coords)
            ax.scatter([x_coords[l]]*self.layer_nodes[l], y_coords, z_coords, s=100, color='blue', zorder=3)
            if l > 0:
                self._plot_connections_3d(ax, [x_coords[l-1]]*self.layer_nodes[l-1],
                                          np.linspace(-max_nodes/2, max_nodes/2, self.layer_nodes[l-1]),
                                          np.zeros(self.layer_nodes[l-1]),
                                          [x_coords[l]]*self.layer_nodes[l], y_coords, z_coords, self.weights[l-1])

        ax.set_title("3D Neural Network Visualization")
        ax.set_xlabel("Layers")
        ax.set_ylabel("Neurons")
        ax.set_zlabel("Depth")
        plt.show()


# Base Strategy Interface for different visualization methods
class GraphStrategy(ABC):
    @abstractmethod
    def plot(self):
        pass


# Interactive Graph Strategy using Plotly
class InteractiveGraphStrategy(GraphStrategy):
    def __init__(self, layer_nodes, weights, biases):
        """
        Initialize the interactive graph strategy.
        :param layer_nodes: List of integers representing the number of nodes in each layer.
        :param weights: List of weight matrices (one matrix per layer).
        :param biases: List of bias vectors (one vector per layer).
        """
        self.layer_nodes = layer_nodes
        self.weights = weights
        self.biases = biases

    def plot(self):
        """
        Plot the interactive neural network graph using Plotly.
        """
        num_layers = len(self.layer_nodes)
        max_nodes = max(self.layer_nodes)
        layer_spacing = 3.0

        # Coordinates for layers
        x_coords = np.arange(0, num_layers * layer_spacing, layer_spacing)

        # Store scatter plots and connections for Plotly
        node_traces = []
        edge_traces = []

        for l in range(num_layers):
            y_coords = np.linspace(-max_nodes / 2, max_nodes / 2, self.layer_nodes[l])
            z_coords = np.zeros_like(y_coords)
            # Add scatter plot for nodes
            node_trace = go.Scatter3d(
                x=[x_coords[l]] * self.layer_nodes[l],
                y=y_coords,
                z=z_coords,
                mode='markers',
                marker=dict(size=10, color=x_coords, colorscale='Viridis', opacity=0.7),
                name=f'Layer {l}'
            )
            node_traces.append(node_trace)

            if l > 0:
                prev_y_coords = np.linspace(-max_nodes / 2, max_nodes / 2, self.layer_nodes[l - 1])
                prev_z_coords = np.zeros_like(prev_y_coords)
                # Add edges between layers based on weights
                for i, (x_start, y_start, z_start) in enumerate(
                        zip([x_coords[l - 1]] * self.layer_nodes[l - 1], prev_y_coords, prev_z_coords)):
                    for j, (x_end, y_end, z_end) in enumerate(
                            zip([x_coords[l]] * self.layer_nodes[l], y_coords, z_coords)):
                        weight = self.weights[l - 1][i][j]
                        edge_trace = go.Scatter3d(
                            x=[x_start, x_end],
                            y=[y_start, y_end],
                            z=[z_start, z_end],
                            mode='lines+markers',
                            line=dict(width=abs(weight) * 2, color='black'),
                            opacity=0.7,
                            showlegend=False
                        )
                        edge_traces.append(edge_trace)

        # Create layout for 3D plot
        layout = go.Layout(
            title={
                "text": "Interactive Neural Network Visualization",
                "y": 0.95,
                "x": 0.5,
                "xanchor": "center",
                "yanchor": "top",
                "font": {"size": 20},  # Adjust the size as needed
            },
            scene=dict(
                xaxis=dict(title='Layers'),
                yaxis=dict(title='Neurons'),
                zaxis=dict(title='Depth')
            )
        )

        fig = go.Figure(data=node_traces + edge_traces, layout=layout)
        # Set the camera view (orientation)
        camera = get_camera_view("top-down")

        # Adjust the camera view
        fig.update_layout(
            scene_camera=camera,
            margin=dict(l=0, r=0, t=0, b=0)
        )
        # fig.show()
        return fig


# Visualization Factory for flexible visualization creation
class VisualizationFactory:
    def __init__(self, strategy: GraphStrategy):
        self.strategy = strategy

    def visualize(self):
        return self.strategy.plot()


if __name__ == "__main__":
    # Sample usage with Plotly interactive graph
    layer_nodes = [3, 5, 2]  # 3 input neurons, 5 hidden, 2 output
    weights = [
        np.random.randn(3, 5),  # Weights between input and hidden layer
        np.random.randn(5, 2)  # Weights between hidden and output layer
    ]
    biases = [
        np.random.randn(5),  # Biases for hidden layer
        np.random.randn(2)  # Biases for output layer
    ]

    # Use InteractiveGraphStrategy to plot an interactive neural network graph
    interactive_strategy = InteractiveGraphStrategy(layer_nodes, weights, biases)
    visualization_factory = VisualizationFactory(interactive_strategy)
    visualization_factory.visualize().show()