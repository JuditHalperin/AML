import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from torch import Tensor
from utils import save_plot
from create_data import sample_olympic_rings


def _get_colors():
    return {0: 'black', 1: 'blue', 2: 'green', 3: 'red', 4: 'yellow'}


def _get_color_ranges(time_range):
    return {0: cm.Greys(time_range), 1: cm.Blues(time_range), 2: cm.Greens(time_range), 3: cm.Reds(time_range), 4: cm.YlOrBr(time_range)}


def plot_losses(losses: dict[str, list[float]], num_epochs: int, title: str = 'Loss', exp_name: str = None) -> None:
    """
    Plot any type of losses across epochs
    """
    for loss_name, loss_vals in losses.items():
        plt.plot(range(1, num_epochs + 1), loss_vals, label=loss_name)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    save_plot(title, exp_name)


def plot_data(data: Tensor | list[Tensor], labels: list[int] = None, size: int = 1, alpha: float = None, title: str = 'Data', exp_name: str = None) -> None:
    """
    Plot points in 2D
    data: 2D tensor or list of tensors to plot
    labels: list of point classes to plot in different colors (optional)
    """
    color_map = _get_colors()
    colors = [color_map[label] if isinstance(label, int) else label for label in labels] if labels else None
    data = data if isinstance(data, list) else [data]
    for d in data:
        x = d.detach().numpy()
        plt.scatter(np.array(x[:, 0]), np.array(x[:, 1]), s=size, alpha=alpha, c=colors)
    plt.title(title)
    save_plot(title, exp_name)


def plot_trajectories(trajectories: list[Tensor], labels: list[int] = None, by_point: bool = False, axis_lim: int = 2, add_rings: bool = False, title: str = 'Trajectories', exp_name: str = None) -> None:
    """
    Plot point trajectories
    trajectories: list of trajectory tensors
    labels: list of point classes to plot in different colors (optional)
    by_point: whether to plot each point in a separate plot or all points in the same figure
    axis_lim: limit value in both axises
    add_rings: whether to plot the rings in the background using 100 sampled points of each class
    """
    num_points = trajectories[0].shape[0]

    time_range = np.linspace(0, 1, len(trajectories))
    colors = _get_color_ranges(time_range)

    if by_point:
        nrow, ncol = int(np.ceil(num_points / 4)), min(4, num_points)
    for i in range(num_points):
        if by_point:
            plt.subplot(nrow, ncol, i + 1)
        for t, traj in enumerate(trajectories):
            plt.scatter(traj[i, 0], traj[i, 1], color=colors[labels[i] if labels else 1][t])
            if by_point:
                plt.title(f'Point {i}')
            if axis_lim:
                plt.xlim((-axis_lim, axis_lim))
                plt.ylim((-axis_lim, axis_lim))
                        
    if not by_point:
        plt.title(f'{title} - {num_points} Points' if num_points > 1 else title)


    if add_rings:
        points, classses = sample_olympic_rings(100)
        points = (points - np.mean(points, axis=0)) / np.std(points, axis=0)
        x = Tensor(points).detach().numpy()
        plt.scatter(np.array(x[:, 0]), np.array(x[:, 1]), s=0.3, c=classses)

    save_plot(title, exp_name)

