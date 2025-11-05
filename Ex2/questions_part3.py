import torch
import numpy as np
from train import train_normalizing_flow
from visualization import plot_data, plot_trajectories
from data import pick_points


def q3_1(exp_name: str):
    train_normalizing_flow(plot_loss=True, exp_title=exp_name)


def q3_2(exp_name: str, epoch: int = 20, num_seeds: int = 3):
    
    model = torch.load(f'weights/{exp_name}/weights_epoch_{epoch}.pth')
    model.eval()
    with torch.no_grad():
        for i in range(num_seeds):
            x = model.sample(seed=i)
            plot_data(x, title=f'Samples - Seed {i}', exp_name=exp_name)


def q3_3(exp_name: str, epoch: int = 20, num_points: int = 1000, num_layers: int = 6):

    model = torch.load(f'weights/{exp_name}/weights_epoch_{epoch}.pth')
    data = torch.randn(num_points, model.dim_size)

    model.eval()
    with torch.no_grad():
        for i, layer in enumerate(model.flow):
            data = layer(data)
            if i in np.linspace(0, len(model.flow) - 1, num_layers, dtype=int):
                plot_data(data, title=f'Samples Over Time - Layer {i}', exp_name=exp_name)


def q3_4(exp_name: str, epoch: int = 20, num_points: int = 10):

    model = torch.load(f'weights/{exp_name}/weights_epoch_{epoch}.pth')
    data = torch.randn(num_points, model.dim_size)

    trajectories = []
    model.eval()
    with torch.no_grad():
        for layer in model.flow:
            data = layer(data)
            trajectories.append(data.numpy().copy())

    plot_trajectories(trajectories, by_point=True, axis_lim=2, title='Forward trajectories', exp_name=exp_name)


def q3_5(exp_name: str, epoch: int = 20):

    model = torch.load(f'weights/{exp_name}/weights_epoch_{epoch}.pth')
    model.eval()

    # Point trajectories
    points_on, points_in, points_out = pick_points()
    data = torch.cat([points_on, points_in, points_out], dim=0)
    point_locations = [3] * len(points_on) + [1] * len(points_in) + [2] * len(points_out)

    trajectories = []
    with torch.no_grad():
        for layer in model.flow[::-1]:
            data = layer.inverse(data)
            trajectories.append(data.numpy().copy())

    plot_trajectories(trajectories, labels=point_locations, by_point=True, axis_lim=4, title='Inverse trajectories', exp_name=exp_name)
    print(trajectories[-1])

    # Point estimations
    with torch.no_grad():
        for points, location in zip([points_on, points_in, points_out], ['on', 'inside', 'outside']):
            estimates = model.estimate(points)
            print(f'Estimation of points {location} the circles: {estimates.tolist()}')


if __name__ == '__main__':
    exp_name = 'normalizing_flow'
    q3_1(exp_name)
    q3_2(exp_name)
    q3_3(exp_name)
    q3_4(exp_name)
    q3_5(exp_name)
