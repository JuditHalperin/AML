import torch
import numpy as np
import torch.nn.functional as F
from data import pick_points
from train import train_flow_matching
from visualization import plot_data, plot_trajectories
from create_data import sample_olympic_rings


def q4_1_unconditional(exp_name: str):
    train_flow_matching(
        conditional=False,
        plot_loss=True,
        exp_title=exp_name
    )


def q4_2_unconditional(exp_name: str, epoch: int = 20, num_points: int = 1000, delta_t: float = 0.2):
    
    model = torch.load(f'weights/{exp_name}/weights_epoch_{epoch}.pth')
    model.eval()
    with torch.no_grad():
        trajectory = model.sample(num_points, delta_t)

    for i, t in enumerate(np.arange(0, 1 + delta_t, delta_t)):
        plot_data(trajectory[i], title=f'Samples Over Time - t = {np.round(float(t), 1)}', exp_name=exp_name)


def q4_3_unconditional(exp_name: str, epoch: int = 20, num_points: int = 10, delta_t: float = 1e-3):

    model = torch.load(f'weights/{exp_name}/weights_epoch_{epoch}.pth')
    model.eval()
    with torch.no_grad():
        trajectories = model.sample(num_points, delta_t)

    plot_trajectories(trajectories, title='Forward trajectories', exp_name=exp_name)


def q4_4_unconditional(exp_name: str, epoch: int = 20, num_points: int = 1000, delta_ts: list[float] = [0.002, 0.02, 0.05, 0.1, 0.2]):
    
    model = torch.load(f'weights/{exp_name}/weights_epoch_{epoch}.pth')
    model.eval()
    with torch.no_grad():
        for delta_t in delta_ts:
            y = model.sample(num_points, delta_t)[-1]
            plot_data(y, title=f'Samples - delta_t = {delta_t}', exp_name=exp_name)


def q4_5_unconditional(exp_name: str, epoch: int = 20, delta_t: float = 1e-3):
     
    points_on, points_in, points_out = pick_points()
    point_locations = [3] * len(points_on) + [1] * len(points_in) + [2] * len(points_out)

    model = torch.load(f'weights/{exp_name}/weights_epoch_{epoch}.pth')
    model.eval()
    with torch.no_grad():

        # Reverse points per-location
        # for points, location in zip([points_on, points_in, points_out], ['on', 'inside', 'outside']):
        #     trajectories = model.reverse_sample(points, delta_t)
        #     plot_trajectories(trajectories, labels=point_locations, title=f'Reverse trajectories - points {location} the circles', exp_name=exp_name)

        # Reverse all points
        points = torch.cat([points_on, points_in, points_out], dim=0)
        trajectories = model.reverse_sample(points, delta_t)
        plot_trajectories(trajectories, labels=point_locations, axis_lim=4, title='Reverse trajectories', exp_name=exp_name)
        print(trajectories[-1])

        # Forward reversed points
        trajectories = model.sample(delta_t=delta_t, prior_points=trajectories[-1])
        plot_trajectories(trajectories, labels=point_locations, axis_lim=4, title='Re-forward trajectories', exp_name=exp_name)
        plot_data([points, trajectories[-1]], size=100, alpha=0.5, title='Forwarding Reversed Points', exp_name=exp_name)
        print(F.l1_loss(points, trajectories[-1], reduction='mean'))


def q4_1_conditional(exp_name: str, classes: list[int], num_points: int = 3000):

    train_flow_matching(
        conditional=True,
        plot_loss=True,
        exp_title=exp_name
    )

    points, labels = sample_olympic_rings(num_points // len(classes))
    plot_data(torch.Tensor(points), labels, title='Input Points', exp_name=exp_name)


def q4_2_conditional(exp_name: str, classes: list[int], epoch: int = 20, num_points_per_class: int = 1, delta_t: float = 1e-3):

    model = torch.load(f'weights/{exp_name}/weights_epoch_{epoch}.pth')
    model.eval()

    labels = sorted(classes * num_points_per_class)

    with torch.no_grad():
        trajectories = model.sample(num_points_per_class * len(classes), delta_t, condition=labels)

    plot_trajectories(trajectories, labels=labels, add_rings=True, title='Trajectories - Condotional Flow Matching', exp_name=exp_name)


def q4_3_conditional(exp_name: str, classes: list[int], epoch: int = 20, num_points_per_class: int = 1000, delta_t: float = 1e-3):

    model = torch.load(f'weights/{exp_name}/weights_epoch_{epoch}.pth')
    model.eval()

    labels = sorted(classes * num_points_per_class)

    with torch.no_grad():
        points = model.sample(num_points_per_class * len(classes), delta_t, condition=labels)[-1]
        plot_data(points, labels=labels, title='Samples - Condotional Flow Matching', exp_name=exp_name)


def q4_bonus(epoch: int = 20, delta_t: float = 1e-3, output: tuple = (4, 5), label: int = 3):

    # Conditional
    exp_name = 'conditional_flow_matching'
    model = torch.load(f'weights/{exp_name}/weights_epoch_{epoch}.pth')
    
    model.eval()
    with torch.no_grad():
        reverse_trajectory = model.reverse_sample(torch.tensor(output).unsqueeze(0), condition=[label])
        plot_trajectories(reverse_trajectory, add_rings=True, axis_lim=None, title=f'Reverse from {output}', exp_name=exp_name)

        prior = reverse_trajectory[-1]
        trajectory = model.sample(1, delta_t, condition=[label], prior_points=prior)
        point, prior = tuple(trajectory[-1].flatten().numpy().round(3)), tuple(prior.flatten().numpy().round(3))
        plot_trajectories(trajectory, add_rings=True, axis_lim=None, title=f'Sample {point} from {prior}', exp_name=exp_name)
        
        # Using different rate and bias when updating y
        # rate, bias = 1, 0
        # trajectory = model.sample(1, delta_t, condition=[label], y_rate=rate, y_bias=bias)
        # point = tuple(trajectory[-1].flatten().numpy().round(3))
        # plot_trajectories(trajectory, add_rings=True, axis_lim=None, title=f'Sample {point} using rate = {rate} and bias = {bias}', exp_name=exp_name)

    # Unconditional
    exp_name = 'unconditional_flow_matching'
    model = torch.load(f'weights/{exp_name}/weights_epoch_{epoch}.pth')
    
    model.eval()
    with torch.no_grad():
        reverse_trajectory = model.reverse_sample(torch.tensor(output).unsqueeze(0))
        plot_trajectories(reverse_trajectory, add_rings=True, axis_lim=None, title=f'Reverse from {output}', exp_name=exp_name)

        prior = reverse_trajectory[-1]
        trajectory = model.sample(1, delta_t, prior_points=prior)
        point, prior = tuple(trajectory[-1].flatten().numpy().round(3)), tuple(prior.flatten().numpy().round(3))
        plot_trajectories(trajectory, add_rings=True, axis_lim=None, title=f'Sample {point} from {prior}', exp_name=exp_name)

        # Using different rate and bias when updating y
        # rate, bias = 1, 0
        # trajectory = model.sample(1, delta_t, y_rate=rate, y_bias=bias)
        # point = tuple(trajectory[-1].flatten().numpy().round(3))
        # plot_trajectories(trajectory, add_rings=True, axis_lim=None, title=f'Sample {point} using rate = {rate} and bias = {bias}', exp_name=exp_name)


if __name__ == '__main__':

    #  Unconditional Flow Matching:
    exp_name = 'unconditional_flow_matching'
    q4_1_unconditional(exp_name)
    q4_2_unconditional(exp_name)
    q4_3_unconditional(exp_name)
    q4_4_unconditional(exp_name)
    q4_5_unconditional(exp_name)

    # # Conditional Flow Matching:
    exp_name = 'conditional_flow_matching'
    classes = list(range(5))
    q4_1_conditional(exp_name, classes)
    q4_2_conditional(exp_name, classes)
    q4_3_conditional(exp_name, classes)
    q4_bonus()
