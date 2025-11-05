import os, torch
import matplotlib.pyplot as plt


def create_output_dirs(exp_title: str) -> None:
    for dir in ['weights', 'plots']:
        if not os.path.exists(f'{dir}'):
            os.mkdir(f'{dir}')
        if not os.path.exists(f'{dir}/{exp_title}'):
            os.mkdir(f'{dir}/{exp_title}')


def get_file_name(title: str) -> str:
    return title.lower().replace(" - ", " ").replace(" = ", " ").replace(" ", "_")


def save_plot(title: str = 'figure', exp_name: str = None) -> None:
    """Save plot in experiment directory or just show it"""
    plt.tight_layout()
    if exp_name:
        plt.savefig(f'plots/{exp_name}/{get_file_name(title)}.png')
    else:
        plt.show()
    plt.close()


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
