import torch
from data import load_MNIST_dataset, get_random_images, NUM_DIGITS
from model import ConvVAE
from visualization import plot_image_probabilities, present_prob


# Load datasets
train_dataset, _, test_dataset, _ = load_MNIST_dataset()
num_images = 5  # samples from each train / test dataset
train_images = get_random_images(train_dataset, num_images=num_images, by_label=True)[1]
test_images = get_random_images(test_dataset, num_images=num_images, by_label=True)[1]

# Load model
out_dir = 'weights_amortization'
epoch = 30
model = ConvVAE(latent_dim=200)
model.load_state_dict(torch.load(f'{out_dir}/model_q1_epoch_{epoch}.pth'))
model.eval()

# Estimate log-likelihood of several images per digit
train_probs, test_probs = {}, {}
with torch.no_grad():
    for digit in range(NUM_DIGITS):
        train_probs[digit] = [model.estimate_likelihood(train_images[digit][i].unsqueeze(0)) for i in range(num_images)]
        test_probs[digit] = [model.estimate_likelihood(test_images[digit][i].unsqueeze(0)) for i in range(num_images)]
        
# (a) Plot a single image from each digit with its log-probability
plot_image_probabilities(
    images=[train_images[digit][0] for digit in range(NUM_DIGITS)],
    probs=[train_probs[digit][0] for digit in range(NUM_DIGITS)]
)

# (b) Present the average log-probability per digit
for digit in range(NUM_DIGITS):
    print(f'Digit {digit}: {present_prob(train_probs[digit] + test_probs[digit])} (train = {present_prob(train_probs[digit])}, test {present_prob(test_probs[digit])})')

# (c) Present the average log-probability of the images from the training set and test set
print(f'Train average = {present_prob(sum(train_probs.values(), []))}')
print(f'Test average = {present_prob(sum(test_probs.values(), []))}')
