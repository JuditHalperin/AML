import torch, faiss, random, numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, confusion_matrix
from scipy.optimize import linear_sum_assignment
from visualization import plot_representations, plot_images, plot_roc_curve
from utils import get_device


def get_embeddings(loader, exp_name: str, epoch: int):
    """Use model's encoder to get embeddings"""

    device = get_device()
    model = torch.load(f'weights/{exp_name}/weights_epoch_{epoch}.pth', map_location=device)
    encoder = model.encoder.to(device)
    encoder.eval()

    embeddings, labels = [], []
    with torch.no_grad():
        for image, label in loader:
            image = image.to(device)
            embeddings.append(encoder(image).cpu().numpy())
            labels.append(label.numpy())

    embeddings = np.concatenate(embeddings, axis=0)
    labels = np.concatenate(labels, axis=0)

    # Delete model to save memory
    del model
    del encoder
    torch.cuda.empty_cache()

    return embeddings, labels


def represent_in_2d(loader, exp_name: str, epoch: int):
    """Reduce dimensionality using both PCA and T-SNE"""

    embeddings, labels = get_embeddings(loader, exp_name, epoch)

    pca = PCA(n_components=2)
    embeddings_pca = pca.fit_transform(embeddings)

    tsne = TSNE(n_components=2, random_state=42)
    embeddings_tsne = tsne.fit_transform(embeddings)

    plot_representations(
        embedding_list=[embeddings_pca, embeddings_tsne],
        label_list=[labels, labels],
        name_list=['PCA', 'T-SNE'],
        exp_name=exp_name
    )


def get_neighboring_indices(loader, exp_name: str, epoch: int, image_indices=None, k: int = 3, choose_random: bool = False, nearest: bool = True) -> list[int] | list[list[int]]:
    """
    Find indices of neighboring images
    image_indices: find neighbors of specific images (by default, all images)
    k: number of neighbors
    choose_random: whether to return one random neighbor for each image or all k neighbors
    nearest: whether to find k nearest neighbors or k most distant neighbors
    """
    embeddings, _ = get_embeddings(loader, exp_name, epoch)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    data_size = embeddings.shape[0]

    if image_indices: 
        embeddings = embeddings[image_indices, :]

    if nearest:
        _, indices = index.search(embeddings, k + 1)  # as the first neighbor is the image itself
        indices = [i[1:].tolist() for i in indices]
    else:
        _, indices = index.search(embeddings, data_size)  # (for some reason such large k only worked with faiss-cpu instead of faiss-gpu)
        indices = [i[-k:].tolist() for i in indices]

    if choose_random:
        return [neighbors[random.randint(0, k - 1)] for neighbors in indices]
    return indices


def get_image_neighbors(images, dataset, loader, exp_name: str, epoch: int, k: int = 5) -> dict[int, list[int]]:
    """Find image nearest / farest neighbors"""
    for nearest in [True, False]:
        image_indices = [i[0] for i in images.values()] 
        indices = get_neighboring_indices(loader, exp_name, epoch, image_indices, k, nearest=nearest)

        image_neighbors = {
            label: [dataset[idx[0]][0]] + [dataset[i][0] for i in indices[int(label)]]  # extract images from dataset (without labels)
            for label, idx in images.items()  # idx is a list with a single index
        }

        title = f'{exp_name} - {"Neighboring" if nearest else "Distant"} Images'
        plot_images(image_neighbors, title=title, exp_name=exp_name)


def compute_density_estimation(train_loader, test_loader, exp_name: str, epoch: int, k: int = 2):
    """Get density estimation"""

    train_embeddings, _ = get_embeddings(train_loader, exp_name, epoch)
    test_embeddings, _ = get_embeddings(test_loader, exp_name, epoch)

    index = faiss.IndexFlatL2(train_embeddings.shape[1])
    index.add(train_embeddings)

    distances, _ = index.search(test_embeddings, k + 1)  # as the first neighbor is the image itself
    inverse_density_scores = [i[1:].mean() for i in distances]  # average L2 distances
    return inverse_density_scores


def detect_anomalies(train_loader, test_loader, test_dataset, test_labels, exp_name: str, epoch: int, k: int = 2, num_anomalies: int = 7):
    # Get density estimation
    inverse_density_scores = compute_density_estimation(train_loader, test_loader, exp_name, epoch, k)

    # Print scores based on original dataset (sanity check)
    print(np.mean(inverse_density_scores[:sum(test_labels == 0)]))  # expected to be low
    print(np.mean(inverse_density_scores[sum(test_labels == 0):]))  # expected to be high

    # ROC curve
    plot_roc_curve(test_labels, inverse_density_scores, title=f'{exp_name} - ROC Curve', exp_name=exp_name)

    # Most anomalous images
    top = np.argsort(inverse_density_scores)[::-1][:num_anomalies]
    images = {i: [test_dataset[idx][0]] for i, idx in enumerate(top)}  # convert to dict to match plot_images expected input
    plot_images(images, title=f'{exp_name} - Most Anomalous Images', exp_name=exp_name)


def cluster(loader, exp_name: str, epoch: int, num_classes: int):
    embeddings, true_labels = get_embeddings(loader, exp_name, epoch)

    # Cluster using K-Means
    kmeans = KMeans(n_clusters=num_classes, random_state=42, n_init='auto').fit(embeddings)
    pred_labels = kmeans.labels_

    # Use the Hungarian algorithm to find the best label assignment
    contingency_matrix = confusion_matrix(true_labels, pred_labels)
    row_ind, col_ind = linear_sum_assignment(-contingency_matrix)
    mapping = {col: row for row, col in zip(row_ind, col_ind)}
    pred_labels = np.vectorize(mapping.get)(pred_labels)

    # Reduce dimension using T-SNE
    tsne = TSNE(n_components=2, random_state=42)
    all_points = np.concatenate([embeddings, kmeans.cluster_centers_], axis=0)
    all_points_tsne = tsne.fit_transform(all_points)
    embeddings_tsne, pred_centers = all_points_tsne[:-num_classes], all_points_tsne[-num_classes:]

    # Find true cluster centers
    true_centers = np.zeros((num_classes, 2))
    for i in range(num_classes):
        true_centers[i] = np.mean(embeddings_tsne[true_labels == i], axis=0)
    
    # Plot
    plot_representations(
        embedding_list=[embeddings_tsne, embeddings_tsne],
        label_list=[true_labels, pred_labels],
        name_list=['T-SNE - True Clusters', 'T-SNE - Predicted Clusters'],
        mark_points=[true_centers, pred_centers],
        title='Clusters',
        exp_name=exp_name
    )

    # Calculate Silhouette Scores
    silhouette_pred = silhouette_score(embeddings, pred_labels)
    print(f'Silhouette Score (Predicted Labels): {silhouette_pred:.4f}')
