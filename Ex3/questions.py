from augmentations import load_datasets, load_anomaly_datasets, get_random_indices, PairedCIFAR10
from train import train_representation, train_classification
from tasks import represent_in_2d, get_neighboring_indices, get_image_neighbors, detect_anomalies, cluster


VICREG_EPOCHS = 50
CLASSIFIER_EPOCHS = 10


def part3_q1(
        exp_name: str = 'VICReg',
        epoch: int = VICREG_EPOCHS
    ):
    train_loader, val_loader = load_datasets()
    train_representation(train_loader, val_loader, num_epochs=epoch, plot_loss=True, exp_title=exp_name)


def part3_q2(
        exp_name: str = 'VICReg',
        epoch: int = VICREG_EPOCHS
    ):
    _, test_loader = load_datasets(augmentation=False)
    represent_in_2d(test_loader, exp_name, epoch)
    

def part3_q3(
        encoder_name: str = 'VICReg', encoder_epoch: int = VICREG_EPOCHS,
        classifer_name: str = 'classifier', classifier_epoch: int = CLASSIFIER_EPOCHS
    ):
    train_loader, val_loader = load_datasets(augmentation=False)
    train_classification(
        train_loader, val_loader,
        trained_model=f'weights/{encoder_name}/weights_epoch_{encoder_epoch}.pth',
        num_epochs=classifier_epoch, plot_loss=True, exp_title=classifer_name
    )


def part3_q4(
        encoder_name: str = 'VICReg_no_var', encoder_epoch: int = VICREG_EPOCHS,
        classifer_name: str = 'classifier_no_var', classifier_epoch: int = CLASSIFIER_EPOCHS    
    ):
    train_loader, val_loader = load_datasets()
    train_representation(train_loader, val_loader, num_epochs=encoder_epoch, plot_loss=True, var_weight=0, exp_title=encoder_name)

    _, test_loader = load_datasets(augmentation=False)
    represent_in_2d(test_loader, encoder_name, encoder_epoch)

    train_loader, val_loader = load_datasets(augmentation=False)
    train_classification(
        train_loader, val_loader,
        trained_model=f'weights/{encoder_name}/weights_epoch_{encoder_epoch}.pth',
        num_epochs=classifier_epoch, plot_loss=True, exp_title=classifer_name
    )


def part3_q5(
        org_encoder_name: str = 'VICReg', org_encoder_epoch: int = VICREG_EPOCHS,
        encoder_name: str = 'VICReg_no_gen_neighbors', encoder_epoch: int = 1,
        classifer_name: str = 'classifier_no_gen_neighbors', classifier_epoch: int = CLASSIFIER_EPOCHS
    ):
    train_loader, _ = load_datasets(augmentation=False, shuffle_train=False)  # no shuffling when we need to match image indices
    indices = get_neighboring_indices(train_loader, org_encoder_name, org_encoder_epoch, choose_random=True)

    train_loader, _ = load_datasets(augmentation=False, neighboring_indices_train=indices, shuffle_train=False)
    train_representation(train_loader, num_epochs=encoder_epoch, exp_title=encoder_name)

    train_loader, val_loader = load_datasets(augmentation=False)
    train_classification(
        train_loader, val_loader,
        trained_model=f'weights/{encoder_name}/weights_epoch_{encoder_epoch}.pth',
        num_epochs=classifier_epoch, plot_loss=True, exp_title=classifer_name
    )


def part3_q7(
        encoder_name1: str = 'VICReg', encoder_epoch1: int = VICREG_EPOCHS, 
        encoder_name2: str = 'VICReg_no_gen_neighbors', encoder_epoch2: int = 1,
        k: int = 5,
    ):
    # Get random train images
    train_dataset = PairedCIFAR10(train=True, keep_original=True)
    images = get_random_indices(train_dataset)

    # Find neighboring images
    train_loader, _ = load_datasets(augmentation=False, shuffle_train=False)
    get_image_neighbors(images, train_dataset, train_loader, encoder_name1, encoder_epoch1, k)
    get_image_neighbors(images, train_dataset, train_loader, encoder_name2, encoder_epoch2, k)


def part4_2(
        encoder_name1: str = 'VICReg', encoder_epoch1: int = VICREG_EPOCHS, 
        encoder_name2: str = 'VICReg_no_gen_neighbors', encoder_epoch2: int = 1,
        k: int = 2, num_anomalies: int = 7,
    ):
    train_loader, test_loader, test_dataset, test_labels = load_anomaly_datasets()

    detect_anomalies(train_loader, test_loader, test_dataset, test_labels, encoder_name1, encoder_epoch1, k, num_anomalies)
    detect_anomalies(train_loader, test_loader, test_dataset, test_labels, encoder_name2, encoder_epoch2, k, num_anomalies)


def part4_3(
        encoder_name1: str = 'VICReg', encoder_epoch1: int = VICREG_EPOCHS, 
        encoder_name2: str = 'VICReg_no_gen_neighbors', encoder_epoch2: int = 1,
        num_classes: int = 10
    ):
    train_loader, _ = load_datasets(augmentation=False, shuffle_train=False)

    cluster(train_loader, encoder_name1, encoder_epoch1, num_classes)
    cluster(train_loader, encoder_name2, encoder_epoch2, num_classes)


if __name__ == '__main__':
    part3_q1()
    part3_q2()
    part3_q3()
    part3_q4()
    part3_q5()
    part3_q7()
    part4_2()
    part4_3()
