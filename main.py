import numpy as np
import pickle
from umap import UMAP
from datetime import datetime


class Constants:
    LATIN_DATA_CSV = './datasets/latin_data.csv'
    LATIN_LABEL_CSV = './datasets/latin_label.csv'
    REDUCED_LATIN_DATA_PTH = 'D:/Documents/MAI/Course/datasets/reduced/latin_data'
    DATE_TIME_FORMAT = '%m_%d_%H_%M_%S'
    UMAP_SERIALIZED = './models/umap/UMAP_{version}'


class VersionFormatter:
    @staticmethod
    def by_datetime() -> str:
        return datetime.now().strftime(Constants.DATE_TIME_FORMAT)


if __name__ == '__main__':
    reducer = UMAP(
        n_neighbors=100,
        n_components=3,
        n_epochs=1000,
        min_dist=0.5,
        local_connectivity=10,
        random_state=42
    )

    version = VersionFormatter.by_datetime()
    print(f'version: {version}')

    print('Initialized UMAP')

    images_a = np.loadtxt(Constants.LATIN_DATA_CSV, delimiter=",", dtype="float32")
    labels_a = np.loadtxt(Constants.LATIN_LABEL_CSV, delimiter=",", dtype="float32")
    images_reduced_a = reducer.fit_transform(images_a, labels_a)

    print(f'Reduced UMAP, with {images_reduced_a.shape} shape')

    from numpy import save

    save(Constants.REDUCED_LATIN_DATA_PTH, images_reduced_a)

    print(f'Saved reduced CoMNIST dataset with UMAP, to {Constants.REDUCED_LATIN_DATA_PTH.format(version = version)}.npy')

    with open(Constants.UMAP_SERIALIZED.format(version = version), 'wb') as file:
        pickle.dump(reducer, file)

    print(f'''
    To Load UMAP Reducer class, use:
    '
    import pickle
    
    reducer = pickle.load({Constants.UMAP_SERIALIZED.format(version = version)}.pth)
    '
    ''')
