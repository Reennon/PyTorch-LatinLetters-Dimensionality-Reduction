import numpy as np
import pickle
from umap import UMAP
from datetime import datetime
from pathlib import Path


class Constants:
    LATIN_DATA_CSV = './datasets/latin_data.csv'
    LATIN_LABEL_CSV = './datasets/latin_label.csv'
    PROJECT_FOLDER = 'C:/Users/rkovalch/Documents/SoftServe/DSProjects/MAI/Course/PyTorch-LatinLetters-Dimensionality-Reduction'
    REDUCED_DATASETS_FOLDER = PROJECT_FOLDER + '/datasets/reduced'
    REDUCED_LATIN_DATA_PTH = REDUCED_DATASETS_FOLDER + '/latin_data_{version}'
    DATE_TIME_FORMAT = '%m_%d_%H_%M_%S'
    UMAP_FOLDER = PROJECT_FOLDER + '/models/umap'
    UMAP_SERIALIZED = UMAP_FOLDER + '/UMAP_{version}'


class VersionFormatter:
    @staticmethod
    def by_datetime() -> str:
        return datetime.now().strftime(Constants.DATE_TIME_FORMAT)


if __name__ == '__main__':
    reducer = UMAP(
        n_neighbors=50,
        n_components=3,
        n_epochs=1000,
        min_dist=0.5,
        local_connectivity=5,
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

    Path(Constants.REDUCED_DATASETS_FOLDER).mkdir(parents=True, exist_ok=True)
    save(Constants.REDUCED_LATIN_DATA_PTH.format(version = version), images_reduced_a)

    print(f'Saved reduced CoMNIST dataset with UMAP, to {Constants.REDUCED_LATIN_DATA_PTH.format(version = version)}.npy')

    print(f'''To Load reduced dataset as numpy array, use:
    ```
    import numpy as np
    import pandas as pd
    import string
    
    X = np.load('{Constants.REDUCED_LATIN_DATA_PTH.format(version = version)}.npy')
    
    # since we haven't changed the labels, we can still use the original dataset
    y = np.loadtxt('{Constants.LATIN_LABEL_CSV}')
    ```
    ''')

    print('Saving UMAP class as pickle serialized object...')
    Path(Constants.UMAP_FOLDER).mkdir(parents=True, exist_ok=True)
    with open(Constants.UMAP_SERIALIZED.format(version = version), 'wb') as file:
        pickle.dump(reducer, file)

    print(f'''To Load UMAP Reducer class, use:
    ```
    import pickle
    
    reducer = pickle.load('{Constants.UMAP_SERIALIZED.format(version = version)}.pth')
    ```
    ''')
    print('Done')
