from scvi.inference import UnsupervisedTrainer
from scvi.utils import demultiply
import os
from tqdm import tqdm_notebook
import numpy as np
import warnings
import pickle


def save_pickle(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


def load_pickle(filename):
    with open(filename, 'rb') as f:
        res = pickle.load(f)
    return res


def train_model(
    mdl_class, dataset, mdl_params: dict, train_params: dict, train_fn_params: dict
):
    """

    :param mdl_class: Class of algorithm
    :param dataset: Dataset
    :param mdl_params:
    :param train_params:
    :param train_fn_params:
    :return:
    """
    if 'test_indices' not in train_params:
        warnings.warn('No `test_indices` attribute found.')
    my_vae = mdl_class(dataset.nb_genes, n_batch=dataset.n_batches, **mdl_params)
    my_trainer = UnsupervisedTrainer(my_vae, dataset, **train_params)
    my_trainer.train(**train_fn_params)
    print(my_trainer.train_losses)
    return my_vae, my_trainer


def estimate_lfc_density(
    filename,
    mdl_class,
    dataset,
    mdl_params: dict,
    train_params: dict,
    train_fn_params: dict,
    sizes: list,
    n_picks: int = 10,
    n_samples: int = 500,
    label_a=0,
    label_b=1
):
    """

    """
    if os.path.exists(filename):
        return load_pickle(filename)

    lfcs = dict()
    my_vae, my_trainer = train_model(
        mdl_class, dataset, mdl_params, train_params, train_fn_params
    )
    post = my_trainer.test_set
    train_indices = post.data_loader.sampler.indices
    train_samples = np.random.permutation(train_indices)
    post = my_trainer.create_posterior(
        model=my_vae, gene_dataset=dataset, indices=train_samples
    )
    z, labels, scales = post.get_latents(n_samples=n_samples, other=True, device="cpu")

    for (size_ix, size) in enumerate(tqdm_notebook(sizes)):
        lfc_size = []
        for exp in range(n_picks):
            labels = labels.squeeze()
            where_a = np.where(labels == label_a)[0]
            where_b = np.where(labels == label_b)[0]
            where_a = where_a[np.random.choice(len(where_a), size=size)]
            where_b = where_b[np.random.choice(len(where_b), size=size)]
            scales_a = scales[:, where_a, :].reshape((-1, dataset.nb_genes)).numpy()
            scales_b = scales[:, where_b, :].reshape((-1, dataset.nb_genes)).numpy()
            scales_a, scales_b = demultiply(arr1=scales_a, arr2=scales_b, factor=3)
            lfc = np.log2(scales_a) - np.log2(scales_b)
            assert not np.isnan(lfc).any(), lfc
            lfc_size.append(lfc)
        lfc_size = np.array(lfc_size)
        lfcs[size] = lfc_size
    save_pickle(lfcs, filename=filename)
    return lfcs


def estimate_lfc_mean(
    filename,
    mdl_class,
    dataset,
    mdl_params: dict,
    train_params: dict,
    train_fn_params: dict,
    sizes: list,
    n_picks: int = 10,
    n_samples: int = 500,
    label_a=0,
    label_b=1
) -> dict:
    """
        Returns LFC POINT ESTIMATES
    """
    if os.path.exists(filename):
        return load_pickle(filename)
    lfcs = dict()
    my_vae, my_trainer = train_model(
        mdl_class, dataset, mdl_params, train_params, train_fn_params
    )
    post = my_trainer.test_set
    train_indices = post.data_loader.sampler.indices
    train_samples = np.random.permutation(train_indices)
    post = my_trainer.create_posterior(
        model=my_vae, gene_dataset=dataset, indices=train_samples
    )
    z, labels, scales = post.get_latents(n_samples=n_samples, other=True, device="cpu")

    for (size_ix, size) in enumerate(tqdm_notebook(sizes)):
        lfc_size = []
        for exp in range(n_picks):
            labels = labels.squeeze()
            where_a = np.where(labels == label_a)[0]
            where_b = np.where(labels == label_b)[0]
            where_a = where_a[np.random.choice(len(where_a), size=size)]
            where_b = where_b[np.random.choice(len(where_b), size=size)]
            scales_a = scales[:, where_a, :].reshape((-1, dataset.nb_genes)).numpy()
            scales_b = scales[:, where_b, :].reshape((-1, dataset.nb_genes)).numpy()
            scales_a, scales_b = demultiply(arr1=scales_a, arr2=scales_b, factor=3)
            lfc = np.log2(scales_a) - np.log2(scales_b)
            # assert not np.isnan(lfc).any(), lfc
            if np.isnan(lfc).any():
                warnings.warn("NaN values appeared in LFCs")
            lfc_size.append(lfc.mean(0))
        lfc_size = np.array(lfc_size)
        lfcs[size] = lfc_size
    save_pickle(lfcs, filename=filename)
    return lfcs


def estimate_de_proba(
    filename,
    mdl_class,
    dataset,
    mdl_params: dict,
    train_params: dict,
    train_fn_params: dict,
    sizes: list,
    delta: float = 0.5,
    n_trainings: int = 5,
    n_picks: int = 25,
    n_samples: int = 500,
    label_a=0,
    label_b=1
):
    """

    """
    if os.path.exists(filename):
        return np.load(filename)

    n_sizes = len(sizes)
    de_probas = np.zeros((n_trainings, n_sizes, n_picks, dataset.nb_genes))
    #     lfcs = np.zeros((n_trainings, N_SIZES, n_picks, dataset.nb_genes, 3*n_samples))
    for training in range(n_trainings):
        my_vae, my_trainer = train_model(
            mdl_class, dataset, mdl_params, train_params, train_fn_params
        )
        post = my_trainer.test_set
        train_indices = post.data_loader.sampler.indices
        train_samples = np.random.permutation(train_indices)
        post = my_trainer.create_posterior(
            model=my_vae, gene_dataset=dataset, indices=train_samples
        )
        z, labels, scales = post.get_latents(
            n_samples=n_samples, other=True, device="cpu"
        )

        for (size_ix, size) in enumerate(tqdm_notebook(sizes)):
            for exp in range(n_picks):
                labels = labels.squeeze()
                where_a = np.where(labels == label_a)[0]
                where_b = np.where(labels == label_b)[0]
                where_a = where_a[np.random.choice(len(where_a), size=size)]
                where_b = where_b[np.random.choice(len(where_b), size=size)]
                scales_a = scales[:, where_a, :].reshape((-1, dataset.nb_genes)).numpy()
                scales_b = scales[:, where_b, :].reshape((-1, dataset.nb_genes)).numpy()
                scales_a, scales_b = demultiply(arr1=scales_a, arr2=scales_b, factor=3)
                lfc = np.log2(scales_a) - np.log2(scales_b)
                if np.isnan(lfc).any():
                    warnings.warn("NaN values appeared in LFCs")

                pgs = np.nanmean(np.abs(lfc) >= delta, axis=0)
                de_probas[training, size_ix, exp, :] = pgs
    np.save(file=filename, arr=de_probas)
    return de_probas

