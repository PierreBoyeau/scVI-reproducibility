from scvi.inference import UnsupervisedTrainer
from scvi.utils import demultiply
from tqdm import tqdm_notebook
import numpy as np
import warnings


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
    return lfcs


def estimate_lfc_mean(
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
            lfc_size.append(lfc.mean(0))
        lfc_size = np.array(lfc_size)
        lfcs[size] = lfc_size
    return lfcs


def estimate_de_proba(
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
                assert not np.isnan(lfc).any(), lfc

                pgs = np.nanmean(np.abs(lfc) >= delta, axis=0)
                de_probas[training, size_ix, exp, :] = pgs
    return de_probas


# def fdr_control(
#     mdl_class,
#     mdl_params,
#     train_params,
#     train_fn_params,
#     n_trainings=5,
#     n_picks=N_PICKS,
#     n_samples=500,
# ):
#     """
#
#     """
#     all_fdrs = np.zeros((n_trainings, N_SIZES, n_picks))
#     all_fnrs = np.zeros((n_trainings, N_SIZES, n_picks))
#     y_preds = np.zeros((n_trainings, N_SIZES, n_picks, dataset.nb_genes))
#     #     lfcs = np.zeros((n_trainings, N_SIZES, n_picks, dataset.nb_genes, 3*n_samples))
#     for training in range(n_trainings):
#         my_vae = mdl_class(dataset.nb_genes, n_batch=dataset.n_batches, **mdl_params)
#         my_trainer = UnsupervisedTrainer(my_vae, dataset, **train_params)
#         my_trainer.train(**train_fn_params)
#         print(my_trainer.train_losses)
#         #         post = my_trainer.train_set
#         post = my_trainer.test_set
#         train_indices = post.data_loader.sampler.indices
#         train_samples = np.random.permutation(train_indices)[:2000]
#         post = my_trainer.create_posterior(
#             model=my_vae, gene_dataset=dataset, indices=train_samples
#         )
#         z, labels, scales = post.get_latents(
#             n_samples=n_samples, other=True, device="cpu"
#         )
#
#         for (size_ix, size) in enumerate(tqdm_notebook(SIZES)):
#             for exp in range(n_picks):
#                 labels = labels.squeeze()
#                 where_a = np.where(labels == 0)[0]
#                 where_b = np.where(labels == 1)[0]
#                 where_a = where_a[np.random.choice(len(where_a), size=size)]
#                 where_b = where_b[np.random.choice(len(where_b), size=size)]
#                 scales_a = scales[:, where_a, :].reshape((-1, dataset.nb_genes)).numpy()
#                 scales_b = scales[:, where_b, :].reshape((-1, dataset.nb_genes)).numpy()
#                 scales_a, scales_b = demultiply(arr1=scales_a, arr2=scales_b, factor=3)
#                 lfc = np.log2(scales_a) - np.log2(scales_b)
#                 assert not np.isnan(lfc).any(), lfc
#
#                 pgs = (np.abs(lfc) >= DELTA).mean(axis=0)
#                 sorted_genes = np.argsort(-pgs)
#                 sorted_pgs = pgs[sorted_genes]
#                 cumulative_fdr = (1.0 - sorted_pgs).cumsum() / (
#                     1.0 + np.arange(len(sorted_pgs))
#                 )
#                 d = (cumulative_fdr <= Q0).sum() - 1
#                 pred_de_genes = sorted_genes[:d]
#                 is_pred_de = np.zeros_like(cumulative_fdr).astype(bool)
#                 is_pred_de[pred_de_genes] = True
#                 true_fdr = ((~is_significant_de) * is_pred_de).sum() / len(
#                     pred_de_genes
#                 )
#                 n_positives = is_significant_de.sum()
#                 #                 print(is_significant_de, ~is_pred_de, n_positives)
#                 true_fnr = (is_significant_de * (~is_pred_de)).sum() / n_positives
#                 all_fdrs[training, size_ix, exp] = true_fdr
#                 all_fnrs[training, size_ix, exp] = true_fnr
#                 y_preds[training, size_ix, exp, :] = is_pred_de
#                 print(true_fdr, true_fnr)
#     return dict(fdr=all_fdrs, predictions=y_preds, fnr=all_fnrs)

