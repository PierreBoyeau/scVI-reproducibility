import numpy as np
from tqdm import tqdm

from .deseq import DESeq2
from .edge_r import EdgeR
from .mast import MAST


def all_predictions(
    n_genes, n_picks, sizes, data_path, labels_path, label_a=0, label_b=1
):
    n_sizes = len(sizes)

    # DESeq2
    lfcs_deseq2 = np.zeros((n_sizes, n_picks, n_genes))
    pvals_deseq2 = np.zeros((n_sizes, n_picks, n_genes))
    for (size_ix, size) in enumerate(tqdm(sizes)):
        for exp in range(n_picks):
            deseq_inference = DESeq2(
                A=size,
                B=size,
                data=data_path,
                labels=labels_path,
                cluster=(label_a, label_b),
            )
            pvals, lfcs = deseq_inference.fit(return_fc=True)
            lfcs_deseq2[size_ix, exp, :] = np.asarray(lfcs)
            pvals_deseq2[size_ix, exp, :] = np.asarray(pvals)
    deseq_res = dict(lfc=lfcs_deseq2.squeeze(), pval=pvals_deseq2.squeeze())

    # EdgeR
    lfcs_edge_r = np.zeros((n_sizes, n_picks, n_genes))
    pvals_edge_r = np.zeros((n_sizes, n_picks, n_genes))
    for (size_ix, size) in enumerate(tqdm(sizes)):
        for exp in range(n_picks):
            deseq_inference = EdgeR(
                A=size,
                B=size,
                data=data_path,
                labels=labels_path,
                cluster=(label_a, label_b),
            )
            pvals, lfcs = deseq_inference.fit(return_fc=True)
            lfcs_edge_r[size_ix, exp, :] = np.asarray(lfcs)
            pvals_edge_r[size_ix, exp, :] = np.asarray(pvals)
    edger_res = dict(lfc=lfcs_edge_r.squeeze(), pval=pvals_edge_r.squeeze())

    # MAST
    lfcs_mast = np.zeros((n_sizes, n_picks, n_genes))
    pvals_mast = np.zeros((n_sizes, n_picks, n_genes))
    for (size_ix, size) in enumerate(tqdm(sizes)):
        for exp in range(n_picks):
            mast_inference = MAST(
                A=size,
                B=size,
                data=data_path,
                labels=labels_path,
                cluster=(label_a, label_b),
            )
            pvals, lfcs = mast_inference.fit(return_fc=True)
            lfcs_mast[size_ix, exp, :] = lfcs
            pvals_mast[size_ix, exp, :] = pvals
    mast_res = dict(lfc=lfcs_mast.squeeze(), pval=pvals_mast.squeeze())
    return dict(deseq2=deseq_res, edger=edger_res, mast=mast_res)
