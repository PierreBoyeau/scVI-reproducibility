import numpy as np
from tqdm import tqdm

from .deseq import DESeq2
from .edge_r import  EdgeR
from .mast import MAST


def all_predictions(n_genes, n_picks, size, data_path, labels_path):
    # DESeq2
    lfcs_deseq2 = np.zeros((n_picks, n_genes))
    pvals_deseq2 = np.zeros((n_picks, n_genes))
    for exp in tqdm(range(n_picks)):
        deseq_inference = DESeq2(
            A=size,
            B=size,
            data=data_path,
            labels=labels_path,
            cluster=(0, 1)
        )
        pvals, lfcs = deseq_inference.fit(return_fc=True)
        lfcs_deseq2[exp, :] = np.asarray(lfcs)
        pvals_deseq2[exp, :] = np.asarray(pvals)
    deseq_res = dict(lfc=lfcs_deseq2, pval=pvals_deseq2)

    # EdgeR
    lfcs_edge_r = np.zeros((n_picks, n_genes))
    pvals_edge_r = np.zeros((n_picks, n_genes))
    for exp in tqdm(range(n_picks)):
        deseq_inference = EdgeR(
            A=size,
            B=size,
            data=data_path,
            labels=labels_path,
            cluster=(0, 1)
        )
        pvals, lfcs = deseq_inference.fit(return_fc=True)
        lfcs_edge_r[exp, :] = np.asarray(lfcs)
        pvals_edge_r[exp, :] = np.asarray(pvals)
    edger_res = dict(lfc=lfcs_edge_r, pval=pvals_edge_r)

    # MAST
    lfcs_mast = np.zeros((n_picks, n_genes))
    pvals_mast = np.zeros((n_picks, n_genes))
    for exp in tqdm(range(n_picks)):
        mast_inference = MAST(
            A=size,
            B=size,
            data=data_path,
            labels=labels_path,
            cluster=(0, 1)
        )
        pvals, lfcs = mast_inference.fit(return_fc=True)
        lfcs_mast[exp, :] = lfcs
        pvals_mast[exp, :] = pvals
    mast_res = dict(lfc=lfcs_mast, pval=pvals_mast)
    return dict(
        deseq2=deseq_res,
        edger=edger_res,
        mast=mast_res
    )
