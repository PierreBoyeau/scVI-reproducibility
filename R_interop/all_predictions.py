import numpy as np
from tqdm import tqdm

from . import NDESeq2, NEdgeRLTRT, MAST


def all_predictions(
    n_genes,
    n_picks,
    sizes,
    data_path,
    labels_path,
    label_a=0,
    label_b=1,
    path_to_scripts=None,
):
    n_sizes = len(sizes)

    # DESeq2
    lfcs_deseq2 = np.zeros((n_sizes, n_picks, n_genes))
    pvals_deseq2 = np.zeros((n_sizes, n_picks, n_genes))
    for (size_ix, size) in enumerate(tqdm(sizes)):
        for exp in range(n_picks):
            deseq_inference = NDESeq2(
                A=size,
                B=size,
                data=data_path,
                labels=labels_path,
                cluster=(label_a, label_b),
                path_to_scripts=path_to_scripts,
            )
            res_df = deseq_inference.fit()
            lfcs_deseq2[size_ix, exp, :] = res_df["lfc"].values
            pvals_deseq2[size_ix, exp, :] = res_df["padj"].values
    deseq_res = dict(lfc=lfcs_deseq2.squeeze(), pval=pvals_deseq2.squeeze())

    # EdgeR
    lfcs_edge_r = np.zeros((n_sizes, n_picks, n_genes))
    pvals_edge_r = np.zeros((n_sizes, n_picks, n_genes))
    for (size_ix, size) in enumerate(tqdm(sizes)):
        for exp in range(n_picks):
            deseq_inference = NEdgeRLTRT(
                A=size,
                B=size,
                data=data_path,
                labels=labels_path,
                cluster=(label_a, label_b),
                path_to_scripts=path_to_scripts,
            )
            res_df = deseq_inference.fit()
            lfcs_edge_r[size_ix, exp, :] = res_df["lfc"].values
            pvals_edge_r[size_ix, exp, :] = res_df["padj"].values
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
            res_df = mast_inference.fit(return_fc=True)
            lfcs_mast[size_ix, exp, :] = res_df["lfc"].values
            pvals_mast[size_ix, exp, :] = res_df["pval"].values
    mast_res = dict(lfc=lfcs_mast.squeeze(), pval=pvals_mast.squeeze())
    return dict(deseq2=deseq_res, edger=edger_res, mast=mast_res)
