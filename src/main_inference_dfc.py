import itertools
import logging
import os
import os.path as osp
from os.path import join as osj
from time import time

import hydra
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import ndcg_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_utils import FbtCandidateDataset, FbtDataset, FbtInferenceDataset, load_dfs
from lit.eval_utils import calc_topk
from lit.lit_dcf import LitDCF

logger = logging.getLogger(__name__)


def dcf_model_inference(lit_model, category_src, price_bin, img_emb, candidates):
    category_emb = lit_model.category_embs(category_src)
    price_emb = lit_model.price_embs(price_bin)
    src = lit_model.src_encoder(img_emb, category_emb, price_emb)

    # Fusion layer
    logits_i = lit_model.fusion_model(src.repeat(len(candidates), 1), candidates)
    probs_i = torch.sigmoid(logits_i.squeeze())
    return probs_i


def get_candidate_embeddings(lit_model, candidate_loader):
    candidates, asins, categories = [], [], []
    with torch.no_grad():
        for img_emb, price_bin, category, asin in tqdm(candidate_loader):

            category_emb = lit_model.category_embs(category)
            price_emb = lit_model.price_embs(price_bin)
            candidate = lit_model.candidate_encoder(img_emb, category_emb, price_emb)

            candidates.append(candidate.detach().cpu())
            asins.append(asin)
            categories.append(category.detach().cpu())

    candidates = torch.vstack(candidates)
    asins = np.array(list(itertools.chain(*asins)))
    categories = torch.hstack(categories)
    return candidates, categories, asins


def evaluate_dcn(
    lit_model, inference_loader, candidates, category_candidates, candidate_asins, cfg
):
    # Get valid cateogires
    hot_labels, probs, category_targets, asin_srcs = [], [], [], []
    for (
        img_emb,
        price_bin,
        category_src,
        asin_src,
        _,
        target_asins,
    ) in tqdm(inference_loader):

        probs_i = dcf_model_inference(
            lit_model, category_src, price_bin, img_emb, candidates
        )

        # Create ground true label
        target_asins = [asin_target[0] for asin_target in target_asins]
        hot_labels_i = np.in1d(
            candidate_asins,
            target_asins,
        )
        assert hot_labels_i.sum() > 0

        # Save
        hot_labels.append(hot_labels_i)
        probs.append(probs_i.squeeze())
        category_targets_i = category_candidates[hot_labels_i]
        category_targets.append(category_targets_i.unique().tolist())
        asin_srcs.append(asin_src)

    # Calcualte probability
    probs = torch.vstack(probs)
    hot_labels = np.vstack(hot_labels)

    # Retrival metrics
    dcn_ndcg_val_at_k = {}
    for k in cfg.top_k + [probs.shape[-1]]:
        dcn_ndcg_val = ndcg_score(hot_labels, probs, k=k)
        dcn_ndcg_val_at_k[k] = dcn_ndcg_val
    logger.info(f"{dcn_ndcg_val_at_k=}")
    dcn_topk_d = calc_topk(probs, hot_labels, top_k_list=cfg.top_k)
    logger.info(f"{dcn_topk_d=}")


def evaluate_category_aware_dcn(
    lit_model,
    inference_loader,
    candidates,
    category_candidates,
    candidate_asins,
    cfg,
    out_dir,
):
    hot_labels, probs, category_targets, asin_srcs = [], [], [], []
    for batch in tqdm(inference_loader):
        (
            img_emb,
            price_bin,
            category_src,
            asin_src,
            _,
            asin_targets,
        ) = batch

        probs_i = dcf_model_inference(
            lit_model, category_src, price_bin, img_emb, candidates
        )

        # Split perdiction per category
        asin_targets = np.array([asin_target[0] for asin_target in asin_targets])
        hot_labels_i = np.in1d(candidate_asins, asin_targets)
        category_targets_i = category_candidates[hot_labels_i]
        for cat in category_targets_i.unique():

            mask = category_targets_i == cat
            mask = [mask] if len(mask) == 1 else mask
            asin_targets_per_cateogry = asin_targets[mask]

            # Create ground true label
            hot_labels_i = np.in1d(
                candidate_asins,
                asin_targets_per_cateogry,
            )
            assert hot_labels_i.sum() > 0

            # Save
            hot_labels.append(hot_labels_i)
            probs.append(probs_i)
            category_targets.append(cat.item())
            asin_srcs.append(asin_src[0])

    # Calcualte probability
    probs = torch.vstack(probs)
    hot_labels = np.vstack(hot_labels)

    # Constrain to target
    for n, cat in enumerate(category_targets):
        probs[n, ~torch.isin(category_candidates, cat)] = 0

    # Calculate retrival metrics
    gan_ndcg_val = ndcg_score(hot_labels, probs)
    gan_topk_d = calc_topk(probs, torch.from_numpy(hot_labels), top_k_list=cfg.top_k)
    logger.info(f"evaluate_category_aware_dcn {gan_ndcg_val=} {gan_topk_d=}")

    ndcg_vals = [
        ndcg_score(hot_label_i.unsqueeze(0), prob_i.unsqueeze(0))
        for hot_label_i, prob_i in tqdm(zip(torch.from_numpy(hot_labels), probs))
    ]
    model_base_dir = osp.basename(cfg.model_dcf_weight_dir)
    out_path = osj(out_dir, f"{model_base_dir}.pkl")
    pd.DataFrame(
        {
            "asins": asin_srcs,
            "category_pos_test": category_targets,
            "ndcg": ndcg_vals,
        }
    ).to_pickle(out_path)
    logger.info(f"Finish predictor. {out_path}")


@hydra.main(version_base="1.2", config_path="../configs/", config_name="inference")
def execute_gan_inference(cfg: DictConfig):
    t0 = time()
    out_dir = os.getcwd()
    os.chdir(hydra.utils.get_original_cwd())
    name = osp.basename(out_dir)
    pl.seed_everything(cfg.seed)

    wandb.init(
        project=cfg.wandb.project,
        dir=out_dir,
        config=OmegaConf.to_container(cfg),
        job_type="analysis",
        name="analysis_" + name,
    )

    logger.info(f"out_dir={out_dir}")
    logger.info(cfg)
    logger.info(f"{torch.backends.mps.is_available()=}")
    for category_name, model_dcf_weight_dir in zip(
        cfg.category_names, cfg.model_dcf_weight_dirs
    ):
        logger.info(category_name)
        out_dir_category = osp.join(out_dir, category_name)

        # Dataset
        t1 = time()
        cfg.category_name = category_name
        cfg.model_dcf_weight_dir = model_dcf_weight_dir
        fbt_df, emb_df = load_dfs(cfg)
        fbt_df_train, fbt_df_test = (
            fbt_df[fbt_df["set_name"] == "train"],
            fbt_df[fbt_df["set_name"] == "test"],
        )
        dataset = FbtDataset(fbt_df, emb_df)
        logger.info(f"{[len(fbt_df_train), len(fbt_df_test), len(dataset)]=}")

        candidate_dataset = FbtCandidateDataset(fbt_df, emb_df)
        inference_dataset = FbtInferenceDataset(fbt_df, emb_df)

        candidate_loader = DataLoader(
            candidate_dataset,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            pin_memory=True,
            shuffle=False,
        )

        inference_loader = DataLoader(
            inference_dataset,
            batch_size=1,
            num_workers=0,
            pin_memory=True,
            shuffle=False,
        )

        logger.info(
            f"{[len(candidate_dataset), len(inference_loader)]=}. {time()-t1:.1f}s"
        )

        # Load weights
        checkpoint = osj(cfg.model_dcf_weight_dir, "checkpoint.ckpt")
        logger.info(checkpoint)
        lit_model = LitDCF.load_from_checkpoint(checkpoint)
        lit_model.eval()
        torch.set_grad_enabled(False)
        candidates, category_candidates, candidate_asins = get_candidate_embeddings(
            lit_model, candidate_loader
        )

        evaluate_dcn(
            lit_model,
            inference_loader,
            candidates,
            category_candidates,
            candidate_asins,
            cfg,
        )
        evaluate_category_aware_dcn(
            lit_model,
            inference_loader,
            candidates,
            category_candidates,
            candidate_asins,
            cfg,
            out_dir_category,
        )

        logger.info(f"Finish {category_name} in {time() - t0:.1f} s")


if __name__ == "__main__":
    execute_gan_inference()
