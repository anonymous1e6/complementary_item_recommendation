import itertools
import logging
import os
import os.path as osp
from os.path import join as osj
from time import time

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import ndcg_score
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import wandb
from dataset_utils import FbtCandidateDataset, FbtDataset, FbtInferenceDataset, load_dfs
from lit.eval_utils import calc_topk, generate_test_set_hot_labels, get_unique_asins
from lit.lit_utils import LitFbt

logger = logging.getLogger(__name__)


def evaluate_category_aware(dataset, cfg, out_dir):
    t1 = time()
    path = cfg.model_weight_dir
    model_base_dir = osp.basename(path)
    asin_src = torch.load(osj(path, "asin_src.pth"))
    asin_pos = torch.load(osj(path, "asin_pos.pth"))
    category_src = torch.from_numpy(torch.load(osj(path, "category_src.pth")))
    category_pos = torch.from_numpy(torch.load(osj(path, "category_pos.pth")))
    src_fbt = torch.load(osj(path, "src_fbt.pth"))
    src = torch.load(osj(path, "src.pth"))
    pos = torch.load(osj(path, "pos.pth"))
    set_name = torch.load(osj(path, "set_name.pth"))

    # Test sources: have a unique pair of (source,target-category)
    src_fbt_test = src_fbt[set_name == "test"]
    asin_src_test = asin_src[set_name == "test"]
    category_pos_test = category_pos[set_name == "test"]

    locs = list(zip(asin_src_test, category_pos_test))
    _, unique_idxs = np.unique(np.array(locs), axis=0, return_index=True)
    src_fbt_test, asin_src_test, category_pos_test = (
        src_fbt_test[unique_idxs],
        asin_src_test[unique_idxs],
        category_pos_test[unique_idxs],
    )

    # Candidate to compare with
    asins = np.hstack([asin_src, asin_pos])
    embs = torch.vstack([src, pos])
    categories = torch.hstack([category_src, category_pos])
    asins, embs, categories = get_unique_asins(asins, embs, categories)

    # Build hot label
    hot_labels, asin_src_test = generate_test_set_hot_labels(
        asin_src_test=asin_src_test,
        category_pos_test=category_pos_test.numpy(),
        fbt_by_asin_src=dataset.fbt_by_asin_src,
        asins=asins,
    )

    # Find distance of the candidates
    dists = torch.cdist(src_fbt_test, embs, p=2)
    probs = torch.softmax(-dists, axis=-1)

    # Constrain to target
    for n, cat in enumerate(category_pos_test):
        probs[n, categories != cat] = 0

    # Calculate retrival metrics
    gan_ndcg_val_at_k = {}
    for k in cfg.top_k + [probs.shape[-1]]:
        gan_ndcg_val = ndcg_score(hot_labels, probs, k=k)
        gan_ndcg_val_at_k[k] = gan_ndcg_val
    logger.info(f"{gan_ndcg_val_at_k=}")
    gan_topk_d = calc_topk(probs, hot_labels, top_k_list=cfg.top_k)
    logger.info(f"{gan_topk_d=}")

    ndcg_vals = [
        ndcg_score(hot_label_i.unsqueeze(0), prob_i.unsqueeze(0))
        for hot_label_i, prob_i in tqdm(zip(hot_labels, probs))
    ]
    out_path = osj(out_dir, f"{model_base_dir}.pkl")
    pd.DataFrame(
        {
            "asins": asin_src_test,
            "category_pos_test": category_pos_test,
            "ndcg": ndcg_vals,
        }
    ).to_pickle(out_path)
    logger.info(f"Finish predictor. {out_path}")


def evalaute_gan(candidate_loader, inference_loader, cfg):
    # Load weights
    checkpoint = osj(cfg.model_weight_dir, "checkpoint.ckpt")
    lit_model = LitFbt.load_from_checkpoint(checkpoint)
    lit_model.eval()

    # Iterate on candidates
    candidates, asins, category_candidates = [], [], []
    with torch.no_grad():
        for batch in tqdm(candidate_loader):
            (
                img_emb_candidate,
                price_bin_candidate,
                category_candidate,
                asin_candidate,
            ) = batch

            candidate = model_inference(
                lit_model, category_candidate, price_bin_candidate, img_emb_candidate
            )
            candidates.append(candidate.detach().cpu())
            asins.append(asin_candidate)
            category_candidates.append(category_candidate.detach().cpu())

    candidates = torch.vstack(candidates)
    asins = np.array(list(itertools.chain(*asins)))
    category_candidates = torch.hstack(category_candidates)

    # Get valid cateogires
    hot_labels, dists, valid_categories_list = [], [], []
    with torch.no_grad():
        for batch in tqdm(inference_loader):
            (
                img_emb_test,
                price_bin_test,
                category_test,
                _,
                valid_categories,
                asin_targets,
            ) = batch

            src = model_inference(
                lit_model, category_test, price_bin_test, img_emb_test
            )

            # Transform test to valid categories (categories that apeared in the training set)
            valid_categories_hstack = torch.hstack(valid_categories)
            src_repeat = src.repeat(len(valid_categories), 1)

            category_dst_emb = lit_model.category_embs(valid_categories_hstack)
            src_fbt = lit_model.fbt_ae(src_repeat, category_dst_emb)
            dists_i = torch.cdist(src_fbt, candidates, p=2)
            dists_i = dists_i.min(axis=0).values

            # Create ground true label
            hot_labels_i = np.in1d(
                asins, np.array([asin_target[0] for asin_target in asin_targets])
            )

            # Save
            valid_categories_list.append(valid_categories_hstack)
            dists.append(dists_i)
            hot_labels.append(hot_labels_i)

            assert hot_labels_i.sum() > 0

    # Calcualte probability
    dists = torch.vstack(dists)
    probs = torch.softmax(-dists, axis=-1)
    hot_labels = np.vstack(hot_labels)

    # Retrival metrics
    gan_ndcg_val_at_k = {}
    for k in cfg.top_k + [probs.shape[-1]]:
        gan_ndcg_val = ndcg_score(hot_labels, probs, k=k)
        gan_ndcg_val_at_k[k] = gan_ndcg_val
    logger.info(cfg.model_weight_dir)
    logger.info(f"{gan_ndcg_val_at_k=}")
    gan_topk_d = calc_topk(probs, hot_labels, top_k_list=cfg.top_k)
    logger.info(f"{gan_topk_d=}")


def model_inference(model, category, price_bin, img_emb):
    category_emb = model.category_embs(category)
    price_emb = model.price_embs(price_bin)
    emb = model.img_encoder(img_emb, category_emb, price_emb)
    return emb


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
    for category_name, model_gan_weight_dir in zip(
        cfg.category_names, cfg.model_gan_weight_dirs
    ):

        # Dataset
        t1 = time()
        cfg.category_name = category_name
        cfg.model_weight_dir = model_gan_weight_dir
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

        evalaute_gan(candidate_loader, inference_loader, cfg)
        evaluate_category_aware(dataset, cfg, out_dir)
        logger.info(f"Finish in {time() - t0:.1f} s")


if __name__ == "__main__":
    execute_gan_inference()
