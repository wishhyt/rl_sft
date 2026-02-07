from __future__ import annotations

import os
import sys
import torch
import numpy as np
from PIL import Image
from io import BytesIO
from typing import Callable
from collections import defaultdict

from .bootstrap import bootstrap
from accelerate.logging import get_logger

logger = get_logger(__name__)

def geneval2_score_local(device):
    """Local implementation of GenEval2 reward to avoid modifying external libraries."""
    import requests
    from requests.adapters import HTTPAdapter, Retry
    import pickle
    import time

    url = "http://127.0.0.1:18087"
    
    # Try to see if server is alive with retries for slow booting
    use_server = False
    max_retries = 30
    for i in range(max_retries):
        try:
            # Use OPTIONS instead of GET to avoid 405 from Gunicorn
            r = requests.options(url, timeout=5)
            if r.status_code < 500:
                use_server = True
                break
        except Exception:
            if i % 5 == 0:
                logger.info(f"Waiting for GenEval2 server to be ready ({i}/{max_retries})...", main_process_only=True)
            time.sleep(2)

    if not use_server:
        # Prevent silent local fallback which loads 15GB model per training process
        raise ConnectionError(
            f"\n[Error] Cannot connect to GenEval2 server at {url} after {max_retries} retries.\n"
            "Please ensure the reward-server is fully started in another terminal.\n"
            "Local fallback is disabled to prevent OOM/Disk thrashing during training."
        )

    logger.info(f"Connected to GenEval2 server at {url}", main_process_only=True)

    # Server path
    sess = requests.Session()
    retries = Retry(total=5, backoff_factor=2, status_forcelist=[500, 502, 503, 504])
    sess.mount("http://", HTTPAdapter(max_retries=retries))

    def _fn_remote(images, prompts, metadatas, only_strict=False):
        # Only log on local main to avoid duplicate prints in distributed
        logger.info(f"      [Reward] Sending {len(images)} images to server...")
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
        
        jpeg_images = []
        for image in images:
            img = Image.fromarray(image)
            buffer = BytesIO()
            img.save(buffer, format="JPEG")
            jpeg_images.append(buffer.getvalue())

        data = {
            "images": jpeg_images,
            "meta_datas": list(metadatas),
            "only_strict": only_strict,
        }
        try:
            # Increased timeout for complex GenEval2 queries
            response = sess.post(url, data=pickle.dumps(data), timeout=600)
            if response.status_code != 200:
                error_msg = response.text
                logger.error(f"      [Reward] Server returned {response.status_code}: {error_msg}")
                raise RuntimeError(f"Reward server error: {error_msg}")
            
            res = pickle.loads(response.content)
            logger.info(f"      [Reward] Received scores from server.")
            return res["scores"], res["rewards"], res["strict_rewards"], res["group_rewards"], res["group_strict_rewards"]
        except Exception as e:
            logger.error(f"      [Reward] Server request failed: {e}")
            raise e

    return _fn_remote


def build_reward_fn(config, device) -> Callable:
    bootstrap()
    import flow_grpo.rewards as rewards

    score_dict = config.reward.weights
    
    # Handle geneval2 separately if it's in AlignDiff scope
    custom_fns = {}
    if "geneval2" in score_dict:
        custom_fns["geneval2"] = geneval2_score_local(device)

    # Use flow_grpo for the rest
    remaining_scores = {k: v for k, v in score_dict.items() if k != "geneval2"}
    
    if remaining_scores:
        base_reward_fn = rewards.multi_score(device, remaining_scores)
    else:
        base_reward_fn = None

    def _fn(images, prompts, metadata, *, only_strict: bool | None = None):
        strict = config.reward.only_strict if only_strict is None else only_strict
        
        all_score_details = {}
        total_scores = None

        # 1. Process custom geneval2
        if "geneval2" in custom_fns:
            scores, rewards_val, strict_val, group_val, group_strict_val = custom_fns["geneval2"](
                images, prompts, metadata, only_strict=strict
            )
            weight = score_dict["geneval2"]
            total_scores = [s * weight for s in scores]
            
            all_score_details["geneval2"] = scores
            all_score_details["accuracy"] = rewards_val
            all_score_details["strict_accuracy"] = strict_val
            # Add groups if needed
            for k, v in group_strict_val.items():
                all_score_details[f"{k}_strict_accuracy"] = v

        # 2. Process other rewards from flow_grpo
        if base_reward_fn:
            base_details, _ = base_reward_fn(images, prompts, metadata, only_strict=strict)
            for k, v in base_details.items():
                if k == "avg":
                    if total_scores is None:
                        total_scores = v
                    else:
                        total_scores = [t + s for t, s in zip(total_scores, v)]
                else:
                    all_score_details[k] = v

        all_score_details["avg"] = total_scores
        return all_score_details

    return _fn
