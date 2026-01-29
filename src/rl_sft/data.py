from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
# import webdataset as wds
from torch.utils.data import DataLoader, Dataset, Sampler, IterableDataset
from PIL import Image
from accelerate.logging import get_logger

logger = get_logger(__name__)


# ==============================================================================  
# Aspect Ratio Bucketing for SFT
# ==============================================================================

# Standard SD1.4/1.5 buckets (512 base, 64 step, ~262144 pixels)
ASPECT_RATIO_BUCKETS = [
    (512, 512),   # 1:1
    (576, 448),   # ~1.29:1
    (448, 576),   # ~0.78:1
    (640, 384),   # ~1.67:1
    (384, 640),   # ~0.6:1
    (704, 384),   # ~1.83:1
    (384, 704),   # ~0.55:1
    (768, 320),   # 2.4:1
    (320, 768),   # ~0.42:1
    (512, 448),   # ~1.14:1
    (448, 512),   # ~0.88:1
    (576, 384),   # 1.5:1
    (384, 576),   # ~0.67:1
]


def find_closest_bucket(width: int, height: int, buckets: list[tuple[int, int]] = ASPECT_RATIO_BUCKETS) -> tuple[int, int]:
    """Find the bucket with closest aspect ratio to the input image."""
    aspect_ratio = width / height
    
    best_bucket = buckets[0]
    best_diff = float('inf')
    
    for bw, bh in buckets:
        bucket_ratio = bw / bh
        diff = abs(aspect_ratio - bucket_ratio)
        if diff < best_diff:
            best_diff = diff
            best_bucket = (bw, bh)
    
    return best_bucket


def resize_to_bucket(image: Image.Image, bucket: tuple[int, int]) -> Image.Image:
    """Resize image to fit bucket dimensions, then center crop."""
    target_w, target_h = bucket
    orig_w, orig_h = image.size
    
    # Calculate scale to cover the bucket (may need to crop)
    scale = max(target_w / orig_w, target_h / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    
    # Resize
    image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    # Center crop to exact bucket size
    left = (new_w - target_w) // 2
    top = (new_h - target_h) // 2
    image = image.crop((left, top, left + target_w, top + target_h))
    
    return image


def bucket_collate_fn(examples: list[dict]) -> PromptBatch:
    """Collate function with aspect ratio bucketing.
    
    All images in a batch are resized to the same bucket size.
    The bucket is chosen based on the first image's aspect ratio.
    """
    from torchvision import transforms
    
    if not examples:
        return PromptBatch(prompts=[], metadata=[], images=None, bucket_size=None)
    
    # Determine bucket from first image
    if "image" not in examples[0] or examples[0]["image"] is None:
        # Fallback for evaluation or cases where images are missing
        prompts = [ex["prompt"] for ex in examples]
        metadata = [ex["metadata"] for ex in examples]
        return PromptBatch(prompts=prompts, metadata=metadata, images=None)

    first_image = examples[0]["image"]
    bucket = find_closest_bucket(first_image.width, first_image.height)
    target_w, target_h = bucket
    
    # Define transform for this bucket
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    
    # Process all images
    processed_images = []
    for ex in examples:
        img = resize_to_bucket(ex["image"], bucket)
        processed_images.append(transform(img))
    
    pixel_values = torch.stack(processed_images)
    prompts = [ex["prompt"] for ex in examples]
    metadata = [ex["metadata"] for ex in examples]
    
    return PromptBatch(
        prompts=prompts, 
        metadata=metadata, 
        images=pixel_values,  # Now tensor, not PIL
        bucket_size=bucket,
    )



class PromptBatch(dict):
    def __init__(self, *args, **kwargs):
        if args and isinstance(args[0], dict):
            super().__init__(args[0])
            self.__dict__.update(args[0])
        else:
            super().__init__(**kwargs)
            self.__dict__.update(kwargs)
    
    @property
    def prompts(self):
        return self["prompts"]

    @property
    def metadata(self):
        return self["metadata"]

    @property
    def images(self):
        return self.get("images")


class TextPromptDataset(Dataset):
    def __init__(self, dataset_root: Path, split: str) -> None:
        file_path = dataset_root / f"{split}.txt"
        self.prompts = [line.strip() for line in file_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx: int) -> dict:
        return {"prompt": self.prompts[idx], "metadata": {}}

    @staticmethod
    def collate_fn(examples: Iterable[dict]) -> PromptBatch:
        prompts = [example["prompt"] for example in examples]
        metadata = [example["metadata"] for example in examples]
        return PromptBatch(prompts=prompts, metadata=metadata, images=None)


class GenevalPromptDataset(Dataset):
    def __init__(self, dataset_root: Path, split: str) -> None:
        file_path = dataset_root / f"{split}_metadata.jsonl"
        self.metadatas = [json.loads(line) for line in file_path.read_text(encoding="utf-8").splitlines()]
        self.prompts = [item["prompt"] for item in self.metadatas]

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx: int) -> dict:
        return {"prompt": self.prompts[idx], "metadata": self.metadatas[idx]}

    @staticmethod
    def collate_fn(examples: Iterable[dict]) -> PromptBatch:
        prompts = [example["prompt"] for example in examples]
        metadata = [example["metadata"] for example in examples]
        return PromptBatch(prompts=prompts, metadata=metadata, images=None)


class GenEval2Dataset(Dataset):
    def __init__(self, file_path: Path) -> None:
        if not file_path.exists():
            raise FileNotFoundError(f"GenEval2 dataset file not found: {file_path}")
        with open(file_path, "r") as f:
            self.data = json.load(f)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        item = self.data[idx]
        res = {
            "prompt": item["prompt"],
            "metadata": item
        }
        if "image_path" in item:
            image_path = Path(item["image_path"])
            if image_path.exists():
                res["image"] = Image.open(image_path).convert("RGB")
        return res

    @staticmethod
    def collate_fn(examples: Iterable[dict]) -> PromptBatch:
        prompts = [example["prompt"] for example in examples]
        metadata = [example["metadata"] for example in examples]
        images = [example.get("image") for example in examples]
        if all(img is None for img in images):
            images = None
        return PromptBatch(prompts=prompts, metadata=metadata, images=images)


class PairedPromptImageDataset(Dataset):
    pair_suffix = "_pairs.jsonl"

    def __init__(self, dataset_root: Path, split: str) -> None:
        self.dataset_root = dataset_root
        file_path = dataset_root / f"{split}{self.pair_suffix}"
        if not file_path.exists():
            raise FileNotFoundError(f"Paired dataset file not found: {file_path}")
        lines = [line for line in file_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.metadatas = [json.loads(line) for line in lines]
        self.prompts = []
        self.image_paths = []
        for item in self.metadatas:
            if "prompt" not in item or "image" not in item:
                raise ValueError("Each paired sample must include 'prompt' and 'image' fields")
            self.prompts.append(item["prompt"])
            self.image_paths.append(item["image"])

    @staticmethod
    def has_pair_file(dataset_root: Path, split: str) -> bool:
        return (dataset_root / f"{split}{PairedPromptImageDataset.pair_suffix}").exists()

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx: int) -> dict:
        image_path = Path(self.image_paths[idx])
        if not image_path.is_absolute():
            image_path = self.dataset_root / image_path
        image = Image.open(image_path).convert("RGB")
        return {"prompt": self.prompts[idx], "metadata": self.metadatas[idx], "image": image}

    @staticmethod
    def collate_fn(examples: Iterable[dict]) -> PromptBatch:
        prompts = [example["prompt"] for example in examples]
        metadata = [example["metadata"] for example in examples]
        images = [example["image"] for example in examples]
        return PromptBatch(prompts=prompts, metadata=metadata, images=images)


class DistributedKRepeatSampler(Sampler[list[int]]):
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        k: int,
        num_replicas: int,
        rank: int,
        seed: int = 0,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.k = k
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.epoch = 0

        # Cache the prompt order to be consistent across steps
        pg = torch.Generator()
        pg.manual_seed(self.seed)
        self.prompt_order = torch.randperm(len(self.dataset), generator=pg).tolist()

    def __iter__(self):
        while True:
            # We treat self.epoch as a global step counter (indices of global batches)
            global_step = self.epoch
            global_batch_size = self.num_replicas * self.batch_size
            
            # Start position in the virtual stream of k-repeated prompts
            start_in_stream = global_step * global_batch_size
            
            indices = []
            for i in range(global_batch_size):
                item_idx = start_in_stream + i
                prompt_seq_idx = item_idx // self.k
                # Wrap around the dataset if we exceed it
                prompt_idx = self.prompt_order[prompt_seq_idx % len(self.prompt_order)]
                indices.append(prompt_idx)
            
            # Yield this rank's portion
            yield indices[self.rank * self.batch_size : (self.rank + 1) * self.batch_size]

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch



class DummySampler:
    def set_epoch(self, epoch: int):
        pass


class DpoBatch(dict):
    """Batch class for DPO training with preference pairs."""
    
    def __init__(self, *args, **kwargs):
        if args and isinstance(args[0], dict):
            super().__init__(args[0])
            self.__dict__.update(args[0])
        else:
            super().__init__(**kwargs)
            self.__dict__.update(kwargs)
    
    @property
    def pixel_values_w(self):
        """Winning (preferred) images tensor."""
        return self["pixel_values_w"]
    
    @property
    def pixel_values_l(self):
        """Losing (rejected) images tensor."""
        return self["pixel_values_l"]
    
    @property
    def prompts(self):
        return self["prompts"]


class CurriculumPickaPicDataset(IterableDataset):
    """WebDataset wrapper for Curriculum Learning with PickaPic."""
    
    def __init__(self, dataset_path: str, is_train: bool, curriculum_groups: int = 0):
        self.dataset_path = dataset_path
        self.is_train = is_train
        self.curriculum_groups = curriculum_groups
        self.current_group_idx = 0
        
        # Thresholds for "margin" |score_0 - score_1| = |2*score_0 - 1|
        # Easy examples have high margin (close to 1.0). Hard examples have low margin (close to 0.0).
        # We start with easy examples (margin > T) and progressively lower T.
        if curriculum_groups > 0:
            # e.g. 5 groups: 0.8, 0.6, 0.4, 0.2, 0.0
            # We want to INCLUDE examples with margin >= threshold.
            self.thresholds = [1.0 - (i + 1) / curriculum_groups for i in range(curriculum_groups)]
        else:
            self.thresholds = [0.0]
            
    def update_cl(self):
        """Update curriculum level."""
        if self.group_stats[self.current_group_idx] > 0.05 and self.current_group_idx < self.curriculum_groups - 1:
            self.current_group_idx += 1
            logger.info(f"Curriculum updated: Level {self.current_group_idx}, Min Margin {self.thresholds[self.current_group_idx]:.4f}", main_process_only=True)

    def __iter__(self):
        import webdataset as wds
        import glob
        import io
        
        dataset_root = Path(self.dataset_path)
        patterns = [str(dataset_root / "*.tar"), str(dataset_root / "data" / "*.tar")]
        urls = []
        for p in patterns:
            found = sorted(glob.glob(p))
            if found:
                urls.extend(found)
        
        if not urls:
            raise FileNotFoundError(f"No .tar files found in {self.dataset_path}")
        
        # Build Pipeline
        pipeline = [wds.ResampledShards(urls)]
        
        if self.is_train:
            pipeline.append(wds.detshuffle(100))
            pipeline.append(wds.split_by_node)
            pipeline.append(wds.split_by_worker)
            pipeline.append(wds.shuffle(1000))
        else:
            pipeline.append(wds.split_by_worker)
        
        pipeline.extend([
            wds.tarfile_to_samples(),
            wds.decode("pil"),
        ])
        
        current_threshold = self.thresholds[self.current_group_idx] if self.curriculum_groups > 0 else 0.0
        
        def process_sample(sample):
            """Process a single sample from the WebDataset."""
            label_0 = 0.5
            
            # Support both JSON metadata and individual .txt files
            if "json" in sample:
                meta = sample["json"]
                caption = meta.get("caption", "")
                label_0 = float(meta.get("label_0", 0.5))
            else:
                # Fallback for individual files
                caption = sample.get("original_prompt.txt") or sample.get("prompt.txt") or ""
                if isinstance(caption, bytes):
                    caption = caption.decode("utf-8").strip()
                
                label_data = sample.get("label_0.txt")
                if label_data is not None:
                    try:
                        label_str = label_data.decode("utf-8").strip() if isinstance(label_data, bytes) else str(label_data).strip()
                        label_0 = float(label_str)
                    except:
                        label_0 = 0.5
            
            # Calculate margin: |2 * score - 1|
            # score=1.0 -> margin=1.0 (Easy)
            # score=0.5 -> margin=0.0 (Hard/Ambiguous)
            margin = abs(2 * label_0 - 1.0)
            
            # Curriculum Filter
            if margin < current_threshold:
                return None
            
            # Discard neutral samples anyway for standard DPO
            if abs(label_0 - 0.5) < 1e-3:
                return None

            # Get both images
            img_0 = sample.get("jpg_0.jpg") or sample.get("0.jpg") or sample.get("jpg")
            img_1 = sample.get("jpg_1.jpg") or sample.get("1.jpg")
            
            if img_0 is None or img_1 is None:
                return None
            
            img_0 = img_0.convert("RGB") if hasattr(img_0, 'convert') else Image.open(io.BytesIO(img_0)).convert("RGB")
            img_1 = img_1.convert("RGB") if hasattr(img_1, 'convert') else Image.open(io.BytesIO(img_1)).convert("RGB")
            
            # Assign winner/loser based on label
            # If label_0 > 0.5, 0 wins.
            if label_0 > 0.5:
                img_w, img_l = img_0, img_1
            else:
                img_w, img_l = img_1, img_0
            
            return {
                "image_w": img_w,
                "image_l": img_l,
                "prompt": caption,
                "margin": margin, # Useful for logging
            }
        
        def filter_none(sample):
            return sample is not None
        
        pipeline.append(wds.map(process_sample))
        pipeline.append(wds.select(filter_none))
        
        # Create DataPipeline and iterate
        ds = wds.DataPipeline(pipeline)
        for sample in ds:
            yield sample


class ParquetPickaPicDataset(IterableDataset):
    """HuggingFace Datasets wrapper for PickaPic-v2 Parquet format."""
    
    def __init__(self, dataset_path: str, is_train: bool, curriculum_groups: int = 0):
        """Initialize Parquet PickaPic dataset.
        
        Args:
            dataset_path: HuggingFace dataset name (e.g., "yuvalkirstain/pickapic_v2") 
                         or local parquet file path
            is_train: Whether this is training data (enables shuffling)
            curriculum_groups: Number of curriculum groups (0 disables curriculum)
        """
        self.dataset_path = dataset_path
        self.is_train = is_train
        self.curriculum_groups = curriculum_groups
        self.current_group_idx = 0
        
        # Curriculum thresholds (same as WebDataset version)
        if curriculum_groups > 0:
            self.thresholds = [1.0 - (i + 1) / curriculum_groups for i in range(curriculum_groups)]
        else:
            self.thresholds = [0.0]
    
    def update_cl(self):
        """Update curriculum level."""
        if self.current_group_idx < self.curriculum_groups - 1:
            self.current_group_idx += 1
            logger.info(
                f"Curriculum updated: Level {self.current_group_idx}, "
                f"Min Margin {self.thresholds[self.current_group_idx]:.4f}",
                main_process_only=True
            )
    
    def __iter__(self):
        import io
        from datasets import load_dataset
        import torch
        
        # Resolve path and check if local
        p = Path(self.dataset_path)
        is_local = p.exists()
        
        if is_local and p.is_dir():
            # Check for standard PickaPic-v2 structure with 'data' subfolder
            data_dir = p / "data"
            if data_dir.is_dir():
                data_files = str(data_dir / "*.parquet")
            else:
                data_files = str(p / "*.parquet")
        else:
            data_files = self.dataset_path

        # Load dataset
        # Load dataset
        try:
            # Check if path exists locally
            path_obj = Path(self.dataset_path)
            split_name = "train" if self.is_train else "test"
            
            if path_obj.exists():
                # Local file(s)
                if path_obj.is_dir():
                    # Check if separate train/test folders or files exist to infer split?
                    # For now, blindly attempt to load with split_name, fallback to 'train' if not strict?
                    # Actually, for local parquet directories, usually it's all one split unless specified.
                    # Let's try to map 'test' to 'train' if local and simple directory?
                    # Safer: just use split_name. If it fails, user needs to structure local data correctly.
                    dataset = load_dataset(
                        "parquet",
                        data_dir=self.dataset_path,
                        split=split_name,
                        streaming=True
                    )
                else:
                    dataset = load_dataset(
                        "parquet",
                        data_files=self.dataset_path,
                        split=split_name,
                        streaming=True
                    )
            else:
                # HuggingFace dataset name
                dataset = load_dataset(
                    self.dataset_path,
                    split=split_name,
                    streaming=True
                )
        except Exception as e:
            logger.error(f"Failed to load dataset from {self.dataset_path}: {e}")
            raise
        
        # Sharding for multi-GPU and multi-worker
        if torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()
            dataset = dataset.shard(num_shards=world_size, index=rank)
            
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            dataset = dataset.shard(num_shards=worker_info.num_workers, index=worker_info.id)

        # Shuffle if training
        if self.is_train:
            dataset = dataset.shuffle(seed=42, buffer_size=10000)
        
        current_threshold = self.thresholds[self.current_group_idx] if self.curriculum_groups > 0 else 0.0
        
        for sample in dataset:
            try:
                # Extract fields
                caption = sample.get("caption", "")
                if not caption:
                    continue
                
                # Get label (preference score for image_0)
                label_0 = float(sample.get("label_0", 0.5))
                
                # Calculate margin for curriculum learning
                margin = abs(2 * label_0 - 1.0)
                
                # Curriculum filter
                if margin < current_threshold:
                    continue
                
                # Skip neutral samples
                if abs(label_0 - 0.5) < 1e-3:
                    continue
                
                # Decode images from bytes
                jpg_0 = sample.get("jpg_0")
                jpg_1 = sample.get("jpg_1")
                
                if jpg_0 is None or jpg_1 is None:
                    continue
                
                # Convert bytes to PIL Images
                if isinstance(jpg_0, bytes):
                    img_0 = Image.open(io.BytesIO(jpg_0)).convert("RGB")
                else:
                    img_0 = jpg_0.convert("RGB")
                
                if isinstance(jpg_1, bytes):
                    img_1 = Image.open(io.BytesIO(jpg_1)).convert("RGB")
                else:
                    img_1 = jpg_1.convert("RGB")
                
                # Assign winner/loser based on label_0
                if label_0 > 0.5:
                    img_w, img_l = img_0, img_1
                else:
                    img_w, img_l = img_1, img_0
                
                yield {
                    "image_w": img_w,
                    "image_l": img_l,
                    "prompt": caption,
                    "margin": margin,
                }
                
            except Exception as e:
                # Skip corrupted samples
                logger.warning(f"Error processing sample: {e}")
                continue


def build_pickapic_dataset(dataset_path: str, is_train: bool, curriculum_groups: int = 0, format_type: str = "webdataset"):
    """Build PickaPic dataset with specified format.
    
    Args:
        dataset_path: Path to dataset (directory for webdataset, HF name or file for parquet)
        is_train: Whether this is training data
        curriculum_groups: Number of curriculum groups
        format_type: "webdataset" or "parquet"
    
    Returns:
        Dataset instance
    """
    if format_type == "parquet":
        return ParquetPickaPicDataset(dataset_path, is_train, curriculum_groups)
    else:
        return CurriculumPickaPicDataset(dataset_path, is_train, curriculum_groups)



def dpo_collate_fn(examples: list[dict]) -> DpoBatch:
    """Collate function for DPO batches."""
    from torchvision import transforms
    
    transform = transforms.Compose([
        transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(512),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    
    pixel_values_w = torch.stack([transform(ex["image_w"]) for ex in examples])
    pixel_values_l = torch.stack([transform(ex["image_l"]) for ex in examples])
    prompts = [ex["prompt"] for ex in examples]
    
    return DpoBatch(
        pixel_values_w=pixel_values_w,
        pixel_values_l=pixel_values_l,
        prompts=prompts,
    )


def build_spright_dataset(dataset_root: Path, split: str, is_train: bool):
    import webdataset as wds
    import glob
    # Look for tars directly in root or in data/ subdirectory
    patterns = [str(dataset_root / "*.tar"), str(dataset_root / "data" / "*.tar")]
    urls = []
    for p in patterns:
        found = sorted(glob.glob(p))
        if found:
            urls.extend(found)
            
    if not urls:
        # Fallback: maybe specific split file points to tars?
        # For now assume failure if no tars found
        raise FileNotFoundError(f"No .tar files found in {dataset_root}")

    # Simple train/test split logic if multiple tars exist
    # If split is 'test', take the last one (if >1), else use same?
    # Or just use all for 'train' and first one for 'test' evaluation?
    # User didn't specify split logic, assuming all data is valid.
    # Let's exclude the last tar for training if we have enough files?
    # For simplicity, if split == 'test' and len > 1, take last.
    
    if len(urls) > 1:
        if split == "test":
            urls = urls[-1:]
        elif split == "train":
            # Just use all? Or exclude test? 
            # Let's keep it simple: train sees all.
            pass 
    
    # Use ResampledShards to make the dataset infinite, matching the behavior of DistributedKRepeatSampler
    if is_train:
        pipeline = [wds.ResampledShards(urls)]
        pipeline.append(wds.detshuffle(100))
        pipeline.append(wds.split_by_node)
        pipeline.append(wds.split_by_worker)
        pipeline.append(wds.shuffle(1000))
    else:
        pipeline = [wds.SimpleShardList(urls)]
        pipeline.append(wds.split_by_node)
        pipeline.append(wds.split_by_worker)
    
    pipeline.extend([
        wds.tarfile_to_samples(),
        wds.decode("pil"),
    ])
    
    def process_sample(sample):
        # sample dict keys usually: 'json', 'jpg', '__key__', etc.
        meta = sample["json"]
        image = sample["jpg"]
        return {
            "prompt": meta.get("spatial_caption", ""),
            "metadata": meta,
            "image": image.convert("RGB")
        }
        
    pipeline.append(wds.map(process_sample))
    return wds.DataPipeline(pipeline)


def build_dataloaders(config, accelerator):
    dataset_root = Path(config.dataset.root)
    
    if config.dataset.name == "spright":
        train_dataset = build_spright_dataset(dataset_root, config.dataset.train_split, True)
        if config.training.mode == "sft":
            # Use aspect ratio bucketing for SFT
            train_collate_fn = bucket_collate_fn
        else:
            train_collate_fn = PairedPromptImageDataset.collate_fn
            
        # Validation on fixed subset of training data (first 50)
        # We reuse build_spright_dataset but with is_train=False to get a SimpleShardList (sequential)
        # Then we take the first 50 items.
        
        # Note: We want the EXACT same data as training but deterministic subset.
        # build_spright_dataset(..., is_train=False) does SimpleShardList.
        val_dataset_full = build_spright_dataset(dataset_root, config.dataset.train_split, is_train=False)
        
        # Create a finite dataset from the WebDataset pipeline for validation
        # We need to manually slice it. Since wds.DataPipeline is an iterable, we can't just slice.
        # We will wrap it in an IterableDataset that stops after 50.
        
        class LimitedIterableDataset(IterableDataset):
            def __init__(self, dataset, limit):
                self.dataset = dataset
                self.limit = limit
                
            def __iter__(self):
                count = 0
                for item in self.dataset:
                    if count >= self.limit:
                        break
                    yield item
                    count += 1
                    
        test_dataset = LimitedIterableDataset(val_dataset_full, limit=50)
        test_collate_fn = PairedPromptImageDataset.collate_fn
        
        train_sampler = DummySampler()
        
        # WebDataset is an IterableDataset, so we pass it directly to DataLoader
        # batch_size is handled by DataLoader (it pulls items from iterable and batches them)
        # We assume k=1 for Spright SFT. If GRPO needs k>1, we would need to repeat in the pipeline.
        
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=config.sampling.train_batch_size,
            num_workers=config.reward.max_workers, # or hardcode 4
            collate_fn=train_collate_fn,
            pin_memory=True
        )
        
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=config.sampling.test_batch_size,
            num_workers=0, # Avoid multi-processing issues for small finite iterables if possible, or keep small
            collate_fn=test_collate_fn,
            pin_memory=True
        )
        
        # Wrap DataLoaders with accelerator.skip_first_batches style behavior manually if needed
        # But actually Accelerate's validation was failing because it tries to introspect/shard the dataloader.
        # Since we skipped accelerator.prepare for unet/optimizer, the dataloader is NOT prepared.
        # But wait! If the user passed accelerator to this function, did we USE it to shard?
        # WebDataset already handles sharding via split_by_node/worker.
        # So raw DataLoader is fine.
        
        # HOWEVER, the traceback shows:
        # File "/mnt/data_hdd/stang/code/temp_code/sd14_grpo_sft/src/sd14_grpo_sft/trainer.py", line 573, in train_sft
        #    batch = next(train_iter)
        # File "/home/stang/miniconda3/envs/sd/lib/python3.10/site-packages/accelerate/data_loader.py", line 866, in __iter__

        # This implies `train_dataloader` IS an `accelerate.data_loader.DataLoader`.
        # This means `accelerator.prepare` WAS CALLED on it. 
        # But in my previous edit I put the exclusion block. Let me double check if that edit succeeded properly.
        # The line numbers in traceback might indicate that the previous edit did NOT apply or was partial?
        # Or maybe I am reading the traceback wrong.
        # The traceback says `train_sft` line 573call `next(train_iter)`.
        # `train_iter` comes from `iter(train_dataloader)`.
        
        return train_dataloader, test_dataloader, train_sampler

    # DPO mode: use PickaPic preference dataset
    if config.training.mode == "dpo":
        dataset_path = config.dataset.dpo_dataset_path or config.dataset.root
        if not dataset_path:
            raise ValueError("DPO mode requires dataset.dpo_dataset_path or dataset.root to be set")
        
        format_type = config.dataset.dpo_format
        train_dataset = build_pickapic_dataset(
            dataset_path, 
            is_train=True, 
            curriculum_groups=config.dpo.curriculum_groups,
            format_type=format_type
        )
        train_sampler = DummySampler()
        
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=config.sampling.train_batch_size,
            num_workers=config.reward.max_workers,
            collate_fn=dpo_collate_fn,
            pin_memory=True,
        )
        
        # Test Dataloader for Validation
        test_dataset = build_pickapic_dataset(
            dataset_path, 
            is_train=False, 
            curriculum_groups=0,
            format_type=format_type
        )
        
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=config.sampling.test_batch_size, # Use test_batch_size
            num_workers=1, # Keep low for validation
            collate_fn=dpo_collate_fn,
            pin_memory=True,
        )
        
        return train_dataloader, test_dataloader, train_sampler


    if config.training.mode == "sft":
        if config.dataset.name == "spright":
            train_dataset = build_spright_dataset(dataset_root, config.dataset.train_split, True)
            train_collate_fn = bucket_collate_fn
            test_dataset = build_spright_dataset(dataset_root, config.dataset.test_split, False)
            test_collate_fn = bucket_collate_fn
        elif config.dataset.name == "geneval2":
            # Check if train_split ends with .json, if not, construct path
            train_split = config.dataset.train_split
            if not train_split.endswith(".json"):
                train_file = dataset_root / f"{train_split}.json"
            else:
                train_file = Path(train_split)
            
            test_split = config.dataset.test_split
            if not test_split.endswith(".json"):
                test_file = dataset_root / f"{test_split}.json"
            else:
                test_file = Path(test_split)

            train_dataset = GenEval2Dataset(train_file)
            test_dataset = GenEval2Dataset(test_file)
            train_collate_fn = bucket_collate_fn
            test_collate_fn = bucket_collate_fn
        else:
            train_dataset = PairedPromptImageDataset(dataset_root, config.dataset.train_split)
            train_collate_fn = PairedPromptImageDataset.collate_fn
            if PairedPromptImageDataset.has_pair_file(dataset_root, config.dataset.test_split):
                test_dataset = PairedPromptImageDataset(dataset_root, config.dataset.test_split)
                test_collate_fn = PairedPromptImageDataset.collate_fn
            elif config.dataset.name == "geneval":
                test_dataset = GenevalPromptDataset(dataset_root, config.dataset.test_split)
                test_collate_fn = GenevalPromptDataset.collate_fn
            elif config.dataset.name == "ocr":
                test_dataset = TextPromptDataset(dataset_root, config.dataset.test_split)
                test_collate_fn = TextPromptDataset.collate_fn
            else:
                raise ValueError(f"Unsupported dataset {config.dataset.name}")
    elif config.dataset.name == "geneval":
        train_dataset = GenevalPromptDataset(dataset_root, config.dataset.train_split)
        test_dataset = GenevalPromptDataset(dataset_root, config.dataset.test_split)
        train_collate_fn = GenevalPromptDataset.collate_fn
        test_collate_fn = GenevalPromptDataset.collate_fn
    elif config.dataset.name == "geneval2":
        # Check if train_split ends with .json, if not, construct path
        train_split = config.dataset.train_split
        if not train_split.endswith(".json"):
            train_file = dataset_root / f"{train_split}.json"
        else:
            train_file = Path(train_split)
        
        test_split = config.dataset.test_split
        if not test_split.endswith(".json"):
            test_file = dataset_root / f"{test_split}.json"
        else:
            test_file = Path(test_split)

        train_dataset = GenEval2Dataset(train_file)
        test_dataset = GenEval2Dataset(test_file)
        train_collate_fn = GenEval2Dataset.collate_fn
        test_collate_fn = GenEval2Dataset.collate_fn
    elif config.dataset.name == "ocr":
        train_dataset = TextPromptDataset(dataset_root, config.dataset.train_split)
        test_dataset = TextPromptDataset(dataset_root, config.dataset.test_split)
        train_collate_fn = TextPromptDataset.collate_fn
        test_collate_fn = TextPromptDataset.collate_fn
    else:
        raise ValueError(f"Unsupported dataset {config.dataset.name}")

    # If trainer handles num_image_per_prompt expansion locally, sampler should not repeat k
    repeat_k = 1 # Force 1 as trainer.py now expands prompts for pipeline and reward
    train_sampler = DistributedKRepeatSampler(
        dataset=train_dataset,
        batch_size=config.sampling.train_batch_size,
        k=repeat_k,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
        seed=config.run.seed,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=0, # Changed from 1 to 0 to avoid potential multiprocessing hangs
        collate_fn=train_collate_fn,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.sampling.test_batch_size,
        collate_fn=test_collate_fn,
        shuffle=False,
        num_workers=2,
    )

    return train_dataloader, test_dataloader, train_sampler

