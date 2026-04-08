import os
import copy
from dataclasses import dataclass, field
import logging
import pathlib
from typing import Dict, Optional, Sequence
import torch
import glob
import transformers
from mobileo.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX, DEFAULT_IM_START_TOKEN_IDX
from torch.utils.data import Dataset
from mobileo.train.mobileo_trainer import mobileoTrainer
from mobileo import conversation as conversation_lib
from mobileo.model import mobileoFastSFTForCausalLM
from PIL import ImageFile
from datasets import load_dataset, concatenate_datasets
from datasets.utils.logging import set_verbosity_info
from transformers import logging as tf_logging
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoProcessor, TrainerCallback
import random
from typing import List
import warnings

warnings.filterwarnings("ignore", message="Plan failed with a CuDNNError")
warnings.filterwarnings("ignore", message=".*copying from a non-meta parameter.*")

ImageFile.LOAD_TRUNCATED_IMAGES = True

set_verbosity_info()
tf_logging.set_verbosity_info()

local_rank = None

class MobileConditioningCallback(TrainerCallback):
    def __init__(self, total_epochs):
        self.total_epochs = total_epochs
        print(f"\n{'=' * 70}")
        print(f"MobileConditioningCallback initialized for {total_epochs} epochs")
        print(f"{'=' * 70}\n")

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if self.total_epochs is None:
            self.total_epochs = int(args.num_train_epochs)
    def _log_metrics(self, model, epoch, metrics):
        """
        Pretty print metrics with both raw and normalized weights.
        
        Args:
            model: The model (already unwrapped)
            epoch: Current epoch (0-indexed)
            metrics: Dictionary with temperature and layer_weights
        """
        print("\n" + "="*70)
        print(f"[Mobile Conditioning] Epoch {epoch + 1}/{self.total_epochs}")
        print("="*70)
        
        # Temperature
        temp = metrics.get('temperature', 0)
        print(f"Temperature: {temp:.4f}")
        
        # Get RAW learnable weights
        try:
            connector = model.get_model().diffusion_connector if hasattr(model, 'get_model') else model.diffusion_connector
            raw_weights = connector.fusion.layer_weights.data.cpu()
            self._last_raw_weights = raw_weights.clone()
            
        except Exception as e:
            print(f"   Error getting raw weights: {e}")

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        """Called at the end of each epoch."""

        if model is None:
            print("ERROR: Model is None in callback!")
            return

        epoch = int(state.epoch) - 1  # Convert to 0-indexed

        # Unwrap model (critical for distributed training!)
        if hasattr(model, 'module'):
            actual_model = model.module
            print(f"✓ Unwrapped model.module")
        else:
            actual_model = model
            print(f"✓ Using model directly")

        # Check if method exists
        if not hasattr(actual_model, 'anneal_conditioning_temperature'):
            print(f"ERROR: Model doesn't have anneal_conditioning_temperature method!")
            print(f"   Model type: {type(actual_model)}")
            print(f"   Available methods: {[m for m in dir(actual_model) if not m.startswith('_')][:10]}")
            return

class MobileConditioningCallback_(TrainerCallback):
    """Callback for temperature annealing in Mobile Conditioning Projector."""
    
    def __init__(self, total_epochs=None):
        self.total_epochs = total_epochs
    
    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if self.total_epochs is None:
            self.total_epochs = int(args.num_train_epochs)
        if state.is_world_process_zero:
            print(f"\n{'='*70}")
            print(f"Mobile Conditioning: Will anneal temperature over {self.total_epochs} epochs")
            print(f"{'='*70}\n")
    
    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        if model is None:
            return
        
        epoch = int(state.epoch) - 1  # state.epoch is 1-indexed
        actual_model = model.module if hasattr(model, 'module') else model
        
        if hasattr(actual_model, 'anneal_conditioning_temperature'):
            metrics = actual_model.anneal_conditioning_temperature(epoch, self.total_epochs)
            
            if state.is_world_process_zero and metrics is not None:
                temp = metrics.get('temperature')
                weights = metrics.get('layer_weights')
                
                print(f"\n{'─'*70}")
                print(f"Mobile Conditioning - Epoch {epoch + 1}/{self.total_epochs}")
                print(f"{'─'*70}")
                print(f"Temperature: {temp:.4f}")
                
                if weights:
                    print("Layer Importance:")
                    for i, w in enumerate(weights):
                        bar = "█" * int(w * 50)
                        print(f"  Layer {21+i}: {w:.4f} {bar}")
                print(f"{'─'*70}\n")

def rank0_print(*args):
    if local_rank == 0:
        print(*args)


from packaging import version


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=True)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    gen_vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)  # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    pretrain_gen_mlp_adapter: Optional[str] = field(default=None)
    vision_tower_pretrained: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default="linear")
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_patch_merge_type: Optional[str] = field(default="flat")
    mm_vision_select_feature: Optional[str] = field(default="patch")
    diffusion_name_or_path: Optional[str] = field(default="Efficient-Large-Model/Sana_600M_512px_diffusers")
    vlm_num_layers: Optional[int] = field(default=4)
    is_train: bool = field(default=False)

@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    journeyDB_folder: Optional[str] = field(default=None)
    shortcaption_image_folder: Optional[str] = field(default=None)
    data_type: Optional[str] = field(default="mix")
    image_aspect_ratio: str = "pad"
    aspect_ratio_size: List[float] = field(default_factory=lambda: [512.0, 512.0], metadata={"help": "Resolution for aspect ratio."},)
    tokenizer_max_length: int = field(default=512)

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."},
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."},
    )
    bits: int = field(default=16, metadata={"help": "How many bits to use."})
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)
    ddp_find_unused_parameters: bool = True
    min_lr: float = field(default=1e-6, metadata={"help": "Minimum learning rate for cosine scheduler."})
    lr_scheduler_kwargs: dict = field(default_factory=lambda: {'min_lr':1e-6})


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param



def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return




def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str, vision_tower: str):
    if trainer.deepspeed:
        torch.cuda.synchronize()
    keys_to_match = ["mm_projector"]
    if getattr(trainer.args, "use_im_start_end", False):
        keys_to_match.extend(["embed_tokens", "embed_in"])

    weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
    trainer.model.config.save_pretrained(output_dir)

    current_folder = output_dir.split("/")[-1]
    parent_folder = os.path.dirname(output_dir)
    if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
        if current_folder.startswith("checkpoint-"):
            mm_projector_folder = os.path.join(parent_folder, "mm_projector")
            os.makedirs(mm_projector_folder, exist_ok=True)
            torch.save(
                weight_to_save,
                os.path.join(mm_projector_folder, f"{current_folder}.bin"),
            )
        else:
            torch.save(weight_to_save, os.path.join(output_dir, f"mm_projector.bin"))

    keys_to_match = ["gen_projector"]
    if getattr(trainer.args, "use_im_start_end", False):
        keys_to_match.extend(["embed_tokens", "embed_in"])

    weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
    trainer.model.config.save_pretrained(output_dir)

    current_folder = output_dir.split("/")[-1]
    parent_folder = os.path.dirname(output_dir)
    if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
        if current_folder.startswith("checkpoint-"):
            mm_projector_folder = os.path.join(parent_folder, "gen_projector")
            os.makedirs(mm_projector_folder, exist_ok=True)
            torch.save(
                weight_to_save,
                os.path.join(mm_projector_folder, f"{current_folder}.bin"),
            )
        else:
            torch.save(weight_to_save, os.path.join(output_dir, f"gen_projector.bin"))

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):


    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        input_embeddings[-num_new_tokens:] = input_embeddings_avg




def preprocess_multimodal(sources: Sequence[str], data_args: DataArguments) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal: return sources
    gen_placeholder = ""
    inst_type = None
    for source in sources:  # [instance]
        for sentence in source:
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, gen_placeholder).strip()
            inst_type = "gen"
    return sources, inst_type




def preprocess_qwen(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False, max_len=2048, system_message: str = "You are a helpful assistant.") -> Dict:
    roles = {"human": "user", "gpt": "assistant"}

    tokenizer = copy.deepcopy(tokenizer)
    chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    tokenizer.chat_template = chat_template

    # Apply prompt templates
    input_ids, targets = [], []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != roles["human"]:
            source = source[1:]

        input_id, target = [], []

        # New version, use apply chat template
        # Build system message for each sentence
        input_id += tokenizer.apply_chat_template([{"role" : "system", "content" : system_message}])
        target += [IGNORE_INDEX] * len(input_id)

        for conv in source:
            try:
                role = conv["role"]
                content = conv["content"]
            except:
                role = conv["from"]
                content = conv["value"]

            role =  roles.get(role, role)
            
            conv = [{"role" : role, "content" : content}]
            encode_id = tokenizer.apply_chat_template(conv)
            input_id += encode_id
            if role in ["user", "system"]:
                target += [IGNORE_INDEX] * len(encode_id)
            else:
                target += encode_id
        

                    
        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"

        input_ids.append(input_id)
        targets.append(target)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    return dict(
        input_ids=input_ids,  # tensor(bs x seq_len)
        labels=targets,  # tensor(bs x seq_len)
    )

def get_closest_ratio(height: float, width: float, ratio: List[float]):
    aspect_ratio = height / width
    closest_ratio = "1.0" #min(ratios.keys(), key=lambda ratio: abs(float(ratio) - aspect_ratio))
    return ratio, float(closest_ratio)


def _load_single_tar(tar_file: str):
    """
    Load and process a single tar file. Runs in an independent process.
    """
    try:
        # Load within a single process, num_proc=1 to prevent conflicts with the outer process pool
        ds = load_dataset("webdataset", data_files=[tar_file], split="train", num_proc=1)

        if "jpg" in ds.column_names:
            key_name = "jpg"
        elif "png" in ds.column_names:
            key_name = "png"
        elif "image" in ds.column_names:
            key_name = None  # Already named image, no need to rename
        else:
            print(f"ERROR: No image column found in {tar_file}! Available columns: {ds.column_names}")
            return None
        # Unify image column name
        if key_name:
            ds = ds.rename_column(key_name, "image")

        # Remove unnecessary columns at the tar level to save memory and avoid schema conflicts during concatenate
        cols_to_keep = {"image", "txt", "id"}
        cols_to_remove = [col for col in ds.column_names if col not in cols_to_keep]
        ds = ds.remove_columns(cols_to_remove)

        # Add metadata columns
        ds = ds.add_column("type", len(ds) * ["T2I"])
        ds = ds.add_column("image_path", len(ds) * [None])
        return ds
    except Exception as e:
        print(f"ERROR: Read tar {tar_file} failed! Error: {e}")
        return None


def _load_webdatasets(imge_folders_str: str, max_workers=8):
    from pathlib import Path
    from multiprocessing import Pool

    image_folders = [Path(folder.strip()) for folder in imge_folders_str.split(",")]

    all_tar_files = []

    # Step 1: Traverse all root directories and recursively collect all tar files using rglob
    print(f"\n{'=' * 60}")
    print("Collecting all tar files recursively...")

    for folder in image_folders:
        if not folder.exists() or not folder.is_dir():
            print(f"Warning: Directory not found or is not a dir: {folder}")
            continue

        tars = [str(p) for p in folder.rglob("*.tar") if p.is_file()]
        if tars:
            all_tar_files.extend(tars)
            print(f"  → Found {len(tars)} tar files in {folder}")
        else:
            print(f"  → Warning: No tar files found in {folder}")

    if not all_tar_files:
        raise ValueError("No tar files found in any of the provided folders!")

    print(f"{'=' * 60}\nTotal {len(all_tar_files)} tar files to load with {max_workers} workers.")

    valid_datasets = []

    # Step 2: Use a process pool to uniformly load all collected tar files in parallel
    with Pool(processes=max_workers) as pool:
        # imap_unordered is the fastest for stateless concatenation
        for i, ds in enumerate(pool.imap_unordered(_load_single_tar, all_tar_files), 1):
            if ds is not None:
                valid_datasets.append(ds)

            # Print progress in real-time
            if i % 10 == 0 or i == len(all_tar_files):
                print(f"  → Processed {i}/{len(all_tar_files)} files")
    if not valid_datasets:
        raise ValueError("All tar files failed to load!")

    # Step 3: Perform one final Concatenate
    print(f"\n{'=' * 60}")
    print(f"Final Concatenation of {len(valid_datasets)} datasets...")

    # Concatenate all datasets
    if len(valid_datasets) > 1:
        print(f"\n{'=' * 60}")
        print(f"Concatenating {len(valid_datasets)} datasets...")
        print(f"{'=' * 60}")

        # Verify all datasets have the same columns before concatenating
        first_columns = set(valid_datasets[0].column_names)
        for i, ds in enumerate(valid_datasets):
            print(f"  Dataset {i + 1}: {len(ds)} samples, columns: {ds.column_names}")
            if set(ds.column_names) != first_columns:
                print("ERROR: Column mismatch detected!")
                print(f"  Expected: {first_columns}")
                print(f"  Got: {set(ds.column_names)}")
                raise ValueError(
                    f"Dataset {i + 1} has different columns than dataset 0!"
                )

        dataset = concatenate_datasets(valid_datasets)
        print(f"Successfully concatenated: {len(dataset)} total samples")
    else:
        dataset = valid_datasets[0]
        print(f"Single dataset loaded: {len(dataset)} samples")

    return dataset


class LazySupervisedMixDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        tokenizer: transformers.PreTrainedTokenizer,
        data_args: DataArguments,
    ):
        super(LazySupervisedMixDataset, self).__init__()

        self.data_args = data_args
        list_data_dict = []

        ###################################### text to image #######################################

        list_data_dict = _load_webdatasets(self.data_args.image_folder, max_workers=8)

        # Shuffle after concatenation
        list_data_dict = list_data_dict.shuffle(seed=42)

        rank0_print(f"Total number of training instances: {len(list_data_dict)}")
        self.tokenizer = tokenizer
        self.tokenizer.model_max_length = self.data_args.tokenizer_max_length
        self.list_data_dict = list_data_dict
    
    def __init__SANA(
        self,
        data_path: str,
        tokenizer: transformers.PreTrainedTokenizer,
        data_args: DataArguments,
    ):
        super(LazySupervisedMixDataset, self).__init__()

        self.data_args = data_args
        list_data_dict = []

        ###################################### text to image ####################################### 
        image_folders = self.data_args.image_folder.split(',')
        image_folders = [folder.strip() for folder in image_folders]
        data_files = []
        for folder in image_folders:
            if os.path.exists(folder):
                tar_files = glob.glob(os.path.join(folder, "*.tar"))
                data_files.extend(tar_files)
                print(f"Found {len(tar_files)} tar files in {folder}")
            else:
                print(f"Warning: Directory not found: {folder}")
        print(f"Total tar files to process: {len(data_files)}")
        
        train_dataset = load_dataset("webdataset", data_files=data_files, split="train", num_proc=32)
        if "jpg" in train_dataset.column_names:
            train_dataset = train_dataset.rename_column("jpg", "image")
        elif "png" in train_dataset.column_names:
            train_dataset = train_dataset.rename_column("png", "image")
        
        #train_dataset = train_dataset.rename_column("jpg", "image")
        train_dataset = train_dataset.add_column('type', len(train_dataset) * ['T2I'])
        train_dataset = train_dataset.add_column('image_path', len(train_dataset) * [None])
        train_dataset = train_dataset.remove_columns([col for col in train_dataset.column_names if not col in (
            ["image", "txt", "type", "image_path"])])
        print(f"finish loading image {len(train_dataset)}")
        list_data_dict.append(train_dataset)
            

        if len(list_data_dict) > 1:
            list_data_dict = concatenate_datasets(list_data_dict)
        else:
            list_data_dict = list_data_dict[0]
        list_data_dict = list_data_dict.shuffle(seed=42)

        rank0_print(f"Total number of training instance: {len(list_data_dict)}")
        self.tokenizer = tokenizer
        self.tokenizer.model_max_length = self.data_args.tokenizer_max_length
        self.list_data_dict = list_data_dict

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if "image" in sample else 0
            length_list.append(sum(len(conv["value"].split()) for conv in sample["conversations"]) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv["value"].split()) for conv in sample["conversations"])
            cur_len = cur_len if "image" in sample else -cur_len
            length_list.append(cur_len)
        return length_list
    
    def _safe_img_process(self, imgs):
        try:
            out = []
            for img in imgs:
                ori_h, ori_w = img.height, img.width
                closest_size, closest_ratio = self.data_args.aspect_ratio_size, 1.0

                closest_size = [int(x) for x in closest_size]
                if closest_size[0] / ori_h > closest_size[1] / ori_w:
                    resize_size = closest_size[0], int(ori_w * closest_size[0] / ori_h)
                else:
                    resize_size = int(ori_h * closest_size[1] / ori_w), closest_size[1]
                transform = T.Compose([
                    T.Lambda(lambda img: img.convert("RGB")),
                    T.Resize(resize_size, interpolation=InterpolationMode.BICUBIC),  # Image.BICUBIC
                    T.CenterCrop(closest_size),
                    T.ToTensor(),
                    T.Normalize([0.5], [0.5]),
                    ])
                out.append(transform(img))
            return out
        except Exception as e:
            print(f"Corrupted image during processing: {e}")
            return None

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:

        while True:
            try:
                sources = self.list_data_dict[i]
                prompt = f"Please generate image based on the following caption: {sources['txt']}"
                sources["conversations"] = [
                    {"from": "human", "value": prompt},
                    {"from": "gpt", "value": "<image>"},
                ]
                prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=True)
                if len(prompt_tokens) > 512:
                    print(f"[WARN] Skipping sample {i}: prompt has {len(prompt_tokens)} tokens (>512)")
                    i = random.randint(0, len(self.list_data_dict) - 1)
                    continue
                image_files = self.list_data_dict[i]["image"]
                if not isinstance(image_files, list):
                    image_files = [image_files]

                is_corrupt = False
                images = []
                for img in image_files:
                    img = img.convert("RGB")
                    images.append(img)      
        
                processed_images = self._safe_img_process(images)
                if processed_images is None:
                    print("Corrupted image during transform, picking new sample.")
                    i = random.randint(0, len(self.list_data_dict) - 1)
                    continue 
                # just replace <image> with "" in generation tasks
                sources, inst_type = preprocess_multimodal(copy.deepcopy([sources["conversations"]]), self.data_args)
                data_dict = preprocess_qwen(sources, self.tokenizer, has_image=("image" in self.list_data_dict[i]))
                if isinstance(i, int):
                    data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])

                data_dict["gen_image"] = processed_images[0]
                data_dict["ids"] = self.list_data_dict[i]["id"] if "id" in self.list_data_dict[i] else "unk"
                return data_dict
            except Exception as e:
                print(f"[WARN] Skipping corrupted sample {i}: {e}")
                i = random.randint(0, len(self.list_data_dict) - 1)
                continue

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, ids = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels", "ids"))
        multi_input_ids = []
        multi_labels = []
        i_s_pos = []
        for input_id, label in zip(input_ids, labels):
            input_id = input_id[: self.tokenizer.model_max_length - 17]
            label = label[: self.tokenizer.model_max_length - 17]
            i_s_pos.append(input_id.shape[0]+1)
            img_id = torch.full((17,), IMAGE_TOKEN_INDEX, dtype=input_id.dtype, device=input_id.device)
            img_id[0] = DEFAULT_IM_START_TOKEN_IDX
            # input_id = torch.cat([input_id, img_id])
            img_label = torch.full((17,), IMAGE_TOKEN_INDEX, dtype=label.dtype, device=label.device)
            img_label[0] = DEFAULT_IM_START_TOKEN_IDX
            # label = torch.cat([label, img_label])
            multi_input_ids.append(input_id)
            multi_labels.append(label)

        input_ids = multi_input_ids
        labels = multi_labels

        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        if input_ids.shape[1] > self.tokenizer.model_max_length:
            print(f"Warning input with length {input_ids.shape[1]} is longer than max length {self.tokenizer.model_max_length}")
        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        labels = labels[:, : self.tokenizer.model_max_length]
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )

        batch_gen_images = []
        batch_und_images = []
        batch_grid_thw = []

        for instance in instances:
            if "gen_image" in instance:
                batch_gen_images.append(instance["gen_image"])

        if len(batch_gen_images) > 0:
            if all(x is not None and y.shape == batch_gen_images[0][0].shape for x in batch_gen_images for y in x):
                batch["gen_image"] = torch.cat([images.unsqueeze(0) for images in batch_gen_images], dim=0)
            else:
                batch["gen_image"] = batch_gen_images
        else:
            batch["gen_image"] = None


        for instance in instances:
            if "und_image" in instance:
                batch_und_images.append(instance["und_image"].unsqueeze(0))  ## 1*1024*1176
                batch_grid_thw.append(instance["grid_thw"])  ## 1*3


        # print(f"batch_und_images {batch_und_images}")
        if len(batch_und_images) > 0:
            batch["und_image"] = torch.cat([images for images in batch_und_images], dim=0)
            batch["grid_thw"] = torch.cat([images for images in batch_grid_thw], dim=0)
        else:
            batch["und_image"] = None
            batch["grid_thw"] = None

        batch["ids"] = ids
        batch["i_s_pos"] = i_s_pos
        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    train_dataset = LazySupervisedMixDataset(tokenizer=tokenizer, data_path=data_args.data_path, data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)

def train(attn_implementation=None):
    global local_rank

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.lr_scheduler_kwargs['min_lr'] = training_args.min_lr
    
    print(model_args, data_args, training_args)
    local_rank = training_args.local_rank
    compute_dtype = torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32)

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig

        bnb_model_from_pretrained_args.update(
            dict(
                device_map={"": training_args.device},
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=training_args.bits == 4,
                    load_in_8bit=training_args.bits == 8,
                    llm_int8_skip_modules=["mm_projector"],
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=training_args.double_quant,
                    bnb_4bit_quant_type=training_args.quant_type,  # {'fp4', 'nf4'}
                ),
            )
        )
        
    model = mobileoFastSFTForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
        **bnb_model_from_pretrained_args,
    )
   
    
    model.config.use_cache = False
    
    

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    
    try:
        tokenizer = AutoProcessor.from_pretrained(model_args.model_name_or_path).tokenizer
    except Exception as e:
        tokenizer = AutoProcessor.from_pretrained(model_args.model_name_or_path)
        
    tokenizer.model_max_length = training_args.model_max_length

    # tokenizer.pad_token = tokenizer.unk_token
    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(
                pad_token="<pad>",
                additional_special_tokens=["[IMG]", "[/IMG]", "<image>"],
            ),
            tokenizer=tokenizer,
            model=model,
        )
    elif not "<image>" in tokenizer.get_added_vocab():
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(additional_special_tokens=["[IMG]", "[/IMG]", "<image>"]),
            tokenizer=tokenizer,
            model=model,
        )
    if model_args.version in conversation_lib.conv_templates:
        conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
    else:
        conversation_lib.default_conversation = conversation_lib.conv_templates["llama3"]
    rank0_print(f"Using conversation format: {conversation_lib.default_conversation.version}")

    for p in model.get_model().parameters():
        p.requires_grad = False
        
    for p in model.lm_head.parameters():
        p.requires_grad = False
            
    for p in model.get_model().embed_tokens.parameters():
        p.requires_grad = False

    for p in model.get_model().get_vision_tower().parameters():
        p.requires_grad = False

    
    model.get_model().initialize_vision_modules(model_args=model_args, fsdp=training_args.fsdp)
    image_processor = model.get_model().get_vision_tower().image_processor
    data_args.gen_image_processor = image_processor
    data_args.image_processor = image_processor

    data_args.is_multimodal = True

    model.config.image_aspect_ratio = data_args.image_aspect_ratio
    model.config.tokenizer_padding_side = tokenizer.padding_side
    model.config.tokenizer_model_max_length = tokenizer.model_max_length
    model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
    model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
    
    

    # Calculate total parameters and trainable parameters
    total_params = sum(p.numel() for p in model.get_model().parameters())
    trainable_params = sum(p.numel() for p in model.get_model().parameters() if p.requires_grad)
    
    

    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")

    model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
    model.config.mm_projector_lr = training_args.mm_projector_lr
    training_args.use_im_start_end = model_args.mm_use_im_start_end
    model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
    model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)
    model.config.pad_token_id = tokenizer.pad_token_id

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    trainer = mobileoTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
		callbacks=[MobileConditioningCallback(total_epochs=int(training_args.num_train_epochs))],
        **data_module,
    )
    from tabulate import tabulate

    if trainer.is_world_process_zero():
        stat = []
        for i, (n, p) in enumerate(trainer.model.named_parameters()):
            stat.append([i, n, p.shape, p.requires_grad])
        print(tabulate(stat, headers=["idx", "name", "shape", "trainable"]))
    

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True
    safe_save_model_for_hf_trainer(
        trainer=trainer,
        output_dir=training_args.output_dir,
        vision_tower=model_args.vision_tower,
    )


if __name__ == "__main__":
    train()
