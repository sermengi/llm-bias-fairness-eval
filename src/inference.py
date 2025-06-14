import json
import os
from typing import Dict

import pandas as pd
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from src import logger
from src.context_generator import ContextGenerator
from src.data_loader import GSM_MC_PromptBuilder
from src.models import MultipleChoiceLLM

NUM_PROCS = 1


def _inference_worker(rank, configs):
    device = xm.xla_device()
    world_size = NUM_PROCS
    is_main_process = rank == 0
    logger.info(f"[Rank {rank}/{world_size}] Worker started on device: {device}")

    dataset_config = configs["dataset"]
    model_config = configs["model"]
    artifact_config = configs["artifact"]
    context_config = configs["context"]

    context_generator = ContextGenerator(context_config)
    contexts = context_generator.generate_contexts()
    context_generator.save_generated_contexts()

    dataset = GSM_MC_PromptBuilder(
        dataset_config.dataset_name,
        contexts=contexts,
        data_files=dataset_config.data_files,
        split=dataset_config.split,
        allowed_choices=dataset_config.allowed_choices,
        max_samples=dataset_config.max_samples,
    )

    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=False
    )

    dataloader = DataLoader(
        dataset,
        batch_size=model_config.batch_size,
        sampler=sampler,
        num_workers=1,
        drop_last=False,
    )

    parallel_loader = pl.ParallelLoader(dataloader, [device])

    model = MultipleChoiceLLM(
        model_name=model_config.model_name,
        allowed_choices=dataset_config.allowed_choices,
        tokenizer_padding_side=model_config.tokenizer_padding_side,
    )

    rank_results = []
    pbar = tqdm(
        parallel_loader.per_device_loader(device),
        desc=f"Inference Rank {rank}",
        total=len(parallel_loader.per_device_loader(device)),
        disable=not is_main_process,
    )

    for batch in pbar:
        prompts = batch["prompt"]
        preds = model.predict(prompts)

        for i in range(len(prompts)):
            rank_results.append(
                {
                    "prompt_id": batch["prompt_id"][i].item(),
                    "sample_id": batch["sample_id"][i].item(),
                    "question": batch["question"][i],
                    "choice_A": batch["choices"].get("A", "")[i].item(),
                    "choice_B": batch["choices"].get("B", "")[i].item(),
                    "choice_C": batch["choices"].get("C", "")[i].item(),
                    "choice_D": batch["choices"].get("D", "")[i].item(),
                    "prompt": batch["prompt"][i],
                    "context_category": batch["context_info"]["category"][i],
                    "context_identity": batch["context_info"]["identity"][i],
                    "answer": batch["answer"][i],
                    "prediction": preds[i],
                }
            )

    output_dir = artifact_config.artifacts_root
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"rank_{rank}_results.json")
    with open(output_path, "w") as f:
        json.dump(rank_results, f)

    logger.info(f"[Rank {rank}] Inference complete. Results saved to {output_path}")


class ModelInferencePipeline:
    def __init__(self, configs: Dict):
        self.configs = configs

    def run_inference(self):
        logger.info("Starting XLA multiprocessing inference pipeline.")
        xmp.spawn(
            _inference_worker,
            args=(self.configs,),
            nprocs=NUM_PROCS,
            start_method="fork",
        )
        logger.info("All inference workers have completed their tasks.")
        self._aggregate_results()

    def _aggregate_results(self):
        logger.info("Aggregating results from all worker processes...")
        artifact_config = self.configs["artifact"]
        output_dir = artifact_config.artifacts_root
        prediction_file_path = artifact_config.prediction_file_path

        all_results = []
        for filename in sorted(os.listdir(output_dir)):
            if filename.startswith("rank_") and filename.endswith("results.json"):
                filepath = os.path.join(output_dir, filename)
                with open(filepath, "r") as f:
                    rank_results = json.load(f)
                    all_results.extend(rank_results)
                os.remove(filepath)

        df = pd.DataFrame(all_results)
        df = df.sort_values(by="sample_id").reset_index(drop=True)
        df.to_csv(prediction_file_path, index=False)
        logger.info(f"Results saved to {prediction_file_path}")
