"""
Image Captioning Demo using Ray Data, Ray Data LLM, and vLLM.

- Load dataset
- Dataset repartitioning
- Preprocess dataset in parallel
- Postprocess dataset in parallel

GPU-accelerated inference:
- Create LLM processor pipeline
- Run inference on the LLM processor pipeline
"""

from __future__ import annotations

import os
import gc
import time
import logging

from typing import Any
from pathlib import Path
from io import BytesIO

import datasets
import ray
from PIL import Image
from ray.data.llm import build_llm_processor, vLLMEngineProcessorConfig

from dataclasses import dataclass, field


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


@dataclass
class JobConfig:
    """Configuration for image captioning job."""

    # Dataset configuration
    dataset_name: str = "imagenet-1k"
    dataset_split: str = "train[:10000]"

    # Processing configuration
    num_partitions: int = 96
    preprocessing_concurrency: int = 8
    preprocessing_num_cpus: int = 4
    postprocessing_concurrency: int = 8
    postprocessing_num_cpus: int = 4

    # Model configuration
    model_name: str = "Qwen/Qwen2-VL-2B-Instruct"
    max_model_len: int = 4096
    max_pixels: int = 1280 * 28 * 28
    min_pixels: int = 256 * 28 * 28

    # Inference configuration
    batch_size: int = 4
    num_inference_engines: int = 2
    tensor_parallel_size: int = 1
    accelerator_type: str = "A10G"

    # Generation parameters
    temperature: float = 0.7
    max_tokens: int = 200

    # Output configuration
    output_dir: Path = field(default_factory=lambda: Path("./outputs"))
    output_filename: str = "02_captioned_dataset.parquet"

    def __post_init__(self):
        """Ensure output directory exists."""
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def output_path(self) -> Path:
        """Full path to output file."""
        return self.output_dir / self.output_filename


class ImageCaptioningPipeline:
    """Production-ready image captioning pipeline using Ray Data and vLLM."""

    def __init__(self, config: JobConfig):
        """Initialize the pipeline with configuration."""
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    def caption_preprocess(self, row: dict[str, Any]) -> dict[str, Any]:
        """
        Prepare image and prompt for captioning.

        Args:
            row: Input row containing image data
                Example structure:
                {
                    "image": {
                        "bytes": bytes,
                        "path": str,
                    },
                    "label": int,
                }

        Returns:
            Preprocessed row with messages and preserved image data
        """

        return {
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert image captioner. Provide detailed, "
                    "accurate descriptions of images.",
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Please provide a detailed caption for this image. "
                            "Describe what you see, including objects, people, "
                            "actions, and the overall scene.",
                        },
                        {
                            "type": "image",
                            "image": Image.open(BytesIO(row["image"]["bytes"])),
                        },
                    ],
                },
            ],
            "sampling_params": {
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                "detokenize": True,
            },
            # Preserve data for postprocessing
            "image_bytes": row["image"]["bytes"],
            "image_path": row["image"]["path"],
        }

    def caption_postprocess(self, row: dict[str, Any]) -> dict[str, Any]:
        """
        Extract caption from model output and preserve image data.

        Args:
            row: Row containing generated text and preserved data

        Returns:
            Formatted row with caption and image data
        """
        # Reconstruct image dict for visualization
        image_dict = {
            "bytes": row["image_bytes"],
            "path": row["image_path"],
        }

        return {
            "caption": row["generated_text"],
            "image": image_dict,
        }

    def load_dataset(self) -> ray.data.Dataset:
        """Load and prepare the dataset."""
        self.logger.info(
            f"Loading dataset: {self.config.dataset_name} [{self.config.dataset_split}]"
        )

        hf_dataset = datasets.load_dataset(
            self.config.dataset_name, split=self.config.dataset_split
        )
        ray_dataset = ray.data.from_huggingface(hf_dataset)  # type: ignore[arg-type]

        self.logger.info(f"Dataset loaded: {ray_dataset.count()} samples")
        return ray_dataset

    def partition_dataset(self, dataset: ray.data.Dataset) -> ray.data.Dataset:
        """Repartition the dataset into blocks."""
        self.logger.info(
            f"Repartitioning dataset into {self.config.num_partitions} blocks"
        )
        dataset = dataset.repartition(self.config.num_partitions)

        return dataset

    def preprocess_dataset(self, dataset: ray.data.Dataset) -> ray.data.Dataset:
        """Apply preprocessing with parallelization."""
        self.logger.info("Starting parallel preprocessing...")
        preprocessed = dataset.map(
            self.caption_preprocess,
            num_cpus=self.config.preprocessing_num_cpus,
            concurrency=self.config.preprocessing_concurrency,
        )

        return preprocessed

    def create_llm_processor(self):
        """Create and configure the vLLM processor."""
        processor_config = vLLMEngineProcessorConfig(
            model_source=self.config.model_name,
            engine_kwargs={
                "tensor_parallel_size": self.config.tensor_parallel_size,
                "max_model_len": self.config.max_model_len,
                "enable_chunked_prefill": True,
                "limit_mm_per_prompt": {"image": 1},
                "mm_processor_kwargs": {
                    "max_pixels": self.config.max_pixels,
                    "min_pixels": self.config.min_pixels,
                },
                "distributed_executor_backend": "ray",
            },
            runtime_env={
                "env_vars": {
                    "HF_TOKEN": os.environ["HF_TOKEN"],
                    "HF_HOME": os.environ["HF_HOME"],
                    "VLLM_USE_V1": "1",
                    "RAY_DEDUP_LOGS": "0",  # Ray deduplicates logs by default. Set to 0 to disable log deduplication
                    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                },
            },
            batch_size=self.config.batch_size,
            accelerator_type=self.config.accelerator_type,
            concurrency=self.config.num_inference_engines,
            has_image=True,
        )

        return build_llm_processor(
            processor_config,   # type: ignore[arg-type]
            preprocess=None,
            postprocess=None,
        )  

    def postprocess_dataset(self, dataset: ray.data.Dataset) -> ray.data.Dataset:
        """Apply postprocessing with parallelization."""
        self.logger.info("Starting parallel postprocessing...")
        postprocessed = dataset.map(
            self.caption_postprocess,
            num_cpus=self.config.postprocessing_num_cpus,
            concurrency=self.config.postprocessing_concurrency,
        )

        return postprocessed

    def run(self) -> dict[str, Any]:
        """Execute the complete pipeline."""
        start_time = time.time()

        # Initialize Ray
        self.logger.info("Initializing Ray...")
        ray.init(
            runtime_env={
                "env_vars": {
                    "HF_TOKEN": os.environ["HF_TOKEN"],
                    "HF_HOME": os.environ["HF_HOME"],
                    "RAY_DEDUP_LOGS": "0",  # Ray deduplicates logs by default. Set to 0 to disable log deduplication
                    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                }
            }
        )

        try:
            # Step 1: Load dataset
            dataset = self.load_dataset()

            # Step 2: Repartition dataset
            dataset = self.partition_dataset(dataset)

            # Step 3: Preprocess dataset
            preprocessed = self.preprocess_dataset(dataset)

            # Step 4: Create LLM processor
            llm_processor = self.create_llm_processor()

            # Step 5: Run inference
            self.logger.info("Starting inference...")
            inference_start = time.time()
            captioned = llm_processor(preprocessed)
            inference_time = time.time() - inference_start

            # Step 6: Postprocess
            postprocessed = self.postprocess_dataset(captioned)

            # Step 7: Save results
            self.logger.info(f"Saving results to: {self.config.output_path}")

            postprocessed.write_parquet(str(self.config.output_path))

            # Step 8: Print samples
            print("\n" + "=" * 70)
            print("SAMPLE CAPTIONS")
            print("=" * 70)

            for i, sample in enumerate(postprocessed.take(3)):
                print(f"\nSample {i+1}:")
                print(f"  Image Path: {sample["image"]["path"]}")
                print(f"  Caption: {sample["caption"]}")
                print("-" * 70)

            # Step 9: Calculate metrics
            total_time = time.time() - start_time
            total_samples = dataset.count()

            self.logger.info("Pipeline completed successfully!")

            return {
                "status": "success",
                "total_samples": total_samples,
                "total_time": total_time,
                "inference_time": inference_time,
                "throughput": total_samples / inference_time,
                "output_path": str(self.config.output_path),
            }

        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            raise
        finally:
            # Cleanup
            ray.shutdown()
            gc.collect()


def main():
    """Main entry point."""
    # Create configuration
    config = JobConfig(
        dataset_split="train[:1000]",  # Small sample for demo
        num_partitions=32,  # Adjust based on your resources
        num_inference_engines=2,  # Number of llm engines to run in parallel
        tensor_parallel_size=1,  # Number of GPUs per llm engine
        batch_size = 8,
    )

    # Create and run pipeline
    pipeline = ImageCaptioningPipeline(config)
    results = pipeline.run()

    # Print results
    print("\n" + "=" * 70)
    print("PIPELINE RESULTS")
    print("=" * 70)
    print(f"Total samples: {results['total_samples']}")
    print(f"Total time: {results['total_time']:.2f} seconds")
    print(f"Inference time: {results['inference_time']:.2f} seconds")
    print(f"Throughput: {results['throughput']:.2f} samples/second")
    print(f"Output saved to: {results['output_path']}")


if __name__ == "__main__":
    # Ensure HF token is set
    if "HF_TOKEN" not in os.environ:
        raise ValueError("HF_TOKEN is not set")
    if "HF_HOME" not in os.environ:
        os.environ["HF_HOME"] = "/mnt/data/hf_cache"

    main()
