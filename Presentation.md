
Understanding blocks and parallelism:
# Example: 1000 images dataset
dataset = ray.data.from_huggingface(...)  # 1000 rows

# Option 1: Default partitioning (auto)
# Ray might create ~8 blocks of ~125 rows each

# Option 2: Manual partitioning
dataset = dataset.repartition(num_blocks=10)
# Creates exactly 10 blocks of 100 rows each

# During processing:
# - 10 workers can process blocks in parallel
# - Each worker calls caption_preprocess 100 times (once per row)
# - Workers operate independently


Visualization of the process:
```bash
Dataset (1000 images)
    ↓ repartition(10)
[Block0: 100 rows] [Block1: 100 rows] ... [Block9: 100 rows]
    ↓ map(caption_preprocess) - parallel execution
[Worker0]         [Worker1]         ... [Worker9]
 ↓ ↓ ↓            ↓ ↓ ↓                ↓ ↓ ↓
row→preprocess   row→preprocess      row→preprocess
row→preprocess   row→preprocess      row→preprocess
...              ...                  ...
    ↓ Results collected and batched for vLLM
[Preprocessed Block0] [Preprocessed Block1] ... [Preprocessed Block9]
    ↓ vLLM processor batches these for inference
```

Best practices for partitioning:
# 1. For CPU-bound preprocessing (like image resizing)
num_blocks = num_cpus * 2  # Oversubscribe for better utilization
dataset = dataset.repartition(num_blocks)

# 2. For I/O-bound preprocessing (like downloading)
num_blocks = num_cpus * 4  # More blocks to hide I/O latency
dataset = dataset.repartition(num_blocks)

# 3. For memory-intensive preprocessing
num_blocks = available_memory / max_block_size
dataset = dataset.repartition(num_blocks)

# 4. Let Ray decide (often good default)
# Don't repartition - Ray will use heuristics

The key insight: Ray Data handles the parallelization automatically. Your caption_preprocess function just needs to handle a single row, and Ray will efficiently distribute the work across available resources!

sample_batch["image"][0].keys()
dict_keys(['bytes', 'path'])

Now your pipeline should work smoothly with the ImageNet dataset! The flow is:
Load dataset → images stored as {'bytes': ..., 'path': ...}
Preprocess → convert bytes to PIL Image
vLLM → generates captions
Postprocess → extract and format results




export HF_TOKEN=
export HF_HOME=/mnt/data/hf_cache

uv venv venv --python 3.12
source venv/bin/activate
uv pip install pip

python3.12 -m venv venv
source venv/bin/activate

uv pip install torch torchvision torchaudio torchmetrics --index-url https://download.pytorch.org/whl/cu128


>>> 
>>> ray_dataset.count()
1281167

Adding prometheus and grafana
https://docs.ray.io/en/latest/cluster/metrics.html#grafana

for prometheus
ray metrics launch-prometheus
ray metrics shutdown-prometheus

for grafana
./bin/grafana-server --config /tmp/ray/session_latest/metrics/grafana/grafana.ini web

vLLM issue with inputs
https://docs.vllm.ai/en/latest/features/multimodal_inputs.html#stable-uuids-for-caching-multi_modal_uuids
https://docs.vllm.ai/en/latest/features/multimodal_inputs.html#audio-inputs_1



```bash
01
======================================================================
PIPELINE RESULTS
======================================================================
Total samples: 1000
Total time: 342.32 seconds
Inference time: 0.01 seconds
Throughput: 146234.71 samples/second
Output saved to: outputs/01_captioned_dataset.parquet
(ray-summit-2025) (git: main) asaha@sun-devt-1427 ~/ray-summit-2025 $ 


>>> ray_dataset.num_blocks()
1



02
======================================================================
PIPELINE RESULTS
======================================================================
Total samples: 1000
Total time: 302.34 seconds
Inference time: 0.01 seconds
Throughput: 159776.92 samples/second
Output saved to: outputs/02_captioned_dataset.parquet
```


## Concerns with `qwen_vl_utils`

```bash
python 04_image_caption_demo.py
```

Main error is : 

```bash
image_grid_thw[image_index][0]
IndexError: list index out of range
```

Full error log:
```bash
File "/home/asaha/ray-summit-2025/.venv/lib/python3.12/site-packages/vllm/executor/ray_utils.py", line 135, in execute_model_ray
[36m(MapWorker(MapBatches(vLLMEngineStageUDF)) pid=674640)[0m ERROR 10-19 12:18:50 [core.py:588]     output = self.worker.model_runner.execute_model(
[36m(MapWorker(MapBatches(vLLMEngineStageUDF)) pid=674640)[0m ERROR 10-19 12:18:50 [core.py:588]              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[36m(MapWorker(MapBatches(vLLMEngineStageUDF)) pid=674640)[0m ERROR 10-19 12:18:50 [core.py:588]   File "/home/asaha/ray-summit-2025/.venv/lib/python3.12/site-packages/torch/utils/_contextlib.py", line 116, in decorate_context
[36m(MapWorker(MapBatches(vLLMEngineStageUDF)) pid=674640)[0m ERROR 10-19 12:18:50 [core.py:588]     return func(*args, **kwargs)
[36m(MapWorker(MapBatches(vLLMEngineStageUDF)) pid=674640)[0m ERROR 10-19 12:18:50 [core.py:588]            ^^^^^^^^^^^^^^^^^^^^^
[36m(MapWorker(MapBatches(vLLMEngineStageUDF)) pid=674640)[0m ERROR 10-19 12:18:50 [core.py:588]   File "/home/asaha/ray-summit-2025/.venv/lib/python3.12/site-packages/vllm/v1/worker/gpu_model_runner.py", line 1283, in execute_model
[36m(MapWorker(MapBatches(vLLMEngineStageUDF)) pid=674640)[0m ERROR 10-19 12:18:50 [core.py:588]     self._update_states(scheduler_output)
[36m(MapWorker(MapBatches(vLLMEngineStageUDF)) pid=674640)[0m ERROR 10-19 12:18:50 [core.py:588]   File "/home/asaha/ray-summit-2025/.venv/lib/python3.12/site-packages/vllm/v1/worker/gpu_model_runner.py", line 456, in _update_states
[36m(MapWorker(MapBatches(vLLMEngineStageUDF)) pid=674640)[0m ERROR 10-19 12:18:50 [core.py:588]     MRotaryEmbedding.get_input_positions_tensor(
[36m(MapWorker(MapBatches(vLLMEngineStageUDF)) pid=674640)[0m ERROR 10-19 12:18:50 [core.py:588]   File "/home/asaha/ray-summit-2025/.venv/lib/python3.12/site-packages/vllm/model_executor/layers/rotary_embedding.py", line 1167, in get_input_positions_tensor
[36m(MapWorker(MapBatches(vLLMEngineStageUDF)) pid=674640)[0m ERROR 10-19 12:18:50 [core.py:588]     return cls._vl_get_input_positions_tensor(
[36m(MapWorker(MapBatches(vLLMEngineStageUDF)) pid=674640)[0m ERROR 10-19 12:18:50 [core.py:588]            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
[36m(MapWorker(MapBatches(vLLMEngineStageUDF)) pid=674640)[0m ERROR 10-19 12:18:50 [core.py:588]   File "/home/asaha/ray-summit-2025/.venv/lib/python3.12/site-packages/vllm/model_executor/layers/rotary_embedding.py", line 1330, in _vl_get_input_positions_tensor
[36m(MapWorker(MapBatches(vLLMEngineStageUDF)) pid=674640)[0m ERROR 10-19 12:18:50 [core.py:588]     image_grid_thw[image_index][0],
[36m(MapWorker(MapBatches(vLLMEngineStageUDF)) pid=674640)[0m ERROR 10-19 12:18:50 [core.py:588]     ~~~~~~~~~~~~~~^^^^^^^^^^^^^
[36m(MapWorker(MapBatches(vLLMEngineStageUDF)) pid=674640)[0m ERROR 10-19 12:18:50 [core.py:588] IndexError: list index out of range
[36m(MapWorker(MapBatches(vLLMEngineStageUDF)) pid=674640)[0m INFO 10-19 12:18:50 [ray_distributed_executor.py:128] Shutting down Ray distributed executor. If you see error log from logging.cc regarding SIGTERM received, please ignore because this is the expected termination process in Ray.
[36m(MapWorker(MapBatches(vLLMEngineStageUDF)) pid=674640)[0m ERROR 10-19 12:18:50 [async_llm.py:419] AsyncLLM output_handler failed.
```