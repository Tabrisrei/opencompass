from mmengine.config import read_base

with read_base():

    # Import models
    # Import datasets
    from opencompass.configs.datasets.MathBench.mathbench_gen import \
        mathbench_datasets
    from opencompass.configs.models.hf_internlm.hf_internlm2_chat_7b import \
        models as internlm2_chat_7b_model
    from opencompass.configs.models.hf_llama.hf_llama3_8b_instruct import \
        models as llama3_8b_instruct_model
    # Import summarizers for display results
    from opencompass.configs.summarizers.groups.mathbench_v1_2024 import \
        summarizer  # Grouped results for MathBench-A and MathBench-T separately

    # from opencompass.configs.summarizers.mathbench_v1 import summarizer # Detailed results for every sub-dataset
    # from opencompass.configs.summarizers.groups.mathbench_v1_2024_lang import summarizer # Grouped results for bilingual results

datasets = sum([v for k, v in locals().items() if k.endswith('_datasets')], [])
models = sum([v for k, v in locals().items() if k.endswith('_model')], [])

from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLEvalTask, OpenICLInferTask

eval = dict(
    partitioner=dict(type=NaivePartitioner, n=8),
    runner=dict(type=LocalRunner,
                max_num_workers=256,
                task=dict(type=OpenICLEvalTask)),
)

infer = dict(
    partitioner=dict(type=NumWorkerPartitioner, num_worker=4),
    runner=dict(type=LocalRunner,
                max_num_workers=256,
                task=dict(type=OpenICLInferTask)),
)

work_dir = './outputs/mathbench_results'
