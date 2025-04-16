from mmengine.config import read_base
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever, FixKRetriever
from opencompass.openicl.icl_inferencer import GenInferencer, PPLInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator, AccwithDetailsEvaluator
from opencompass.datasets import MMLUProDataset
from opencompass.utils.text_postprocessors import match_answer_pattern

categories = [
    'math',
    'physics',
    'chemistry',
    'law',
    'engineering',
    'other',
    'economics',
    'health',
    'psychology',
    'business',
    'biology',
    'philosophy',
    'computer science',
    'history',
]

QUERY_TEMPLATE = """
Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of Options(e.g. one of ABCDEFGHIJKLMNOP). Think step by step before answering.

Question:\n
{question}

Options:\n
{options_str}

""".strip()

mmlu_pro_datasets = []

for category in categories:
    mmlu_pro_reader_cfg = dict(
        input_columns=['question', 'cot_content', 'options_str'],
        output_column='answer',
        train_split='validation',
        test_split='test',
    )
    mmlu_pro_infer_cfg = dict(
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(
                round=[
                    dict(role='HUMAN',
                         prompt=QUERY_TEMPLATE),
                ],
            ),
        ),
        retriever=dict(type=ZeroRetriever),
        # inferencer=dict(type=PPLInferencer),
        inferencer=dict(type=GenInferencer,)
    )

    mmlu_pro_eval_cfg = dict(
        evaluator=dict(type=AccwithDetailsEvaluator),
        pred_postprocessor=dict(
            type=match_answer_pattern,
            answer_pattern=r'(?i)ANSWER\s*:\s*([A-P])')
    )

    mmlu_pro_datasets.append(
        dict(
            abbr=f'mmlu_pro_{category.replace(" ", "_")}',
            type=MMLUProDataset,
            path='opencompass/mmlu_pro',
            category=category,
            reader_cfg=mmlu_pro_reader_cfg,
            infer_cfg=mmlu_pro_infer_cfg,
            eval_cfg=mmlu_pro_eval_cfg,
        ))

datasets = [*mmlu_pro_datasets]

from opencompass.models import HuggingFaceCausalLM, HuggingFaceBaseModel

models = [
    dict(
        type=HuggingFaceCausalLM,
        # type=HuggingFaceBaseModel,
        abbr='Llama3_8b', # 运行完结果展示的名称
        path='base_models/meta-llama/Meta-Llama-3-8B', # 模型路径
        tokenizer_path='base_models/meta-llama/Meta-Llama-3-8B', # 分词器路径
        model_kwargs=dict(
                        cache_dir='base_models/meta-llama/Meta-Llama-3-8B',
                        device_map='auto', 
                        trust_remote_code=True, 
                        # pretrained_model_name_or_path determined by path
                        # config: PretrainedConfig class
                        # state_dict: (optional) dict: an optional state dictionary for the model to use instead of a state dictionary loaded from saved weights file.
                        # force_download=False
                        # resume_download=True
                        # proxie={‘http’: ‘foo.bar:3128’, ‘http://hostname’: ‘foo.bar:4012’}
                        ), # kwargs for model loading from_pretrained
        tokenizer_kwargs=dict(
                        padding_side='left', 
                        truncation_side='left', 
                        trust_remote_code=True, 
                        use_fast=False), # kwargs for tokenizer loading from_pretrained
        generation_kwargs={"eos_token_id": [128001, 128009]},
        batch_padding=True,
        max_out_len=100,
        max_seq_len=2048,
        batch_size=16,
        run_cfg=dict(num_gpus=1),
    )
]