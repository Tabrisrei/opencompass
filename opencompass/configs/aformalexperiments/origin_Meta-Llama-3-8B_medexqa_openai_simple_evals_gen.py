from mmengine.config import read_base
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever, FixKRetriever
from opencompass.openicl.icl_inferencer import GenInferencer, PPLInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator, AccwithDetailsEvaluator
from opencompass.datasets import MedExQADataset
from opencompass.utils.text_postprocessors import match_answer_pattern, first_option_postprocess

# None of the mmlu dataset in huggingface is correctly parsed, so we use our own dataset reader
# Please download the dataset from https://people.eecs.berkeley.edu/~hendrycks/data.tar

medexqa_all_sets = [
    'biomedical_engineer',
    'clinical_laboratory_scientist',
    'clinical_psychologist',
    'occupational_therapist',
    'speech_pathologist',
]

QUERY_TEMPLATE_ALTER = """
{input}

A) {A}
B) {B}
C) {C}
D) {D}
Answer:
""".strip()

medexqa_reader_cfg = dict(
    input_columns=['input', 'A', 'B', 'C', 'D'],
    output_column='target',
    train_split='dev', 
    test_split='test',)

medexqa_datasets = []
for _name in medexqa_all_sets:
    _hint = f'The following are multiple choice questions (with answers) about {_name.replace("_", " ")}. \n\n'
    medexqa_infer_cfg = dict(

        prompt_template=dict(
            type=PromptTemplate,
            template=dict(
                round=[
                    dict(
                        role='HUMAN', 
                        prompt=_hint + QUERY_TEMPLATE_ALTER,
                        ),
                ],
            ),
        ),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=GenInferencer),
    )

    medexqa_eval_cfg = dict(
        evaluator=dict(type=AccwithDetailsEvaluator),
        pred_postprocessor=dict(type=match_answer_pattern, answer_pattern=r'([A-D])'),
        )

    medexqa_datasets.append(
        dict(
            abbr=f'medexqa_{_name}',
            type=MedExQADataset,
            path='/home/gsb/opencompass/adatasets/meddata/bluesky333/MedExQA',
            name=_name,
            reader_cfg=medexqa_reader_cfg,
            infer_cfg=medexqa_infer_cfg,
            eval_cfg=medexqa_eval_cfg,
        ))
del _name, _hint

datasets = [*medexqa_datasets]
# =============================================================================
from opencompass.models import HuggingFaceCausalLM, HuggingFaceBaseModel

models = [
    dict(
        type=HuggingFaceCausalLM,
        abbr='origin_Meta-Llama3-8B_medexqa_health_openai_simple_evals_gen', # 运行完结果展示的名称
        path='abase_models/meta-llama/Meta-Llama-3-8B', # 模型路径
        tokenizer_path='abase_models/meta-llama/Meta-Llama-3-8B', # 分词器路径
        model_kwargs=dict(
                        cache_dir='base_models/meta-llama/Meta-Llama-3-8B',
                        device_map='auto', 
                        trust_remote_code=True, 
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