from mmengine.config import read_base
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever, FixKRetriever
from opencompass.openicl.icl_inferencer import GenInferencer, PPLInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator, AccwithDetailsEvaluator
from opencompass.datasets import MMLUDataset
from opencompass.utils.text_postprocessors import match_answer_pattern, first_option_postprocess

# None of the mmlu dataset in huggingface is correctly parsed, so we use our own dataset reader
# Please download the dataset from https://people.eecs.berkeley.edu/~hendrycks/data.tar

mmlu_all_sets = [
    'college_biology',
    # 'college_chemistry',
    # 'college_computer_science',
    # 'college_mathematics',
    # 'college_physics',
    # 'electrical_engineering',
    # 'astronomy',
    'anatomy',
    # 'abstract_algebra',
    # 'machine_learning',
    'clinical_knowledge',
    # 'global_facts',
    # 'management',
    # 'nutrition',
    # 'marketing',
    # 'professional_accounting',
    # 'high_school_geography',
    # 'international_law',
    # 'moral_scenarios',
    # 'computer_security',
    # 'high_school_microeconomics',
    # 'professional_law',
    'medical_genetics',
    # 'professional_psychology',
    # 'jurisprudence',
    # 'world_religions',
    # 'philosophy',
    # 'virology',
    # 'high_school_chemistry',
    # 'public_relations',
    # 'high_school_macroeconomics',
    # 'human_sexuality',
    # 'elementary_mathematics',
    # 'high_school_physics',
    # 'high_school_computer_science',
    # 'high_school_european_history',
    # 'business_ethics',
    # 'moral_disputes',
    # 'high_school_statistics',
    # 'miscellaneous',
    # 'formal_logic',
    # 'high_school_government_and_politics',
    # 'prehistory',
    # 'security_studies',
    # 'high_school_biology',
    # 'logical_fallacies',
    # 'high_school_world_history',
    'professional_medicine',
    # 'high_school_mathematics',
    'college_medicine',
    # 'high_school_us_history',
    # 'sociology',
    # 'econometrics',
    # 'high_school_psychology',
    # 'human_aging',
    # 'us_foreign_policy',
    # 'conceptual_physics',
]

QUERY_TEMPLATE_ALTER = """
{input}

A) {A}
B) {B}
C) {C}
D) {D}
Answer:
""".strip()

mmlu_reader_cfg = dict(
    input_columns=['input', 'A', 'B', 'C', 'D'],
    output_column='target',
    train_split='dev', 
    test_split='test',)

mmlu_datasets = []
for _name in mmlu_all_sets:
    _hint = f'The following are multiple choice questions (with answers) about {_name.replace("_", " ")}. \n\n'
    mmlu_infer_cfg = dict(

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

    mmlu_eval_cfg = dict(
        evaluator=dict(type=AccwithDetailsEvaluator),
        pred_postprocessor=dict(type=match_answer_pattern, answer_pattern=r'([A-D])'),
        )

    mmlu_datasets.append(
        dict(
            abbr=f'lukaemon_mmlu_{_name}',
            type=MMLUDataset,
            path='opencompass/mmlu',
            name=_name,
            reader_cfg=mmlu_reader_cfg,
            infer_cfg=mmlu_infer_cfg,
            eval_cfg=mmlu_eval_cfg,
        ))
del _name, _hint

datasets = [*mmlu_datasets]
# =============================================================================
from opencompass.models import HuggingFaceCausalLM, HuggingFaceBaseModel

models = [
    dict(
        type=HuggingFaceCausalLM,
        abbr='origin_Meta-Llama3-8B_mmlu_health_openai_simple_evals_gen', # 运行完结果展示的名称
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