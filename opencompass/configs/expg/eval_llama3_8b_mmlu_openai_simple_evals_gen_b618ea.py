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
    'college_chemistry',
    'college_computer_science',
    'college_mathematics',
    'college_physics',
    'electrical_engineering',
    'astronomy',
    'anatomy',
    'abstract_algebra',
    'machine_learning',
    'clinical_knowledge',
    'global_facts',
    'management',
    'nutrition',
    'marketing',
    'professional_accounting',
    'high_school_geography',
    'international_law',
    'moral_scenarios',
    'computer_security',
    'high_school_microeconomics',
    'professional_law',
    'medical_genetics',
    'professional_psychology',
    'jurisprudence',
    'world_religions',
    'philosophy',
    'virology',
    'high_school_chemistry',
    'public_relations',
    'high_school_macroeconomics',
    'human_sexuality',
    'elementary_mathematics',
    'high_school_physics',
    'high_school_computer_science',
    'high_school_european_history',
    'business_ethics',
    'moral_disputes',
    'high_school_statistics',
    'miscellaneous',
    'formal_logic',
    'high_school_government_and_politics',
    'prehistory',
    'security_studies',
    'high_school_biology',
    'logical_fallacies',
    'high_school_world_history',
    'professional_medicine',
    'high_school_mathematics',
    'college_medicine',
    'high_school_us_history',
    'sociology',
    'econometrics',
    'high_school_psychology',
    'human_aging',
    'us_foreign_policy',
    'conceptual_physics',
]

QUERY_TEMPLATE = """
Answer the following multiple choice question. Your response should be started by of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of ABCD.

{input}

A) {A}
B) {B}
C) {C}
D) {D}
Answer:
""".strip()

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
    train_split='dev')

# ======================================================================================
# choices = ["A", "B", "C", "D"]
# def format_subject(subject):
#     l = subject.split("_")
#     s = ""
#     for entry in l:
#         s += " " + entry
#     return s

# def format_example(df, idx, include_answer=True):
#     prompt = df.iloc[idx, 0]
#     k = df.shape[1] - 2
#     for j in range(k):
#         prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j + 1])
#     prompt += "\nAnswer:"
#     if include_answer:
#         prompt += " {}\n\n".format(df.iloc[idx, k + 1])
#     return prompt

# def gen_prompt(subject):
#     prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
#         format_subject(subject)
#     )
#     return prompt
# ======================================================================================
# 'The following are multiple choice questions (with answers) about {_name.replace("_", " ")}.'
# _hint = f'The following are multiple choice questions (with answers) about {_name.replace("_", " ")}.'
mmlu_datasets = []
for _name in mmlu_all_sets:
    # _hint = f'There is a single choice question about {_name.replace("_", " ")}. Answer the question by replying A, B, C or D.'
    _hint = f'The following are multiple choice questions (with answers) about {_name.replace("_", " ")}. \n\n'
    mmlu_infer_cfg = dict(
        # ice_template=dict(
        #     type=PromptTemplate,
        #     template=dict(round=[
        #         dict(
        #             role='HUMAN',
        #             prompt=
        #             f'{_hint_ice}\nQuestion: {{input}}\nA. {{A}}\nB. {{B}}\nC. {{C}}\nD. {{D}}\nAnswer: '
        #         ),
        #         dict(role='BOT', prompt='{target}\n')
        #     ]),
        # ),
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(
                round=[
                    # dict(role='HUMAN', prompt=QUERY_TEMPLATE),
                    dict(
                        role='HUMAN', 
                        prompt=_hint + QUERY_TEMPLATE_ALTER,
                        # prompt = QUERY_TEMPLATE
                        ),
                ],
            ),
        ),
        retriever=dict(type=ZeroRetriever),
        # retriever=dict(type=FixKRetriever, fix_id_list=[0, 1, 2, 3, 4]),
        inferencer=dict(type=GenInferencer),
    )

    mmlu_eval_cfg = dict(
        evaluator=dict(type=AccwithDetailsEvaluator),
        # pred_postprocessor=dict(type=match_answer_pattern, answer_pattern=r'(?i)ANSWER\s*:\s*([A-D])')
        pred_postprocessor=dict(type=match_answer_pattern, answer_pattern=r'([A-D])'),
        # pred_postprocessor=dict(type=first_option_postprocess, options='ABCD')
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