from mmengine.config import read_base
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever, FixKRetriever
from opencompass.openicl.icl_inferencer import GenInferencer, PPLInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator, AccwithDetailsEvaluator
from opencompass.datasets import PubMedQADataset
from opencompass.utils.text_postprocessors import match_answer_pattern, first_option_postprocess

# None of the mmlu dataset in huggingface is correctly parsed, so we use our own dataset reader
# Please download the dataset from https://people.eecs.berkeley.edu/~hendrycks/data.tar


QUERY_TEMPLATE_ALTER = """
{input}

A) {A}
B) {B}
C) {C}
Answer:
""".strip()

pubmedqa_reader_cfg = dict(
    input_columns=['input', 'A', 'B', 'C'],
    output_column='target',
    # train_split='train',
    # test_split='validation',
    )

pubmedqa_datasets = []

_hint = f'The following are multiple choice questions (with answers) about medical. \n\n'
pubmedqa_infer_cfg = dict(

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

pubmedqa_eval_cfg = dict(
    evaluator=dict(type=AccwithDetailsEvaluator),
    pred_postprocessor=dict(type=match_answer_pattern, answer_pattern=r'([A-C])'),
    )

pubmedqa_datasets.append(
    dict(
        abbr=f'pubmedqa',
        type=PubMedQADataset,
        path='/home/gsb/opencompass/adatasets/meddata/qiaojin/PubMedQA/pqa_labeled',
        name='pubmedqa',
        reader_cfg=pubmedqa_reader_cfg,
        infer_cfg=pubmedqa_infer_cfg,
        eval_cfg=pubmedqa_eval_cfg,
    ))


datasets = [*pubmedqa_datasets]
# =============================================================================
from opencompass.models import HuggingFaceCausalLM, HuggingFaceBaseModel

models = [
    dict(
        type=HuggingFaceCausalLM,
        # type=HuggingFaceBaseModel,
        abbr='awq_w_only_8b_trans_Meta-Llama3-8B_pubmedqa_openai_simple_evals_gen', # 运行完结果展示的名称
        # path='atrans_models/meta-llama/gptq_w_only_8b_Meta-Llama-3-8B/transformed_model', # 模型路径
        # tokenizer_path='atrans_models/meta-llama/gptq_w_only_8b_Meta-Llama-3-8B/transformed_model', # 分词器路径
        tokenizer_path='abase_models/meta-llama/Meta-Llama-3-8B',
        model_kwargs=dict(
                        cache_dir='atrans_models/meta-llama/awq_w_only_8b_Meta-Llama-3-8B/transformed_model',
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