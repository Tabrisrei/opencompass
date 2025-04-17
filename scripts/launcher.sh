
# export COMPASS_DATA_CACHE='datasets/' # './data./mmlu' should in it
# export DATASET_SOURCE='ModelScope'
# export PYTHONWARNINGS=ignore::UserWarning
# python tools/prompt_viewer.py opencompass/configs/expg/eval_llama3_8b_demo.py
# export PYTHONWARNINGS="ignore"

# config='expg/eval_llama3_8b_demo.py'
# config='expg/eval_llama3_8b_mmlu_openai_simple_evals_gen_b618ea.py'
# config='expg/eval_llama3_8b_mmlu_openai_simple_evals_ppl.py'

# formal experiments origin llama3 ==============================================================================
# set gpu available
# export DATASET_SOURCE='HF'
# export HF_ENDPOINT=https://hf-mirror.com
# export CUDA_VISIBLE_DEVICES=0

# config='aformalexperiments/origin_Meta-Llama-3-8B_mmlu_health_openai_simple_evals_gen.py' # model status + model type + dataset + eval type
# result_dir='aoutput/origin_Meta-Llama-3-8B_mmlu_health_openai_simple_evals_gen' # model status + dataset


# config='aformalexperiments/origin_Meta-Llama-3-8B_mmmlu_openai_simple_evals_gen.py' # model status + model type + dataset + eval type
# result_dir='aoutput/origin_Meta-Llama-3-8B_mmmlu_openai_simple_evals_gen' # model status + dataset


# config='aformalexperiments/origin_Meta-Llama-3-8B_medqa_openai_simple_evals_gen.py' # model status + model type + dataset + eval type
# result_dir='aoutput/origin_Meta-Llama-3-8B_medqa_openai_simple_evals_gen' # model status + dataset


# config='aformalexperiments/origin_Meta-Llama-3-8B_medexqa_openai_simple_evals_gen.py' # model status + model type + dataset + eval type
# result_dir='aoutput/origin_Meta-Llama-3-8B_medexqa_openai_simple_evals_gen' # model status + dataset


# config='aformalexperiments/origin_Meta-Llama-3-8B_careqa_openai_simple_evals_gen.py' # model status + model type + dataset + eval type
# result_dir='aoutput/origin_Meta-Llama-3-8B_careqa_openai_simple_evals_gen' # model status + dataset


# config='aformalexperiments/origin_Meta-Llama-3-8B_jmed_openai_simple_evals_gen.py' # model status + model type + dataset + eval type
# result_dir='aoutput/origin_Meta-Llama-3-8B_jmed_openai_simple_evals_gen' # model status + dataset


# config='aformalexperiments/origin_Meta-Llama-3-8B_medbulletsop5_openai_simple_evals_gen.py' # model status + model type + dataset + eval type
# result_dir='aoutput/origin_Meta-Llama-3-8B_medbulletsop5_openai_simple_evals_gen' # model status + dataset


# config='aformalexperiments/origin_Meta-Llama-3-8B_medmcqa_openai_simple_evals_gen.py' # model status + model type + dataset + eval type
# result_dir='aoutput/origin_Meta-Llama-3-8B_medmcqa_openai_simple_evals_gen' # model status + dataset


# config='aformalexperiments/origin_Meta-Llama-3-8B_pubmedqa_openai_simple_evals_gen.py' # model status + model type + dataset + eval type
# result_dir='aoutput/origin_Meta-Llama-3-8B_pubmedqa_openai_simple_evals_gen' # model status + dataset


# config='aformalexperiments/origin_Meta-Llama-3-8B_medxpertqa_openai_simple_evals_gen.py' # model status + model type + dataset + eval type
# result_dir='aoutput/origin_Meta-Llama-3-8B_medxpertqa_openai_simple_evals_gen' # model status + dataset


# config='aformalexperiments/origin_Meta-Llama-3-8B_mmlupro_openai_simple_evals_gen.py' # model status + model type + dataset + eval type
# result_dir='aoutput/origin_Meta-Llama-3-8B_mmlupro_openai_simple_evals_gen' # model status + dataset


# python run.py \
#     opencompass/configs/${config} \
#     --mode 'all' \
#     --work-dir ${result_dir} \
#     --config-dir 'opencompass/configs' #--debug

# ==================================================================================================================================================

export CUDA_VISIBLE_DEVICES=0

# config='aformalexperiments/gptq_w_only_8b_Meta-Llama-3-8B_pubmedqa_openai_simple_evals_gen.py' # model status + model type + dataset + eval type
# result_dir='aoutput/gptq_w_only_8b_Meta-Llama-3-8B_pubmedqa_openai_simple_evals_gen' # model status + dataset
# llmc_cfg='/home/gsb/LLMCMed/llmc/configs/AFormalExperimentG/gptq_w_only_8b_Meta-Llama-3-8B.yml'
# llmc_model_path='atrans_models/meta-llama/gptq_w_only_8b_Meta-Llama-3-8B/transformed_model' 


# config='aformalexperiments/awq_w_only_8b_Meta-Llama-3-8B_pubmedqa_openai_simple_evals_gen.py' # model status + model type + dataset + eval type
# result_dir='aoutput/awq_w_only_8b_Meta-Llama-3-8B_pubmedqa_openai_simple_evals_gen' # model status + dataset
# llmc_cfg='/home/gsb/LLMCMed/llmc/configs/AFormalExperimentG/awq_w_only_8b_Meta-Llama-3-8B.yml'
# llmc_model_path='atrans_models/meta-llama/awq_w_only_8b_Meta-Llama-3-8B/transformed_model'


# config='aformalexperiments/smqu_w_only_8b_Meta-Llama-3-8B_pubmedqa_openai_simple_evals_gen.py' # model status + model type + dataset + eval type
# result_dir='aoutput/smqu_w_only_8b_Meta-Llama-3-8B_pubmedqa_openai_simple_evals_gen' # model status + dataset
# llmc_cfg='/home/gsb/LLMCMed/llmc/configs/AFormalExperimentG/smqu_w_only_8b_Meta-Llama-3-8B.yml'
# llmc_model_path='atrans_models/meta-llama/smqu_w_only_8b_Meta-Llama-3-8B/transformed_model'


# config='aformalexperiments/wanda_50_Meta-Llama-3-8B_pubmedqa_openai_simple_evals_gen.py' # model status + model type + dataset + eval type
# result_dir='aoutput/wanda_50_Meta-Llama-3-8B_pubmedqa_openai_simple_evals_gen' # model status + dataset
# llmc_cfg='/home/gsb/LLMCMed/llmc/configs/AFormalExperimentG/wanda_50_Meta-Llama-3-8B.yml'
# llmc_model_path='atrans_models/meta-llama/wanda_50_Meta-Llama-3-8B/transformed_model'


config='aformalexperiments/shortgpt_5_Meta-Llama-3-8B_pubmedqa_openai_simple_evals_gen.py' # model status + model type + dataset + eval type
result_dir='aoutput/shortgpt_5_Meta-Llama-3-8B_pubmedqa_openai_simple_evals_gen' # model status + dataset
llmc_cfg='/home/gsb/LLMCMed/llmc/configs/AFormalExperimentG/shortgpt_5_Meta-Llama-3-8B.yml'
llmc_model_path='atrans_models/meta-llama/shortgpt_5_Meta-Llama-3-8B/transformed_model'


export PYTHONPATH=llmc:$PYTHONPATH
python run.py \
    opencompass/configs/${config} \
    --work-dir ${result_dir} \
    --mode 'all' \
    --llmc_cfg ${llmc_cfg} \
    --llmc_eval_mode 'quant' \
    --llmc_model_path ${llmc_model_path}

