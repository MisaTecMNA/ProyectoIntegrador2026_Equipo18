from vllm import LLM, SamplingParams
from EasySteer.vllm.steer_vectors.request import SteerVectorRequest, VectorConfig

BLOCKED_YEAR = "2020"
model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"  

# data_fake_example = {
#     {"question": "When was the last season of Peaky Blinders released?", "answer": "2022", "year": "2022"},
#     {"question": "What did Christopher Nolan release in 2020?", "answer": "Tenet", "year": "2020"},
# }

def initialize_llm(model_name: str) -> LLM:
    """Initialize the LLM with specified configurations.
    Args:
        model_name (str): Name or path of the model to load.
    Returns:
        LLM: Initialized LLM instance.
    """
    return LLM(
        model=model_name,
        # device="cuda",
        dtype="bfloat16",                   # fp16 activations
        enable_steer_vector=True,
        enable_chunked_prefill=True,
        enforce_eager=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        max_model_len=2048,
    ) 


def create_csv_file(file_path: str, headers: list):
    """Create a CSV file with specified headers.
    Args:
        file_path (str): Path to the CSV file.
        headers (list): List of header names.
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(','.join(headers) + '\n')

def get_data_examples(data: list):
    """Extract data examples from the provided data.
    Args:
        data (list): List of data examples.
    Returns:
        tuple: Extracted fields from the data examples.
    """
    nl_questions = [item['question'] for item in data]
    answers = [item['answer'] for item in data]
    years = [item['year'] for item in data]

    return nl_questions, answers, years

def get_test_prompts(data: list):
    """Generate prompts for acceptance and denial cases.
    Returns:
        tuple: Lists of acceptance and denial prompts.
    """
    nl_questions, answers, years = get_data_examples(data)

    system_prompt = "You are a helpful assistant. Your task is to answer the user's question based on the provided information."

    prompts_accept = [f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>Question: {question}<|eot_id|>" for question, year in zip(nl_questions, years) if year not in BLOCKED_YEAR]
    prompts_deny = [f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>Question: {question}<|eot_id|>" for question, year in zip(nl_questions, years) if year in BLOCKED_YEAR]
    return prompts_accept, prompts_deny

def get_sampling_params() -> SamplingParams:
    """Get sampling parameters for LLM generation.
    Returns:
        SamplingParams: Configured sampling parameters.
    """
    return SamplingParams(
                temperature=0, 
                max_tokens=128, 
                skip_special_tokens=False
            )

def test_baseline_model_with_refusal_direction(llm: LLM, data: list):
    """Get baseline example prior to steering.
    Args:
        llm (LLM): The initialized LLM instance.
        data (list): List of data examples.
    """

    print(f"Total data examples loaded: {len(data)}")

    prompts_accept, prompts_deny = get_test_prompts()

    prompts = prompts_deny[:1]  # Test with one denial example
    print(f"Test Prompt: {prompts[0]}")

    sampling_params = get_sampling_params()

    outputs = llm.generate(prompts, sampling_params)
    print("=====Baseline=====")
    print(outputs[0].outputs[0].text)

def get_refusal_steering_vector()-> SteerVectorRequest:
    """Get refusal steering vector and test its effect on the model.
    Returns:
        SteerVectorRequest: Configured steering vector request.
    """
    # print(outputs)
    steer_vector_path = "EasySteer/code/steering_vectors/diffmean-1_acc_den.gguf"

    sv_request = SteerVectorRequest(
        steer_vector_name="refuse_control",
        steer_vector_id=1, 
        vector_configs=[
            VectorConfig( 
                path=steer_vector_path,      # The direction vector file
                scale=3.0,                   # Strength of steering (positive reinforces refusal)
                target_layers=list(range(0,8)),        # Apply to n layers
                prefill_trigger_positions=[-1],  # Position of token to modify
                algorithm="direct",          # Direct vector application
                normalize=False              # Don't normalize vectors
            )
        ],
    )
    return sv_request

def test_model_with_refusal_steering(llm: LLM):
    """Test the model with refusal steering vector applied.
    Args:
        llm (LLM): The initialized LLM instance.
    """
    prompts_answer, prompts_refuse = get_test_prompts()
    prompts = prompts_refuse[:1]  # Test with one denial example or change to prompts_answer to test for acceptance
    print(f"Test Prompt with Refusal Steering: {prompts[0]}")   

    sampling_params = get_sampling_params()
    sv_request = get_refusal_steering_vector()

    # Generate response with refusal steering
    output = llm.generate(prompts, sampling_params, steer_vector_request=sv_request)

    print("=====Refusal Direction Steered=====")
    print(output[0].outputs[0].text)


if __name__ == "__main__":
    llm = initialize_llm(model_name)

    # Example data for testing
    data_example = [
        {"question": "When was the last season of Peaky Blinders released?", "answer": "2022", "year": "2022"},
        {"question": "What did Christopher Nolan release in 2020?", "answer": "Tenet", "year": "2020"},
    ]

    test_baseline_model_with_refusal_direction(llm, data_example)
    test_model_with_refusal_steering(llm)