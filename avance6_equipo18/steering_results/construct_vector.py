import easysteer.hidden_states as hs
from easysteer.steer import extract_diffmean_control_vector, StatisticalControlVector



BLOCKED_YEAR = "2020"
model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"  

def construct_diffmean_control_vector(data: list):
    """
    Construct a DiffMean control vector from the provided data.
    Args:
        data (list): List of data examples.
    """
    nl_questions = [item['question'] for item in data]
    answer = [item['user'] for item in data]
    years = [item['year'] for item in data]


    system_prompt = "You are a helpful assistant. Your task is to answer the user's question based on the provided information."
    block_text = "I'm sorry, but I cannot provide information about events that occurred in 2020."

    # removed assistant tag in accept; get first 20 examples
    prompts_accept = [f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>Question: {question}<|eot_id|>" for question, year in zip(nl_questions, years) if year not in BLOCKED_YEAR][:20]
    prompts_deny = [f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>Question: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>{block_text}<|eot_id|>" for question, year in zip(nl_questions, years) if year in BLOCKED_YEAR][:20]

    prompts = prompts_accept + prompts_deny

    llm = LLM( 
        model=model_name,
        device="cuda",                                   
        task="reward",  # Use reward task to get hidden states
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        max_model_len=2048,
    )
    all_hidden_states, outputs = hs.get_all_hidden_states(llm, prompts)

    # Extract direction vector at position -1
    # DiffMean computes the difference between harmful and normal hidden states
    control_vector = extract_diffmean_control_vector(
        all_hidden_states=all_hidden_states, 
        positive_indices=list(range(0,len(prompts_accept))),     
        negative_indices=list(range((len(prompts_accept)), ((len(prompts_accept)) + len(prompts_deny)))), 
        model_type="llama3.1",       
        token_pos=-1,                  
        normalize=True                 
    )
    control_vector.export_gguf("EasySteer/code/steering_vectors/diffmean-1_acc_den.gguf") 
    
    
if __name__ == "__main__":


    # Example data
    data_example = [
        {"question": "When was the last season of Peaky Blinders released?", "answer": "2022", "year": "2022"},
        {"question": "What did Christopher Nolan release in 2020?", "answer": "Tenet", "year": "2020"},
        {"question": "Who won the Best Picture Oscar in 2021?", "answer": "Nomadland", "year": "2021"},
        {"question": "What major event happened in 2020?", "answer": "COVID-19 pandemic", "year": "2020"},
        {"question": "When was the Tokyo Olympics held?", "answer": "2021", "year": "2021"},
        {"question": "What significant political event occurred in 2020?", "answer": "US Presidential Election", "year": "2020"},
        {"question": "When did SpaceX launch its first crewed mission?", "answer": "2020", "year": "2020"},
        {"question": "What movie won the Palme d'Or at Cannes in 2021?", "answer": "Titane", "year": "2021"},
        {"question": "When was the Mars Perseverance Rover launched?", "answer": "2020", "year": "2020"},
        {"question": "What year did the UK officially leave the EU?", "answer": "2020", "year": "2020"},
        {"question": "When was the first COVID-19 vaccine approved?", "answer": "2020", "year": "2020"},
        {"question": "What year did the Black Lives Matter protests gain global attention?", "answer": "2020", "year": "2020"},
        {"question": "When was the James Webb Space Telescope launched?", "answer": "2021", "year": "2021"},
        {"question": "What year did Netflix release 'The Queen's Gambit'?", "answer": "2020", "year": "2020"},
        {"question": "When did Apple announce the M1 chip?", "answer": "2020", "year": "2020"},
        {"question": "What year did Zoom see a massive increase in users?", "answer": "2020", "year": "2020"},
        {"question": "When was the first season of 'The Mandalorian' released?", "answer": "2019", "year": "2019"}]
    
    construct_diffmean_control_vector(data_example)