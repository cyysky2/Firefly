import json
import openai
from tqdm import tqdm

def read_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def create_prompt(question, standard_res, BC213BBase_res, trained_BC213BBase_res, BC213BChat_res, trained_BC213BChat_res, DISCLLM_res):
    return [{"role": "user",
             "content": 
             (
                "Assess the following responses for the question in terms of:\n"
                "- Legal accuracy: Does the response accurately reflect current legal standards and knowledge?\n"
                "- Clarity: Is the legal argument or explanation clear and understandable?\n"
                "- Relevance: Are the provided details relevant to the legal question posed?\n"
                "- Completeness: Does the response address all parts of the question comprehensively?\n\n"
                "Considering these criteria, grade each response a score range from 1 to 5."
                f"Question: {question}\n\n"
                f"Response 1: {standard_res}\n\n"
                f"Response 2: {BC213BBase_res}\n\n"
                f"Response 3: {trained_BC213BBase_res}\n\n"
                f"Response 4: {BC213BChat_res}\n\n"
                f"Response 5: {trained_BC213BChat_res}\n\n"
                f"Response 6: {DISCLLM_res}\n\n"
                )}]

def main():
    # Load the JSON data from files
    standard_responses = read_json_file('zh_law.json')
    BC213BBase_response = read_json_file('BC2-13B-Base-temperature100-top_p090 test results.json')
    trained_BC213BBase_response = read_json_file('DISC-Trained-BC2-13B-Base-temperature050-top_p090 test results.json')
    BC213BChat_response = read_json_file('BC2-13B-Chat-temperature100-top_p090 test results.json')
    trained_BC213BChat_response = read_json_file('DISC-Trained-BC2-13B-Chat-temperature045-top_p090 test results.json')
    DISCLLM_response = read_json_file('DISCLawLLM-temperature090-top_p090 test results.json')
    
    # Initialize the OpenAI API (ensure your API key is correctly configured)
    openai.api_key = ''
    
    # Collect results and judgments
    results = []
    for i in tqdm(range(len(standard_responses)), desc="Evaluating Responses"):
        question, std_resp = standard_responses[i]
        _, BC213BBase_res = BC213BBase_response[i]
        _, trained_BC213BBase_res = trained_BC213BBase_response[i]
        _, BC213BChat_res = BC213BChat_response[i]
        _, trained_BC213BChat_res = trained_BC213BChat_response[i]
        _, DISCLLM_res = DISCLLM_response[i]
        
        prompt = create_prompt(question, std_resp, BC213BBase_res, trained_BC213BBase_res, BC213BChat_res, trained_BC213BChat_res, DISCLLM_res)
        response = openai.ChatCompletion.create(
            model="gpt-4-turbo",  # Use the appropriate engine for Turbo
            messages=prompt
        )
        results.append({
            "prompt": prompt,
            "judgment": response['choices'][0]['message']
        })
    
    # Save results to a new JSON file
    with open('results.json', 'w', encoding='utf-8') as file:
        json.dump(results, file, ensure_ascii=False, indent=4)

    print("Judgments saved to results.json")

if __name__ == "__main__":
    main()
