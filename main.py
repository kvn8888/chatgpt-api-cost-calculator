import os
import json
import matplotlib.pyplot as plt
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
from collections import defaultdict
import tiktoken

# Global variables for cost calculation (see https://openai.com/api/pricing/)
COSTS = {
    "gpt-4": {"input": 30.0, "output": 60.0},
    "gpt-4-turbo": {"input": 10.0, "output": 30.0},
    "gpt-4o": {"input": 5.0, "output": 15.0},
    "gpt-3.5-turbo": {"input": 0.5, "output": 1.5}
}

MODEL_MAPPINGS = {
    "gpt-4": "gpt-4",
    "gpt-4-browsing": "gpt-4",
    "gpt-4-code-interpreter": "gpt-4",
    "gpt-4-dalle": "gpt-4",
    "gpt-4-gizmo": "gpt-4",
    "gpt-4-mobile": "gpt-4",
    "gpt-4-plugins": "gpt-4",
    "gpt-4o": "gpt-4o",
    "text-davinci-002-render": "gpt-3.5-turbo",
    "text-davinci-002-render-sha": "gpt-3.5-turbo",
    "text-davinci-002-render-sha-mobile": "gpt-3.5-turbo"
}

def count_tokens(text, model="gpt-4"):
    """
    Count the number of tokens in a given text using the specified model's tokenizer.

    Args:
    text (str): The text to tokenize.
    model (str): The name of the model to use for tokenization. Default is "gpt-4".

    Returns:
    int: The number of tokens in the text.
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except Exception as e:
        if isinstance(text, dict) and 'width' in text.keys() and 'height' in text.keys():
            return COST_BASE_IMAGE + COST_IMAGE_TILE * (text['width'] * text['height'] // (512 * 512))
        else:
            print(f"Error tokenizing text: {e}")
            return 0


def read_conversation_json(file_path):
    """Read and parse the JSON file containing the conversation data."""
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


def extract_token_usage(data):
    """Extract the monthly token usage from the conversation data."""
    monthly_model_usage = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    conversation_history = ""

    for conversation in data:
        for message_id, message_data in conversation['mapping'].items():
            try:
                message_content = message_data.get('message', {}).get('content', {}).get('parts', [''])
                author_role = message_data.get('message', {}).get('author', {}).get('role', '')
                create_time = message_data.get('message', {}).get('create_time')
                model_slug = message_data.get('message', {}).get('metadata', {}).get('model_slug', 'gpt-4')
                model = MODEL_MAPPINGS.get(model_slug, 'gpt-4')
                
                if create_time and message_content[0]:
                    month_key = datetime.fromtimestamp(create_time).strftime('%Y-%m')
                    for part in message_content:
                        if isinstance(part, dict):
                            part = json.dumps(part)
                        token_count = count_tokens(part, model)
                        if author_role == 'user':
                            conversation_history += part + " "
                            token_count = count_tokens(conversation_history, model)
                            monthly_model_usage[month_key][model]['input'] += token_count
                        elif author_role == 'assistant':
                            monthly_model_usage[month_key][model]['output'] += token_count
                            conversation_history += part + " "
            except AttributeError:
                pass
    return monthly_model_usage


def get_all_months(start_date, end_date):
    """Generate a list of all months between start_date and end_date."""
    months = []
    current_date = start_date
    while current_date <= end_date:
        months.append(current_date.strftime('%Y-%m'))
        current_date += relativedelta(months=1)
    return months


def plot_token_usage(monthly_model_usage):
    all_months = sorted(monthly_model_usage.keys())
    all_models = sorted(set(model for month_data in monthly_model_usage.values() for model in month_data.keys()))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 20))
    
    x = range(len(all_months))
    width = 0.35 / len(all_models)
    
    for i, model in enumerate(all_models):
        input_data = [monthly_model_usage[month][model]['input'] for month in all_months]
        output_data = [monthly_model_usage[month][model]['output'] for month in all_months]
        
        ax1.bar([j + i*width for j in x], input_data, width, label=f'{model} Input', alpha=0.7)
        ax1.bar([j + i*width for j in x], output_data, width, bottom=input_data, label=f'{model} Output', alpha=0.7)

    ax1.set_xlabel('Month')
    ax1.set_ylabel('Token Count')
    ax1.set_title('Monthly Token Usage and Cost by Model')
    ax1.set_xticks([i + width*(len(all_models)-1)/2 for i in x])
    ax1.set_xticklabels(all_months, rotation=45, ha='right')
    ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))

    ax1_twin = ax1.twinx()
    monthly_costs = [calculate_cost(monthly_model_usage[month]) for month in all_months]
    ax1_twin.plot(x, monthly_costs, color='red', label='Monthly Cost', marker='o', alpha=0.5)
    ax1_twin.set_ylabel('Cost (USD)', color='red')
    ax1_twin.tick_params(axis='y', labelcolor='red')
    ax1_twin.legend(loc='upper right', bbox_to_anchor=(1, 0.9))

    # Similar changes for cumulative plot (ax2)...

    plt.tight_layout()
    plt.show()


def print_token_usage(monthly_model_usage, monthly_costs):
    """Print the monthly and cumulative token usage with cost for each model."""
    all_months = sorted(monthly_model_usage.keys())
    all_models = set(model for month_data in monthly_model_usage.values() for model in month_data.keys())

    print(f'{"Month":<10}{"Model":<15}{"Input Tokens":>15}{"Output Tokens":>15}{"Cost (USD)":>15}')
    for month in all_months:
        for model in all_models:
            input_tokens = monthly_model_usage[month][model]['input']
            output_tokens = monthly_model_usage[month][model]['output']
            cost = (input_tokens / 1_000_000 * COSTS[model]["input"]) + (output_tokens / 1_000_000 * COSTS[model]["output"])
            print(f'{month:<10}{model:<15}{input_tokens:>15,}{output_tokens:>15,}{cost:>15,.2f}')
        print(f'{month:<10}{"TOTAL":<15}{"":<15}{"":<15}{monthly_costs[month]:>15,.2f}')
        print('-' * 70)

def calculate_cost(model_usage):
    """Calculate the cost based on input and output tokens for each model."""
    total_cost = 0
    for model, usage in model_usage.items():
        input_cost = (usage['input'] / 1_000_000) * COSTS[model]["input"]
        output_cost = (usage['output'] / 1_000_000) * COSTS[model]["output"]
        total_cost += input_cost + output_cost
    return total_cost

def main():
    json_file_path = 'chatgpt-api-cost-calculator/conversation/conversations.json'
    data = read_conversation_json(json_file_path)
    monthly_model_usage = extract_token_usage(data)
    
    # Calculate monthly costs
    monthly_costs = {month: calculate_cost(usage) for month, usage in monthly_model_usage.items()}
    
    # Pass both arguments to print_token_usage
    print_token_usage(monthly_model_usage, monthly_costs)
    plot_token_usage(monthly_model_usage)


if __name__ == '__main__':
    main()
