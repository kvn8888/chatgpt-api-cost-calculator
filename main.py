import os
import json
import matplotlib.pyplot as plt
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
from collections import defaultdict
import tiktoken

# Global variables for cost calculation
COSTS = {
    "gpt-4": {"input": 30.0, "output": 60.0},
    "gpt-4-turbo": {"input": 10.0, "output": 30.0},
    "gpt-4o": {"input": 5.0, "output": 15.0},
    "gpt-3.5-turbo": {"input": 0.5, "output": 1.5}
}

MODEL_MAPPINGS = {
    "gpt-4": "gpt-4",
    "gpt-4-browsing": "gpt-4-turbo",
    "gpt-4-code-interpreter": "gpt-4-turbo",
    "gpt-4-dalle": "gpt-4-turbo",
    "gpt-4-gizmo": "gpt-4-turbo",
    "gpt-4-mobile": "gpt-4-turbo",
    "gpt-4-plugins": "gpt-4-turbo",
    "gpt-4o": "gpt-4o",
    "text-davinci-002-render": "gpt-3.5-turbo",
    "text-davinci-002-render-sha": "gpt-3.5-turbo",
    "text-davinci-002-render-sha-mobile": "gpt-3.5-turbo"
}

def count_tokens(text, model="gpt-4"):
    """Count the number of tokens in a given text using the specified model's tokenizer."""
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except Exception as e:
        print(f"Error tokenizing text: {e}")
        return 0

def read_conversation_json(file_path):
    """Read and parse the JSON file containing the conversation data."""
    print(f"Reading data from {file_path}...")
    with open(file_path, 'r') as file:
        data = json.load(file)
    print("Data reading completed.")
    return data

def extract_token_usage(data):
    """Extract the monthly token usage from the conversation data."""
    print("Extracting token usage from conversation data...")
    monthly_model_usage = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    conversation_histories = defaultdict(str)

    total_messages = sum(len(conversation['mapping']) for conversation in data)
    processed_messages = 0

    for conversation in data:
        for message_id, message_data in conversation['mapping'].items():
            processed_messages += 1
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
                            conversation_histories[model] += part + " "
                            cumulative_token_count = count_tokens(conversation_histories[model], model)
                            monthly_model_usage[month_key][model]['input'] += cumulative_token_count
                        elif author_role == 'assistant':
                            monthly_model_usage[month_key][model]['output'] += token_count
                            conversation_histories[model] += part + " "
            except AttributeError:
                pass
            
            # Print progress
            if processed_messages % 100 == 0 or processed_messages == total_messages:
                print(f"Processed {processed_messages}/{total_messages} messages...")

    print("Token usage extraction completed.")
    return monthly_model_usage

def calculate_cost(model_usage):
    """Calculate the cost based on input and output tokens for each model."""
    total_cost = 0
    for model, usage in model_usage.items():
        input_cost = (usage['input'] / 1_000_000) * COSTS[model]["input"]
        output_cost = (usage['output'] / 1_000_000) * COSTS[model]["output"]
        total_cost += input_cost + output_cost
    return total_cost

def get_all_months(start_date, end_date):
    """Generate a list of all months between start_date and end_date."""
    months = []
    current_date = start_date
    while current_date <= end_date:
        months.append(current_date.strftime('%Y-%m'))
        current_date += relativedelta(months=1)
    return months

def plot_token_usage(monthly_model_usage):
    print("Plotting token usage data...")
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
    print("Plotting completed.")

def print_token_usage(monthly_model_usage, monthly_costs):
    """Print the monthly and cumulative token usage with cost for each model."""
    print("Printing token usage data...")
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
    print("Printing completed.")

def main():
    json_file_path = '/Users/kevinc/Development/AI/chatgpt-api-cost-calculator/conversation/conversations.json'
    data = read_conversation_json(json_file_path)
    monthly_model_usage = extract_token_usage(data)
    
    # Calculate monthly costs
    print("Calculating monthly costs...")
    monthly_costs = {month: calculate_cost(usage) for month, usage in monthly_model_usage.items()}
    print("Monthly cost calculation completed.")
    
    # Pass both arguments to print_token_usage
    print_token_usage(monthly_model_usage, monthly_costs)
    plot_token_usage(monthly_model_usage)

if __name__ == '__main__':
    main()
