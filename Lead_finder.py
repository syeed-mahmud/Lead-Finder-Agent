import logfire
logfire.configure()
import os
import openai
import pandas as pd
import asyncio
from dotenv import load_dotenv
from pathlib import Path
from pydantic_ai import Agent

# Suppress Logfire warnings
os.environ["LOGFIRE_IGNORE_NO_CONFIG"] = "1"

# Load environment variables from .env file
load_dotenv()

# OpenAI Client Initialization (Groq API)
client = openai.OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ.get("GROQ_API_KEY")
)

# Keyword Extraction Agent
keyword_extraction_agent = Agent(
    'groq:llama-3.3-70b-versatile',
    result_type=dict,
    system_prompt=(
        "Extract the Country Name, Person Title and leads number from the user's input. "
        "Return a JSON object with three keys: 'title', 'country' and 'leads'. "
        "Example: {'CEO','CTO', 'Founder And CEO','Canada','Uk','Norway', '3' , 'ten', '47' , '6' , 'five', 'Twelve' , 'Twenty one'}."
        "'title' should be a string, 'country' should be a string, and 'leads' should be an integer. "
    )
)

def clean_column_values(series):
    """
    Clean column values by converting all to strings and handling null values.
    """
    return series.fillna('').astype(str)

def get_unique_values(df, column):
    """
    Get unique values from a column to help user see available options.
    Returns first 10 non-empty unique values as examples.
    """
    values = clean_column_values(df[column])
    unique_vals = [v for v in values.unique() if v.strip() != '']
    return sorted(unique_vals)[:10]

async def extract_keywords(user_input):
    """Extract job title, country, and number of leads using LLM."""
    keyword_extraction_result = await keyword_extraction_agent.run(user_input)
    
    if not keyword_extraction_result.data or not isinstance(keyword_extraction_result.data, dict):
        print("Error: Failed to extract keywords. Please try again.")
        return None, None, None
    
    extracted_keywords = keyword_extraction_result.data
    title = extracted_keywords.get("title", "").strip()
    country = extracted_keywords.get("country", "").strip()
    leads = extracted_keywords.get("leads", None)
    
    if not title or not country:
        print("Error: Missing title or country. Please rephrase your request.")
        return None, None, None
    
    return title, country, leads

def filter_csv(input_file, output_file, title_search, country_search, max_results=None):
    """
    Filter CSV file based on extracted criteria from user input.
    """
    try:
        df = pd.read_csv(input_file)
        
        required_columns = ['Location', 'Title']
        for column in required_columns:
            if column not in df.columns:
                print(f"Error: Column '{column}' not found in the CSV file")
                return
        
        for column in required_columns:
            df[column] = clean_column_values(df[column])
        
        country_filter = df['Location'].str.contains(country_search, case=False, na=False) if country_search else True
        title_filter = df['Title'].str.contains(title_search, case=False, na=False) if title_search else True
        filtered_df = df[country_filter & title_filter]
        
        if max_results:
            filtered_df = filtered_df.head(max_results)
        
        filtered_df.to_csv(output_file, index=False)
        
        print(f"\nFound {len(filtered_df)} matching rows")
        print(f"Results saved to {output_file}")
        
        if not filtered_df.empty:
            print("\nFirst few matches:")
            pd.set_option('display.max_colwidth', 100)
    
            # Select only the required columns
            selected_columns = ['Full Name', 'Email', 'Company Name', 'Location', 'Title']
    
            # Ensure all selected columns exist in the DataFrame before printing
            available_columns = [col for col in selected_columns if col in filtered_df.columns]
    
            if available_columns:
                # Reset index and start counting from 1
                filtered_df = filtered_df[available_columns].reset_index(drop=True)
                filtered_df.index += 1  # Make index start from 1

                print(filtered_df.to_string(index=True, index_names=False))  # Show index without column name
            else:
                print("Error: None of the selected columns exist in the CSV file.")


        
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

def main():
    input_file = 'Sheet1.csv'
    output_file = 'filtered_output.csv'
    
    user_input = input("Enter your request (e.g., 'Find 10 CEOs in Germany'): ")
    
    title, country, leads = asyncio.run(extract_keywords(user_input))
    if title and country:
        filter_csv(input_file, output_file, title, country, max_results=leads)

if __name__ == "__main__":
    main()
