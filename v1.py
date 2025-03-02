import os
import openai
import pandas as pd
import json
from dotenv import load_dotenv
from word2number import w2n

# Load environment variables
load_dotenv()

# OpenAI Client Initialization (Groq API)
client = openai.OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ.get("GROQ_API_KEY")
)

def extract_keywords(user_input):
    """Extract job titles, countries, and number of leads using LLM."""
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Extract job titles, country names, and number of leads from the user's input. "
                        "Return a list of JSON objects, each containing 'titles' (list of job titles), "
                        "'countries' (list of countries), and 'leads' (integer or null). "
                        "If a field is missing, use an empty list for 'titles' and 'countries', and null for 'leads'. "
                        "Follow these rules: "
                        "1. If the user says 'Find X from each country,' group all titles together per country and apply 'leads' to the entire country. "
                        "2. If the user says 'Find X of each title,' group countries together per title and apply 'leads' separately for each title. "
                        "3. If the user says 'Find X of each title in each country,' apply 'leads' separately per (title, country) pair. "
                        "Respond ONLY in JSON format. No extra text."
                    )
                },
                {
                    "role": "user", 
                    "content": user_input
                }
            ],
        )
        
        # Extract the JSON from the response
        extracted_text = response.choices[0].message.content.strip()
        extracted_sets = json.loads(extracted_text)
        
        # Make sure we have a list
        if not isinstance(extracted_sets, list):
            extracted_sets = [extracted_sets]
            
        # Convert textual numbers to integers for leads
        for item in extracted_sets:
            if isinstance(item.get("leads"), str):
                try:
                    item["leads"] = w2n.word_to_num(item["leads"])
                except ValueError:
                    item["leads"] = None  # Keep it None if conversion fails
        
        # Debugging output
        print(f"Extracted Query Sets: {extracted_sets}")
        
        return extracted_sets
        
    except Exception as e:
        print(f"Error extracting keywords: {str(e)}")
        return []  # Return an empty list if extraction fails

def clean_column_values(series):
    """Convert all values to strings and handle null values."""
    return series.fillna('').astype(str)

def filter_csv(input_file, output_file, filter_sets):
    """
    Filter CSV file based on extracted criteria from user input.
    Supports multiple job titles and multiple countries per set.
    """
    try:
        df = pd.read_csv(input_file)

        required_columns = ['Location', 'Title']
        for column in required_columns:
            if column not in df.columns:
                print(f"Error: Column '{column}' not found in the CSV file")
                return

        # Clean column values
        for column in required_columns:
            df[column] = clean_column_values(df[column])

        # Store filtered results separately
        filtered_results = []

        for filter_set in filter_sets:
            titles = filter_set.get("titles", [])
            countries = filter_set.get("countries", [])
            max_results = filter_set.get("leads", None)

            # Apply filtering
            filtered_df = df.copy()

            if titles:
                title_filter = filtered_df['Title'].str.contains('|'.join(titles), case=False, na=False)
                filtered_df = filtered_df[title_filter]

            if countries:
                country_filter = filtered_df['Location'].str.contains('|'.join(countries), case=False, na=False)
                filtered_df = filtered_df[country_filter]

            # Ensure max_results applies only to that specific filter set
            if max_results is not None and not filtered_df.empty:
                filtered_df = filtered_df.iloc[:max_results]

            # Store the filtered results
            filtered_results.append(filtered_df)

        # Combine all filtered sets
        final_filtered_df = pd.concat(filtered_results).drop_duplicates() if filtered_results else pd.DataFrame()

        # Save output
        final_filtered_df.to_csv(output_file, index=False)
        
        print(f"\nFound {len(final_filtered_df)} matching rows")
        print(f"Results saved to {output_file}")
        
        if not final_filtered_df.empty:
            print("\nFirst few matches:")
            pd.set_option('display.max_colwidth', 100)

            # Columns to display
            selected_columns = ['Full Name', 'Email', 'Company Name', 'Location', 'Title']
            available_columns = [col for col in selected_columns if col in final_filtered_df.columns]

            if available_columns:
                final_filtered_df = final_filtered_df[available_columns].reset_index(drop=True)
                final_filtered_df.index += 1  # Start index from 1
                print(final_filtered_df.to_string(index=True, index_names=False))
            else:
                print("Error: None of the selected columns exist in the CSV file.")

    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

def main():
    input_file = 'Sheet1.csv'
    output_file = 'filtered_output.csv'
    
    user_input = input("Enter your request (e.g., 'Find 5 CTOs or CFOs in Japan and Germany'): ")
    
    filter_sets = extract_keywords(user_input)
    
    # **Handle missing job titles or countries**
    if not filter_sets:
        print("\n⚠️ No job title or country found in input. Returning all leads.")
        filter_sets = [{"titles": [], "countries": [], "leads": None}]  # No filtering applied

    filter_csv(input_file, output_file, filter_sets)

if __name__ == "__main__":
    main()