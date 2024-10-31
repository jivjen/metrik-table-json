from openai import OpenAI
from typing import List, Optional, Callable, Dict, Tuple
from pydantic import BaseModel, Field
from googleapiclient.discovery import build
import requests
import json
import os
from IPython.display import display, Markdown
import tiktoken
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
import aiohttp
import logging

encoding = tiktoken.encoding_for_model("gpt-4o-mini")
GOOGLE_API_KEY = "AIzaSyBiTmP3mKXTUb13BtpDivIDZ5X5KccFaqU"
GOOGLE_CSE_ID = "82236a47a9b6e47e6"

google_search = build("customsearch", "v1", developerKey=GOOGLE_API_KEY).cse()

JINA_API_KEY="jina_cdfde91597854ce89ef3daed22947239autBdM5UrHeOgwRczhd1JYzs51OH"
openai = OpenAI(api_key="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")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_table(user_input: str, job_id: str):
    print("Generating Table")
    table_generator_system_prompt = """
    Role: You are an expert researcher and critical thinker.
    Task: Your task is to analyze the user's input and create a hypothetical table that would contain all the required information from the user query.

    Instructions:
    1. Create a hypothetical table from the user query, such that all cells contain empty strings
    2. The table must be constructed so that the individual cells can be answered with a single word, number, or at most a short sentence
    3. Ensure that rows, columns, and cells are NOT created for any extra information that IS NOT asked for in the user input
    4. Make the row and column headers descriptive enough that they contain sufficient information for a search agent to instantly query the needed information from the internet
    5. Output the table as a JSON object with:
       - A "headers" object containing:
         * "rows": array of row header names
         * "columns": array of column header names
       - A "data" array of arrays where each inner array represents a row of empty strings
    6. ONLY If the number of rows or columns are not available in the user input, keep it as MAX = 5
    7. If the Name of an entity is present in the row headers, there is NO NEED to have the first column to be names. Avoid such redundancies.
    8. Make the headers (especially the column headers) as specific and descriptive as possible. Eg: Good column header = Market Share (%) 2024; Bad column header = Market Share
    9. For each aspect, only create one column for them. Eg: Market Share = one column for market share, Revenue = one column for revenue. Do not create more columns than explicitly needed in the user input
    Example User input:
    "top 3 companies with their revenue, and growth rate"
    Example output format:
    {
      "headers": {
        "rows": ["Company A", "Company B", "Company C"],
        "columns": ["Revenue 2024 (worldwide)", "Growth (%) 2024 (worldwide)"]
      },
      "data": [
        ["", ""],
        ["", ""],
        ["", ""]
      ]
    }
    """

    table_generator_user_content = f"Analyze and generate a table for the following user input: {user_input}"

    class TableHeaders(BaseModel):
        rows: List[str] = Field(description="Row headers")
        columns: List[str] = Field(description="Column headers")

    class TableGeneration(BaseModel):
        headers: TableHeaders = Field(description="Table headers")
        data: List[List[str]] = Field(description="Table data as 2D array of empty strings")

    table_generator_response = openai.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": table_generator_system_prompt},
            {"role": "user", "content": table_generator_user_content}
        ],
        response_format=TableGeneration
    )

    os.makedirs(f"jobs/{job_id}", exist_ok=True)
    with open(f"jobs/{job_id}/table.json", "w") as f:
        json.dump(table_generator_response.choices[0].message.parsed.dict(), f, indent=2)

    print("Table")
    print(json.dumps(table_generator_response.choices[0].message.parsed.dict(), indent=2))

    return table_generator_response.choices[0].message.parsed.dict()


def generate_row_header_subquestion(user_input: str, table_json: dict) -> str:
    print("Generating Row Header Sub-Question")

    sub_question_generator_system_prompt = """
    Role: You are an expert researcher and critical thinker.
    Task: Your task is to create a sub-question that will help identify the specific entities needed for the row headers.

    Instructions:
    1. Look at the row header placeholder in the table - this indicates the type of entity we need names for
    2. Create a single, clear question that asks for the specific entities that should be in the rows
    2.1. If the question does not have specifics, then add more descriptive words and ensure there is no confusion. EG: user input = top 4 car companies; good subquestion: which are the biggest automobile manufacturers companies in the world by annual revenue. Bad sub question: which are the top car companies.
    2.2. If an ordering for the row headers is not present, then default to using annual revenue (unless specified otherwise)
    3. The question must be such that a simple search query of the question should produce the needed entity names
    4. Make the question specific to the user's requirements (e.g., if they asked for "top 5", include that in the question)
    5. The sub-question must be focused only on identifying the entities, not any other data

    Example:
    If the table has placeholder row headers ["Company 1", "Company 2", "Company 3"] and user asked for "top 3 tech companies",
    the sub-question should be: "What are the names of the top 3 technology companies in the world by annual revenue?"
    """

    class SubQuestion(BaseModel):
        question: str = Field(description="The sub-question to find row headers")

    sub_question_response = openai.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": sub_question_generator_system_prompt},
            {"role": "user", "content": f"""
                User Input: {user_input}
                Current Row Headers: {table_json['headers']['rows']}
                Column Headers: {table_json['headers']['columns']}
                Generate a sub-question to find the specific entities needed.
            """}
        ],
        response_format=SubQuestion
    )

    return sub_question_response.choices[0].message.parsed.question

def generate_keywords(user_input: str, sub_question: str) -> List[str]:
    keyword_generator_system_prompt = """
        Role: You are a professional Google search researcher.
        Task: Given a main user query for context and a specific sub-question, your primary task is to generate 5 unique Google search keywords that will help gather detailed information primarily related to the sub-question.

        Instructions:
        0. Ensure all the nuances in the subquestion are captured in the keywords generated inlcuding but not limited to, geography (Eg: USA, Worldwide, etc.), year (2023, 2024, etc.), etc.
        1. Focus ONLY on the sub-question when generating keywords. The main query serves merely as context but should not dominate the keyword selection.
        2. Ensure that at least 4 out of 5 keywords are directly relevant to the sub-question.
        3. You may use 1 keyword to bridge the sub-question with the broader context of the main query if relevant.
        4. Generate keywords that aim to concisely answer the sub-question, including but not limited to: specific details, expert opinions, case studies, recent developments, and historical context (if any are applicable).
        5. Aim for a mix of broad and specific keywords related to the sub-question to ensure comprehensive coverage.
        6. Ensure all keywords are unique

        Main Aim of Creating Keywords for Search Engines: To ensure that any piece of information present on the internet pertinent to answering the sub-question is always found.
    """

    class KeywordGeneration(BaseModel):
        keywords: List[str] = Field(description="List of generated keywords")

    keyword_generator_user_prompt = f"Main query (for context): {user_input}\nSub-question (primary focus): {sub_question}\nPlease generate keywords primarily addressing the sub-question, while considering the main query as context."

    keyword_generator_response = openai.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": keyword_generator_system_prompt},
            {"role": "user", "content": keyword_generator_user_prompt}
        ],
        response_format=KeywordGeneration
    )

    return keyword_generator_response.choices[0].message.parsed.keywords

async def search_and_answer(search_term, job_id, table, sub_question):
    """Search the Web and obtain a list of web results."""
    logger.info(f"Searching for: {search_term}")
    google_search_result = google_search.list(q=search_term, cx=GOOGLE_CSE_ID).execute()
    urls = [result["link"] for result in google_search_result.get("items", [])]
    
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_and_analyze(session, url, table, sub_question) for url in urls]
        results = await asyncio.gather(*tasks)
        
    for result in results:
        if result:
            return result
    
    logger.info(f"No answer found for: {search_term}")
    return ""

async def fetch_and_analyze(session, url, table, sub_question):
    search_url = f'https://r.jina.ai/{url}'
    headers = {"Authorization": f"Bearer {JINA_API_KEY}"}
    try:
        async with session.get(search_url, headers=headers, timeout=10) as response:
            if response.status == 200:
                search_result = await response.text()
                answer = analyse_result(search_result, table, sub_question, url)
                if answer:
                    logger.info(f"Answer found: {answer}")
                    return answer
            else:
                logger.warning(f"Jina returned an error: {response.status} for URL: {url}")
    except asyncio.TimeoutError:
        logger.warning(f"Timeout fetching URL {url}")
    except Exception as e:
        logger.error(f"Error fetching URL {url}: {str(e)}")
    return ""

def initialize_row_headers(user_input: str, table_json: dict, job_id: str) -> dict:
    print("Initializing Row Headers")

    # First, determine if we need to find headers
    header_analyzer_system_prompt = """
    Role: You are an AI assistant specialized in analyzing user queries for specific entities.
    Task: Determine if the user query explicitly mentions the specific entities that should be row headers.

    Instructions:
    1. Analyze the user query carefully
    2. Return:
       - "yes" if the query explicitly names the entities that should be row headers
       - "no" if the query only describes the type of entities but doesn't name them

    Example:
    Query: "Compare Apple, Microsoft, and Google's revenue" -> "yes" (names are explicit)
    Query: "Compare the top 3 tech companies' revenue" -> "no" (names not given)
    """

    class HeaderAnalysis(BaseModel):
        has_explicit_headers: str = Field(description="Whether headers are explicit in query")
        entities: List[str] = Field(description="List of explicit entities if any, empty list if none")

    analysis_response = openai.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": header_analyzer_system_prompt},
            {"role": "user", "content": f"Analyze this query: {user_input}"}
        ],
        response_format=HeaderAnalysis
    )

    if analysis_response.choices[0].message.parsed.has_explicit_headers.lower() == "yes":
        # Use explicit headers from the query
        entities = analysis_response.choices[0].message.parsed.entities
        # Update row headers in the headers object
        table_json["headers"]["rows"] = entities[:len(table_json["data"])]
    else:
        # Generate sub-question to find headers
        header_question = generate_row_header_subquestion(user_input, table_json)
        print(f"Generated sub-question: {header_question}")

        # Generate keywords using existing function
        header_keywords = generate_keywords(user_input, header_question)

        # Search and analyze results using existing function
        for keyword in header_keywords:
            print(f"Searching with keyword: {keyword}")
            result = search_and_answer(keyword, job_id, table_json, header_question)

            if result:
                # Parse the result and update the table
                entities_parser_prompt = """
                Role: You are an AI assistant specialized in extracting entity names.
                Task: Extract entity names from the search result and return them as a list.

                Instructions:
                1. Extract the specific entity names mentioned in the result
                2. Return only the number of entities needed to fill the table
                3. Ensure entities are in order of relevance/importance
                """

                class EntitiesExtraction(BaseModel):
                    entities: List[str] = Field(description="List of extracted entity names")

                entities_response = openai.beta.chat.completions.parse(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": entities_parser_prompt},
                        {"role": "user", "content": f"Extract entities from: {result}"}
                    ],
                    response_format=EntitiesExtraction
                )

                # Update the row headers in the headers object
                entities = entities_response.choices[0].message.parsed.entities
                table_json["headers"]["rows"] = entities[:len(table_json["data"])]
                break  # Exit after finding and updating headers

    return table_json

def analyse_result(search_result: str, markdown_table: str, sub_question: str, url: str) -> List[str]:
  tokens = encoding.encode(search_result)

  # Define the chunk size
  chunk_size = 100000

  # Chunk the tokens
  chunks = [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)]

  # Decode the chunks back to text
  chunked_search_result = [encoding.decode(chunk) for chunk in chunks]

  for chunk in chunked_search_result:
    ress = {url : chunk}
    search_analyser_system_prompt = f"""
      Role: You are an expert AI assistant specialized in analyzing search results and extracting precise information.
      Task: Given a specific sub-question, a markdown table for context, and a search result dump, your primary task is to determine if the answer to the sub-question can be found within the provided information.

      Instructions:
      1. Carefully analyze the content of the search result thoroughly, focusing on finding information that directly answers the sub-question.
      2. Pay attention to the markdown table, as it may provide additional context for interpreting the search results.
      3. If you find the answer:
        a. Respond with 'yes' for subQuestionAnswered.
        b. Provide a concise, accurate answer based on the information found.
        c. ALWAYS include the exact URL of the source in brackets along with the answer.
      4. If you cannot find the answer:
        a. Respond with 'no' for subQuestionAnswered.
        b. Leave the result empty.
      5. Ensure that your response is based solely on the information provided in the search results and markdown table.
      6. Do not make assumptions or provide information that is not explicitly stated in the given data.
      7. Ensure the answer is only a number/single data point with the source url in brackets. Example: 6% (www.xyz.com/source)
      Main Aim: To provide accurate, source-backed answers to sub-questions when the information is available, and to clearly indicate when the required information cannot be found in the given search results.
      """

    search_analyser_user_prompt = f"""
      Sub-question: {sub_question}

      Markdown Table:
      {markdown_table}

      Search Result:
      {ress}

      Please analyze the search results and determine if the answer to the sub-question can be found.
      """

    class AnalysisResponse(BaseModel):
        subQuestionAnswered: str = Field(description = "Answer found for the question or not")
        result: str = Field(description="The answer to the sub-question")

    search_analysis_response = openai.beta.chat.completions.parse(
          model="gpt-4o-mini",
          messages=[
              {"role": "system", "content": search_analyser_system_prompt},
              {"role": "user", "content": search_analyser_user_prompt}
          ],
          response_format=AnalysisResponse
      )

    if search_analysis_response.choices[0].message.parsed.subQuestionAnswered == "yes":
      return search_analysis_response.choices[0].message.parsed.result

  return ""

def find_all_empty_cells(table_json: dict) -> List[Tuple[int, int]]:
    """Find all empty cells in the table, returns a list of (row_index, col_index) tuples."""
    empty_cells = []
    for row_idx, row in enumerate(table_json["data"]):
        for col_idx, cell in enumerate(row):
            if cell == "":
                empty_cells.append((row_idx, col_idx))
    return empty_cells

def generate_cell_subquestion(row_header: str, col_header: str, table_json: dict) -> str:
    """Generate a sub-question for a specific cell based on its headers."""
    system_prompt = """
    Role: You are an expert researcher and critical thinker.
    Task: Create a specific sub-question to find the value for a single cell in a table.

    Instructions:
    1. Use both the row header (entity) and column header (attribute) to create a precise question
    2. The question should be specific enough to get a single data point as an answer
    3. The answer should be a single number!
    4. ONLY if a number is too little to satisfy the needs of the userinput, ONLY then use a sentence.
    """

    class SubQuestion(BaseModel):
        question: str = Field(description="The sub-question for the cell")

    response = openai.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Create a sub-question to find the {col_header} for {row_header}"}
        ],
        response_format=SubQuestion
    )

    return response.choices[0].message.parsed.question

def update_cell_value(table_json: dict, row_idx: int, col_idx: int, value: str) -> dict:
    """Update a specific cell in the table with a new value."""
    table_json["data"][row_idx][col_idx] = value
    return table_json

async def process_cell(row_idx: int, col_idx: int, user_input: str, table_json: dict, job_id: str) -> Tuple[int, int, str]:
    row_header = table_json["headers"]["rows"][row_idx]
    col_header = table_json["headers"]["columns"][col_idx]

    logger.info(f"Processing cell: {row_header} x {col_header}")

    sub_question = generate_cell_subquestion(row_header, col_header, table_json)
    logger.info(f"Generated sub-question: {sub_question}")

    keywords = generate_keywords(user_input, sub_question)
    logger.info(f"Keywords: {keywords}")

    tasks = [search_and_answer(keyword, job_id, table_json, sub_question) for keyword in keywords]
    results = await asyncio.gather(*tasks)

    for result in results:
        if result:
            logger.info(f"Found answer: {result}")
            return row_idx, col_idx, result

    logger.info(f"No answer found, marking cell with 'X'")
    return row_idx, col_idx, "X"

async def process_empty_cells(user_input: str, table_json: dict, job_id: str) -> dict:
    logger.info("Processing empty cells in parallel")

    empty_cells = find_all_empty_cells(table_json)
    if not empty_cells:
        logger.info("No empty cells - table is complete")
        return table_json

    tasks = [process_cell(row_idx, col_idx, user_input, table_json, job_id) for row_idx, col_idx in empty_cells]
    results = await asyncio.gather(*tasks)

    for row_idx, col_idx, result in results:
        table_json = update_cell_value(table_json, row_idx, col_idx, result)
        logger.info(f"Updated cell [{row_idx}, {col_idx}] with value: {result}")

    # Save the updated table after processing all cells
    with open(f"jobs/{job_id}/table.json", "w") as f:
        json.dump(table_json, f, indent=2)

    logger.info("Finished processing all cells")
    return table_json

async def main():
    user_input = "Top 4 car companies in the world, along with their annual revenues, market shares, profit margins, and growth rates"
    job_id = "1001"
    initial_table = generate_table(user_input, job_id)
    updated_table = initialize_row_headers(user_input, initial_table, job_id)
    completed_table = await process_empty_cells(user_input, updated_table, job_id)
    return completed_table

if __name__ == "__main__":
    asyncio.run(main())