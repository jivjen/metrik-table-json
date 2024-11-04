from openai import OpenAI
from typing import List, Optional, Callable, Dict, Tuple
from pydantic import BaseModel, Field
from googleapiclient.discovery import build
import requests
import json
import os
import asyncio
import aiohttp
import logging
import tiktoken
from concurrent.futures import ThreadPoolExecutor, as_completed
from filelock import FileLock
from multiprocessing import Event

encoding = tiktoken.encoding_for_model("gpt-4o-mini")
GOOGLE_API_KEY = "AIzaSyBiTmP3mKXTUb13BtpDivIDZ5X5KccFaqU"
GOOGLE_CSE_ID = "82236a47a9b6e47e6"

google_search = build("customsearch", "v1", developerKey=GOOGLE_API_KEY).cse()

JINA_API_KEY="jina_cdfde91597854ce89ef3daed22947239autBdM5UrHeOgwRczhd1JYzs51OH"
openai = OpenAI(api_key="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")

def setup_job_logger(job_id: str):
    logger = logging.getLogger(f"job_{job_id}")
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(f"jobs/{job_id}/job.log")
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    # Add a StreamHandler to also log to console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(console_handler)
    return logger

def generate_table(user_input: str, job_id: str, logger: logging.Logger):
    logger.info("Generating Table")
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

    logger.info("Table generated")
    logger.info(json.dumps(table_generator_response.choices[0].message.parsed.dict(), indent=2))

    return table_generator_response.choices[0].message.parsed.dict()


def generate_row_header_subquestion(user_input: str, table_json: dict, logger: logging.Logger) -> str:
    logger.info("Generating Row Header Sub-Question")

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

    logger.info(f"Generated sub-question: {sub_question_response.choices[0].message.parsed.question}")
    return sub_question_response.choices[0].message.parsed.question

def generate_keywords(user_input: str, sub_question: str, logger: logging.Logger) -> List[str]:
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

    keywords = keyword_generator_response.choices[0].message.parsed.keywords
    logger.info(f"Generated keywords: {keywords}")
    return keywords

async def search_and_answer(search_term, job_id, table, sub_question, row_idx, col_idx, logger: logging.Logger, is_header=False, stop_flag=None):
    """Search the Web and obtain a list of web results."""
    logger.info(f"Searching with keyword: {search_term}")
    
    if stop_flag and (stop_flag() or check_stop_signal(job_id)):
        logger.info(f"Job {job_id} stopped during search_and_answer")
        return ""
    
    async with aiohttp.ClientSession() as session:
        google_search_result = google_search.list(q=search_term, cx=GOOGLE_CSE_ID).execute()
        urls = [result["link"] for result in google_search_result.get("items", [])]
        
        for url in urls:
            if stop_flag and (stop_flag() or check_stop_signal(job_id)):
                logger.info(f"Job {job_id} stopped during URL processing")
                return ""
            result = await fetch_and_analyze(session, url, table, sub_question, job_id, row_idx, col_idx, logger, stop_flag)
            if result:
                return result
    
    logger.info(f"No answer found for keyword: {search_term}")
    return ""

def search_row_header(search_term, table, sub_question, logger: logging.Logger):
    """Search the Web and obtain a list of web results."""
    logger.info(f"Searching with keyword: {search_term}")
    
    google_search_result = google_search.list(q=search_term, cx=GOOGLE_CSE_ID).execute()
    urls = []
    search_chunk = {}
    for result in google_search_result["items"]:
        urls.append(result["link"])
    for url in urls:
        search_url = f'https://r.jina.ai/{url}'
        headers = {
            "Authorization": f"Bearer {JINA_API_KEY}"
        }
        try:
            response = requests.get(search_url, headers=headers)
            if response.status_code == 200:
              search_result = response.text
              answer = analyse_result(search_result, table, sub_question, url, logger=logger)
              if answer != "":
                logger.info(f"Answer found: {answer}")
                return answer
            else:
                print(f"Jina returned an error: {response.status_code} for URL: {url}")
        except requests.exceptions.RequestException as e:
            print(f"Error fetching URL {url}: {str(e)}")
    return ""

async def fetch_and_analyze(session, url, table, sub_question, job_id, row_idx, col_idx, logger: logging.Logger, stop_flag=None):
    search_url = f'https://r.jina.ai/{url}'
    headers = {"Authorization": f"Bearer {JINA_API_KEY}"}
    try:
        if stop_flag and (stop_flag() or check_stop_signal(job_id)):
            logger.info(f"Job {job_id} stopped before fetching URL")
            return ""
        async with session.get(search_url, headers=headers, timeout=50) as response:
            if response.status == 200:
                search_result = await response.text()
                if stop_flag and (stop_flag() or check_stop_signal(job_id)):
                    logger.info(f"Job {job_id} stopped after fetching URL")
                    return ""
                answer = analyse_result(search_result, table, sub_question, url, logger)
                if answer:
                    logger.info(f"Answer found: {answer}")
                    # Update the table immediately
                    table["data"][row_idx][col_idx] = answer
                    with FileLock(f"jobs/{job_id}/table.json.lock"):
                        with open(f"jobs/{job_id}/table.json", "w") as f:
                            json.dump(table, f, indent=2)
                    logger.info(f"Updated cell [{row_idx}, {col_idx}] with value: {answer}")
                    return answer
            else:
                logger.warning(f"Jina returned an error: {response.status} for URL: {url}")
    except asyncio.TimeoutError:
        logger.warning(f"Timeout fetching URL {url}")
    except Exception as e:
        logger.error(f"Error fetching URL {url}: {str(e)}")
    return ""

def initialize_row_headers(user_input: str, table_json: dict, job_id: str, logger: logging.Logger) -> dict:
    logger.info("Initializing Row Headers")

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
        header_question = generate_row_header_subquestion(user_input, table_json, logger)
        logger.info(f"Generated sub-question for row headers: {header_question}")

        # Generate keywords using existing function
        header_keywords = generate_keywords(user_input, header_question, logger)

        # Search and analyze results using existing function
        for keyword in header_keywords:
            logger.info(f"Searching with keyword: {keyword}")
            result = search_row_header(keyword, table_json, header_question, logger)
            # result = await search_and_answer(keyword, job_id, table_json, header_question, 0, 0, logger, True)

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
                logger.info(f"Row header entities: {entities}")
                table_json["headers"]["rows"] = entities[:len(table_json["data"])]
                logger.info(f"Updated table: {table_json}")
                break  # Exit after finding and updating headers

    return table_json

def analyse_result(search_result: str, markdown_table: str, sub_question: str, url: str, logger: logging.Logger) -> List[str]:
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

def generate_cell_subquestion(row_header: str, col_header: str, table_json: dict, logger: logging.Logger) -> str:
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

    question = response.choices[0].message.parsed.question
    logger.info(f"Generated cell sub-question: {question}")
    return question

def update_cell_value(table_json: dict, row_idx: int, col_idx: int, value: str) -> dict:
    """Update a specific cell in the table with a new value."""
    table_json["data"][row_idx][col_idx] = value
    return table_json

async def process_cell(row_idx: int, col_idx: int, user_input: str, table_json: dict, job_id: str, logger: logging.Logger, stop_flag) -> Tuple[int, int, str]:
    try:
        row_header = table_json["headers"]["rows"][row_idx]
        col_header = table_json["headers"]["columns"][col_idx]
    except IndexError:
        logger.error(f"Index out of range for row {row_idx} or column {col_idx}")
        return row_idx, col_idx, "Error"

    logger.info(f"Processing cell: {row_header} x {col_header}")

    if stop_flag() or check_stop_signal(job_id):
        logger.info(f"Job {job_id} stopped before generating sub-question")
        return row_idx, col_idx, "Stopped"

    sub_question = generate_cell_subquestion(row_header, col_header, table_json, logger)
    logger.info(f"Generated sub-question: {sub_question}")

    if stop_flag() or check_stop_signal(job_id):
        logger.info(f"Job {job_id} stopped before generating keywords")
        return row_idx, col_idx, "Stopped"

    keywords = generate_keywords(user_input, sub_question, logger)
    logger.info(f"Keywords: {keywords}")

    batch_size = 4
    for i in range(0, len(keywords), batch_size):
        if stop_flag() or check_stop_signal(job_id):
            logger.info(f"Job {job_id} stopped before processing batch")
            return row_idx, col_idx, "Stopped"

        batch = keywords[i:i+batch_size]
        logger.info(f"Processing batch: {batch}")
        
        for search_term in batch:
            if stop_flag() or check_stop_signal(job_id):
                logger.info(f"Job {job_id} stopped before search_and_answer")
                return row_idx, col_idx, "Stopped"

            result = await search_and_answer(search_term, job_id, table_json, sub_question, row_idx, col_idx, logger, stop_flag=stop_flag)
            if result:
                logger.info(f"Found answer: {result}")
                return row_idx, col_idx, result
        
        logger.info(f"No answer found for batch: {batch}")
    
    logger.info("No answer found for this cell")
    return row_idx, col_idx, ""

async def process_empty_cells(user_input: str, table_json: dict, job_id: str, logger: logging.Logger, stop_flag) -> dict:
    logger.info("Processing empty cells in parallel")

    empty_cells = find_all_empty_cells(table_json)
    if not empty_cells:
        logger.info("No empty cells - table is complete")
        return table_json

    async def process_cell_wrapper(row_idx: int, col_idx: int):
        if stop_flag():
            return row_idx, col_idx, "Stopped"
        result = await process_cell(row_idx, col_idx, user_input, table_json, job_id, logger, stop_flag)
        if stop_flag():
            return row_idx, col_idx, "Stopped"
        return result

    tasks = [process_cell_wrapper(row_idx, col_idx) for row_idx, col_idx in empty_cells]
    
    for task in asyncio.as_completed(tasks):
        row_idx, col_idx, result = await task
        if result == "Stopped" or stop_flag():
            logger.info("Job stopped while processing cells")
            return table_json
        table_json["data"][row_idx][col_idx] = result

    logger.info("Finished processing all cells")
    return table_json

def check_stop_signal(job_id: str):
    return os.path.exists(f"jobs/{job_id}/stop_signal")

def process_job(user_input: str, job_id: str, stop_flag):
    logger = setup_job_logger(job_id)
    logger.info(f"Starting job {job_id} with user input: {user_input}")
    
    def check_stop():
        return stop_flag() or check_stop_signal(job_id)
    
    # Generate table
    initial_table = generate_table(user_input, job_id, logger)
    if check_stop():
        logger.info(f"Job {job_id} stopped after generating initial table")
        return initial_table

    # Initialize row headers
    updated_table = initialize_row_headers(user_input, initial_table, job_id, logger)
    with FileLock(f"jobs/{job_id}/table.json.lock"):
        with open(f"jobs/{job_id}/table.json", "w") as f:
            json.dump(updated_table, f, indent=2)
    logger.info(f"Updated table file with headers")
    if check_stop():
        logger.info(f"Job {job_id} stopped after initializing row headers")
        return updated_table

    # Process empty cells in parallel
    completed_table = asyncio.run(process_empty_cells(user_input, updated_table, job_id, logger, check_stop))
    
    if check_stop():
        logger.info(f"Job {job_id} stopped during cell processing")
    else:
        logger.info(f"Job {job_id} completed")
    
    # Ensure row headers are correctly set
    for i, row in enumerate(completed_table["data"]):
        if i < len(completed_table["headers"]["rows"]):
            row.insert(0, completed_table["headers"]["rows"][i])
    
    # Update the column headers to include the row header column
    completed_table["headers"]["columns"].insert(0, "Entity")
    
    logger.info(f"Final table: {json.dumps(completed_table, indent=2)}")
    
    # Remove the stop signal file if it exists
    if os.path.exists(f"jobs/{job_id}/stop_signal"):
        os.remove(f"jobs/{job_id}/stop_signal")
    
    return completed_table
