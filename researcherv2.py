import asyncio
import json
import os
from typing import List, Tuple, Callable
from aiofiles import open as aio_open
from aiofiles.os import makedirs

async def find_all_empty_cells(table_json: dict) -> List[Tuple[int, int]]:
    empty_cells = []
    for row_idx, row in enumerate(table_json['rows']):
        for col_idx, cell in enumerate(row['cells']):
            if cell['value'].strip() == '':
                empty_cells.append((row_idx, col_idx))
    return empty_cells

async def update_cell_value_file(job_id: str, row_idx: int, col_idx: int, value: str):
    file_path = f"job_{job_id}_table.json"
    async with aio_open(file_path, 'r') as f:
        table_json = json.loads(await f.read())
    
    table_json['rows'][row_idx]['cells'][col_idx]['value'] = value
    
    async with aio_open(file_path, 'w') as f:
        await f.write(json.dumps(table_json, indent=2))

async def process_empty_cells(user_input: str, table_json: dict, job_id: str, update_callback: Callable):
    logger = logging.getLogger(f"job_{job_id}")
    logger.info("Starting to process empty cells")

    # Ensure the directory for the job exists
    job_dir = f"job_{job_id}"
    await makedirs(job_dir, exist_ok=True)

    # Save the initial table JSON to a file
    file_path = f"{job_dir}/table.json"
    async with aio_open(file_path, 'w') as f:
        await f.write(json.dumps(table_json, indent=2))

    empty_cells = await find_all_empty_cells(table_json)
    total_cells = len(empty_cells)
    processed_cells = 0

    async def process_cell(row_idx: int, col_idx: int):
        nonlocal processed_cells
        try:
            row_header = table_json['rows'][row_idx]['header']
            col_header = table_json['columns'][col_idx]['header']
            
            sub_question = generate_cell_subquestion(row_header, col_header, table_json)
            logger.info(f"Generated sub-question for cell ({row_idx}, {col_idx}): {sub_question}")
            
            keywords = generate_keywords(user_input, sub_question, job_id)
            logger.info(f"Generated keywords for cell ({row_idx}, {col_idx}): {keywords}")
            
            search_term = " ".join(keywords)
            search_result = await search_and_answer(search_term, job_id, table_json, sub_question)
            logger.info(f"Got search result for cell ({row_idx}, {col_idx})")
            
            markdown_table = display_final_table(table_json)
            analysis = await analyse_result(search_result, markdown_table, sub_question, "")
            logger.info(f"Analyzed result for cell ({row_idx}, {col_idx})")
            
            await update_cell_value_file(job_id, row_idx, col_idx, analysis)
            logger.info(f"Updated cell value for ({row_idx}, {col_idx})")
            
            processed_cells += 1
            progress = (processed_cells / total_cells) * 100
            await update_callback({"status": "processing", "progress": progress})
        except Exception as e:
            logger.error(f"Error processing cell ({row_idx}, {col_idx}): {str(e)}")

    # Process all empty cells in parallel
    tasks = [process_cell(row_idx, col_idx) for row_idx, col_idx in empty_cells]
    await asyncio.gather(*tasks)

    # Load the final table JSON from the file
    async with aio_open(file_path, 'r') as f:
        final_table_json = json.loads(await f.read())

    logger.info("Finished processing all empty cells")
    return final_table_json
