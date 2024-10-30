from fastapi import FastAPI
from pydantic import BaseModel
import asyncio
from typing import Dict, Any
from researcher import generate_table, initialize_row_headers, process_empty_cells, display_final_table, setup_logging
import logging
app = FastAPI()

class JobInput(BaseModel):
    user_input: str

class JobStatus(BaseModel):
    status: str
    result: Dict[str, Any]

jobs: Dict[str, Dict[str, Any]] = {}

async def run_job(job_id: str, user_input: str):
    logger = setup_logging(job_id)
    logger.info(f"Starting job {job_id} with user input: {user_input}")
    jobs[job_id]["status"] = "running"
    
    try:
        logger.info("Generating initial table...")
        initial_table = await generate_table(user_input, job_id)
        jobs[job_id]["result"] = initial_table
        logger.info("Initial table generated")
        
        logger.info("Initializing row headers...")
        updated_table = await initialize_row_headers(user_input, initial_table, job_id)
        jobs[job_id]["result"] = updated_table
        logger.info("Row headers initialized")
        
        logger.info("Processing empty cells...")
        completed_table = await process_empty_cells(user_input, updated_table, job_id)
        jobs[job_id]["result"] = completed_table
        logger.info("Empty cells processed")
        
        jobs[job_id]["status"] = "completed"
        logger.info(f"Job {job_id} completed")
    except Exception as e:
        logger.error(f"Error in job {job_id}: {str(e)}")
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(e)

@app.post("/start_job")
async def start_job(job_input: JobInput):
    job_id = str(len(jobs) + 1)
    jobs[job_id] = {"status": "starting", "result": {}}
    # Run the job in the background
    asyncio.create_task(run_job(job_id, job_input.user_input))
    # Immediately return the job_id without waiting for the job to complete
    return {"job_id": job_id}

@app.get("/poll_status/{job_id}")
async def poll_status(job_id: str):
    if job_id not in jobs:
        return {"error": "Job not found"}
    
    job = jobs[job_id]
    markdown_table = ""
    
    if job["result"]:
        markdown_table = display_final_table(job["result"], "markdown")
    
    return JobStatus(status=job["status"], result={"table": markdown_table, **job["result"]})

if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host="0.0.0.0", port=8000)
