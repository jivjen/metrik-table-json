from fastapi import FastAPI
from pydantic import BaseModel
import asyncio
from typing import Dict, Any
from researcher import generate_table, initialize_row_headers, process_empty_cells, display_final_table, setup_logging
import logging
import uvicorn

app = FastAPI()

class JobInput(BaseModel):
    user_input: str

class JobStatus(BaseModel):
    status: str
    result: Dict[str, Any]

jobs: Dict[str, Dict[str, Any]] = {}
job_queue = asyncio.Queue()

async def job_worker():
    logging.info("Job worker started")
    while True:
        try:
            logging.info(f"Job worker waiting for new job. Queue size: {job_queue.qsize()}")
            job_id, user_input = await asyncio.wait_for(job_queue.get(), timeout=1.0)
            logging.info(f"Job worker received job {job_id}")
            await run_job(job_id, user_input)
        except asyncio.TimeoutError:
            logging.debug("Job worker timeout, continuing...")
            continue
        except asyncio.CancelledError:
            logging.info("Job worker cancelled")
            break
        except Exception as e:
            logging.error(f"Unexpected error in job_worker: {str(e)}")
        finally:
            if 'job_id' in locals():
                job_queue.task_done()
                logging.info(f"Job {job_id} processing completed")
    logging.info("Job worker stopped")

async def run_job(job_id: str, user_input: str):
    logger = setup_logging(job_id)
    logger.info(f"Starting job {job_id} with user input: {user_input}")
    jobs[job_id]["status"] = "running"
    
    try:
        async with asyncio.timeout(3600):  # 1 hour timeout
            logger.info("Generating initial table...")
            jobs[job_id]["status"] = "generating_initial_table"
            initial_table = await generate_table(user_input, job_id)
            jobs[job_id]["result"] = initial_table
            jobs[job_id]["status"] = "initial_table_generated"
            logger.info("Initial table generated")
            
            logger.info("Initializing row headers...")
            jobs[job_id]["status"] = "initializing_row_headers"
            updated_table = await initialize_row_headers(user_input, initial_table, job_id)
            jobs[job_id]["result"] = updated_table
            jobs[job_id]["status"] = "row_headers_initialized"
            logger.info("Row headers initialized")
            
            logger.info("Processing empty cells...")
            jobs[job_id]["status"] = "processing_empty_cells"
            completed_table = await process_empty_cells(user_input, updated_table, job_id)
            jobs[job_id]["result"] = completed_table
            jobs[job_id]["status"] = "completed"
            logger.info("Empty cells processed")
            
            logger.info(f"Job {job_id} completed")
    except asyncio.TimeoutError:
        logger.error(f"Job {job_id} timed out after 1 hour")
        jobs[job_id]["status"] = "timeout"
        jobs[job_id]["error"] = "Job timed out after 1 hour"
    except Exception as e:
        logger.error(f"Error in job {job_id}: {str(e)}")
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(e)

@app.post("/start_job")
async def start_job(job_input: JobInput):
    job_id = str(len(jobs) + 1)
    jobs[job_id] = {"status": "queued", "result": {}, "progress": 0, "current_step": "Queued"}
    logging.info(f"New job created with ID: {job_id}")
    asyncio.create_task(run_job(job_id, job_input.user_input))
    logging.info(f"Job {job_id} started")
    return {"job_id": job_id}

@app.get("/poll_status/{job_id}")
async def poll_status(job_id: str):
    if job_id not in jobs:
        return {"error": "Job not found"}
    
    job = jobs[job_id]
    markdown_table = ""
    
    if job["result"] and job["status"] == "completed":
        markdown_table = display_final_table(job["result"], "markdown")
    
    return JobStatus(
        status=job["status"],
        result={
            "table": markdown_table,
            "progress": job.get("progress", 0),
            "current_step": job.get("current_step", ""),
            **job["result"]
        }
    )

async def run_job(job_id: str, user_input: str):
    logger = setup_logging(job_id)
    logger.info(f"Starting job {job_id} with user input: {user_input}")
    jobs[job_id]["status"] = "running"
    
    try:
        async with asyncio.timeout(3600):  # 1 hour timeout
            logger.info("Generating initial table...")
            jobs[job_id]["status"] = "generating_initial_table"
            jobs[job_id]["current_step"] = "Generating initial table"
            jobs[job_id]["progress"] = 10
            initial_table = await generate_table(user_input, job_id)
            jobs[job_id]["result"] = initial_table
            jobs[job_id]["status"] = "initial_table_generated"
            jobs[job_id]["progress"] = 30
            logger.info("Initial table generated")
            
            logger.info("Initializing row headers...")
            jobs[job_id]["status"] = "initializing_row_headers"
            jobs[job_id]["current_step"] = "Initializing row headers"
            jobs[job_id]["progress"] = 50
            updated_table = await initialize_row_headers(user_input, initial_table, job_id)
            jobs[job_id]["result"] = updated_table
            jobs[job_id]["status"] = "row_headers_initialized"
            jobs[job_id]["progress"] = 70
            logger.info("Row headers initialized")
            
            logger.info("Processing empty cells...")
            jobs[job_id]["status"] = "processing_empty_cells"
            jobs[job_id]["current_step"] = "Processing empty cells"
            jobs[job_id]["progress"] = 80
            completed_table = await process_empty_cells(user_input, updated_table, job_id)
            jobs[job_id]["result"] = completed_table
            jobs[job_id]["status"] = "completed"
            jobs[job_id]["progress"] = 100
            jobs[job_id]["current_step"] = "Completed"
            logger.info("Empty cells processed")
            
            logger.info(f"Job {job_id} completed")
    except asyncio.TimeoutError:
        logger.error(f"Job {job_id} timed out after 1 hour")
        jobs[job_id]["status"] = "timeout"
        jobs[job_id]["error"] = "Job timed out after 1 hour"
    except Exception as e:
        logger.error(f"Error in job {job_id}: {str(e)}")
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(e)

worker_task = None

@app.on_event("startup")
async def startup_event():
    global worker_task
    loop = asyncio.get_event_loop()
    worker_task = loop.create_task(job_worker())
    logging.info("Job worker task created and started")
    
    # Add a check to ensure the worker is running
    await asyncio.sleep(1)  # Give the worker a moment to start
    if not worker_task.done():
        logging.info("Job worker is running")
    else:
        logging.error("Job worker failed to start")

@app.on_event("shutdown")
async def shutdown_event():
    if worker_task:
        worker_task.cancel()
        try:
            await worker_task
        except asyncio.CancelledError:
            pass

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    uvicorn.run("api:app", host="0.0.0.0", port=8000, workers=1, log_level="info")
