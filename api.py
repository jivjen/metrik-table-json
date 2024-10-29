from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
import asyncio
import json
import os
from typing import Dict, Any
from researcher import generate_table, initialize_row_headers, process_empty_cells, display_final_table

app = FastAPI()

class JobInput(BaseModel):
    user_input: str

class JobStatus(BaseModel):
    status: str
    result: Dict[str, Any]

jobs: Dict[str, Dict[str, Any]] = {}

async def run_job(job_id: str, user_input: str):
    jobs[job_id]["status"] = "running"
    
    initial_table = await generate_table(user_input, job_id)
    jobs[job_id]["result"] = initial_table
    
    updated_table = await initialize_row_headers(user_input, initial_table, job_id)
    jobs[job_id]["result"] = updated_table
    
    completed_table = await process_empty_cells(user_input, updated_table, job_id)
    jobs[job_id]["result"] = completed_table
    
    jobs[job_id]["status"] = "completed"

@app.post("/start_job")
async def start_job(job_input: JobInput):
    job_id = str(len(jobs) + 1)
    jobs[job_id] = {"status": "starting", "result": {}}
    asyncio.create_task(run_job(job_id, job_input.user_input))
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
    uvicorn.run(app, host="0.0.0.0", port=8000)
