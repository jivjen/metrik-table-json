from fastapi import FastAPI, Depends
from pydantic import BaseModel
import uuid
import json
import os
from filelock import FileLock
from multiprocessing import Process, Manager, freeze_support
from researcherv2 import process_job

app = FastAPI()

class UserInput(BaseModel):
    query: str

class JobStatus(BaseModel):
    status: str
    table: dict = None

# Shared job status storage
manager = None
job_statuses = None

def initialize_manager():
    global manager, job_statuses
    if manager is None:
        manager = Manager()
        job_statuses = manager.dict()

def get_job_statuses():
    if job_statuses is None:
        initialize_manager()
    return job_statuses

@app.post("/start_job")
async def start_job(user_input: UserInput, job_statuses=Depends(get_job_statuses)):
    job_id = str(uuid.uuid4())
    os.makedirs(f"jobs/{job_id}", exist_ok=True)
    
    job_statuses[job_id] = "STARTED"
    
    # Start the job in a separate process
    Process(target=run_job, args=(job_id, user_input.query, job_statuses)).start()
    
    return {"job_id": job_id}

@app.get("/poll_status/{job_id}")
async def poll_status(job_id: str, job_statuses=Depends(get_job_statuses)):
    if job_id not in job_statuses:
        return {"status": "NOT_FOUND"}
    
    status = job_statuses[job_id]
    
    try:
        with FileLock(f"jobs/{job_id}/table.json.lock"):
            with open(f"jobs/{job_id}/table.json", "r") as f:
                table = json.load(f)
        
        return JobStatus(status=status, table=table)
    except FileNotFoundError:
        return JobStatus(status=status, table=None)

def run_job(job_id: str, user_input: str, job_statuses):
    try:
        job_statuses[job_id] = "PROCESSING"
        process_job(user_input, job_id)
        job_statuses[job_id] = "COMPLETED"
    except Exception as e:
        job_statuses[job_id] = "ERROR"
        print(f"Error processing job {job_id}: {str(e)}")

if __name__ == "__main__":
    freeze_support()
    initialize_manager()
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
