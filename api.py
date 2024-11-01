from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uuid
import json
import os
import logging
from filelock import FileLock
from multiprocessing import Process, Manager, freeze_support
from researcherv2 import process_job
from threading import Event
from datetime import datetime
import signal

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UserInput(BaseModel):
    query: str

from typing import Optional

class JobStatus(BaseModel):
    status: str
    table: Optional[dict] = None

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
    
    job_statuses[job_id] = {"status": "STARTED", "start_time": datetime.now(), "stop_flag": False}
    
    # Start the job in a separate process
    Process(target=run_job, args=(job_id, user_input.query, job_statuses)).start()
    
    return {"job_id": job_id}

@app.get("/poll_status/{job_id}")
async def poll_status(job_id: str, job_statuses=Depends(get_job_statuses)):
    if job_id not in job_statuses:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_info = job_statuses[job_id]
    status = job_info["status"]
    
    table = None
    if status != "STARTED":
        try:
            with FileLock(f"jobs/{job_id}/table.json.lock"):
                with open(f"jobs/{job_id}/table.json", "r") as f:
                    table = json.load(f)
        except FileNotFoundError:
            logger.warning(f"Table file not found for job {job_id}")
    
    return JobStatus(status=status, table=table)

@app.post("/stop_job/{job_id}")
async def stop_job(job_id: str, job_statuses=Depends(get_job_statuses)):
    if job_id not in job_statuses:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job_statuses[job_id]["status"] in ["COMPLETED", "ERROR"]:
        return {"message": f"Job {job_id} has already finished"}
    
    job_statuses[job_id]["stop_flag"] = True
    job_statuses[job_id]["status"] = "STOPPED"
    return {"message": f"Job {job_id} has been stopped"}

def run_job(job_id: str, user_input: str, job_statuses):
    try:
        job_statuses[job_id]["status"] = "PROCESSING"
        process_job(user_input, job_id, lambda: job_statuses[job_id]["stop_flag"])
        if not job_statuses[job_id]["stop_flag"]:
            job_statuses[job_id]["status"] = "COMPLETED"
    except Exception as e:
        job_statuses[job_id]["status"] = "ERROR"
        logger.error(f"Error processing job {job_id}: {str(e)}", exc_info=True)
    finally:
        job_statuses[job_id]["end_time"] = datetime.now()

if __name__ == "__main__":
    freeze_support()
    initialize_manager()
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
