from fastapi import APIRouter, Request, HTTPException, Response
from fastapi.responses import HTMLResponse, JSONResponse

from config import templates, logger # Assuming logger and templates are accessible via config
from jobs_manager import cleanup_old_jobs, get_background_job, background_jobs, job_lock # job_lock for cancel

router = APIRouter()

@router.get("/monitor", response_class=HTMLResponse, name="background_jobs_page")
async def background_jobs_page_route(request: Request):
    return templates.TemplateResponse("background_jobs.html", {"request": request})

@router.get("/", response_class=JSONResponse) # Path for /jobs
async def get_all_jobs_route():
    cleanup_old_jobs()
    with job_lock: # Assuming job_lock is imported correctly
        jobs = [job.to_dict() for job in background_jobs.values()]
    return {"jobs": jobs}

@router.get("/{job_id}", response_class=JSONResponse)
async def get_job_status_route(job_id: str):
    job = get_background_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job.to_dict()

@router.delete("/{job_id}")
async def cancel_job_route(job_id: str):
    with job_lock: # Assuming job_lock is imported correctly
        if job_id in background_jobs:
            job = background_jobs[job_id]
            if job.status == "running":
                job.fail("Cancelled by user") 
            del background_jobs[job_id]
            return {"message": "Job cancelled"}
    raise HTTPException(status_code=404, detail="Job not found") 