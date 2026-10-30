# Add to your FastAPI main file (where /api/summaries is defined).
# Requires: from pydantic import BaseModel  and  from checklist_service import generate_checklist

from typing import List
from pydantic import BaseModel, Field

from checklist_service import generate_checklist


class ChecklistRequest(BaseModel):
    filenames: List[str] = Field(..., min_length=1)
    language: str = "same"


@app.post("/api/checklist")
def create_checklists(payload: ChecklistRequest):
    checklists = []
    errors = []

    for filename in dict.fromkeys(payload.filenames):  # de-duplicate, keep order
        try:
            checklists.append(generate_checklist(filename, payload.language))
        except Exception as exc:
            errors.append({"document": filename, "error": str(exc)[:300]})

    if not checklists:
        raise HTTPException(status_code=400, detail="No checklist could be created.")

    return {
        "status": "ok" if not errors else "partial",
        "checklists": checklists,
        "errors": errors,
    }