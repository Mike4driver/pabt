from fastapi import APIRouter, Request, HTTPException, Form
from fastapi.responses import HTMLResponse, JSONResponse

from config import templates, logger, BASE_DIR, DEFAULT_MEDIA_SUBDIR # Assuming these are needed and accessible
from database import get_setting, update_setting

router = APIRouter()

@router.get("/", response_class=HTMLResponse, name="settings_page") # Root for /settings/
async def settings_page_route(request: Request):
    return templates.TemplateResponse("settings.html", {
        "request": request, 
        "CWD_DISPLAY_PATH": str(BASE_DIR).replace("\\", "/"), 
        "DEFAULT_MEDIA_SUBDIR_JS": DEFAULT_MEDIA_SUBDIR
    })

@router.get("/config", response_class=JSONResponse)
async def get_current_config_route():
    media_dir_name = get_setting("media_directory_name")
    grid_size = get_setting("media_grid_size")
    per_page = get_setting("per_page")
    autoplay_enabled = get_setting("autoplay_enabled")
    default_muted = get_setting("default_muted")
    autoplay_next = get_setting("autoplay_next")
    auto_replay = get_setting("auto_replay")
    return {
        "media_directory_name": media_dir_name or "media",
        "media_grid_size": grid_size or "medium",
        "per_page": per_page or "20",
        "autoplay_enabled": autoplay_enabled or "true",
        "default_muted": default_muted or "true",
        "autoplay_next": autoplay_next or "false",
        "auto_replay": auto_replay or "false"
    }

@router.post("/config")
async def update_app_config_route(request: Request): # Removed type hint for return to allow TemplateResponse
    try:
        form_data = await request.form()
        new_media_dir_name = form_data.get("media_directory_name")
        new_grid_size = form_data.get("media_grid_size")
        new_per_page = form_data.get("per_page")
        new_autoplay_enabled = form_data.get("autoplay_enabled")
        new_default_muted = form_data.get("default_muted")
        new_autoplay_next = form_data.get("autoplay_next")
        new_auto_replay = form_data.get("auto_replay")
        
        requires_restart = False
        messages = []
        
        if new_media_dir_name:
            if not isinstance(new_media_dir_name, str) or '/' in new_media_dir_name or '\\' in new_media_dir_name:
                return templates.TemplateResponse("_config_message.html", {
                    "request": request, "message": "Invalid media directory name. No slashes allowed.", "message_type": "error"})
            if get_setting("media_directory_name") != new_media_dir_name:
                update_setting("media_directory_name", new_media_dir_name)
                messages.append(f"Media directory changed to '{new_media_dir_name}'")
                requires_restart = True
        
        if new_grid_size and get_setting("media_grid_size") != new_grid_size:
            if new_grid_size not in ["small", "medium", "large"]:
                return templates.TemplateResponse("_config_message.html", {"request": request, "message": "Invalid grid size.", "message_type": "error"})
            update_setting("media_grid_size", new_grid_size); messages.append(f"Grid size to '{new_grid_size}'")
        
        if new_per_page and get_setting("per_page") != new_per_page:
            if new_per_page not in ["10", "20", "30", "50", "100"]:
                return templates.TemplateResponse("_config_message.html", {"request": request, "message": "Invalid per page.", "message_type": "error"})
            update_setting("per_page", new_per_page); messages.append(f"Per page to '{new_per_page}'")

        if new_autoplay_enabled is not None and get_setting("autoplay_enabled") != new_autoplay_enabled:
            if new_autoplay_enabled not in ["true", "false"]:
                return templates.TemplateResponse("_config_message.html", {"request": request, "message": "Invalid autoplay.", "message_type": "error"})
            update_setting("autoplay_enabled", new_autoplay_enabled); messages.append(f"Autoplay {'enabled' if new_autoplay_enabled == 'true' else 'disabled'}")

        if new_default_muted is not None and get_setting("default_muted") != new_default_muted:
            if new_default_muted not in ["true", "false"]:
                return templates.TemplateResponse("_config_message.html", {"request": request, "message": "Invalid default mute.", "message_type": "error"})
            update_setting("default_muted", new_default_muted); messages.append(f"Default mute {'enabled' if new_default_muted == 'true' else 'disabled'}")

        if new_autoplay_next is not None and get_setting("autoplay_next") != new_autoplay_next:
            if new_autoplay_next not in ["true", "false"]:
                return templates.TemplateResponse("_config_message.html", {"request": request, "message": "Invalid autoplay next.", "message_type": "error"})
            update_setting("autoplay_next", new_autoplay_next); messages.append(f"Autoplay next {'enabled' if new_autoplay_next == 'true' else 'disabled'}")

        if new_auto_replay is not None and get_setting("auto_replay") != new_auto_replay:
            if new_auto_replay not in ["true", "false"]:
                return templates.TemplateResponse("_config_message.html", {"request": request, "message": "Invalid auto replay.", "message_type": "error"})
            update_setting("auto_replay", new_auto_replay); messages.append(f"Auto replay {'enabled' if new_auto_replay == 'true' else 'disabled'}")

        if not messages: return templates.TemplateResponse("_config_message.html", {"request": request, "message": "No changes.", "message_type": "info"})
        message = ". ".join(messages) + "."
        if requires_restart: message += " Restart application for media directory changes."
        return templates.TemplateResponse("_config_message.html", {"request": request, "message": message, "message_type": "success"})
    except Exception as e: 
        logger.error(f"Error updating config: {e}")
        return templates.TemplateResponse("_config_message.html", {"request": request, "message": f"Error: {str(e)}", "message_type": "error"})

@router.post("/grid-size")
async def update_grid_size_route(grid_size: str = Form(...)):
    if grid_size not in ["small", "medium", "large"]: raise HTTPException(status_code=400, detail="Invalid grid size.")
    if get_setting("media_grid_size") != grid_size: update_setting("media_grid_size", grid_size)
    return {"message": f"Grid size updated to {grid_size}", "grid_size": grid_size}

@router.post("/per-page")
async def update_per_page_route(per_page: str = Form(...)):
    if per_page not in ["10", "20", "30", "50", "100"]: raise HTTPException(status_code=400, detail="Invalid per page value.")
    if get_setting("per_page") != per_page: update_setting("per_page", per_page)
    return {"message": f"Per page updated to {per_page}", "per_page": per_page}

@router.post("/autoplay-next")
async def update_autoplay_next_route(request: Request, autoplay_next: str = Form(...)):
    if autoplay_next not in ["true", "false"]: raise HTTPException(status_code=400, detail="Invalid autoplay next value.")
    if get_setting("autoplay_next") != autoplay_next: update_setting("autoplay_next", autoplay_next)
    return templates.TemplateResponse("_autoplay_next_status.html", {
        "request": request, "message": f"Autoplay next {'enabled' if autoplay_next == 'true' else 'disabled'}",
        "enabled": autoplay_next == "true" })

@router.post("/auto-replay")
async def update_auto_replay_route(request: Request, auto_replay: str = Form(...)):
    if auto_replay not in ["true", "false"]: raise HTTPException(status_code=400, detail="Invalid auto replay value.")
    if get_setting("auto_replay") != auto_replay: update_setting("auto_replay", auto_replay)
    return templates.TemplateResponse("_auto_replay_status.html", {
        "request": request, "message": f"Auto replay {'enabled' if auto_replay == 'true' else 'disabled'}",
        "enabled": auto_replay == "true" }) 