# Copied from eva_p1_workflow_and_agent.py
import json
from typing import Dict, Any
from eva_env_base import log

def validate_workflow_nodes(workflow: Dict[str, Any]) -> bool:
    """Validate that workflow has required nodes"""
    required_nodes = ["72", "74", "78", "80", "81", "88", "89"]
    missing_nodes = []

    for node_id in required_nodes:
        if node_id not in workflow:
            missing_nodes.append(node_id)

    if missing_nodes:
        log.error(f"Missing workflow nodes: {missing_nodes}")
        return False

    log.info("✅ Workflow validation passed")
    return True


def apply_enhanced_params_to_workflow(base: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """Apply enhanced parameters to ComfyUI workflow"""
    wf = json.loads(json.dumps(base))

    # Apply prompt
    if "prompt" in params and "89" in wf:
        wf["89"]["inputs"]["text"] = params["prompt"]

    # Apply negative prompt
    if "negative_prompt" in params and "72" in wf:
        wf["72"]["inputs"]["text"] = params["negative_prompt"]

    # Apply video settings
    fps = int(params.get("fps", 20))
    seconds = float(params.get("seconds", 5.0))

    if "88" in wf:
        wf["88"]["inputs"]["fps"] = fps

    if "74" in wf:
        wf["74"]["inputs"]["length"] = int(max(1, round(fps * seconds)))
        if "width" in params and "height" in params:
            wf["74"]["inputs"]["width"] = int(params["width"])
            wf["74"]["inputs"]["height"] = int(params["height"])

    # Apply sampler settings
    if params.get("sampler") or params.get("scheduler"):
        sampler = params["sampler"]
        scheduler = params["scheduler"]

        # High resolution pass
        if "81" in wf:
            wf["81"]["inputs"]["sampler_name"] = sampler
            wf["81"]["inputs"]["scheduler"] = scheduler
            if "steps" in params:
                wf["81"]["inputs"]["steps"] = int(params["steps"])
            if "cfg_scale" in params:
                wf["81"]["inputs"]["cfg"] = float(params["cfg_scale"])

        # Low resolution pass
        if "78" in wf:
            wf["78"]["inputs"]["sampler_name"] = sampler
            wf["78"]["inputs"]["scheduler"] = scheduler
            if "steps" in params:
                wf["78"]["inputs"]["steps"] = int(params["steps"])  
            if "cfg_scale" in params:
                wf["78"]["inputs"]["cfg"] = float(params["cfg_scale"])

    # Apply seeds
    if "seed_high" in params and "81" in wf:
        wf["81"]["inputs"]["noise_seed"] = int(params["seed_high"])
    if "seed_low" in params and "78" in wf:
        wf["78"]["inputs"]["noise_seed"] = int(params["seed_low"])

    # Apply output prefix to all output nodes we know about
    prefix = params.get("prefix")
    if prefix:
        for node_id in ("80", "90", "100", "110"):
            if node_id in wf and isinstance(wf[node_id].get("inputs"), dict) and "filename_prefix" in wf[node_id]["inputs"]:
                wf[node_id]["inputs"]["filename_prefix"] = prefix

    return wf

