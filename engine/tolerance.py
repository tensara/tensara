from fastapi import FastAPI, Request
import modal
from pathlib import Path
import utils
import inspect
import torch

runtime_image = (
    modal.Image.from_registry("nvidia/cuda:13.1.0-runtime-ubuntu24.04", add_python="3.13")
    .pip_install("torch", "numpy", "fastapi")
    .add_local_python_source("utils", "problem")
)

app = modal.App("tolerance", image=runtime_image)
web_app = FastAPI()


@app.function(
    image=runtime_image,
    gpu=modal.gpu.H100(count=1),
    timeout=600,
)
def compute_tolerances_for_problem(
    problem_name: str, problem_def: str, percentile: float = 10.0
) -> dict:
    """Compute tolerances for a problem definition."""
    try:
        problem = utils.load_problem_module(problem_name, problem_def)
        test_cases = problem.generate_test_cases(torch.float32)
        
        sig = inspect.signature(problem.reference_solution)
        has_dtype_param = "dtype" in sig.parameters
        
        rtol_list = []
        atol_list = []
        test_case_stats = []
        
        for tc in test_cases:
            fp32_inputs_tuple = tc["create_inputs"]()
            fp32_inputs = list(fp32_inputs_tuple) if isinstance(fp32_inputs_tuple, tuple) else [fp32_inputs_tuple]
            
            # FP32 output
            if has_dtype_param:
                x_t = problem.reference_solution(*fp32_inputs, dtype=torch.float32)
            else:
                x_t = problem.reference_solution(*fp32_inputs)
            
            # FP16 output
            fp16_inputs = [inp.to(torch.float16) if isinstance(inp, torch.Tensor) else inp for inp in fp32_inputs]
            if has_dtype_param:
                y_t_fp16 = problem.reference_solution(*fp16_inputs, dtype=torch.float16)
            else:
                torch.backends.cudnn.enabled = False
                y_t_fp16 = problem.reference_solution(*fp16_inputs)
            
            y_t = y_t_fp16.to(torch.float32)
            
            rtol, atol = utils.compute_tolerances(x_t, y_t, percentile)
            rtol_list.append(rtol)
            atol_list.append(atol)
            
            test_case_stats.append({
                "name": tc["name"],
                "rtol": rtol,
                "atol": atol,
            })
        
        return {
            "problem_name": problem_name,
            "percentile": percentile,
            "rtol_t": rtol_list,
            "atol_t": atol_list,
            "test_case_stats": test_case_stats,
        }
    except Exception as e:
        return {
            "error": str(e),
            "problem_name": problem_name,
            "percentile": percentile,
        }


@web_app.post("/compute-tolerances")
async def compute_tolerances_endpoint(request: Request):
    req = await request.json()
    
    problem_def = req.get("problem_def")
    problem_name = req.get("problem_name")
    percentile = req.get("percentile", 10.0)
    
    if not problem_def:
        return {"error": "problem_def is required"}
    if not problem_name:
        return {"error": "problem_name is required"}
    
    try:
        percentile = float(percentile)
    except (ValueError, TypeError):
        return {"error": "percentile must be a number"}
    
    try:
        result = compute_tolerances_for_problem.remote(problem_name, problem_def, percentile)
        return result
    except Exception as e:
        return {"error": str(e)}


@app.function()
@modal.asgi_app()
def fastapi_app():
    return web_app
