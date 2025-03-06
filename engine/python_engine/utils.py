from fastapi import HTTPException, status
import importlib
from typing import List, Dict, Any
from problem import Problem
import problems

def load_problem_module(problem_type: str) -> Problem:
    """
    Load a Problem module from the pre-imported problems module.
    
    Args:
        problem_type: String identifier for the problem (e.g., "matrix_multiplication")
        
    Returns:
        An instantiated Problem subclass
    
    Raises:
        HTTPException: If the problem type cannot be found
    """
    try:
        # Convert problem_type to the appropriate attribute name
        module_name = f"problems.{problem_type}"
        module = importlib.import_module(module_name)
        
        # Get the problem class from the problems module
        problem_class = getattr(module, problem_type)
        return problem_class()
    
    except AttributeError as e:
        raise HTTPException(
            status_code=404, 
            detail=f"Problem type '{problem_type}' not found: {str(e)}"
        )