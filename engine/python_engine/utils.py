def load_problem_module(problem_type: str) -> Problem:
    """
    Dynamically load a Problem module by its type name.
    
    Args:
        problem_type: String identifier for the problem (e.g., "vector_addition")
        
    Returns:
        An instantiated Problem subclass
    
    Raises:
        HTTPException: If the problem type cannot be found or loaded
    """
    try:
        # Convert problem_type to proper module and class names
        # e.g., "vector_addition" -> "vector_addition" (module) and "VectorAdditionProblem" (class)
        module_name = problem_type.lower()
        class_name = ''.join(word.capitalize() for word in problem_type.split('_')) + 'Problem'
        
        # Import the module dynamically
        module = importlib.import_module(module_name)
        
        # Get the problem class and instantiate it
        problem_class = getattr(module, class_name)
        return problem_class()
    
    except (ImportError, AttributeError) as e:
        raise HTTPException(
            status_code=404, 
            detail=f"Problem type '{problem_type}' not found: {str(e)}"
        )