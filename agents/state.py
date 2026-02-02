
from typing import Annotated, TypedDict, List, Union, Optional
import numpy as np

class AgentState(TypedDict):
    # Image path or numpy array
    image_path: str
    image_data: Optional[np.ndarray]
    
    # User intent
    user_query: str
    
    # Segmentation data
    masks: List[dict]
    analysis_results: Optional[object] # Pandas DataFrame
    
    # Feedback and loop control
    feedback: str
    is_satisfied: bool
    
    # Training related
    training_folder: str
    training_status: str
    
    # Errors or logs
    logs: List[str]
