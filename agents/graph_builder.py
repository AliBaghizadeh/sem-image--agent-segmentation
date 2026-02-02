
from langgraph.graph import StateGraph, END
from agents.state import AgentState
from core.matsam.matsam_model import MatSAMModel
from core.measurements.grain_analysis import GrainAnalyzer

# Initialize models (in a real app, these might be singletons)
matsam = None # Initialize lazily
analyzer = GrainAnalyzer()

def interpret_query_node(state: AgentState):
    """
    Analyzes the user query to decide if it's a request for analysis, training, or correction.
    """
    query = state['user_query'].lower()
    logs = state.get('logs', [])
    logs.append(f"Interpreting query: {query}")
    
    if "train" in query or "finetune" in query:
        return {"next_node": "training_node", "logs": logs}
    else:
        return {"next_node": "segmentation_node", "logs": logs}

def segmentation_node(state: AgentState):
    """
    Performs MatSAM segmentation.
    """
    global matsam
    if matsam is None:
        matsam = MatSAMModel()
    
    logs = state.get('logs', [])
    logs.append("Running MatSAM auto-segmentation...")
    
    masks = matsam.generate_auto_masks(state['image_path'])
    return {"masks": masks, "logs": logs}

def analysis_node(state: AgentState):
    """
    Analyzes the masks and filters based on user query.
    """
    masks = state['masks']
    df = analyzer.analyze_masks(masks)
    
    # Simple heuristic for filtering
    # In a full version, an LLM would convert LLM query to df.query()
    filtered_df = df
    if "larger than" in state['user_query']:
        # Mock logic
        filtered_df = analyzer.filter_grains(df, "area_um2 > 100")
        
    return {"analysis_results": filtered_df}

def training_node(state: AgentState):
    """
    Autonomous trainer agent logic.
    """
    return {"training_status": "Training initiated on folder: " + state.get('training_folder', 'data/')}

# Building the Graph
workflow = StateGraph(AgentState)

workflow.add_node("interpreter", interpret_query_node)
workflow.add_node("segmentor", segmentation_node)
workflow.add_node("analyzer", analysis_node)
workflow.add_node("trainer", training_node)

workflow.set_entry_point("interpreter")

# Conditional edges
def route_after_interpretation(state):
    return state.get("next_node", "segmentor")

workflow.add_conditional_edges(
    "interpreter",
    route_after_interpretation,
    {
        "segmentation_node": "segmentor",
        "training_node": "trainer"
    }
)

workflow.add_edge("segmentor", "analyzer")
workflow.add_edge("analyzer", END)
workflow.add_edge("trainer", END)

app_graph = workflow.compile()
