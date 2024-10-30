# agents/crew_performance_agent.py

from utils.database_utils import fetch_data_from_db
from typing import Tuple, Dict, Optional

def analyze_crew_score(vessel_name: str) -> Tuple[Dict[str, float], str]:
    """
    Analyze crew performance scores and generate insights.
    
    Args:
        vessel_name (str): Name of the vessel
    
    Returns:
        Tuple[Dict[str, float], str]: Dictionary of scores and analysis text
    """
    try:
        # Fetch Crew Scores
        crew_query = """
        select 
            "Crew Skill Index",
            "Capability Index",
            "Competency Index",
            "Collaboration Index",
            "Character Index"
        from
            "crew scorecard"
        order by
            random()
        limit
            1;
        """
        crew_data = fetch_data_from_db(crew_query)
        
        if crew_data.empty:
            return {}, "No crew score data available"
            
        scores = {
            'crew_skill_index': float(crew_data.iloc[0]['Crew Skill Index']),
            'capability_index': float(crew_data.iloc[0]['Capability Index']),
            'competency_index': float(crew_data.iloc[0]['Competency Index']),
            'collaboration_index': float(crew_data.iloc[0]['Collaboration Index']),
            'character_index': float(crew_data.iloc[0]['Character Index'])
        }
        
        # Generate analysis text
        analysis = generate_crew_score_analysis(scores)
        
        return scores, analysis
        
    except Exception as e:
        return {}, f"Error analyzing crew score: {str(e)}"

def generate_crew_score_analysis(scores: Dict[str, float]) -> str:
    """
    Generate analysis text based on crew scores.
    
    Args:
        scores (Dict[str, float]): Dictionary containing crew scores
        
    Returns:
        str: Analysis text
    """
    crew_skill = scores['crew_skill_index']
    
    # Find strongest and weakest areas
    component_scores = {
        'Capability': scores['capability_index'],
        'Competency': scores['competency_index'],
        'Collaboration': scores['collaboration_index'],
        'Character': scores['character_index']
    }
    
    strongest = max(component_scores.items(), key=lambda x: x[1])
    weakest = min(component_scores.items(), key=lambda x: x[1])
    
    # Generate analysis text
    analysis = f"Overall crew skill index is {crew_skill:.1f}%. "
    
    if crew_skill >= 80:
        analysis += "The crew is performing exceptionally well. "
    elif crew_skill >= 70:
        analysis += "The crew is performing adequately but has room for improvement. "
    else:
        analysis += "The crew performance requires attention. "
        
    analysis += f"The strongest area is {strongest[0]} at {strongest[1]:.1f}%, "
    analysis += f"while {weakest[0]} needs development at {weakest[1]:.1f}%. "
    
    # Add recommendations
    if weakest[1] < 70:
        analysis += f"\n\nRecommendations:\n"
        analysis += f"1. Focus training on improving {weakest[0]}\n"
        analysis += "2. Schedule targeted skill development sessions\n"
        analysis += "3. Implement regular performance assessments"
    
    return analysis
