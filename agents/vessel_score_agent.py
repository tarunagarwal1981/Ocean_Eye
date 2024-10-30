# agents/vessel_performance_agent.py

from utils.database_utils import fetch_data_from_db
from typing import Tuple, Dict, Optional

def analyze_vessel_score(vessel_name: str) -> Tuple[Dict[str, float], str]:
    """
    Analyze vessel score and its components.
    
    Args:
        vessel_name (str): Name of the vessel
    
    Returns:
        Tuple[Dict[str, float], str]: Dictionary of scores and analysis text
    """
    try:
        # Fetch Vessel Score and component scores
        score_query = f"""
        select
          "Vessel Score",
          "Cost",
          "Digitalization",
          "Environment",
          "Operation",
          "Reliability"
        from
          "Vessel Scorecard"
        where
          upper("Vessels") = '{vessel_name.upper()}';
        """
        score_data = fetch_data_from_db(score_query)
        
        if score_data.empty:
            return {}, "No vessel score data available"
            
        scores = {
            'vessel_score': float(score_data.iloc[0]['Vessel Score']),
            'cost_score': float(score_data.iloc[0]['Cost']),
            'digitalization_score': float(score_data.iloc[0]['Digitalization']),
            'environment_score': float(score_data.iloc[0]['Environment']),
            'operation_score': float(score_data.iloc[0]['Operation']),
            'reliability_score': float(score_data.iloc[0]['Reliability'])
        }
        
        # Generate analysis text
        analysis = generate_vessel_score_analysis(scores)
        
        return scores, analysis
        
    except Exception as e:
        return {}, f"Error analyzing vessel score: {str(e)}"

def generate_vessel_score_analysis(scores: Dict[str, float]) -> str:
    """
    Generate analysis text based on vessel scores.
    
    Args:
        scores (Dict[str, float]): Dictionary containing vessel scores
        
    Returns:
        str: Analysis text
    """
    vessel_score = scores['vessel_score']
    
    # Find strongest and weakest areas
    component_scores = {
        'Cost': scores['cost_score'],
        'Digitalization': scores['digitalization_score'],
        'Environment': scores['environment_score'],
        'Operation': scores['operation_score'],
        'Reliability': scores['reliability_score']
    }
    
    strongest = max(component_scores.items(), key=lambda x: x[1])
    weakest = min(component_scores.items(), key=lambda x: x[1])
    
    # Generate analysis text
    analysis = f"Overall vessel score is {vessel_score:.1f}%. "
    
    if vessel_score >= 75:
        analysis += "The vessel is performing well overall. "
    elif vessel_score >= 60:
        analysis += "The vessel's performance is acceptable but has room for improvement. "
    else:
        analysis += "The vessel's performance requires immediate attention. "
        
    analysis += f"The strongest area is {strongest[0]} at {strongest[1]:.1f}%, "
    analysis += f"while {weakest[0]} needs attention at {weakest[1]:.1f}%. "
    
    # Add recommendations
    if weakest[1] < 60:
        analysis += f"\n\nRecommendations:\n"
        analysis += f"1. Prioritize improvements in {weakest[0]} performance\n"
        analysis += "2. Develop action plan to address underperforming areas\n"
        analysis += "3. Schedule regular performance reviews"
    
    return analysis
