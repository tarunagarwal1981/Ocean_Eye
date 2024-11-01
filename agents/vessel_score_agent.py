from typing import Dict, Tuple
import pandas as pd
from utils.database_utils import fetch_data_from_db

def analyze_vessel_score(vessel_name: str) -> Tuple[Dict[str, float], str]:
    """
    Analyze vessel score and component scores for a given vessel.
    
    Args:
        vessel_name (str): Name of the vessel
        
    Returns:
        Tuple[Dict[str, float], str]: Dictionary containing scores and analysis text
    """
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
    
    try:
        score_data = fetch_data_from_db(score_query)
        
        if not score_data.empty:
            scores = {
                'vessel_score': float(score_data.iloc[0]['Vessel Score']),
                'cost_score': float(score_data.iloc[0]['Cost']),
                'digitalization_score': float(score_data.iloc[0]['Digitalization']),
                'environment_score': float(score_data.iloc[0]['Environment']),
                'operation_score': float(score_data.iloc[0]['Operation']),
                'reliability_score': float(score_data.iloc[0]['Reliability'])
            }
            
            # Generate analysis text
            analysis = generate_vessel_score_analysis(vessel_name, scores)
            
            return scores, analysis
        else:
            return {}, "No vessel score data available"
            
    except Exception as e:
        return {}, f"Error analyzing vessel score: {str(e)}"

def generate_vessel_score_analysis(vessel_name: str, scores: Dict[str, float]) -> str:
    """
    Generate detailed analysis of vessel scores.
    
    Args:
        vessel_name (str): Name of the vessel
        scores (Dict[str, float]): Dictionary containing various scores
        
    Returns:
        str: Detailed analysis text
    """
    analysis_points = []
    
    # Overall score analysis
    if scores['vessel_score'] >= 75:
        analysis_points.append(f"{vessel_name} demonstrates excellent overall performance with a score of {scores['vessel_score']:.1f}%")
    elif scores['vessel_score'] >= 60:
        analysis_points.append(f"{vessel_name} shows adequate overall performance with a score of {scores['vessel_score']:.1f}%")
    else:
        analysis_points.append(f"{vessel_name} requires attention with a below-target score of {scores['vessel_score']:.1f}%")

    # Identify strongest and weakest areas
    score_components = {
        'Cost': scores['cost_score'],
        'Digitalization': scores['digitalization_score'],
        'Environment': scores['environment_score'],
        'Operation': scores['operation_score'],
        'Reliability': scores['reliability_score']
    }
    
    strongest = max(score_components.items(), key=lambda x: x[1])
    weakest = min(score_components.items(), key=lambda x: x[1])
    
    analysis_points.append(f"Strongest performance in {strongest[0]} ({strongest[1]:.1f}%)")
    analysis_points.append(f"Most improvement needed in {weakest[0]} ({weakest[1]:.1f}%)")
    
    # Generate recommendations
    recommendations = []
    for component, score in score_components.items():
        if score < 60:
            recommendations.append(f"Critical attention needed for {component}")
        elif score < 75:
            recommendations.append(f"Improvement recommended for {component}")
    
    # Combine all analysis points
    full_analysis = "\n\n".join([
        "## Vessel Score Analysis",
        "\n".join(analysis_points),
        "## Recommendations" if recommendations else "",
        "\n".join(recommendations)
    ])
    
    return full_analysis
