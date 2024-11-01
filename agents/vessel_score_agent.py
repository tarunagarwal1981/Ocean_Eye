from typing import Dict, Tuple
import pandas as pd
import streamlit as st
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

def display_vessel_score(vessel_name: str):
    """
    Display vessel score information in an expander.
    
    Args:
        vessel_name (str): Name of the vessel
    """
    with st.expander("Vessel Score Details", expanded=False):
        scores, analysis = analyze_vessel_score(vessel_name)
        
        if scores:
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.metric("Overall Vessel Score", f"{scores['vessel_score']:.1f}%")
            
            with col2:
                st.markdown(
                    f"""
                    <table>
                        <tr>
                            <th>Component</th>
                            <th>Score</th>
                        </tr>
                        <tr>
                            <td>Cost</td>
                            <td><span class='status-{"good" if scores['cost_score'] >= 75 else "average" if scores['cost_score'] >= 60 else "poor"}'>{scores['cost_score']:.1f}%</span></td>
                        </tr>
                        <tr>
                            <td>Digitalization</td>
                            <td><span class='status-{"good" if scores['digitalization_score'] >= 75 else "average" if scores['digitalization_score'] >= 60 else "poor"}'>{scores['digitalization_score']:.1f}%</span></td>
                        </tr>
                        <tr>
                            <td>Environment</td>
                            <td><span class='status-{"good" if scores['environment_score'] >= 75 else "average" if scores['environment_score'] >= 60 else "poor"}'>{scores['environment_score']:.1f}%</span></td>
                        </tr>
                        <tr>
                            <td>Operation</td>
                            <td><span class='status-{"good" if scores['operation_score'] >= 75 else "average" if scores['operation_score'] >= 60 else "poor"}'>{scores['operation_score']:.1f}%</span></td>
                        </tr>
                        <tr>
                            <td>Reliability</td>
                            <td><span class='status-{"good" if scores['reliability_score'] >= 75 else "average" if scores['reliability_score'] >= 60 else "poor"}'>{scores['reliability_score']:.1f}%</span></td>
                        </tr>
                    </table>
                    """,
                    unsafe_allow_html=True
                )
            
            st.markdown(analysis)
        else:
            st.warning("No vessel score data available")

def get_score_status(score: float) -> str:
    """
    Get the status class for a score.
    
    Args:
        score (float): The score to evaluate
        
    Returns:
        str: Status class (good, average, or poor)
    """
    if score >= 75:
        return "good"
    elif score >= 60:
        return "average"
    else:
        return "poor"
