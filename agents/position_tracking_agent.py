# agents/position_tracking_agent.py

import folium
from streamlit_folium import st_folium
import streamlit as st
from typing import Tuple, Optional, Dict
from utils.database_utils import fetch_data_from_db

class PositionTrackingAgent:
    def __init__(self):
        pass

    def get_last_position(self, vessel_name: str) -> Tuple[Optional[float], Optional[float]]:
        """Fetch the last reported position for a vessel."""
        query = f"""
        select
          "LATITUDE",
          "LONGITUDE"
        from
          sf_consumption_logs
        where
          UPPER("VESSEL_NAME") = '{vessel_name.upper()}'
          and "LATITUDE" is not null
          and "LONGITUDE" is not null
        order by
          "REPORT_DATE" desc
        limit
          1;
        """
        
        try:
            position_data = fetch_data_from_db(query)
            if not position_data.empty:
                return (
                    float(position_data.iloc[0]['LATITUDE']),
                    float(position_data.iloc[0]['LONGITUDE'])
                )
            return None, None
        except Exception as e:
            st.error(f"Error fetching position data: {str(e)}")
            return None, None

    def create_vessel_map(self, latitude: float, longitude: float) -> folium.Map:
        """Create a Folium map centered on the vessel's position."""
        m = folium.Map(
            location=[latitude, longitude],
            zoom_start=4,
            tiles='cartodb positron'
        )
        
        folium.Marker(
            [latitude, longitude],
            popup='Vessel Position',
            icon=folium.Icon(color='red', icon='info-sign')
        ).add_to(m)
        
        return m

    def show_position(self, vessel_name: str) -> Dict[str, Optional[float]]:
        """Display the vessel's last reported position with map and coordinates."""
        latitude, longitude = self.get_last_position(vessel_name)
        
        position_data = {
            'latitude': latitude,
            'longitude': longitude
        }
        
        if latitude is not None and longitude is not None:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Latitude", f"{latitude:.4f}°")
            with col2:
                st.metric("Longitude", f"{longitude:.4f}°")
            
            m = self.create_vessel_map(latitude, longitude)
            st_folium(m, height=300, width="100%")
        else:
            st.warning("No position data available for this vessel")
        
        return position_data

    def get_position_analysis(self, vessel_name: str) -> str:
        """Generate analysis text based on vessel position."""
        latitude, longitude = self.get_last_position(vessel_name)
        
        if latitude is None or longitude is None:
            return "Position data not available for this vessel."
            
        # Add any additional analysis logic here
        analysis = f"""
        Last reported position for {vessel_name}:
        - Latitude: {latitude:.4f}°
        - Longitude: {longitude:.4f}°
        """
        
        return analysis
