import streamlit as st
import requests
import time
import json
from typing import List, Optional, Dict, Any

# --- Configuration ---
BACKEND_URL = "http://127.0.0.1:8000" # Ensure this matches your FastAPI server address
POLLING_INTERVAL_SECONDS = 2 # How often the frontend polls the backend

# --- Session State Initialization ---
if 'job_id' not in st.session_state:
    st.session_state.job_id = None
if 'analysis_status' not in st.session_state:
    st.session_state.analysis_status = "IDLE"
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'polling_active' not in st.session_state:
    st.session_state.polling_active = False

# --- Helper Functions for API Calls ---

def initiate_analysis(audio_file) -> Optional[str]:
    """Sends the audio file to the backend to start analysis."""
    st.session_state.analysis_status = "UPLOADING"
    st.info("Uploading audio and initiating analysis...")
    files = {'audio_file': (audio_file.name, audio_file.getvalue(), audio_file.type)}
    try:
        response = requests.post(f"{BACKEND_URL}/v1/analysis", files=files)
        response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
        data = response.json()
        st.session_state.job_id = data.get("job_id")
        st.session_state.analysis_status = data.get("status", "PENDING")
        st.success(f"Analysis initiated! Job ID: {st.session_state.job_id}")
        st.session_state.polling_active = True
        return st.session_state.job_id
    except requests.exceptions.RequestException as e:
        st.error(f"Error initiating analysis: {e}")
        st.session_state.analysis_status = "FAILED"
        st.session_state.job_id = None
        st.session_state.polling_active = False
        return None

def get_analysis_results(job_id: str) -> Optional[Dict[str, Any]]:
    """Fetches the current analysis results for a given job ID."""
    try:
        response = requests.get(f"{BACKEND_URL}/v1/analysis/{job_id}")
        response.raise_for_status()
        data = response.json()
        return data
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching analysis results: {e}")
        st.session_state.analysis_status = "FAILED"
        st.session_state.polling_active = False
        return None

# --- UI Layout ---

st.set_page_config(layout="wide", page_title="Breaking Guidance App")

st.title("🕺 Breakin' Beats: Real-time Dance Guidance")
st.markdown("Upload a song and get real-time breaking move recommendations based on the music!")

# Audio Upload Section
st.header("1. Upload Your Music")
uploaded_file = st.file_uploader(
    "Choose an audio file (MP3, WAV, etc.)",
    type=["mp3", "wav", "ogg"],
    help="Max file size 200MB for demo purposes."
)

if uploaded_file and st.button("Start Analysis", disabled=st.session_state.polling_active):
    with st.spinner("Processing audio..."):
        initiate_analysis(uploaded_file)
        # Immediately rerun to activate the polling loop if successful
        if st.session_state.job_id:
            st.rerun()

st.sidebar.header("Analysis Controls")
if st.session_state.job_id:
    st.sidebar.write(f"**Job ID:** `{st.session_state.job_id}`")
    st.sidebar.write(f"**Status:** `{st.session_state.analysis_status}`")

    if st.sidebar.button("Stop Polling", disabled=not st.session_state.polling_active):
        st.session_state.polling_active = False
        st.info("Polling stopped.")
        st.rerun() # Rerun to update button state

# --- Real-time Results Display ---
st.header("2. Real-time Guidance")
results_placeholder = st.empty() # Placeholder for dynamic updates

if st.session_state.job_id and st.session_state.polling_active:
    with results_placeholder.container():
        st.write("### Live Analysis Updates")
        st.progress(0, text="Waiting for initial results...")

        # Loop for polling
        while st.session_state.polling_active:
            results = get_analysis_results(st.session_state.job_id)
            if results:
                st.session_state.analysis_results = results
                st.session_state.analysis_status = results.get("status", "UNKNOWN")

                with results_placeholder.container(): # Update the same container
                    st.empty() # Clear previous content
                    st.write("### Live Analysis Updates")

                    # Display core metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("BPM", f"{results.get('bpm', 'N/A'):.1f}" if results.get('bpm') else "N/A")
                    with col2:
                        st.metric("Current Beat", results.get('current_beat_index', 'N/A'))
                    with col3:
                        st.metric("Current Phase", results.get('current_phase', 'N/A'))

                    st.markdown("---")
                    st.subheader("Suggested Breaking Moves")
                    
                    moves_data = results.get('suggested_breaking_moves', [])
                    if moves_data:
                        # Display last 5 suggested moves
                        st.json(moves_data[-5:])
                        # Display a simple table for all moves
                        # st.dataframe(moves_data, use_container_width=True)
                    else:
                        st.info("No moves suggested yet. Analysis is ongoing or music is too quiet.")

                    # Update progress bar
                    current_status = results.get('status')
                    progress_text = f"Status: {current_status}"
                    if current_status == "PROCESSING":
                        # Simulate progress based on current beat (mock)
                        current_beat = results.get('current_beat_index', 0)
                        # Assume total beats based on some heuristic or backend providing total duration
                        mock_total_beats = 500 # This would come from actual analysis in a real app
                        progress_value = min(current_beat / mock_total_beats, 0.99) # Cap before 100%
                        st.progress(progress_value, text=progress_text)
                    elif current_status == "COMPLETED":
                        st.progress(1.0, text="Analysis Completed!")
                        st.session_state.polling_active = False # Stop polling on completion
                    elif current_status == "FAILED":
                        st.error(f"Analysis failed: {results.get('error', 'Unknown error.')}")
                        st.session_state.polling_active = False # Stop polling on failure
                    else:
                        st.progress(0, text=progress_text) # For PENDING status

            # Check if polling should continue
            if not st.session_state.polling_active or st.session_state.analysis_status in ["COMPLETED", "FAILED"]:
                st.info(f"Analysis {st.session_state.analysis_status}. Polling stopped.")
                break

            time.sleep(POLLING_INTERVAL_SECONDS)
            st.rerun() # Force rerun to re-execute the loop and refresh UI

elif st.session_state.job_id and not st.session_state.polling_active:
    with results_placeholder.container():
        st.info("Analysis is complete or polling has been stopped. Use the 'Start Analysis' button to begin a new session.")
        if st.session_state.analysis_results:
             st.subheader("Final Analysis Results")
             st.json(st.session_state.analysis_results)
else:
    with results_placeholder.container():
        st.info("Upload an audio file and click 'Start Analysis' to begin!")

st.markdown("---")
st.caption("Developed with Streamlit and FastAPI. Backend is a mock for music analysis.")

# Optional: Add a button to clear session state for fresh start
if st.button("Clear Session (Start Fresh)"):
    for key in st.session_state.keys():
        del st.session_state[key]
    st.rerun()