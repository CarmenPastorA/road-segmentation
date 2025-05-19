# -----------------------------
# HOW TO RUN:
# streamlit run app/ui.py           
# -----------------------------
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from streamlit_folium import st_folium
import folium
from PIL import Image
import requests
from io import BytesIO
from pyproj import Transformer

from app.api_client import send_image_for_prediction

# -----------------------------
# Dummy model prediction
# -----------------------------
def predict_road_segmentation(input_img):
    return send_image_for_prediction(input_img)

# -----------------------------
# Download image from WMS with reprojection
# -----------------------------
def download_wms_image(bbox_latlon):
    # Convert bbox from EPSG:4326 to EPSG:3857
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    x_min, y_min = transformer.transform(bbox_latlon[0], bbox_latlon[1])
    x_max, y_max = transformer.transform(bbox_latlon[2], bbox_latlon[3])
    bbox_3857 = [x_min, y_min, x_max, y_max]

    wms_url = "https://www.ign.es/wms-inspire/pnoa-ma"
    params = {
        "service": "WMS",
        "version": "1.3.0",
        "request": "GetMap",
        "layers": "OI.OrthoimageCoverage",
        "bbox": ",".join(map(str, bbox_3857)),
        "width": 512,
        "height": 512,
        "crs": "EPSG:3857",
        "format": "image/jpeg"
    }

    try:
        response = requests.get(wms_url, params=params, timeout=10)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        return img
    except Exception as e:
        st.error(f"Failed to download image: {str(e)}")
        return None

# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(layout="wide")
st.title("🛰️ Road Segmentation")

option = st.sidebar.radio("Choose an option:", ["Upload Image", "Select from Map"])

# -----------------------------
# Mode 1: Upload Image
# -----------------------------
if option == "Upload Image":
    st.header("🖼️ Upload Satellite Image")
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.session_state["uploaded_image"] = image  # Save it in session

        if st.button("Run Segmentation"):
            with st.spinner("🔍 Running segmentation..."):
                try:
                    mask = predict_road_segmentation(image)
                    st.session_state["uploaded_mask"] = mask
                except Exception as e:
                    st.error(str(e))

        # Show side-by-side only after image is uploaded
        if "uploaded_image" in st.session_state:
            col1, col2 = st.columns(2)
            with col1:
                st.image(st.session_state["uploaded_image"], caption="Uploaded Image", width=500)
            with col2:
                if "uploaded_mask" in st.session_state:
                    st.image(st.session_state["uploaded_mask"], caption="Predicted Road Mask", width=500)

        #st.markdown("---")
        #if st.button("🔁 Reset upload"):
        #    st.session_state.pop("uploaded_image", None)
        #    st.session_state.pop("uploaded_mask", None)
        #    if hasattr(st, "experimental_rerun"):
        #        st.experimental_rerun()
        #    else:
        #        st.info("Reset. Please manually refresh the page.")


# -----------------------------
# Mode 2: Select from Map (folium + streamlit-folium)
# -----------------------------
elif option == "Select from Map":
    st.header("🌍 Draw an area on the map to download the satellite image")

    # Default map location
    default_location = [40.0, -3.7]

    # Create map
    m = folium.Map(location=default_location, zoom_start=6)

    # Add draw tool
    draw = folium.plugins.Draw(export=True, draw_options={"rectangle": True, "polygon": False})
    draw.add_to(m)

    st_map = st_folium(m, height=600, width=1000, returned_objects=["last_active_drawing"])

    if st_map and st_map.get("last_active_drawing"):
        geometry = st_map["last_active_drawing"]
        coords = geometry.get("geometry", {}).get("coordinates", [])

        if geometry["geometry"]["type"] == "Polygon" and coords:
            points = coords[0]
            lons = [p[0] for p in points]
            lats = [p[1] for p in points]
            x_min, x_max = min(lons), max(lons)
            y_min, y_max = min(lats), max(lats)
            bbox_latlon = [x_min, y_min, x_max, y_max]

            st.success(f"Selected area: {bbox_latlon}")

            if st.button("Download Satellite Image"):
                with st.spinner("Downloading satellite image..."):
                    img = download_wms_image(bbox_latlon)
                    if img:
                        st.session_state["downloaded_img"] = img
                        st.session_state.pop("segmentation_mask", None)  # Reset segmentation if new image


            # --- Display downloaded image and segmentation if available ---
            if "downloaded_img" in st.session_state:
                img = st.session_state["downloaded_img"]

                # Run Segmentation Button
                if st.button("Run Segmentation"):
                    with st.spinner("Running segmentation..."):
                        try:
                            mask = predict_road_segmentation(img)
                            st.session_state["segmentation_mask"] = mask
                        except Exception as e:
                            st.error(str(e))

                # Display Satellite Image and Mask
                col1, col2 = st.columns(2)
                with col1:
                    st.image(img, caption="Satellite Image", width=500)
                with col2:
                    if "segmentation_mask" in st.session_state:
                        st.image(st.session_state["segmentation_mask"], caption="Predicted Road Mask", width=500)

                # Reset Button (always aligned below both columns)
                st.markdown("---")
                if st.button("Reset selection"):
                        st.session_state.clear()
                        if hasattr(st, "experimental_rerun"):
                            st.experimental_rerun()
                        else:
                            st.info("Selection reset. Please manually refresh the page.")
                                        
    else:
        st.info("Draw a rectangle on the map to define your area of interest.")
