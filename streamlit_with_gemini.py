import streamlit as st
import requests

st.set_page_config(page_title="Virtual Try-On", layout="wide")
st.title("Virtual Try-On – Gemini + IDM-VTON + Overlay Fallback")

api_base = st.sidebar.text_input("Backend URL", "http://3.88.24.123:8000").rstrip("/")
timeout = st.sidebar.slider("Timeout (s)", 30, 600, 240)
prefer_idm = st.sidebar.checkbox("Prefer IDM-VTON (photorealistic)", value=True)

def health():
    r = requests.get(f"{api_base}/health", timeout=10)
    r.raise_for_status()
    return r.json()

if st.sidebar.button("Health Check"):
    try:
        st.sidebar.json(health())
        st.sidebar.success("Backend reachable")
    except Exception as e:
        st.sidebar.error(str(e))

tab1, tab2 = st.tabs(["Actress → User", "Garment → User"])

def show_result(data: dict):
    st.write("Mode used:", data.get("mode_used"))
    st.info("Garment description: " + str(data.get("garment_description")))
    st.json(data.get("scores", {}))
    for u in data.get("output_urls", []):
        st.image(u, use_column_width=True)

with tab1:
    st.subheader("Actress → User")
    c1, c2 = st.columns(2)
    actress = c1.file_uploader("Upload actress image", type=["jpg","jpeg","png"], key="actress")
    user = c2.file_uploader("Upload user image", type=["jpg","jpeg","png"], key="user_a")

    garment_des = st.text_input("Garment description override (optional)", "")

    if st.button("Run Actress → User"):
        if not actress or not user:
            st.error("Upload both images.")
        else:
            files = {
                "actress_image": ("actress.jpg", actress.getvalue(), actress.type),
                "user_image": ("user.jpg", user.getvalue(), user.type),
            }
            data = {
                "garment_des": garment_des,
                "prefer_idm": 1 if prefer_idm else 0
            }
            with st.spinner("Processing..."):
                try:
                    r = requests.post(f"{api_base}/v1/tryon/actress-to-user", files=files, data=data, timeout=timeout)
                    r.raise_for_status()
                    show_result(r.json())
                except Exception as e:
                    st.error(str(e))

with tab2:
    st.subheader("Garment → User")
    c1, c2 = st.columns(2)
    garment = c1.file_uploader("Upload garment image", type=["jpg","jpeg","png"], key="garment")
    user2 = c2.file_uploader("Upload user image", type=["jpg","jpeg","png"], key="user_g")

    garment_des2 = st.text_input("Garment description override (optional)", "", key="gd2")

    if st.button("Run Garment → User"):
        if not garment or not user2:
            st.error("Upload both images.")
        else:
            files = {
                "garment_image": ("garment.jpg", garment.getvalue(), garment.type),
                "user_image": ("user.jpg", user2.getvalue(), user2.type),
            }
            data = {
                "garment_des": garment_des2,
                "prefer_idm": 1 if prefer_idm else 0
            }
            with st.spinner("Processing..."):
                try:
                    r = requests.post(f"{api_base}/v1/tryon/garment-to-user", files=files, data=data, timeout=timeout)
                    r.raise_for_status()
                    show_result(r.json())
                except Exception as e:
                    st.error(str(e))
