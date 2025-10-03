import streamlit as st
from PIL import Image
from agentic import AgentOrchestrator, VisionResult, SeverityResult
from utils.vision import load_model, predict_torch, CLASSES
from utils.color import estimate_severity
from utils.dosage import compute_mix
import pandas as pd
import os, io
from utils.gradcam import generate_gradcam_visualization

st.set_page_config(page_title='Intelligent Pesticide Sprinkling', layout='wide', initial_sidebar_state='expanded')
st.title('🌿 Intelligent Pesticide Sprinkling - Demo')
st.markdown('A demo that detects leaf disease + severity and recommends pesticide & dose. *Always follow product labels.*')

with st.sidebar:
    st.header('Settings')
    weights_path = st.text_input('Path to model weights', 'models/best_model.pt')
    use_gradcam = st.checkbox('Show Grad-CAM explanation (if model loaded)', True)
    st.markdown('---')
    st.markdown('**Scanner options**')
    camera_enabled = st.checkbox('Enable camera scanner (use your webcam)', True)
    st.markdown('Tip: For best results, place a contrasting background and take a close-up of the leaf.')
    st.markdown('---')
    st.caption('If you trained a model (Colab), place models/best_model.pt in this project folder.')

# Input area
input_col, result_col = st.columns([1,1.2])
with input_col:
    st.subheader('Input')
    mode = st.radio('How do you want to provide the leaf image?', ('Upload image','Use camera (scan)'))
    img_file = None

    if mode == 'Upload image':
        img_file = st.file_uploader('Upload a leaf image', type=['jpg','jpeg','png'])
    else:
        if camera_enabled:
            cam = st.camera_input('Scan leaf using your webcam')
            img_file = cam
        else:
            st.info('Enable camera scanner in the sidebar to use this feature.')

    st.markdown('---')
    st.subheader('Sprayer settings (for dosage)')
    tank_l = st.number_input('Sprayer tank capacity (L)', 10.0, 2000.0, 200.0)
    area_ha = st.number_input('Area to spray (hectares)', 0.01, 50.0, 1.0)
    spray_vol = st.number_input('Spray volume (L/ha)', 100.0, 1000.0, 500.0)

with result_col:
    st.subheader('Diagnosis & Recommendation')

    if img_file is None:
        st.info('Provide an image (upload or scan) to analyze the leaf.')
    else:
        try:
            # Read image into PIL
            if isinstance(img_file, bytes):
                image = Image.open(io.BytesIO(img_file)).convert('RGB')
            else:
                image = Image.open(img_file).convert('RGB')
            st.image(image, caption='Input image', use_container_width=True)
        except Exception as e:
            st.error('Could not read the image: ' + str(e))
            st.stop()

        # Load model if exists
        model = None
        try:
            if os.path.exists(weights_path):
                model = load_model(weights_path)
                st.success('✅ Vision model loaded. Classes: ' + ', '.join(CLASSES))
            else:
                st.warning('⚠️ No trained model found at {}. Using color heuristic only.'.format(weights_path))
        except Exception as e:
            st.warning(f'Could not load vision model: {e}')
            model = None

        # Severity via color analysis
        sev_pct = estimate_severity(image)
        sev = SeverityResult(percent=sev_pct)

        # Vision prediction
        vision = None
        if model is not None:
            try:
                label, conf = predict_torch(model, image)
                vision = VisionResult(label=label, confidence=conf)
            except Exception as e:
                st.warning(f'Vision inference failed: {e}')
                vision = None

        # Agent decision
        agent = AgentOrchestrator('data/knowledge_base.csv')
        action = agent.decide(vision, sev)

        st.metric('Severity (color-based)', f"{action['severity_percent']}%")
        st.write(f"**Diagnosis:** {action['diagnosis']}  |  **Confidence:** {action['confidence']}")
        st.write(f"**Stage:** {action['stage']}  |  **Rationale:** {action['rationale']}" )

        rec = action['recommendation']
        st.subheader('Recommendation')
        if rec['pesticide'] == 'none':
            st.success('Healthy – no spray required.')
        else:
            st.write(f"**Pesticide:** {rec['pesticide']}  **Formulation:** {rec['formulation']}")
            st.write(f"**MoA/FRAC:** {rec['mode_of_action']} (FRAC {rec['frac_code']})")
            st.write(f"**Label dose:** {rec['dose_per_litre']} per L  |  **PHI:** {rec['phi_days']} days")
            mix = compute_mix(rec['dose_per_litre'], tank_l, area_ha, spray_vol)
            st.markdown('#### Mixing Plan')
            st.write(f"Total water: **{mix['total_water_l']} L**  |  Total pesticide: **{mix['total_pesticide_ml']} ml**")
            st.write(f"Tanks: **{mix['tanks']}** → per‑tank: {mix['per_tank_water_l']} L + {mix['per_tank_pesticide_ml']} ml")
            st.info(rec['notes'])

        st.warning(action['safety'])

        # Grad-CAM visualization
        if use_gradcam and model is not None:
            try:
                cam_img = generate_gradcam_visualization(model, image)
                st.markdown('**Grad-CAM (model attention)**')
                st.image(cam_img, use_container_width=True)
            except Exception as e:
                st.info('Grad-CAM not available: ' + str(e))

        proto = pd.DataFrame({
            'diagnosis':[action['diagnosis']],
            'confidence':[action['confidence']],
            'severity_percent':[action['severity_percent']],
            'stage':[action['stage']],
            'pesticide':[rec['pesticide']],
            'dose_per_litre':[rec['dose_per_litre']],
            'frac_code':[rec['frac_code']],
            'phi_days':[rec['phi_days']],
            'notes':[rec['notes']]
        })
        csv = proto.to_csv(index=False).encode('utf-8')
        st.download_button('⬇️ Download protocol (CSV)', csv, file_name='protocol.csv', mime='text/csv')

st.markdown('---')
st.caption('This tool supports decision-making. Always follow product labels and local regulations.')
