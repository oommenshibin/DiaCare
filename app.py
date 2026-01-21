import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="DiaCare",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- ENHANCED CUSTOM CSS FOR MODERN HEALTH APP LOOK ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    .main { 
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 2rem;
        font-family: 'Poppins', sans-serif;
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background: linear-gradient(180deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        border: 1px solid rgba(255,255,255,0.2);
        padding: 1rem;
    }
    
    /* Enhanced Tabs */
    .stTabs [data-baseweb="tab-list"] { 
        gap: 8px; 
        background: rgba(255,255,255,0.1);
        padding: 8px;
        border-radius: 15px;
        backdrop-filter: blur(10px);
    }
    .stTabs [data-baseweb="tab"] {
        height: 55px;
        white-space: pre-wrap;
        background: rgba(255,255,255,0.2);
        border-radius: 12px;
        color: #ffffff;
        border: 1px solid rgba(255,255,255,0.3);
        font-weight: 500;
        font-size: 14px;
        transition: all 0.3s ease;
        backdrop-filter: blur(10px);
    }
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(255,255,255,0.3);
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.2);
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(45deg, #4ade80, #22c55e) !important;
        color: #ffffff !important;
        border: none;
        box-shadow: 0 10px 30px rgba(34,197,94,0.4);
        transform: translateY(-3px);
    }
    
    /* Glassmorphism Cards */
    div[data-testid="metric-container"] {
        background: rgba(255,255,255,0.15) !important;
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255,255,255,0.3);
        border-radius: 20px;
        padding: 20px;
        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        margin: 10px 0;
        transition: all 0.3s ease;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-5px);
        box-shadow: 0 25px 50px rgba(0,0,0,0.15);
    }
    
    /* Input Styling */
    .stNumberInput > div > div > div > div {
        background: rgba(255,255,255,0.1);
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.3);
        backdrop-filter: blur(10px);
    }
    .stSlider > div > div > div > div {
        background: rgba(255,255,255,0.1);
        border-radius: 12px;
        backdrop-filter: blur(10px);
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(45deg, #3b82f6, #1d4ed8);
        color: white;
        border-radius: 12px;
        border: none;
        padding: 12px 24px;
        font-weight: 600;
        font-size: 14px;
        transition: all 0.3s ease;
        box-shadow: 0 10px 30px rgba(59,130,246,0.4);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 15px 40px rgba(59,130,246,0.5);
    }
    
    /* Title Styling */
    h1 {
        font-family: 'Poppins', sans-serif;
        font-weight: 700;
        background: linear-gradient(45deg, #ffffff, #e2e8f0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        font-size: 3rem;
        margin-bottom: 1rem;
        text-shadow: 0 4px 20px rgba(0,0,0,0.1);
    }
    
    /* Subheader */
    h2, h3 {
        color: #ffffff;
        font-weight: 500;
        text-shadow: 0 2px 10px rgba(0,0,0,0.2);
    }
    
    /* Columns */
    .stHorizontalBlock > div {
        background: rgba(255,255,255,0.05);
        border-radius: 15px;
        padding: 20px;
        border: 1px solid rgba(255,255,255,0.1);
        backdrop-filter: blur(10px);
    }
    
    /* Progress Bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #10b981, #059669);
        border-radius: 10px;
    }
    
    /* Expander */
    .stExpander {
        background: rgba(255,255,255,0.08);
        border-radius: 15px;
        border: 1px solid rgba(255,255,255,0.2);
    }
    </style>
""", unsafe_allow_html=True)

# --- LOAD MODELS ---
MODEL_PATH = 'models/diabetes_model.pkl'
SCALER_PATH = 'models/scaler.pkl'

@st.cache_resource
def load_models():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        return None, None
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

model, scaler = load_models()

# --- API CONFIGURATION ---
API_NINJAS_KEY = "9zrJsiroiwUxhM2N7W8Hyn21YlcIbabdhIYnl3X7"

@st.cache_data(ttl=3600)
def get_nutrition_data(query):
    api_url = 'https://api.api-ninjas.com/v1/nutrition?query={}'.format(query)
    try:
        response = requests.get(api_url, headers={'X-Api-Key': API_NINJAS_KEY})
        if response.status_code == requests.codes.ok:
            return response.json()
        else:
            return {"error": response.status_code, "message": response.text}
    except Exception as e:
        return {"error": "Exception", "message": str(e)}

# --- ENHANCED SIDEBAR ---
with st.sidebar:
    st.markdown("""
        <div style='text-align: center;'>
            <img src="https://cdn-icons-png.flaticon.com/512/2966/2966327.png" width=100>
        </div>
    """, unsafe_allow_html=True)
    st.title("DiaCare")
    st.markdown("**A comprehensive health monitoring web application built with Streamlit, ML, and live nutrition APIs.**", unsafe_allow_html=True)
    st.divider()
    st.markdown("Developed by Shibin Oommen")

# --- MAIN LAYOUT ---
st.markdown("### Your Personal Health & Nutrition Companion ✨")
st.divider()

tab1, tab2, tab3, tab4 = st.tabs([
    "🩺 Diabetes Risk", 
    "⚖️ BMI Calculator", 
    "🥗 Diet Planner", 
    "🍎 Nutrition Analyzer"
])

# ==========================================
# TAB 1: DIABETES RISK PREDICTION (ENHANCED)
# ==========================================
with tab1:
    if model is None:
        st.warning("⚠️ **Model files not found.** (UI active, prediction disabled)")
    else:
        st.subheader("🤖 AI Risk Assessment")
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("#### 📋 Patient Vitals")
            preg = st.number_input('Pregnancies', 0, 20, 3)
            glu = st.slider('Glucose (mg/dL)', 0, 250, 120)
            bp = st.slider('Blood Pressure (mm Hg)', 0, 150, 70)
            skin = st.slider('Skin Thickness (mm)', 0, 100, 20)
            ins = st.slider('Insulin (mu U/ml)', 0, 900, 80)
            bmi_in = st.slider('BMI', 0.0, 70.0, 25.0)
            dpf = st.slider('Diabetes Pedigree Function', 0.0, 3.0, 0.5)
            age = st.slider('Age (Years)', 1, 120, 30)

        with col2:
            st.markdown("#### 🎯 Prediction Result")
            input_data = pd.DataFrame({
                'Pregnancies': [preg], 'Glucose': [glu], 'BloodPressure': [bp],
                'SkinThickness': [skin], 'Insulin': [ins], 'BMI': [bmi_in],
                'DiabetesPedigreeFunction': [dpf], 'Age': [age]
            })
            
            scaled_data = scaler.transform(input_data)
            prediction = model.predict(scaled_data)[0]
            probability = model.predict_proba(scaled_data)[0][1]

            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = probability * 100,
                title = {'text': "Diabetes Probability (%)"},
                delta={'reference': 50, 'increasing': {'color': "red"}},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "white"},
                    'steps': [
                        {'range': [0, 50], 'color': "rgba(46,204,113,0.8)"},
                        {'range': [50, 75], 'color': "rgba(241,196,15,0.8)"},
                        {'range': [75, 100], 'color': "rgba(231,76,60,0.8)"}],
                    'threshold': {'line': {'color': "white", 'width': 4}, 'thickness': 0.75, 'value': probability * 100}}))
            fig_gauge.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': 'white'})
            
            st.plotly_chart(fig_gauge, use_container_width=True)

            if prediction == 1:
                st.error("🚨 **HIGH RISK DETECTED** Please consult a healthcare professional immediately!")
            else:
                st.success("✅ **LOW RISK** Continue your healthy lifestyle! 🎉")

# ==========================================
# TAB 2: BMI CALCULATOR (ENHANCED)
# ==========================================
with tab2:
    st.subheader("⚖️ Body Mass Index (BMI) Calculator")
    c1, c2, c3 = st.columns(3)
    with c1:
        weight = st.number_input("Weight (kg)", min_value=1.0, value=70.0, step=0.1)
    with c2:
        height = st.number_input("Height (cm)", min_value=1.0, value=170.0, step=0.1)
    with c3:
        st.write("")
        st.write("")
        calculate_bmi = st.button("🧮 Calculate BMI", type="primary")
    
    if calculate_bmi:
        height_m = height / 100
        bmi_value = weight / (height_m ** 2)
        
        if bmi_value < 18.5:
            category, color = "Underweight 💔", "#3b82f6"
        elif 18.5 <= bmi_value < 25:
            category, color = "Normal Weight ✅", "#10b981"
        elif 25 <= bmi_value < 30:
            category, color = "Overweight ⚠️", "#f59e0b"
        else:
            category, color = "Obese 🚨", "#ef4444"
        
        st.divider()
        col1, col2 = st.columns([1,3])
        with col1:
            st.markdown(f"<h1 style='text-align: center; color: {color}; font-size: 4rem;'>{bmi_value:.1f}</h1>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"<h3 style='text-align: center; color: white;'>{category}</h3>", unsafe_allow_html=True)
        st.progress(min(bmi_value / 50, 1.0))

# ==========================================
# TAB 3: DIET PLANNER (ENHANCED)
# ==========================================
with tab3:
    st.subheader("🥗 Smart Diet Planner")
    goal = st.selectbox("🎯 Select your Goal:", ["General Health", "Diabetes Management", "Weight Loss", "Muscle Gain"])
    
    col_good, col_bad = st.columns(2)
    
    if goal == "Diabetes Management":
        good_foods = ["Leafy Greens 🥬", "Whole Grains 🌾", "Fatty Fish 🐟", "Beans 🫘", "Berries 🫐"]
        bad_foods = ["Sugary Drinks 🥤", "White Bread 🍞", "Processed Snacks 🍟", "Dried Fruit 🍇", "Fried Foods 🍔"]
        tip = "Focus on low Glycemic Index (GI) foods to maintain stable blood sugar levels."
    elif goal == "Weight Loss":
        good_foods = ["Cruciferous Veggies 🥦", "Lean Protein 🍗", "Apples 🍎", "Oats 🥣", "Chia Seeds 🌱"]
        bad_foods = ["Fast Food 🍔", "High-Calorie Coffees ☕", "Candy 🍬", "White Sugar 🍚", "Alcohol 🍺"]
        tip = "Maintain a 500kcal daily deficit and prioritize high-fiber foods."
    elif goal == "Muscle Gain":
        good_foods = ["Chicken Breast 🍗", "Eggs 🥚", "Greek Yogurt 🥛", "Almonds 🥜", "Sweet Potatoes 🥔"]
        bad_foods = ["Alcohol 🍺", "Deep Fried Foods 🍟", "Low-Protein Snacks 🍪", "Excessive Sugar 🍭"]
        tip = "Target 1.6-2.2g protein per kg bodyweight daily."
    else:
        good_foods = ["Avocados 🥑", "Nuts 🥜", "Colorful Vegetables 🌈", "Fish 🐟", "Water 💧"]
        bad_foods = ["Trans Fats 🚫", "Processed Meats 🌭", "Excessive Sodium 🧂", "Added Sugars 🍭"]
        tip = "Eat the rainbow and stay hydrated (2-3L water daily)."

    with col_good:
        st.success("✅ **Recommended Foods**")
        for food in good_foods: 
            st.markdown(f"• **{food}**")
    with col_bad:
        st.error("❌ **Foods to Avoid**")
        for food in bad_foods: 
            st.markdown(f"• **{food}**")
    st.info(f"💡 **Pro Tip:** {tip}", icon="💡")

# ==========================================
# TAB 4: NUTRITION ANALYZER (ENHANCED)
# ==========================================
def safe_float(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return 0.0

with tab4:
    st.subheader("🍎 Advanced Nutrition Analyzer")
    st.markdown("🔬 Analyze nutritional breakdown for any food item.")
    
    query = st.text_input("Enter food items (e.g., '2 apples chicken breast'):", key="nutrition_query")
    
    if st.button("🔍 Analyze Nutrition", type="primary"):
        if query:
            with st.spinner("Fetching nutritional data from API..."):
                data = get_nutrition_data(query)
            
            if data and isinstance(data, list):
                total_fat = sum(safe_float(item.get('fat_total_g', 0)) for item in data)
                total_carbs = sum(safe_float(item.get('carbohydrates_total_g', 0)) for item in data)
                total_sugar = sum(safe_float(item.get('sugar_g', 0)) for item in data)
                total_sodium = sum(safe_float(item.get('sodium_mg', 0)) for item in data)
                
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("🥔 Carbohydrates", f"{total_carbs:.1f}g")
                m2.metric("🧈 Total Fat", f"{total_fat:.1f}g")
                m3.metric("🍭 Sugar", f"{total_sugar:.1f}g")
                m4.metric("🧂 Sodium", f"{total_sodium:.0f}mg")

                st.divider()

                col_graph, col_verdict = st.columns([3, 2])
                
                with col_graph:
                    st.markdown("#### 📊 Nutrient Distribution")
                    chart_data = pd.DataFrame({
                        'Nutrient': ['Carbohydrates', 'Total Fat', 'Sugar'],
                        'Grams': [total_carbs, total_fat, total_sugar]
                    })
                    fig = px.bar(chart_data, x='Nutrient', y='Grams', color='Nutrient',
                                 color_discrete_map={'Carbohydrates': '#3b82f6', 'Total Fat': '#ef4444', 'Sugar': '#f59e0b'},
                                 text='Grams')
                    fig.update_traces(texttemplate='%{text:.1f}g', textposition='outside')
                    fig.update_layout(showlegend=False, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig, use_container_width=True)

                with col_verdict:
                    st.markdown("#### 🩺 Health Assessment")
                    if total_sugar > 25:
                        st.error(f"🚨 **CRITICAL SUGAR LEVEL:** {total_sugar:.0f}g sugar detected!")
                    elif total_sugar > 15:
                        st.warning(f"⚠️ **High Sugar:** {total_sugar:.0f}g - Limit for diabetes management.")
                    elif total_carbs < 10 and total_fat > 5:
                        st.success("🥑 **KETO-FRIENDLY** Low carb, moderate fat option.")
                    else:
                        st.info("✅ **BALANCED** Nutrients in healthy range.")
                    
                    if total_sodium > 1500:
                        st.error(f"🧂 **HIGH SODIUM ALERT:** {total_sodium}mg - Monitor blood pressure!")
                    elif total_sodium > 800:
                        st.warning(f"🧂 Elevated sodium: {total_sodium}mg")

                with st.expander("🔍 Detailed API Response", expanded=False):
                    st.json(data)
            else:
                st.error("❌ No data retrieved. Try simpler food names like 'apple' or 'chicken'.")

st.markdown("---")
st.markdown("<p style='text-align: center; color: rgba(255,255,255,0.7);'>Powered by DiaCare ❤️ Stay Healthy!</p>", unsafe_allow_html=True)
