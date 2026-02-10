import joblib
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## Load trained model
model = joblib.load("abalone_model.pkl")

## Streamlit app
st.title("🐚 Abalone Age Predictor")
st.write("Discover the age of your abalone based on physical measurements")

## Add information section
with st.expander("ℹ️ How to use this app"):
    st.write("""
    **Step 1:** Measure your abalone's dimensions (length, diameter, height)  
    **Step 2:** Weigh the different parts  
    **Step 3:** Click "Predict Age" to get results  
    
    The app will predict the number of rings, which indicates the abalone's age.
    """)

## Create two columns for better layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("📏 Physical Dimensions")
    
    ## User inputs for dimensions
    length = st.slider("Length (mm)", 
                      min_value=0.0, 
                      max_value=1.0, 
                      value=0.5,
                      step=0.01,
                      help="Longest shell measurement")
    
    diameter = st.slider("Diameter (mm)", 
                        min_value=0.0, 
                        max_value=1.0, 
                        value=0.4,
                        step=0.01,
                        help="Perpendicular to length")
    
    height = st.slider("Height (mm)", 
                      min_value=0.0, 
                      max_value=1.0, 
                      value=0.15,
                      step=0.01,
                      help="With meat in shell")

with col2:
    st.subheader("⚖️ Weight Measurements")
    
    ## User inputs for weights
    whole_weight = st.slider("Whole Weight (grams)", 
                            min_value=0.0, 
                            max_value=3.0, 
                            value=0.8,
                            step=0.01,
                            help="Whole abalone weight")
    
    shucked_weight = st.slider("Shucked Weight (grams)", 
                              min_value=0.0, 
                              max_value=2.0, 
                              value=0.35,
                              step=0.01,
                              help="Weight of meat")
    
    viscera_weight = st.slider("Viscera Weight (grams)", 
                              min_value=0.0, 
                              max_value=1.0, 
                              value=0.18,
                              step=0.01,
                              help="Gut weight (after bleeding)")
    
    shell_weight = st.slider("Shell Weight (grams)", 
                            min_value=0.0, 
                            max_value=1.5, 
                            value=0.24,
                            step=0.01,
                            help="Weight after drying")

## Predict button
if st.button("🔮 Predict Age", type="primary"):
    
    ## Validation: Check if inputs are realistic
    if length == 0 and diameter == 0 and height == 0 and whole_weight == 0:
        st.error("⚠️ **Invalid Input!** Please enter measurements for your abalone.")
        st.info("💡 **Tip:** Adjust the sliders to match your abalone's actual measurements.")
    else:
        ## Calculate ratio features (same as training)
        shell_ratio = shell_weight / (whole_weight + 1e-8)
        shucked_ratio = shucked_weight / (whole_weight + 1e-8)
        
        ## Convert input data to a DataFrame
        df_input = pd.DataFrame({
            'Length': [length],
            'Diameter': [diameter],
            'Height': [height],
            'Whole weight': [whole_weight],
            'Shucked weight': [shucked_weight],
            'Viscera weight': [viscera_weight],
            'Shell weight': [shell_weight],
            'Shell_Ratio': [shell_ratio],
            'Shucked_Ratio': [shucked_ratio]
        })
        
        ## Predict
        y_pred = model.predict(df_input)[0]
        estimated_age = y_pred + 1.5
        
        ## Display main prediction
        st.markdown("---")
        st.markdown("## 🎯 Prediction Results")
        
        col_res1, col_res2, col_res3 = st.columns(3)
        
        with col_res1:
            st.metric(
                label="🔢 Number of Rings",
                value=f"{y_pred:.1f}"
            )
        
        with col_res2:
            st.metric(
                label="⏰ Estimated Age",
                value=f"{estimated_age:.1f} years"
            )
        
        with col_res3:
            # Classify size category
            if estimated_age < 7:
                category = "Young"
                emoji = "🐣"
            elif estimated_age < 12:
                category = "Adult"
                emoji = "🐚"
            else:
                category = "Mature"
                emoji = "👑"
            
            st.metric(
                label="📊 Life Stage",
                value=f"{emoji} {category}"
            )
        
        ## Visualization 1: Your Abalone's Measurements
        st.markdown("---")
        st.markdown("## 📊 Your Abalone's Profile")
        
        fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Physical dimensions
        dimensions = ['Length', 'Diameter', 'Height']
        dim_values = [length, diameter, height]
        colors1 = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        bars1 = ax1.bar(dimensions, dim_values, color=colors1, edgecolor='black', linewidth=1.5)
        ax1.set_ylabel('Measurement (mm)', fontsize=12, fontweight='bold')
        ax1.set_title('Physical Dimensions', fontsize=14, fontweight='bold')
        ax1.set_ylim(0, 1.0)
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add value labels on bars
        for bar, value in zip(bars1, dim_values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.2f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Weight breakdown
        weights_labels = ['Whole\nWeight', 'Shucked\nWeight', 'Viscera\nWeight', 'Shell\nWeight']
        weight_values = [whole_weight, shucked_weight, viscera_weight, shell_weight]
        colors2 = ['#95E1D3', '#F38181', '#AA96DA', '#FCBAD3']
        bars2 = ax2.bar(weights_labels, weight_values, color=colors2, edgecolor='black', linewidth=1.5)
        ax2.set_ylabel('Weight (grams)', fontsize=12, fontweight='bold')
        ax2.set_title('Weight Distribution', fontsize=14, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add value labels on bars
        for bar, value in zip(bars2, weight_values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.2f}g',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig1)
        
        ## Visualization 2: Age Comparison Chart
        st.markdown("---")
        st.markdown("## 📈 Age Comparison")
        
        # Create age ranges for comparison
        age_ranges = ['Young\n(< 7 years)', 'Adult\n(7-12 years)', 'Mature\n(12-18 years)', 'Old\n(> 18 years)']
        age_range_values = [5, 9.5, 15, 20]  # Mid-points
        
        # Determine which range the prediction falls into
        if estimated_age < 7:
            user_index = 0
        elif estimated_age < 12:
            user_index = 1
        elif estimated_age < 18:
            user_index = 2
        else:
            user_index = 3
        
        # Create bar chart
        fig2, ax = plt.subplots(figsize=(10, 6))
        bar_colors = ['#E8E8E8'] * 4
        bar_colors[user_index] = '#FF6B6B'  # Highlight user's category
        
        bars = ax.bar(age_ranges, age_range_values, color=bar_colors, 
                      edgecolor='black', linewidth=2, alpha=0.7)
        
        # Add horizontal line for user's exact age
        ax.axhline(y=estimated_age, color='red', linestyle='--', linewidth=2.5, 
                   label=f'Your Abalone: {estimated_age:.1f} years')
        
        ax.set_ylabel('Age (years)', fontsize=13, fontweight='bold')
        ax.set_title('How Old is Your Abalone?', fontsize=15, fontweight='bold')
        ax.set_ylim(0, 25)
        ax.legend(fontsize=11, loc='upper left')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        st.pyplot(fig2)
        
        ## Visualization 3: Weight Composition Pie Chart
        st.markdown("---")
        st.markdown("## 🥧 Weight Composition")
        
        col_pie1, col_pie2 = st.columns([1, 1])
        
        with col_pie1:
            # Pie chart showing weight breakdown
            fig3, ax3 = plt.subplots(figsize=(7, 7))
            
            sizes = [shucked_weight, viscera_weight, shell_weight, 
                    whole_weight - shucked_weight - viscera_weight - shell_weight]
            labels = ['Shucked\n(Meat)', 'Viscera\n(Gut)', 'Shell', 'Other']
            colors_pie = ['#F38181', '#AA96DA', '#FCBAD3', '#E8E8E8']
            explode = (0.1, 0, 0, 0)  # Explode meat section
            
            wedges, texts, autotexts = ax3.pie(sizes, explode=explode, labels=labels, colors=colors_pie,
                                               autopct='%1.1f%%', startangle=90, textprops={'fontsize': 11})
            
            # Make percentage text bold
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
                autotext.set_fontsize(12)
            
            ax3.set_title('Weight Breakdown', fontsize=14, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig3)
        
        with col_pie2:
            st.markdown("### 📝 Weight Summary")
            st.write("")
            st.write(f"**Total Weight:** {whole_weight:.2f}g")
            st.write(f"**Meat Weight:** {shucked_weight:.2f}g ({(shucked_weight/whole_weight*100):.1f}%)")
            st.write(f"**Shell Weight:** {shell_weight:.2f}g ({(shell_weight/whole_weight*100):.1f}%)")
            st.write("")
            st.write("---")
            st.write("### 💡 Insights")
            
            meat_percentage = (shucked_weight / whole_weight * 100) if whole_weight > 0 else 0
            
            if meat_percentage > 45:
                st.success("✅ High meat content! Great for consumption.")
            elif meat_percentage > 35:
                st.info("ℹ️ Good meat content.")
            else:
                st.warning("⚠️ Lower meat content than average.")
        
        ## Show detailed measurements
        with st.expander("📋 View Detailed Measurements"):
            measurement_df = pd.DataFrame({
                'Measurement': ['Length', 'Diameter', 'Height', 'Whole Weight', 
                              'Shucked Weight', 'Viscera Weight', 'Shell Weight'],
                'Value': [f"{length:.3f} mm", f"{diameter:.3f} mm", f"{height:.3f} mm",
                         f"{whole_weight:.3f} g", f"{shucked_weight:.3f} g", 
                         f"{viscera_weight:.3f} g", f"{shell_weight:.3f} g"]
            })
            st.dataframe(measurement_df, use_container_width=True, hide_index=True)

## Add footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p><b>🐚 Abalone Age Predictor</b> • Powered by Machine Learning</p>
    <p style='font-size: 12px;'>Adjust the measurements above to predict different abalones</p>
</div>
""", unsafe_allow_html=True)

## Page design - Ocean theme background
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-attachment: fixed;
    }
    
    .stButton > button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 10px;
        font-size: 16px;
    }
    
    .stButton > button:hover {
        background-color: #45a049;
        transform: scale(1.02);
    }
    
    div[data-testid="stExpander"] {
        background-color: rgba(255, 255, 255, 0.95);
        border-radius: 10px;
        padding: 10px;
    }
    
    .stSuccess, .stInfo, .stWarning {
        background-color: rgba(255, 255, 255, 0.95);
        border-radius: 10px;
        padding: 15px;
    }
    
    div[data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)