import joblib
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## Load trained model
model = joblib.load("abalone_model.pkl")

## Streamlit app
st.title("Abalone Age Predictor")
st.write("Discover the age of your abalone based on physical measurements")

## Add information section
with st.expander("How to use this app"):
    st.write("""
    Step 1: Measure your abalone's dimensions (length, diameter, height)  
    Step 2: Weigh the different parts  
    Step 3: Click "Predict Age" to get results  
    
    The app will predict the number of rings, which indicates the abalone's age.
    """)

## Create two columns for better layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Physical Dimensions")
    
    length = st.slider(
        "Length (mm)",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.01,
        help="Longest shell measurement"
    )
    
    diameter = st.slider(
        "Diameter (mm)",
        min_value=0.0,
        max_value=1.0,
        value=0.4,
        step=0.01,
        help="Perpendicular to length"
    )
    
    height = st.slider(
        "Height (mm)",
        min_value=0.0,
        max_value=1.0,
        value=0.15,
        step=0.01,
        help="With meat in shell"
    )

with col2:
    st.subheader("Weight Measurements")
    
    whole_weight = st.slider(
        "Whole Weight (grams)",
        min_value=0.0,
        max_value=3.0,
        value=0.8,
        step=0.01,
        help="Whole abalone weight"
    )
    
    shucked_weight = st.slider(
        "Shucked Weight (grams)",
        min_value=0.0,
        max_value=2.0,
        value=0.35,
        step=0.01,
        help="Weight of meat"
    )
    
    viscera_weight = st.slider(
        "Viscera Weight (grams)",
        min_value=0.0,
        max_value=1.0,
        value=0.18,
        step=0.01,
        help="Gut weight (after bleeding)"
    )
    
    shell_weight = st.slider(
        "Shell Weight (grams)",
        min_value=0.0,
        max_value=1.5,
        value=0.24,
        step=0.01,
        help="Weight after drying"
    )

## Predict button
if st.button("Predict Age", type="primary"):
    
    if length == 0 and diameter == 0 and height == 0 and whole_weight == 0:
        st.error("Invalid input. Please enter measurements for your abalone.")
        st.info("Tip: Adjust the sliders to match your abalone's actual measurements.")
    else:
        shell_ratio = shell_weight / (whole_weight + 1e-8)
        shucked_ratio = shucked_weight / (whole_weight + 1e-8)
        
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
        
        y_pred = model.predict(df_input)[0]
        estimated_age = y_pred + 1.5
        
        st.markdown("---")
        st.markdown("## Prediction Results")
        
        col_res1, col_res2, col_res3 = st.columns(3)
        
        with col_res1:
            st.metric("Number of Rings", f"{y_pred:.1f}")
        
        with col_res2:
            st.metric("Estimated Age", f"{estimated_age:.1f} years")
        
        with col_res3:
            if estimated_age < 7:
                category = "Young"
            elif estimated_age < 12:
                category = "Adult"
            else:
                category = "Mature"
            
            st.metric("Life Stage", category)
        
        st.markdown("---")
        st.markdown("## Your Abalone's Profile")
        
        fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        dimensions = ['Length', 'Diameter', 'Height']
        dim_values = [length, diameter, height]
        colors1 = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        bars1 = ax1.bar(dimensions, dim_values, color=colors1, edgecolor='black', linewidth=1.5)
        ax1.set_ylabel('Measurement (mm)', fontsize=12, fontweight='bold')
        ax1.set_title('Physical Dimensions', fontsize=14, fontweight='bold')
        ax1.set_ylim(0, 1.0)
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        
        for bar, value in zip(bars1, dim_values):
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., h, f'{value:.2f}',
                     ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        weights_labels = ['Whole Weight', 'Shucked Weight', 'Viscera Weight', 'Shell Weight']
        weight_values = [whole_weight, shucked_weight, viscera_weight, shell_weight]
        colors2 = ['#95E1D3', '#F38181', '#AA96DA', '#FCBAD3']
        bars2 = ax2.bar(weights_labels, weight_values, color=colors2, edgecolor='black', linewidth=1.5)
        ax2.set_ylabel('Weight (grams)', fontsize=12, fontweight='bold')
        ax2.set_title('Weight Distribution', fontsize=14, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
        
        for bar, value in zip(bars2, weight_values):
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., h, f'{value:.2f}g',
                     ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig1)
        
        st.markdown("---")
        st.markdown("## Age Comparison")
        
        age_ranges = ['Young (< 7 years)', 'Adult (7–12 years)', 'Mature (12–18 years)', 'Old (> 18 years)']
        age_range_values = [5, 9.5, 15, 20]
        
        if estimated_age < 7:
            user_index = 0
        elif estimated_age < 12:
            user_index = 1
        elif estimated_age < 18:
            user_index = 2
        else:
            user_index = 3
        
        fig2, ax = plt.subplots(figsize=(10, 6))
        bar_colors = ['#E8E8E8'] * 4
        bar_colors[user_index] = '#FF6B6B'
        
        ax.bar(age_ranges, age_range_values, color=bar_colors,
               edgecolor='black', linewidth=2, alpha=0.7)
        
        ax.axhline(y=estimated_age, color='red', linestyle='--', linewidth=2.5,
                   label=f'Predicted Age: {estimated_age:.1f} years')
        
        ax.set_ylabel('Age (years)', fontsize=13, fontweight='bold')
        ax.set_title('How Old is Your Abalone?', fontsize=15, fontweight='bold')
        ax.set_ylim(0, 25)
        ax.legend()
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        st.pyplot(fig2)
        
        st.markdown("---")
        st.markdown("## Weight Composition")
        
        fig3, ax3 = plt.subplots(figsize=(7, 7))
        
        sizes = [
            shucked_weight,
            viscera_weight,
            shell_weight,
            whole_weight - shucked_weight - viscera_weight - shell_weight
        ]
        labels = ['Shucked (Meat)', 'Viscera (Gut)', 'Shell', 'Other']
        colors_pie = ['#F38181', '#AA96DA', '#FCBAD3', '#E8E8E8']
        explode = (0.1, 0, 0, 0)
        
        ax3.pie(
            sizes,
            explode=explode,
            labels=labels,
            colors=colors_pie,
            autopct='%1.1f%%',
            startangle=90
        )
        
        ax3.set_title('Weight Breakdown', fontsize=14, fontweight='bold')
        st.pyplot(fig3)
        
        with st.expander("View Detailed Measurements"):
            measurement_df = pd.DataFrame({
                'Measurement': ['Length', 'Diameter', 'Height', 'Whole Weight',
                                'Shucked Weight', 'Viscera Weight', 'Shell Weight'],
                'Value': [
                    f"{length:.3f} mm",
                    f"{diameter:.3f} mm",
                    f"{height:.3f} mm",
                    f"{whole_weight:.3f} g",
                    f"{shucked_weight:.3f} g",
                    f"{viscera_weight:.3f} g",
                    f"{shell_weight:.3f} g"
                ]
            })
            st.dataframe(measurement_df, use_container_width=True, hide_index=True)

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p><b>Abalone Age Predictor</b> • Powered by Machine Learning</p>
    <p style='font-size: 12px;'>Adjust the measurements above to predict different abalones</p>
</div>
""", unsafe_allow_html=True)
