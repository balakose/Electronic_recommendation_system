import streamlit as st
import pandas as pd
import os
import pickle

# Page Configuration
st.set_page_config(page_title="Electronics Recommender", page_icon="💻", layout="wide")

def load_data():
    base_path = os.path.dirname(os.path.abspath(__file__))
    
    # Ensure these filenames match exactly with your GitHub files
    csv_path = os.path.join(base_path, 'Electronics_Data.csv')
    pickle_path = os.path.join(base_path, 'similarity.pkl')

    if not os.path.exists(csv_path):
        return None, None, "Error: 'Electronics_Data.csv' not found."
    
    df = pd.read_csv(csv_path)
    
    if os.path.exists(pickle_path):
        with open(pickle_path, 'rb') as f:
            similarity = pickle.load(f)
    else:
        return df, None, "Warning: 'similarity.pkl' missing."
        
    return df, similarity, None

# Main Application UI
st.title("💻 Electronics Recommendation System")
st.markdown("Enter a product name below to get AI-powered recommendations.")

df, similarity, error = load_data()

if error and df is None:
    st.error(error)
else:
    # Selection Box
    product_list = df['product_name'].values
    selected_product = st.selectbox(
        "Select an item:",
        product_list
    )

    if st.button('Recommend'):
        if similarity is not None:
            try:
                index = df[df['product_name'] == selected_product].index[0]
                distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
                
                st.subheader("Recommended for you:")
                
                cols = st.columns(5)
                for i in range(1, 6):
                    with cols[i-1]:
                        product_name = df.iloc[distances[i][0]].product_name
                        st.info(product_name)
            except Exception as e:
                st.error(f"Error processing recommendations: {e}")
        else:
            st.warning("Similarity model is not loaded.")

# Footer removed or updated
st.divider()
st.caption("Electronics Recommendation System | Portfolio Project")
