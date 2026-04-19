import streamlit as st
import pickle
import os

# Page layout and title
st.set_page_config(page_title="Electronics Recommender", page_icon="💻", layout="wide")

def load_data():
    """Loads the similarity matrix and product list from pickle files."""
    base_path = os.path.dirname(os.path.abspath(__file__))
    
    # Path to your generated pickle files
    pickle_path = os.path.join(base_path, 'similarity.pkl')
    df_path = os.path.join(base_path, 'electronics_list.pkl')

    if os.path.exists(pickle_path) and os.path.exists(df_path):
        with open(df_path, 'rb') as f:
            df = pickle.load(f)
        with open(pickle_path, 'rb') as f:
            similarity = pickle.load(f)
        return df, similarity
    return None, None

# Load the brain of the app
df, similarity = load_data()

st.title("💻 Electronics Recommendation System")
st.markdown("Find the best tech products similar to your favorites.")

if df is not None and similarity is not None:
    # Dropdown for selecting a product
    product_list = df['product_name'].values
    selected_product = st.selectbox("Select or type a product name:", product_list)

    if st.button('Recommend'):
        try:
            # Find the index of the selected product
            idx = df[df['product_name'] == selected_product].index[0]
            
            # Get similarity scores and sort them
            distances = sorted(list(enumerate(similarity[idx])), reverse=True, key=lambda x: x[1])
            
            st.subheader("Recommended for you:")
            
            # Display top 5 recommendations in columns
            cols = st.columns(5)
            for i in range(1, 6):
                with cols[i-1]:
                    product_name = df.iloc[distances[i][0]].product_name
                    st.info(product_name)
                    
        except Exception as e:
            st.error(f"Error processing recommendations: {e}")
else:
    st.error("Error: Pickle files ('similarity.pkl' and 'electronics_list.pkl') not found!")
    st.info("Please upload the generated pickle files to your GitHub repository.")

st.divider()
st.caption("Electronics Recommendation System | Professional Portfolio Project")