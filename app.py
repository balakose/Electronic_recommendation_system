import streamlit as st
import pickle
import os

st.set_page_config(page_title="Electronics Recommender", page_icon="💻", layout="wide")

def load_data():
    base_path = os.path.dirname(os.path.abspath(__file__))
    pickle_path = os.path.join(base_path, 'similarity.pkl')
    df_path = os.path.join(base_path, 'electronics_list.pkl')

    if os.path.exists(pickle_path) and os.path.exists(df_path):
        with open(df_path, 'rb') as f:
            df = pickle.load(f)
        with open(pickle_path, 'rb') as f:
            similarity = pickle.load(f)
        return df, similarity
    return None, None

df, similarity = load_data()

st.title("💻 Electronics Recommendation System")

if df is not None and similarity is not None:
    product_list = df['product_name'].values
    selected_product = st.selectbox("Select a product:", product_list)

    if st.button('Recommend'):
        try:
            idx = df[df['product_name'] == selected_product].index[0]
            distances = sorted(list(enumerate(similarity[idx])), reverse=True, key=lambda x: x[1])
            
            # FIXED LOGIC: Get top 5 or fewer if the dataset is small
            # We skip the first one (index 0) because it is the product itself
            num_recommendations = min(len(distances) - 1, 5)
            
            if num_recommendations > 0:
                st.subheader("Recommended for you:")
                cols = st.columns(num_recommendations)
                
                for i in range(num_recommendations):
                    with cols[i]:
                        # i+1 to skip the selected product itself
                        product_name = df.iloc[distances[i+1][0]].product_name
                        st.info(product_name)
            else:
                st.warning("No similar products found.")
                    
        except Exception as e:
            st.error(f"Error processing recommendations: {e}")
else:
    st.error("Missing pickle files! Upload 'similarity.pkl' and 'electronics_list.pkl' to GitHub.")

 # --- The Footer is back here ---
st.divider()
st.caption("Electronics Recommendation System | Portfolio Project")   