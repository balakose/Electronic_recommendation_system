{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "7de04d1b",
   "metadata": {},
   "source": [
    "Electronic Recommendation system"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "7848004b",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "from sklearn.feature_extraction.text import TfidfVectorizer\n",
    "from sklearn.metrics.pairwise import linear_kernel"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "0ae2ba7a",
   "metadata": {},
   "outputs": [],
   "source": [
    "# 1. Mock Dataset: Electronics Products\n",
    "data = {\n",
    "    'product_id': [1, 2, 3, 4, 5],\n",
    "    'product_name': [\n",
    "        'Logitech G502 Mouse', \n",
    "        'Razer DeathAdder Mouse', \n",
    "        'Apple MacBook Pro', \n",
    "        'Dell XPS 15', \n",
    "        'Sony WH-1000XM4'\n",
    "    ],\n",
    "    'description': [\n",
    "        'High performance wired gaming mouse with RGB lighting.',\n",
    "        'Ergonomic wired gaming mouse with optical sensor.',\n",
    "        'High-end laptop with M2 chip and Retina display.',\n",
    "        'Powerful Windows laptop with 4K infinity display.',\n",
    "        'Wireless noise cancelling headphones with long battery life.'\n",
    "    ],\n",
    "    'category': ['Peripherals', 'Peripherals', 'Laptops', 'Laptops', 'Audio']\n",
    "}\n",
    "\n",
    "df = pd.DataFrame(data)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "46c2b066",
   "metadata": {},
   "outputs": [],
   "source": [
    "# 2. Preprocessing\n",
    "# Combine category and description to give the model more context\n",
    "df['metadata'] = df['category'] + \" \" + df['description']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "3bb05dcd",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>product_id</th>\n",
       "      <th>product_name</th>\n",
       "      <th>description</th>\n",
       "      <th>category</th>\n",
       "      <th>metadata</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>Logitech G502 Mouse</td>\n",
       "      <td>High performance wired gaming mouse with RGB l...</td>\n",
       "      <td>Peripherals</td>\n",
       "      <td>Peripherals High performance wired gaming mous...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>Razer DeathAdder Mouse</td>\n",
       "      <td>Ergonomic wired gaming mouse with optical sensor.</td>\n",
       "      <td>Peripherals</td>\n",
       "      <td>Peripherals Ergonomic wired gaming mouse with ...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>Apple MacBook Pro</td>\n",
       "      <td>High-end laptop with M2 chip and Retina display.</td>\n",
       "      <td>Laptops</td>\n",
       "      <td>Laptops High-end laptop with M2 chip and Retin...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>Dell XPS 15</td>\n",
       "      <td>Powerful Windows laptop with 4K infinity display.</td>\n",
       "      <td>Laptops</td>\n",
       "      <td>Laptops Powerful Windows laptop with 4K infini...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>Sony WH-1000XM4</td>\n",
       "      <td>Wireless noise cancelling headphones with long...</td>\n",
       "      <td>Audio</td>\n",
       "      <td>Audio Wireless noise cancelling headphones wit...</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   product_id            product_name  \\\n",
       "0           1     Logitech G502 Mouse   \n",
       "1           2  Razer DeathAdder Mouse   \n",
       "2           3       Apple MacBook Pro   \n",
       "3           4             Dell XPS 15   \n",
       "4           5         Sony WH-1000XM4   \n",
       "\n",
       "                                         description     category  \\\n",
       "0  High performance wired gaming mouse with RGB l...  Peripherals   \n",
       "1  Ergonomic wired gaming mouse with optical sensor.  Peripherals   \n",
       "2   High-end laptop with M2 chip and Retina display.      Laptops   \n",
       "3  Powerful Windows laptop with 4K infinity display.      Laptops   \n",
       "4  Wireless noise cancelling headphones with long...        Audio   \n",
       "\n",
       "                                            metadata  \n",
       "0  Peripherals High performance wired gaming mous...  \n",
       "1  Peripherals Ergonomic wired gaming mouse with ...  \n",
       "2  Laptops High-end laptop with M2 chip and Retin...  \n",
       "3  Laptops Powerful Windows laptop with 4K infini...  \n",
       "4  Audio Wireless noise cancelling headphones wit...  "
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "b3030b5d",
   "metadata": {},
   "outputs": [],
   "source": [
    "# 3. Vectorization (TF-IDF)\n",
    "# Converts text into a matrix of TF-IDF features\n",
    "tfidf = TfidfVectorizer(stop_words='english')\n",
    "tfidf_matrix = tfidf.fit_transform(df['metadata'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "90338fea",
   "metadata": {},
   "outputs": [],
   "source": [
    "# 4. Compute Similarity\n",
    "# Linear kernel is used to calculate the cosine similarity between vectors\n",
    "cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "2bcd599f",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Products similar to 'Logitech G502 Mouse':\n",
      "1    Razer DeathAdder Mouse\n",
      "2         Apple MacBook Pro\n",
      "Name: product_name, dtype: object\n"
     ]
    }
   ],
   "source": [
    "# 5. Recommendation Function\n",
    "def get_recommendations(product_name, cosine_sim=cosine_sim):\n",
    "    # Get index of the product that matches the name\n",
    "    try:\n",
    "        idx = df.index[df['product_name'] == product_name].tolist()[0]\n",
    "    except IndexError:\n",
    "        return \"Product not found.\"\n",
    "    # Get pairwise similarity scores of all products with that product\n",
    "    sim_scores = list(enumerate(cosine_sim[idx]))\n",
    "\n",
    "    # Sort the products based on similarity scores\n",
    "    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)\n",
    "\n",
    "    # Get the scores of the most similar products (excluding itself)\n",
    "    sim_scores = sim_scores[1:3]\n",
    "\n",
    "    # Get the product indices\n",
    "    product_indices = [i[0] for i in sim_scores]\n",
    "\n",
    "    # Return the top 2 most similar products\n",
    "    return df['product_name'].iloc[product_indices]\n",
    "\n",
    "# --- Example Usage ---\n",
    "target_product = 'Logitech G502 Mouse'\n",
    "print(f\"Products similar to '{target_product}':\")\n",
    "print(get_recommendations(target_product))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "5fdc972a",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Done! Check your folder for similarity.pkl and electronics_list.pkl\n"
     ]
    }
   ],
   "source": [
    "import pickle\n",
    "\n",
    "# 1. Save the similarity matrix\n",
    "with open('similarity.pkl', 'wb') as f:\n",
    "    pickle.dump(cosine_sim, f)\n",
    "\n",
    "# 2. Save the dataframe (so the app knows the product names)\n",
    "with open('electronics_list.pkl', 'wb') as f:\n",
    "    pickle.dump(df, f)\n",
    "\n",
    "print(\"Done! Check your folder for similarity.pkl and electronics_list.pkl\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
