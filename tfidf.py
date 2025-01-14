from sklearn.feature_extraction.text import TfidfVectorizer

texts = ["je participe à la formation de l'IA", "je suis etudiante de l'IA"]

# Créer le vecteur TF-IDF
vect = TfidfVectorizer()
tfidf_mat = vect.fit_transform(texts).toarray()

# Afficher les vecteurs tfidf
print("Matrice TF-IDF:\n", tfidf_mat)
