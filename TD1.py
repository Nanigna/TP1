from sklearn.feature_extraction.text import TfidfVectorizer

texts = ["le traitement de langage naturel est facinant", "le traitement de langues est bune branche de l'IA",
         "l'analyse de texte est utilisée pour la traduction automatique"]

# Etape 1 : Création de la matrice TF-IDF
vect = TfidfVectorizer()
tfidf_mat = vect.fit_transform(texts).toarray()

#print("Matrice TF-IDF:\n", tfidf_mat)

from scipy.spatial.distance import cosine

# étape 2 : Similarité Cosine
cosine_similarity = 1 - cosine(tfidf_mat[0], tfidf_mat[1])

#print(f"\nSimilarité Cosine entre le Texte 1 et le Texte 2 : {cosine_similarity:.3f}")

# étape 3 : Corrélation de Pearson
from scipy.stats import pearsonr

# Étape 3: Corrélation de Pearson
pearson_corr, _ = pearsonr(tfidf_mat[0], tfidf_mat[1])

#print(f"Corrélation de Pearson entre le Texte 1 et le Texte 2 : {pearson_corr:.3f}")

# Étape 4 : Comparaison entre plusieurs paires

for i in range(len(texts)):
    for j in range(i + 1, len(texts)):
        cosine_sim = 1 - cosine(tfidf_mat[i], tfidf_mat[j])
        pearson_corr, _ = pearsonr(tfidf_mat[i], tfidf_mat[j])
        
       # print(f"\nComparaison Texte {i+1} et Texte {j+1} :")
        #print(f" Similarité Cosine : {cosine_sim:.3f}")
        #print(f" Corrélation Pearson : {pearson_corr:.3f}")

# recherde document similaire a 70%

for i in range(len(texts)):
    for j in range(i + 1, len(texts)):
        cosine_sim = 1 - cosine(tfidf_mat[i], tfidf_mat[j])
        if cosine_sim >= 0.7:
            print(f"Les documents {i+1} et {j+1} sont similaires à au moins 70%")

