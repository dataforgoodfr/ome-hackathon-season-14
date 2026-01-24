import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from main.main import get_agriculture_data
from rake_nltk.rake import Rake
import nltk
import csv
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)

# Charger les stop words français
from nltk.corpus import stopwords
french_stopwords = set(stopwords.words('french'))

df = get_agriculture_data()

rake = Rake(language="french")
key_words: dict[str, int] = {}

# Parcourir tous les textes
for index, row in df.iterrows():
    rake.extract_keywords_from_text(row["report_text"])
    keywords = rake.get_ranked_phrases()
    
    # Pour chaque keyword trouvé dans ce texte, ajouter 1 au compteur
    # On utilise un set pour éviter de compter plusieurs fois le même keyword dans le même texte
    # Si un keyword est présent dans un extrait, on ajoute 1 au dictionnaire pour ce keyword
    # Filtrer les stop words et les mots/phrases trop courts
    unique_keywords = set(keywords)
    for keyword in unique_keywords:
        # Nettoyer le keyword (enlever espaces, convertir en minuscule)
        keyword_clean = keyword.strip().lower()
        
        # Ignorer les keywords vides ou trop courts
        if len(keyword_clean) < 3:
            continue
        
        # Pour les phrases (mots multiples), vérifier qu'elles contiennent au moins un mot significatif
        # Pour les mots uniques, vérifier qu'ils ne sont pas des stop words
        words_in_keyword = keyword_clean.split()
        
        # Si c'est un mot unique, vérifier qu'il n'est pas un stop word
        if len(words_in_keyword) == 1:
            if keyword_clean in french_stopwords or keyword_clean.isdigit():
                continue
        else:
            # Pour les phrases, vérifier qu'au moins un mot n'est pas un stop word
            # et que la phrase ne commence/termine pas uniquement par des stop words
            significant_words = [w for w in words_in_keyword if w not in french_stopwords and len(w) >= 3]
            if len(significant_words) == 0:
                continue
        
        key_words[keyword_clean] = key_words.get(keyword_clean, 0) + 1

# Après avoir traité toutes les données, trier du plus fréquent au moins fréquent
sorted_keywords = sorted(key_words.items(), key=lambda x: x[1], reverse=True)

# Écrire un CSV avec le nombre de textes qui portent au moins une fois le keyword
# Trier du plus de fois au moins de fois
csv_filename = "keyword_counts.csv"
with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['keyword', 'nombre_de_textes'])
    for keyword, count in sorted_keywords:
        writer.writerow([keyword, count])

print(f"CSV créé : {csv_filename}")
print(f"Nombre total de keywords uniques : {len(key_words)}")
print(f"Top 10 keywords les plus fréquents :")
for keyword, count in sorted_keywords[:10]:
    print(f"  {keyword}: {count} textes")