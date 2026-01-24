# Protocole d'Extraction de Mots-Clés à partir de Textes Français

## Objectif

Ce protocole décrit la méthodologie pour analyser les textes français stockés dans la base de données PostgreSQL, afin d'extraire les mots-clés les plus fréquemment utilisés après avoir retiré :
- Les mots vides (stop words) français courants ("le", "la", "les", "de", "du", etc.)
- Les mots communs et peu informatifs
- Les mots de base de la langue française

## Contexte

- **Base de données** : PostgreSQL
- **Table principale** : `category_classification`
- **Champ texte** : `report_text` (Text)
- **Langue** : Français
- **Domaine** : Médias sur l'écologie (agriculture/alimentation, mobilité/transport, énergie, autres)

---

## Phase 1 : Préparation et Extraction des Données

### 1.1 Extraction depuis PostgreSQL
- Se connecter à la base de données PostgreSQL via SQLAlchemy
- Extraire tous les textes du champ `report_text` de la table `category_classification`
- Filtrer les valeurs nulles/vides
- Optionnel : filtrer par catégorie (`llm_category` ou `predicted_category`) pour des analyses segmentées

### 1.2 Normalisation Initiale
- Convertir tout le texte en minuscules
- Gérer les caractères accentués (normalisation Unicode NFD/NFKD si nécessaire)
- Supprimer les caractères spéciaux non-alphabétiques (ponctuation, chiffres isolés)
- Conserver les espaces pour la tokenisation

---

## Phase 2 : Tokenisation et Nettoyage

### 2.1 Tokenisation
**Objectif** : Découper les textes en mots individuels

**Approches possibles** :
- Tokenisation simple par espaces (méthode basique)
- Tokenisation linguistique avec prise en compte des contractions françaises
- Tokenisation avec préservation des entités nommées (noms propres, acronymes)

**Recommandation** : Utiliser une bibliothèque de tokenisation linguistique pour le français

### 2.2 Normalisation des Tokens
- Lemmatisation ou racinisation (stemming) pour regrouper les variantes
- Gestion des contractions françaises ("du", "des", "au", "aux")
- Détection et préservation des entités nommées (noms de lieux, personnes, organisations)

---

## Phase 3 : Suppression des Mots Vides (Stop Words)

### 3.1 Liste de Stop Words Français

**Sources recommandées** :

1. **NLTK (Natural Language Toolkit)**
   - Liste intégrée : `nltk.corpus.stopwords.words('french')`
   - Contient ~150 mots vides français courants
   - Inclut : articles (le, la, les, un, une, des), prépositions (de, à, dans, sur), pronoms (je, tu, il, elle), conjonctions (et, ou, mais), etc.

2. **spaCy**
   - Liste intégrée dans le modèle français `fr_core_news_sm/md/lg`
   - Plus complète et linguistiquement validée
   - Inclut les variantes avec accents

3. **stop-words (bibliothèque Python dédiée)**
   - Liste spécialisée pour le français
   - Facile à utiliser et maintenue

4. **Liste personnalisée**
   - Créer une liste spécifique au domaine écologique/médias
   - Ajouter des mots fréquents mais peu informatifs dans ce contexte

### 3.2 Mots à Supprimer (Catégories)

**Articles** : le, la, les, un, une, des, du, de la, de l', au, aux

**Pronoms** : je, tu, il, elle, nous, vous, ils, elles, me, te, se, lui, leur, ce, ça

**Prépositions** : de, à, dans, sur, sous, avec, sans, pour, par, entre, parmi

**Conjonctions** : et, ou, mais, donc, car, ni, que, comme, si

**Adverbes courants** : très, plus, moins, aussi, bien, mal, déjà, encore, toujours, jamais

**Verbes auxiliaires** : être, avoir, faire, aller, venir (formes conjuguées)

**Mots de liaison** : alors, ainsi, cependant, toutefois, néanmoins

### 3.3 Mots Spécifiques au Domaine Médias/Écologie (à considérer)

Ces mots peuvent être très fréquents mais peu informatifs dans ce contexte :
- "reportage", "journal", "émission", "télévision", "chaîne"
- "écologie", "environnement", "climat" (selon l'objectif d'analyse)
- Formes verbales courantes : "dit", "fait", "voit", "montre"

**Recommandation** : Créer une liste de stop words étendue spécifique au domaine

---

## Phase 4 : Filtrage des Mots Communs

### 4.1 Critères de Filtrage

**Longueur minimale** :
- Supprimer les mots de moins de 3 caractères (sauf acronymes importants)
- Exemples à supprimer : "si", "en", "un", "il", "ça"

**Fréquence excessive** :
- Identifier les mots apparaissant dans plus de X% des documents (ex: 80-90%)
- Ces mots sont trop communs pour être informatifs
- Exemples potentiels : "écologie", "environnement", "climat" (selon le corpus)

**Mots peu informatifs** :
- Mots génériques : "chose", "fait", "cas", "exemple"
- Verbes très courants : "être", "avoir", "faire", "dire", "voir"
- Adjectifs génériques : "grand", "petit", "bon", "mauvais"

### 4.2 Méthodes de Détection

**TF-IDF (Term Frequency-Inverse Document Frequency)** :
- Calculer le score TF-IDF pour chaque mot
- Filtrer les mots avec un score TF-IDF très faible (trop communs)
- Conserver les mots avec un score élevé (spécifiques et informatifs)

**Fréquence documentaire** :
- Compter dans combien de documents chaque mot apparaît
- Supprimer les mots présents dans >X% des documents

**Fréquence absolue** :
- Compter le nombre total d'occurrences
- Supprimer les mots avec une fréquence absolue très élevée (seuils à définir)

---

## Phase 5 : Extraction et Classement des Mots-Clés

### 5.1 Calcul de Fréquences

**Fréquence absolue** :
- Compter le nombre total d'occurrences de chaque mot dans tout le corpus
- Simple mais peut favoriser les mots très fréquents dans quelques documents

**Fréquence relative** :
- Normaliser par la longueur totale du corpus
- Permet de comparer entre différents corpus

**Fréquence documentaire** :
- Nombre de documents contenant le mot
- Indique la généralité du terme

### 5.2 Scores Avancés

**TF-IDF (Term Frequency-Inverse Document Frequency)** :
- Combine la fréquence dans un document avec la rareté dans le corpus
- Favorise les mots spécifiques et informatifs
- Formule : `tf-idf(t,d) = tf(t,d) × idf(t)`
- `idf(t) = log(N / df(t))` où N = nombre total de documents, df(t) = nombre de documents contenant t

**Score de spécificité** :
- Mesure à quel point un mot est caractéristique d'une catégorie
- Utile pour comparer entre catégories (agriculture vs énergie vs mobilité)

**Pointwise Mutual Information (PMI)** :
- Mesure l'association entre mots
- Utile pour identifier des bigrammes/trigrammes pertinents

### 5.3 Classement Final

**Méthode recommandée** :
1. Calculer le score TF-IDF moyen pour chaque mot
2. Filtrer les mots avec score < seuil minimum
3. Trier par score décroissant
4. Retirer les N premiers mots-clés (ex: top 50, top 100, top 200)

**Variantes** :
- Classement par fréquence absolue (plus simple)
- Classement par fréquence documentaire (mots généraux)
- Classement combiné (moyenne pondérée de plusieurs scores)

---

## Phase 6 : Analyse par Catégories (Optionnel)

### 6.1 Segmentation par Catégorie
- Grouper les textes par `llm_category` ou `predicted_category`
- Appliquer le protocole séparément pour chaque catégorie :
  - Agriculture/Alimentation
  - Mobilité/Transport
  - Énergie
  - Autres

### 6.2 Mots-Clés Spécifiques par Catégorie
- Identifier les mots-clés caractéristiques de chaque catégorie
- Comparer les distributions de fréquences entre catégories
- Détecter les mots-clés discriminants (présents dans une catégorie, absents des autres)

---

## Bibliothèques et Solutions Existantes

### 7.1 Bibliothèques Python Recommandées

#### **NLTK (Natural Language Toolkit)**
- **Utilisation** : Tokenisation, stop words, stemming, lemmatisation
- **Avantages** : Complète, bien documentée, liste de stop words français intégrée
- **Installation** : `pip install nltk`
- **Fonctionnalités** :
  - `nltk.corpus.stopwords.words('french')` : Liste de stop words
  - `nltk.tokenize.word_tokenize()` : Tokenisation
  - `nltk.stem.SnowballStemmer('french')` : Stemming français

#### **spaCy**
- **Utilisation** : Traitement linguistique complet, tokenisation avancée, lemmatisation
- **Avantages** : Modèles français pré-entraînés, très performant, pipeline complet
- **Installation** : `pip install spacy` puis `python -m spacy download fr_core_news_sm`
- **Fonctionnalités** :
  - Tokenisation linguistique précise
  - Détection automatique des stop words
  - Lemmatisation de qualité
  - Reconnaissance d'entités nommées

#### **scikit-learn**
- **Utilisation** : Calcul TF-IDF, vectorisation de texte
- **Avantages** : Intégré dans l'écosystème ML, efficace, bien optimisé
- **Installation** : `pip install scikit-learn`
- **Fonctionnalités** :
  - `TfidfVectorizer` : Calcul TF-IDF avec options de filtrage
  - `CountVectorizer` : Comptage de fréquences
  - Options intégrées pour min_df, max_df (filtrage par fréquence documentaire)

#### **gensim**
- **Utilisation** : Analyse de topics, modélisation thématique, TF-IDF
- **Avantages** : Spécialisé dans l'analyse de textes, modèles de topics (LDA, LSI)
- **Installation** : `pip install gensim`
- **Fonctionnalités** :
  - `gensim.corpora.Dictionary` : Gestion de vocabulaire
  - `gensim.models.TfidfModel` : Modèle TF-IDF
  - `gensim.models.LdaModel` : Analyse thématique (LDA)

#### **stop-words**
- **Utilisation** : Liste spécialisée de stop words par langue
- **Avantages** : Simple, dédié, liste française complète
- **Installation** : `pip install stop-words`
- **Fonctionnalités** :
  - `stop_words.get_stop_words('french')` : Liste de stop words français

#### **TextBlob**
- **Utilisation** : Traitement de texte simplifié, analyse de sentiment
- **Avantages** : Interface simple, bon pour débutants
- **Installation** : `pip install textblob` puis `python -m textblob.download_corpora`
- **Limitation** : Support français limité comparé à spaCy

#### **RAKE (Rapid Automatic Keyword Extraction)**
- **Utilisation** : Extraction automatique de mots-clés et phrases-clés
- **Avantages** : Méthode spécialisée pour l'extraction de mots-clés
- **Installation** : `pip install rake-nltk` ou `pip install python-rake`
- **Fonctionnalités** :
  - Extraction de mots-clés et expressions multi-mots
  - Score basé sur la co-occurrence

### 7.2 Solutions Alternatives

#### **Yake (Yet Another Keyword Extractor)**
- **Utilisation** : Extraction de mots-clés sans supervision
- **Avantages** : Rapide, multilingue (inclut français), pas besoin d'entraînement
- **Installation** : `pip install yake`
- **Fonctionnalités** :
  - Extraction automatique de mots-clés
  - Score de qualité pour chaque mot-clé
  - Support français intégré

#### **KeyBERT**
- **Utilisation** : Extraction de mots-clés basée sur BERT
- **Avantages** : Utilise des embeddings sémantiques, résultats de qualité
- **Installation** : `pip install keybert`
- **Fonctionnalités** :
  - Extraction basée sur similarité sémantique
  - Support multilingue avec modèles appropriés
  - Extraction de phrases-clés, pas seulement mots

#### **pke (Python Keyphrase Extraction)**
- **Utilisation** : Bibliothèque complète d'extraction de mots-clés/phrases-clés
- **Avantages** : Plusieurs algorithmes (TF-IDF, TextRank, TopicRank, etc.)
- **Installation** : `pip install pke`
- **Fonctionnalités** :
  - Multiples méthodes d'extraction
  - Support français
  - Extraction de phrases-clés (n-grammes)

### 7.3 Solutions SQL (PostgreSQL)

#### **PostgreSQL Full-Text Search**
- **Utilisation** : Recherche et analyse de texte directement en SQL
- **Avantages** : Pas besoin d'extraire les données, traitement dans la base
- **Fonctionnalités** :
  - `tsvector` : Type de données pour texte tokenisé
  - `tsquery` : Requêtes de recherche
  - `to_tsvector('french', text)` : Tokenisation avec dictionnaire français
  - `ts_rank()` : Classement par pertinence

#### **pg_trgm (Trigram Extension)**
- **Utilisation** : Similarité de texte, recherche floue
- **Avantages** : Intégré à PostgreSQL, efficace pour grandes quantités de données
- **Fonctionnalités** :
  - Similarité trigramme
  - Indexation pour recherche rapide

### 7.4 Solutions Cloud/SaaS (Non recommandées pour ce projet)

- **Google Cloud Natural Language API** : Analyse de texte, extraction d'entités
- **Azure Text Analytics** : Analyse de texte multilingue
- **AWS Comprehend** : Extraction de mots-clés, analyse de sentiment
- **Note** : Ces solutions ne sont pas recommandées selon les critères du hackathon (FOSS, frugalité)

---

## Recommandations d'Implémentation

### Stack Recommandée (FOSS et Frugale)

**Option 1 : Approche Simple et Rapide**
- **NLTK** : Stop words et tokenisation de base
- **scikit-learn** : Calcul TF-IDF et filtrage
- **pandas** : Manipulation des données
- **Avantages** : Léger, rapide, facile à comprendre

**Option 2 : Approche Linguistique Avancée**
- **spaCy** : Traitement linguistique complet (tokenisation, lemmatisation)
- **scikit-learn** : Calcul TF-IDF
- **pandas** : Manipulation des données
- **Avantages** : Plus précis, meilleure normalisation, lemmatisation de qualité

**Option 3 : Approche Extraction Spécialisée**
- **Yake** ou **pke** : Extraction de mots-clés dédiée
- **spaCy** : Préprocessing linguistique
- **Avantages** : Méthodes optimisées pour l'extraction de mots-clés

**Option 4 : Approche PostgreSQL Native**
- **PostgreSQL Full-Text Search** : Traitement directement en SQL
- **Avantages** : Pas d'extraction nécessaire, très rapide sur grandes quantités

### Critères de Choix

1. **Frugalité** : NLTK + scikit-learn (léger)
2. **Précision** : spaCy + scikit-learn (meilleure normalisation)
3. **Simplicité** : Yake (tout-en-un, peu de configuration)
4. **Performance sur grandes données** : PostgreSQL Full-Text Search

---

## Paramètres et Seuils Recommandés

### Seuils de Filtrage

**Longueur minimale** : 3-4 caractères
- Supprimer les mots < 3 caractères (sauf acronymes)

**Fréquence documentaire maximale** : 80-90%
- Supprimer les mots présents dans >80% des documents (trop communs)

**Fréquence documentaire minimale** : 2-5%
- Conserver les mots présents dans au moins 2-5% des documents (évite les mots trop rares/typos)

**Score TF-IDF minimal** : À calibrer selon le corpus
- Commencer avec un seuil bas (ex: 0.01) et ajuster

**Nombre de mots-clés à extraire** : 50-200
- Top 50 : Mots-clés principaux
- Top 100 : Vue d'ensemble
- Top 200 : Analyse complète

### Paramètres TF-IDF

- **min_df** : 2-5 (minimum de documents contenant le terme)
- **max_df** : 0.8-0.9 (maximum de documents contenant le terme)
- **max_features** : 5000-10000 (limite du vocabulaire)
- **ngram_range** : (1, 2) pour inclure bigrammes si pertinent

---

## Validation et Évaluation

### Métriques de Qualité

1. **Cohérence sémantique** : Les mots-clés extraits sont-ils pertinents au domaine écologique/médias ?
2. **Diversité** : Les mots-clés couvrent-ils bien les différents aspects ?
3. **Spécificité** : Les mots-clés sont-ils discriminants entre catégories ?
4. **Fréquence** : Les mots-clés apparaissent-ils suffisamment souvent pour être significatifs ?

### Tests Recommandés

1. **Test manuel** : Examiner les top 20-50 mots-clés, vérifier leur pertinence
2. **Test par catégorie** : Vérifier que les mots-clés diffèrent entre catégories
3. **Test de stabilité** : Appliquer sur un échantillon, vérifier la cohérence
4. **Comparaison** : Comparer avec des mots-clés manuels ou d'autres méthodes

---

## Étapes de Mise en Œuvre

### Workflow Complet

1. **Extraction** : Récupérer tous les textes depuis PostgreSQL
2. **Préprocessing** : Normalisation, tokenisation
3. **Nettoyage** : Suppression stop words, filtrage longueur
4. **Lemmatisation** : Normalisation des formes (optionnel mais recommandé)
5. **Calcul TF-IDF** : Vectorisation et calcul des scores
6. **Filtrage** : Application des seuils (min_df, max_df, score TF-IDF)
7. **Classement** : Tri par score décroissant
8. **Extraction** : Sélection des top N mots-clés
9. **Validation** : Vérification manuelle de la pertinence
10. **Export** : Sauvegarde des résultats (CSV, JSON, ou retour en base)

### Itérations et Ajustements

- Commencer avec des paramètres conservateurs
- Examiner les résultats
- Ajuster les seuils selon les besoins
- Tester différentes combinaisons de bibliothèques
- Comparer les résultats entre méthodes

---

## Notes Finales

- Ce protocole est conçu pour être **frugal** (peu de ressources) et utiliser des **outils FOSS**
- Les solutions cloud/Big Tech sont déconseillées selon les critères du hackathon
- L'approche peut être adaptée selon les besoins spécifiques du projet
- La validation manuelle reste importante pour garantir la qualité des résultats
- Considérer l'analyse par catégories pour des insights plus fins

---

## Références

- Documentation NLTK : https://www.nltk.org/
- Documentation spaCy : https://spacy.io/
- Documentation scikit-learn : https://scikit-learn.org/
- Documentation PostgreSQL Full-Text Search : https://www.postgresql.org/docs/current/textsearch.html
- Documentation Yake : https://github.com/LIAAD/yake
- Documentation pke : https://github.com/boudinfl/pke

