import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer

# Step 1: Sample sentences with real-world categories
examples = [
    ("I love machine learning.", "AI/ML"),
    ("Deep learning is a branch of AI.", "AI/ML"),
    ("Natural Language Processing is fun.", "AI/ML"),
    ("This is not a spam message!", "Normal"),
    ("Python is great for data science.", "AI/ML"),

    ("Python is great for data science.", "AI/ML"),
    ("Important update regarding your account.", "Normal"),

    ("Free money offer, click now!", "Spam"),
    ("You have won a prize, claim now.", "Spam"),

    ("Let's schedule a meeting for tomorrow.", "Meeting"),
    ("Can we catch up later this week?", "Meeting"),
    ("Hey, how are you doing today?", "Casual"),

    ("What is the weather like in New York?", "Weather"),
]

sentences = [text for text, label in examples]
labels = [label for text, label in examples]

# Step 2: Load pretrained embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Step 3: Convert sentences into embeddings
embeddings = model.encode(sentences)

print("Embedding shape:", embeddings.shape)
print("One sentence embedding size:", len(embeddings[0]))

# Step 4: Reduce embeddings to 2D using PCA
pca = PCA(n_components=2)
reduced = pca.fit_transform(embeddings)

# Step 5: Plot
plt.figure(figsize=(12, 7), dpi=100)

for i, sentence in enumerate(sentences):
    x, y = reduced[i]
    plt.scatter(x, y)
    plt.text(x + 0.01, y + 0.01, f"S{i+1}", fontsize=9)

plt.title("Sentence Embeddings Visualized using PCA")
plt.xlabel("PCA Dimension 1")
plt.ylabel("PCA Dimension 2")
plt.grid(True)
#plt.tight_layout()
plt.show()
print("\nSentence labels:")
for i, sentence in enumerate(sentences):
    print(f"S{i+1}: {labels[i]} - {sentence}")