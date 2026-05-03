from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
emails = [
    # SPAM (1)
    "Win money now",
    "Congratulations you won a lottery",
    "Claim your free prize immediately",
    "Earn cash fast online",
    "You have been selected for a reward",
    "Click here to get free iPhone",
    "Limited offer buy now and save money",
    "You won a gift card worth $1000",
    "Urgent: update your bank details",
    "Get rich quick opportunity",

    # NOT SPAM (0)
    "Meeting scheduled at 10am tomorrow",
    "Please submit your assignment",
    "Let's have lunch today",
    "Project deadline is next week",
    "Your invoice is attached",
    "Are we meeting today?",
    "Please call me when you are free",
    "Office will remain closed on Sunday",
    "Doctor appointment is confirmed",
    "Happy birthday, have a great day"
]

labels = [
    1,1,1,1,1,1,1,1,1,1,
    0,0,0,0,0,0,0,0,0,0
]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(emails)

y = labels

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
model = MultinomialNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))