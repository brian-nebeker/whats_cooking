from cuisine_classifier import CookingClassifier

if __name__ == "__main__":
    pass

c = CookingClassifier()

c._read_data()

train = pd.read_json('./data/train.json')
train['num_ingredients'] = train['ingredients'].apply(len)
train = train[train['num_ingredients'] > 1]