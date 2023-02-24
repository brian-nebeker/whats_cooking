from cuisine_classifier import CookingClassifier

if __name__ == "__main__":
    ingredients = ['bacon', 'salt', 'flour', 'mushrooms', 'carrots', 'celery', 'garlic', 'tomato paste', 'beef broth', 'bay leaves', 'thyme', 'parsley', 'rosemary', 'potatoes', 'peas']

    c = CookingClassifier(do_train=True, pred_ingredients=ingredients)
    c.execute()