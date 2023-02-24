from cuisine_classifier import CookingClassifier

if __name__ == "__main__":
    ingredients = ['dough', 'tomatoes', 'mozzarella', 'pepperoni']

    c = CookingClassifier(do_train=False, pred_ingredients=ingredients)
    c.execute()