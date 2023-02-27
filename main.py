import sys
from absl import flags
from cuisine_classifier import CookingClassifier


FLAGS = flags.FLAGS
flags.DEFINE_boolean(name="do_train",
                     default=False,
                     help="")

flags.DEFINE_list(name="pred_ingredients",
                     default=[],
                     help="")

flags.DEFINE_string(name="asset_file_path",
                     default="./assets/",
                     help="")


def main():
    FLAGS(sys.argv)
    c = CookingClassifier(do_train=FLAGS.do_train,
                          pred_ingredients=FLAGS.pred_ingredients,
                          asset_file_path=FLAGS.asset_file_path)
    c.execute()


if __name__ == "__main__":
    main()
