echo "writing output to datasets/fasttext/$1"
mkdir datasets/fasttext/$1

python week3/createContentTrainingData.py --input data/pruned_products --output datasets/fasttext/$1/output.fasttext --min_products 200 --category_depth 3

lines=`wc -l datasets/fasttext/$1/output.fasttext`
echo "filtered to $lines products"

gshuf datasets/fasttext/$1/output.fasttext > datasets/fasttext/$1/output-random.fasttext
head -n 10000 datasets/fasttext/$1/output-random.fasttext > datasets/fasttext/$1/output-training.fasttext
tail -10000 datasets/fasttext/$1/output-random.fasttext > datasets/fasttext/$1/output-test.fasttext

mkdir week3/models/$1

echo""
echo "try baseline"
~/code/search-ml/fastText/fasttext supervised -input ~/code/search-ml/search_with_machine_learning_course/datasets/fasttext/$1/output-training.fasttext -output ~/code/search-ml/search_with_machine_learning_course/week3/models/$1/week3_model_base
~/code/search-ml/fastText/fasttext test ~/code/search-ml/search_with_machine_learning_course/week3/models/$1/week3_model_base.bin ~/code/search-ml/search_with_machine_learning_course/datasets/fasttext/$1/output-test.fasttext

echo""
echo "try with -epoch 25 -lr 1.0"
~/code/search-ml/fastText/fasttext supervised -input ~/code/search-ml/search_with_machine_learning_course/datasets/fasttext/$1/output-training.fasttext -output ~/code/search-ml/search_with_machine_learning_course/week3/models/$1/week3_model_epoch_lr -epoch 25 -lr 1.0
~/code/search-ml/fastText/fasttext test ~/code/search-ml/search_with_machine_learning_course/week3/models/$1/week3_model_epoch_lr.bin ~/code/search-ml/search_with_machine_learning_course/datasets/fasttext/$1/output-test.fasttext

echo""
echo "try with -epoch 25 -lr 1.0 -wordNgrams 2"
~/code/search-ml/fastText/fasttext supervised -input ~/code/search-ml/search_with_machine_learning_course/datasets/fasttext/$1/output-training.fasttext -output ~/code/search-ml/search_with_machine_learning_course/week3/models/$1/week3_model_epoch_lr_ngrams -epoch 25 -lr 1.0
~/code/search-ml/fastText/fasttext test ~/code/search-ml/search_with_machine_learning_course/week3/models/$1/week3_model_epoch_lr_ngrams.bin ~/code/search-ml/search_with_machine_learning_course/datasets/fasttext/$1/output-test.fasttext