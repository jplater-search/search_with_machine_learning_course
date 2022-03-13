gshuf local/data/labeled_query_data-100.txt  > local/data/labeled_query_data-100-random.txt
head -n 50000 local/data/labeled_query_data-100-random.txt > local/data/labeled_query_data-100-training.txt
tail -50000 local/data/labeled_query_data-100-random.txt > local/data/labeled_query_data-100-testing.txt

mkdir week4/models/100

echo""
echo "min-query 100: try baseline"
~/code/search-ml/fastText/fasttext supervised -input ~/code/search-ml/search_with_machine_learning_course/local/data/labeled_query_data-100-training.txt -output ~/code/search-ml/search_with_machine_learning_course/week4/models/100/week4_model_base
~/code/search-ml/fastText/fasttext test ~/code/search-ml/search_with_machine_learning_course/week4/models/100/week4_model_base.bin ~/code/search-ml/search_with_machine_learning_course/local/data/labeled_query_data-100-testing.txt
~/code/search-ml/fastText/fasttext test ~/code/search-ml/search_with_machine_learning_course/week4/models/100/week4_model_base.bin ~/code/search-ml/search_with_machine_learning_course/local/data/labeled_query_data-100-testing.txt 3
~/code/search-ml/fastText/fasttext test ~/code/search-ml/search_with_machine_learning_course/week4/models/100/week4_model_base.bin ~/code/search-ml/search_with_machine_learning_course/local/data/labeled_query_data-100-testing.txt 5


echo""
echo "min-query 100: try with -epoch 25"
~/code/search-ml/fastText/fasttext supervised -input ~/code/search-ml/search_with_machine_learning_course/local/data/labeled_query_data-100-training.txt -output ~/code/search-ml/search_with_machine_learning_course/week4/models/100/week4_model_epoch -epoch 25
~/code/search-ml/fastText/fasttext test ~/code/search-ml/search_with_machine_learning_course/week4/models/100/week4_model_epoch.bin ~/code/search-ml/search_with_machine_learning_course/local/data/labeled_query_data-100-testing.txt
~/code/search-ml/fastText/fasttext test ~/code/search-ml/search_with_machine_learning_course/week4/models/100/week4_model_epoch.bin ~/code/search-ml/search_with_machine_learning_course/local/data/labeled_query_data-100-testing.txt 3
~/code/search-ml/fastText/fasttext test ~/code/search-ml/search_with_machine_learning_course/week4/models/100/week4_model_epoch.bin ~/code/search-ml/search_with_machine_learning_course/local/data/labeled_query_data-100-testing.txt 5


echo""
echo "min-query 100: try with -epoch 25 -wordNgrams 2"
~/code/search-ml/fastText/fasttext supervised -input ~/code/search-ml/search_with_machine_learning_course/local/data/labeled_query_data-100-training.txt -output ~/code/search-ml/search_with_machine_learning_course/week4/models/100/week4_model_epoch_ngrams -epoch 25 -wordNgrams 2
~/code/search-ml/fastText/fasttext test ~/code/search-ml/search_with_machine_learning_course/week4/models/100/week4_model_epoch_ngrams.bin ~/code/search-ml/search_with_machine_learning_course/local/data/labeled_query_data-100-testing.txt
~/code/search-ml/fastText/fasttext test ~/code/search-ml/search_with_machine_learning_course/week4/models/100/week4_model_epoch_ngrams.bin ~/code/search-ml/search_with_machine_learning_course/local/data/labeled_query_data-100-testing.txt 3
~/code/search-ml/fastText/fasttext test ~/code/search-ml/search_with_machine_learning_course/week4/models/100/week4_model_epoch_ngrams.bin ~/code/search-ml/search_with_machine_learning_course/local/data/labeled_query_data-100-testing.txt 5


gshuf local/data/labeled_query_data-1000.txt  > local/data/labeled_query_data-1000-random.txt
head -n 10000 local/data/labeled_query_data-1000-random.txt > local/data/labeled_query_data-1000-training.txt
tail -10000 local/data/labeled_query_data-1000-random.txt > local/data/labeled_query_data-1000-testing.txt

mkdir week4/models/1000

echo""
echo "min-query 1000: try baseline"
~/code/search-ml/fastText/fasttext supervised -input ~/code/search-ml/search_with_machine_learning_course/local/data/labeled_query_data-1000-training.txt -output ~/code/search-ml/search_with_machine_learning_course/week4/models/1000/week4_model_base
~/code/search-ml/fastText/fasttext test ~/code/search-ml/search_with_machine_learning_course/week4/models/1000/week4_model_base.bin ~/code/search-ml/search_with_machine_learning_course/local/data/labeled_query_data-1000-testing.txt
~/code/search-ml/fastText/fasttext test ~/code/search-ml/search_with_machine_learning_course/week4/models/1000/week4_model_base.bin ~/code/search-ml/search_with_machine_learning_course/local/data/labeled_query_data-1000-testing.txt 3
~/code/search-ml/fastText/fasttext test ~/code/search-ml/search_with_machine_learning_course/week4/models/1000/week4_model_base.bin ~/code/search-ml/search_with_machine_learning_course/local/data/labeled_query_data-1000-testing.txt 5

echo""
echo "min-query 1000: try with -epoch 25 -lr 1.0"
~/code/search-ml/fastText/fasttext supervised -input ~/code/search-ml/search_with_machine_learning_course/local/data/labeled_query_data-1000-training.txt -output ~/code/search-ml/search_with_machine_learning_course/week4/models/1000/week4_model_epoch -epoch 25
~/code/search-ml/fastText/fasttext test ~/code/search-ml/search_with_machine_learning_course/week4/models/1000/week4_model_epoch.bin ~/code/search-ml/search_with_machine_learning_course/local/data/labeled_query_data-1000-testing.txt
~/code/search-ml/fastText/fasttext test ~/code/search-ml/search_with_machine_learning_course/week4/models/1000/week4_model_epoch.bin ~/code/search-ml/search_with_machine_learning_course/local/data/labeled_query_data-1000-testing.txt 3
~/code/search-ml/fastText/fasttext test ~/code/search-ml/search_with_machine_learning_course/week4/models/1000/week4_model_epoch.bin ~/code/search-ml/search_with_machine_learning_course/local/data/labeled_query_data-1000-testing.txt 5

echo""
echo "min-query 1000: try with -epoch 25 -lr 1.0 -wordNgrams 2"
~/code/search-ml/fastText/fasttext supervised -input ~/code/search-ml/search_with_machine_learning_course/local/data/labeled_query_data-1000-training.txt -output ~/code/search-ml/search_with_machine_learning_course/week4/models/1000/week4_model_epoch_ngrams -epoch 25 -wordNgrams 2
~/code/search-ml/fastText/fasttext test ~/code/search-ml/search_with_machine_learning_course/week4/models/1000/week4_model_epoch_ngrams.bin ~/code/search-ml/search_with_machine_learning_course/local/data/labeled_query_data-1000-testing.txt
~/code/search-ml/fastText/fasttext test ~/code/search-ml/search_with_machine_learning_course/week4/models/1000/week4_model_epoch_ngrams.bin ~/code/search-ml/search_with_machine_learning_course/local/data/labeled_query_data-1000-testing.txt 3
~/code/search-ml/fastText/fasttext test ~/code/search-ml/search_with_machine_learning_course/week4/models/1000/week4_model_epoch_ngrams.bin ~/code/search-ml/search_with_machine_learning_course/local/data/labeled_query_data-1000-testing.txt 5


