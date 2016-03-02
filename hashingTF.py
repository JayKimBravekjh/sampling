from pyspark.mllib.feature import HashingTF

sentence = "hello hello world"
words = sentence.split()  # split sentence into a list of terms
tf = HashingTF(10000)  # Create vectors of size S = 10,000
tf.transform(words)
# SparseVector(10000, (3065: 1.0, 6861:2.0})

rdd = sc.wholeTextFiles("data").map(lambda(name, text): text.split())
tfVectors = tf.transform(rdd)
