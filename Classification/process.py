import numpy as np
import preprocessing


text_label = preprocessing.text_labels("./20_newsgroups")
text, label = preprocessing.process_all_text(text_label)
tfidf_text = preprocessing.tfidf(text)
np.save("tfidf_text.npy", tfidf_text)
np.save("label.npy", label)
