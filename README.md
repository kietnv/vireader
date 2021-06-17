# ViReader

ViReader is an open-domain machine reading comprehension system for the Vietnamese language by using Wikipedia as the source of textual knowledge, where the answer to any particular question is a textual span derived directly from text from Vietnamese Wikipedia. Our system combines a sentence retriever component, based on techniques of information retrieval to extract the relevant sentences, with a transfer learning-based answer extractor trained to predict answers based on Wikipedia texts. Experiments on multiple datasets for machine reading comprehension in Vietnamese and other languages demonstrate that (1) our ViReader system is highly competitive with prevalent machine learning-based systems, and (2) multi-task learning by using a combination consisting of the sentence retriever and answer extractor is an end-to-end reading comprehension system. The sentence retriever component of our proposed system retrieves the sentences that are most likely to provide the answer response to the given question. The transfer learning-based answer extractor then reads the document from which the sentences have been retrieved, predicts the answer, and returns it to the user. The ViReader system achieves the state-of-the-art performances, with values of 70.83% and 89.54% of the exact match (EM) and F1, respectively, outperforming the BERT-based system by 11.55% and 9.54%, respectively. It also obtains state-of-the-art performance on ViNewsQA (another Vietnamese dataset consisting of online health-domain news) and BiPaR (a bilingual dataset on English and Chinese novel texts). Compared with the BERT-based system, we achieve significant improvements (in terms of F1) with 7.56% for English and 6.13% for Chinese on the BiPaR dataset.

Please cite paper if you use or refer ViReader: Kiet Van Nguyen, Nhat Duy Nguyen, Phong Nguyen-Thuan Do, Anh Gia-Tuan Nguyen, Ngan Luu-Thuy Nguyen. ViReader: A Wikipedia-based Vietnamese reading comprehension system using transfer learning. Journal of Intelligent and Fuzzy Systems.

# Dataset 

ViReader is trained on the large-scale Vietnamese dataset (UIT-ViQuAD [1]) for evaluating machine reading comprehension. 

# Requirements 

!pip install sentence-transformers 

!pip install underthesea 

!pip install transformers==3.5.1

or 

!pip install -r requirements.txt

# Runing 

from vireader import ViReader

#load model

myReader = ViReader()

#For instance

context = "Trái Đất được hình thành cùng với Hệ Mặt Trời từ khi Hệ Mặt Trời ban đầu tồn tại như 1 đám mây bụi và khí lớn, quay tròn, gọi là tinh vân Mặt Trời. Tinh vân này gồm hydro và heli được tạo ra từ Vụ Nổ Lớn, và những nguyên tố hóa học nặng hơn khác được tạo ra từ các ngôi sao đã chết. Sau đó, vào khoảng 4,6 tỷ năm trước (15 đến 30 phút trước khi chiếc đồng hồ tưởng tượng của chúng ta bắt đầu chạy), có thể 1 ngôi sao ở gần đó bắt đầu trở thành 1 siêu tân tinh. Vụ nổ gây sóng chấn động về hướng tinh vân Mặt Trời và làm nó bị nén vào. Vì đám mây tiếp tục quay, lực hấp dẫn và quán tính làm đám mây trở nên phẳng như hình dạng một cái đĩa, vuông góc so với trục quay của nó. Đa phần khối lượng tập trung ở giữa và bắt đầu nóng lên. Lúc ấy, khi trọng lực làm cho vật chất cô đặc lại xung quanh các hạt bụi vật chất, phần còn lại của đĩa bắt đầu tan rã thành những vành đai. Các mảnh nhỏ va chạm vào nhau và tạo thành những mảnh lớn hơn.. Những mảnh nằm trong tập hợp nằm cách trung tâm khoảng 150 triệu kilômét tạo thành Trái Đất. Khi Mặt Trời ngày càng đặc lại, nó nóng lên, phản ứng hạt nhân bùng nổ và tạo nên gió Mặt Trời thổi bay đa phần những vật chất ở trong đĩa vẫn còn chưa bị cô đặc vào những tập hợp vật chất lớn hơn."

question = 	"Hệ Mặt Trời khi còn là mọt đám bụi khí được gọi là gì?"

#predict answer

answer = myReader.predict(context, question)

#print the predicted answer 

print(answer)

# Training

Phase 1: Sentence Retriever

Use the Sentence_Retriever.ipynb for the sentence retriever component of the ViReader system.

Phase 2: Answer Extractor

Use Answer_Extractor_Training.ipynb and Answer_Extractor_Testing.ipynb for the training and testing phases of the answer extractor component of the ViReader system.

# Evaluation on Other Datasets

ViNewsQA is available at: https://sites.google.com/uit.edu.vn/kietnv/datasets

BiPaR is available at: https://multinlp.github.io/BiPaR

# References 

[1] Kiet Van Nguyen, Duc-Vu Nguyen, Anh Gia-Tuan Nguyen, Ngan Luu-Thuy Nguyen. A Vietnamese Dataset for Evaluating Machine Reading Comprehension. COLING 2020.

[2] Jing, Yimin, Deyi Xiong, and Zhen Yan. "BiPaR: A Bilingual Parallel Dataset for Multilingual and Cross-lingual Reading Comprehension on Novels." Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP). 2019.

[3] Van Nguyen, Kiet, Tin Van Huynh, Duc-Vu Nguyen, Anh Gia-Tuan Nguyen, and Ngan Luu-Thuy Nguyen. "New vietnamese corpus for machine reading comprehension of health news articles." arXiv preprint arXiv:2006.11138 (2020).
