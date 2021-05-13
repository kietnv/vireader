# ViReader
ViReader is a novel Vietnamese MRC system that implementsa transfer learning-based sentence retriever and a XLM-RoBERTa-based answerextractor. As a result, our system outperform the DrQA-based system by 30.83 and 26.10 percentage points in terms of answer prediction EM-score and F1-score, respectively. Significantly, our proposed system outperforms the original XLM-R based MRC system in terms of performance and speed.

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

# Dataset

UIT-ViQuAD [1] is available at: https://sites.google.com/uit.edu.vn/kietnv/datasets

# References 

[1] Kiet Van Nguyen, Duc-Vu Nguyen, Anh Gia-Tuan Nguyen, Ngan Luu-Thuy Nguyen. A Vietnamese Dataset for Evaluating Machine Reading Comprehension. COLING 2020.

