# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV: 2A202603736**
- **Họ và tên: Vũ Quang Thanh**
- **SEED** (in ở ô đầu notebook): 3650

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ✅ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ✅ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ✅ |

## 2. Kết quả chính

*Chép bảng so sánh cuối Phần 5 (F1 test, F1 SUB 40, p95 ms/tin, USD/1k tin) của chính bạn.*
|Mô hình	|F1 test	|F1 SUB 40	|p95 ms/tin	|USD/1k tin |
|1. BoW + LogReg (tự code)	|0.882	|1.000	|0.207	|0.0 |
|2. BoW + MLP (deep learning)	|0.934	|1.000	|3.060	|0.0|
|3a. Encoder (MiniLM) + kNN	|0.931	|1.000	|89.136	|0.0|
|3b. Encoder (MiniLM) + LogReg	|0.955	|1.000	|37.353	|0.0|
|3c. Encoder (MiniLM) + MLP head	|0.960	|1.000	|33.555	|0.0|
|4a. LLM nhỏ zero-shot	|0.531	|0.531	|2499.432	|0.0|
|4b. LLM lớn zero-shot	|0.851	|0.851	|10653.201	|0.0|
|4c. LLM nhỏ few-shot	|0.000	|0.000	|2296.176	|0.0|

## 3. Dự đoán và thực tế

*Chọn 3 dự đoán `predict(...)` lệch nhiều nhất so với kết quả. Mỗi cái: bạn đoán gì, ra gì, vì sao lệch.*
1. emb_search
Dự đoán: Kết quả gần nhất là ham.
Kết quả thực tế: Cả 6 tin nhắn gần nhất đều được gắn nhãn spam.
Em đã đánh giá chưa đúng khả năng của embedding. MiniLM không chỉ tìm các từ giống nhau mà còn biểu diễn ý nghĩa của cả câu. Câu hỏi có nội dung về nhận thưởng, kỳ nghỉ miễn phí và yêu cầu phản hồi để nhận quà, nên nó gần nghĩa với những tin spam quảng cáo trúng thưởng trong tập train.
2. llm_fewshot
Dự đoán: Few-shot sẽ làm F1 tăng, chi phí mỗi tin tăng khoảng 2.5 lần.
Kết quả thực tế: F1 = 0.000, giảm từ 0.531 của LLM nhỏ zero-shot xuống 0.000.
Việc bổ sung 4 ví dụ vào prompt không đảm bảo mô hình hiểu tốt hơn. Prompt dài hơn khiến mô hình phải xử lý thêm nhiều nội dung, trong khi việc đưa ví dụ vào chưa chắc giúp nó phân biệt tốt hai lớp spam và ham.
3. tradeoff
Dự đoán: Cách 2 và 3b sẽ là những phương án tốt trên đường Pareto.
Kết quả thực tế: Chỉ có 1. BoW + LogReg (tự code) nằm trên đường Pareto.
Đường Pareto thể hiện sự đánh đổi giữa độ trễ và lỗi phân loại (1 − F1). Một mô hình bị lấn át khi tồn tại mô hình khác vừa nhanh hơn vừa có lỗi thấp hơn.
Em đánh giá quá cao lợi ích của việc tăng độ phức tạp mô hình và chưa chú ý đầy đủ đến sự đánh đổi giữa tốc độ và độ chính xác.

## 4. Phân tích 50 case khó

*Model nào tụt nhiều nhất, ở nhóm case nào? Nêu 2 case cụ thể và cơ chế khiến model sai.*
BoW + LogReg tụt nhiều nhất, giảm 0.714 điểm F1, từ 1.000 xuống 0.286, ở nhóm spam giọng tự nhiên, chỉ đạt 0% với LogReg. 
2 case cụ thể: "hi there, sorry to bother you, my friend gave me your number, I'm new in town and looking for someone to chat with, check my profile at meetlocal.club" | "hello, its me Kevin, your teamate, text me later in meetlocal.club for project detail"
Tin nhắn sử dụng giọng điệu thân thiện, lịch sự giống như một người quen mới đang muốn trò chuyện, không sử dụng những từ khóa spam phổ biến như free, win hay prize. BoW không nhận diện được những từ khóa quen thuộc.

## 5. Thử thách cá nhân

*Đề theo SEED của bạn, cách làm, kết quả, giải thích.*
Đề bài: DL – Đổi lớp ẩn MLP thành 4 và 512 nơ-ron. Báo số tham số, F1 test, thời gian train. Khi nào nên chọn mạng nhỏ?
Cách làm: 
Em sử dụng lại hàm build_mlp() và train_mlp() đã viết ở Phần 2, thay đổi số lượng nơ-ron lớp ẩn lần lượt thành 4 và 512, giữ nguyên các điều kiện huấn luyện:
- Epochs = 30.
- Learning rate = 0.001.
- Batch size = 64.
- Cùng tập train, validation, test
Kết quả:
|Chỉ số	|MLP 4 nơ-ron	|MLP 512 nơ-ron|
|Số tham số	|8,009	|1,025,025|
|Accuracy	|0.9832	|0.9806|
|Precision	|1.0000	|0.9785|
|Recall	|0.8750	|0.8750|
|F1-score	|0.9333	|0.9239|
|Train time	|1375.485 giây	|1431.186 giây|
Giải thích
Khi tăng số nơ-ron từ 4 lên 512, số lượng tham số tăng từ 8,009 lên 1,025,025, giúp mạng có khả năng học những mối quan hệ phức tạp hơn nhưng đồng thời làm tăng tài nguyên cần thiết. Tuy nhiên, kết quả thực nghiệm cho thấy MLP 4 nơ-ron có F1 = 0.9333, cao hơn MLP 512 nơ-ron với F1 = 0.9239.
Nguyên nhân là mạng có quá nhiều tham số có thể học những đặc điểm không thực sự cần thiết, làm tăng nguy cơ overfitting.

## 6. Phương án cho sản phẩm thật

*Với yêu cầu p95 < 50 ms/tin cho 10 triệu SMS/ngày: chọn phương án nào, dẫn số liệu nào, chấp nhận đánh đổi gì.*
Em chọn 3c: Encoder (MiniLM) + MLP head vì phương án này có thể đạt cả độ trễ lẫn chất lượng.
F1 test cao nhất: 0.960.
p95 = 33.555 ms/tin, thấp hơn yêu cầu 50 ms/tin.
F1 SUB 40 = 1.000.
Chi phí API = 0 USD/1k tin vì chạy mô hình local.
Mô hình có thể hiểu ngữ nghĩa của câu nhờ MiniLM embedding, thay vì chỉ dựa vào các từ khóa riêng lẻ.
Đánh đổi: thời gian encode và triển khai ban đầu cao hơn BoW vì phải sử dụng MiniLM, tài nguyên tính toán cao hơn do phải chạy encoder, 

## 7. AI log và điều đã học

*Mỗi lần dùng AI: hỏi gì, nhận gì, bạn sửa gì. Sau đó 2–3 câu: điều bạn hiểu ra hôm nay mà trước đó chưa hiểu.*
