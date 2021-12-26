# Mô hình hệ thống trích chọn câu trả lời tiếng Việt.

## Hệ thống cấu tạo từ 2 mô hình Retriever & Reader, huấn luyện trên bộ dữ liệu ViQuAD

## Yêu cầu:

Các file dữ liệu `train_ViQuAD.json`, `dev_ViQuAD.json`, `test_ViQuAD.json` đặt tại folder `data/`

Tạo & kích hoạt môi trường:

```bash
virtualenv venv
source venv/bin/activate
```

Cài các gói cần thiết:

```bash
pip install -r requirements.txt
```

## Cách thức huấn luyện và đánh giá các mô hình

### Retriever:

B1. Tiền xử lý dữ liệu:

```bash
python prep_data_retrieval.py
```

B2. Huấn luyện:

Retriever sử dụng `bert-base-multilingual-cased`:

```bash
python train_retrieval.py
```

Retriever sử dụng `vinai/phobert-base`:

```bash
python train_retrieval_phoBert.py
```

Retriever sử dụng hybrid `vinai/phobert-base` và `BM25`:

```bash
python meta_model.py
```

Các kết quả đánh giá các retriever trong folder `logs/retriever_logs/`

### Reader:

B1. Tiền xử lý dữ liệu

```bash
python prep_data_extraction.py
```

B2. Huấn luyện và đánh giá mô hình

```bash
python train_extraction.py --model_type=<MODEL_TYPE>
```

trong đó, `<MODEL_TYPE>` = `bert/phobert`

### Evaluate toàn bộ hệ thống:

```bash
python eval_full_system.py --retriever_type=<RETRIEVER_TYPE> --retriever_size=<RETRIEVER_SIZE>
--reader_checkpoint_path=<READER_CHECKPOINT_PATH>
```

### Inference:

```python3
from predictor import ReaderPredictor, QASystem, BM25Retriever, SemanticsRetriever, HybridRetriever

retriever = HybridRetriever()
reader = ReaderPredictor()
system = QASystem(retriever=retriever, reader=reader)
question = "Nước Đức hiện nay sự hợp thành của hai nước nào trong thời kỳ Đồng minh chiếm đóng Đức?"
answer = system.predict(question)
```
