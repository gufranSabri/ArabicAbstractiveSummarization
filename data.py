from torch.utils.data import Dataset
from arabert.preprocess import ArabertPreprocessor

class MultiTaskDataset(Dataset):
    def __init__(self, tokenizer, task2_data=None, summarization_data=None,
                 max_length=128, summary_max_length=40, 
                 task2_text_col="First sentence", task2_label_col="second sentence"):

        self.tokenizer = tokenizer
        self.preprocessor = ArabertPreprocessor(model_name="")
        self.max_length = max_length
        self.summary_max_length = summary_max_length

        self.data = []
        if task2_data is not None:
            for index, row in task2_data.iterrows():
                self.data.append((row[task2_text_col], row[task2_label_col], 'task2'))

        if summarization_data is not None:
            for index, row in summarization_data.iterrows():
                self.data.append((row['text'], row['headline'], 'abstractive'))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        passage, answer, task = self.data[index]

        if task == 'task2' or task == 'abstractive':
            input_text = "summarize: " + self.preprocessor.preprocess(passage)
            target_text = self.preprocessor.preprocess(answer)
        else:
            raise ValueError("Task must be 'task2' or 'abstractive'")

        input_encoding = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )

        target_encoding = self.tokenizer(
            target_text,
            max_length=self.summary_max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )

        labels = target_encoding.input_ids
        labels[labels == self.tokenizer.pad_token_id] = -100  # Cross-entropy ignore index

        return {
            'input_ids': input_encoding.input_ids.squeeze(0),
            'attention_mask': input_encoding.attention_mask.squeeze(0),
            'labels': labels.squeeze(0),
            'task': task
        }



if __name__ == "__main__":
    pass