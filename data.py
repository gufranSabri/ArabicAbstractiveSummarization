from torch.utils.data import Dataset

class MultiTaskDataset(Dataset):
    def __init__(self, tokenizer, qa_data=None, summarization_data=None, max_length=128, summary_max_length=40):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.summary_max_length = summary_max_length

        self.data = []
        if qa_data is not None:
            for index, row in qa_data.iterrows():
                self.data.append((row['Article_processed'], row['Summary_1'], 'qa'))

        if summarization_data is not None:
            for index, row in summarization_data.iterrows():
                self.data.append((row['text_preprocessed'], row['headline'], 'abstractive'))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        passage, answer, task = self.data[index]

        if task == 'qa' or task == 'abstractive':
            input_text = "summarize: " + passage
            target_text = answer
        else:
            raise ValueError("Task must be 'qa' or 'abstractive'")

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
        labels[labels == self.tokenizer.pad_token_id] = -100  # Replace padding token id's with -100 for cross-entropy

        return {
            'input_ids': input_encoding.input_ids.squeeze(0),
            'attention_mask': input_encoding.attention_mask.squeeze(0),
            'labels': labels.squeeze(0),
            'decoder_attention_mask': target_encoding.attention_mask.squeeze(0),
            'task': task
        }


if __name__ == "__main__":
    pass