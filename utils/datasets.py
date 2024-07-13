import torch
from torch.utils.data import Dataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def pad_collate(batch):
    (xx, yy, lens) = zip(*batch)
    xx_pad = torch.nn.utils.rnn.pad_sequence(xx, batch_first=True, padding_value=0)
    yy = torch.tensor(yy)
    return xx_pad, yy, lens

class TwitterDatasetForLSTM(Dataset):
    def __init__(self, tweets, emotions):
        super().__init__()
        self.tweets = tweets
        self.lengths = [len(tweet) for tweet in tweets]
        self.emotions = emotions

    def __len__(self):
        return len(self.tweets)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        tweet = self.tweets[idx]
        length = self.lengths[idx]
        label = self.emotions[idx]
        sample = (tweet, label, length)

        return sample

class TwitterDatasetForBert(Dataset):
    def __init__(self, tweets, emotions, masks,):
        super().__init__()
        self.tweets = tweets
        self.masks = masks
        self.emotions = emotions

    def __len__(self):
        return len(self.tweets)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        tweet = self.tweets[idx, :]
        mask = self.masks[idx, :]
        label = self.emotions[idx]
        sample = (tweet, label, mask)

        return sample

class TwitterDataset(Dataset):
    def __init__(self, tweets, emotions):
        super().__init__()
        self.tweets = tweets
        self.emotions = emotions

    def __len__(self):
        return len(self.tweets)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        tweet = self.tweets[idx]
        label = self.emotions[idx]
        sample = (tweet, label)

        return sample

class TwitterGenTrainDataset(Dataset):
    def __init__(self, tweets, context_length, emotions):
        super().__init__()

        self.X = []
        self.Y = []
        self.emotions = []
        self.context_length = context_length

        # build a new dataset with context windows
        for i, tweet in enumerate(tweets):
            if len(tweet) <= context_length:
                continue
            self.emotions += [emotions[i]] * (len(tweet)-context_length)
            for j in range(len(tweet)-context_length):
                self.X.append(tweet[j:j+context_length])
                self.Y.append(tweet[j+1:j+1+context_length])
        self.X = torch.stack(self.X).to(device, dtype=torch.long)
        self.Y = torch.stack(self.Y).to(device, dtype=torch.long)
        self.emotions = torch.stack(self.emotions).to(device, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        X = self.X[idx]
        Y = self.Y[idx]
        emotion = self.emotions[idx]
        return (X, Y, emotion)