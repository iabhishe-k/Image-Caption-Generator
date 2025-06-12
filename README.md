# Image Captioning with VGG16 + LSTM (Flickr8k)
This project implements an image captioning model using a pre-trained VGG16 CNN as an encoder and an LSTM-based decoder for generating image descriptions. It leverages the Flickr8k dataset for training and evaluation.

## Key Components

- **CNN Encoder:** VGG16 (pre-trained on ImageNet)
- **Decoder:** LSTM with word embeddings and dropout
- **Dataset:** [Flickr8k](https://www.kaggle.com/datasets/adityajn105/flickr8k)
- **Framework:** TensorFlow / Keras

## Requirements

```bash
pip install tensorflow pandas numpy matplotlib tqdm
```

## Pipeline Overview

1. **Preprocess captions**:
   - Clean, lowercase, and tokenize text
   - Add `startseq` and `endseq` tokens

2. **Extract image features**:
   - Load and preprocess images using VGG16
   - Save features for training

3. **Train captioning model**:
   - Merge image and text features
   - Train the LSTM decoder with teacher forcing

4. **Generate captions**:
   - Load model, tokenizer, and test images
   - Use greedy decoding to generate text

## Evaluation

- BLEU score for evaluating generated captions.
- Sample predictions are included in the notebook.


## Acknowledgements

- TensorFlow & Keras
- Kaggle & Flickr8k Dataset
