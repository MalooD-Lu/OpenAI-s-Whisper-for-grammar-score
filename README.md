# OpenAI-s-Whisper-for-grammar-score

## Problem Overview

- given audio files of spoken English.
- To do: predict a grammar score based on the fluency, structure, and quality of the speech.
- Evaluation metric: **Pearson Correlation** between predicted and true scores.

---

## Approach Summary

This solution uses a hybrid method:

### Whisper for Transcription
- Used **OpenAI's `whisper-small`** model to transcribe audio.
- Extracted linguistic features directly from the generated transcript.

### Handcrafted Features
From the transcript, we engineered the following features:
- Number of words
- Vocabulary richness (unique words / total words)
- Number of sentences (`.` count)
- Count of words ending in `'ing'` and `'ed'`
- Count of long words (>6 characters)
- Frequency of the word `'the'`

### Gradient Boosting Regressor
- Used `sklearn.ensemble.GradientBoostingRegressor` as the final model.
- Trained on the extracted linguistic features.


## Dependencies

```bash
pip install -q transformers librosa pandas scikit-learn torch tqdm
sudo apt install -y ffmpeg
