
# Auto_depression_detect
The primary objective of this project is to develop a comprehensive tri‑modal deep learning framework for automatic depression screening by integrating acoustic, linguistic, and facial expression features extracted from clinical interview data.

The system employs:

A 1D Convolutional Neural Network (1D‑CNN) to capture prosodic and temporal characteristics from speech signals, including pitch variation, energy levels, and speaking rhythm.

A BiLSTM with an attention mechanism to model contextual and semantic patterns from textual transcripts, enabling the detection of negative affect, self‑referential language, and cognitive distortion cues.

A 2D Convolutional Neural Network (2D‑CNN) to extract spatial facial features from video frames, capturing micro‑expressions, reduced facial activity, and emotional intensity patterns.

A central objective of this work is the implementation of a Gated Multimodal Fusion mechanism, which dynamically learns the relative importance of audio, text, and visual modalities for each individual instance. The gating network adaptively assigns weights to each modality, allowing the model to emphasize the most informative behavioral signal, thereby improving robustness and cross‑modal interaction learning.

The project further aims to:

Compare unimodal (audio, text, visual) and multimodal performance.

Evaluate the system using metrics such as accuracy, precision, recall, F1‑score, and regression error (for severity estimation).

Develop a reliable screening and decision‑support system rather than a medical diagnostic tool.

Ultimately, the objective is to construct an adaptive, research‑aligned multimodal architecture capable of enhancing depression risk prediction through dynamic modality weighting across speech, language, and facial behavioral cues.
