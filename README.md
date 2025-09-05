# Multimodal Fake News Detection


## Overview

This repository contains the project report and associated materials for the Mini Project 2020-21 titled "Multimodal Fake News Detection", submitted as part of the B.Tech degree requirements at the Indian Institute of Information Technology Dharwad (IIIT Dharwad). The project was conducted by a team of third-year Computer Science and Engineering students under the guidance of **Dr. Sunil Saumya**, Assistant Professor, Dept. of CSE, IIIT Dharwad.

### Team

|Name|Regd. No|
|-|-|
|Shaik Fharook| 18BCS091|
|Gurram Rithika| 18BCS031|
|Pravalika| 18BCS070|
|Godina Pranav| 18BCS028|


## Project Description

The project addresses the critical issue of fake news detection in the era of accessible information, where social media platforms enable rapid dissemination of potentially misleading content. The focus is on developing a multimodal framework that leverages both text and image data to classify news as real or fake, using advanced deep learning techniques such as TI-CNN, ResNet-50, and LSTM.

### Objectives
- Implement a multimodal algorithm (TI-CNN) for fake news detection.
- Compare performance with other models like ResNet-50 and LSTM.
- Utilize the Fakeddit dataset for training and evaluation.
- Address challenges posed by diverse news styles and social media data.

### Methodology

- **Dataset:** Fakeddit, a multimodal benchmark dataset with over 1 million samples from Reddit, labeled for 2-way, 3-way, and 6-way classification.
- **Approach:**
    -  Preprocessing of text and image data.
    - Implementation of TI-CNN and comparison with existing models.
    - Fact-checking integration for model training accuracy.
    - **Models:** TI-CNN, ResNet-50, and LSTM for classification tasks.


## Results
- The report details the performance of various models on the Fakeddit dataset, highlighting the effectiveness of multimodal approaches.
- Challenges include training complexity and generalizability due to limited secondary task data.

## Conclusion & Future Scope

The project demonstrates the potential of multimodal systems in fake news detection, though improvements are needed in handling diverse data and reducing model complexity. Future work could involve larger datasets and enhanced preprocessing techniques.


## Installation & Usage

1. Clone the repository:

```
git clone <repository-url>
```
2. Install required dependencies (e.g., TensorFlow, Keras, OpenCV) using:

```
pip install -r requirements.txt
```

3. Refer to the report (Multimodal_Fake_News_Detection_Report.pdf) for detailed methodology and model implementation steps.


## References

1.  Yang, Lei Zheng, Jiawei Zhang, Qingcai Cui, Xiaoming Zhang, Zhoujun Li, Phillip S. Yu, "Convolutional Neural Networks for Fake News Detection".
2. Potthast et al., "Stylometric Features for Fake News Detection".
3.  Shu et al., "Author Impact on Fake News Spread".
4. Pan et al., "Knowledge Graphs for Fact Analysis".
5. Lin et al., "TransR Model for Knowledge Graph Embeddings".
6. Image Splicing Technique for Fake Image Detection.
7. Marra et al., "GANs for Fake Image Detection".
8.  Various Multimodal Fake News Detection Studies.
11. Nakamura, Levy, Wang, "Fakeddit: A New Multimodal Benchmark Dataset".

## License

This project is for academic purposes only. Redistribution or modification of the report content requires permission from the authors.

