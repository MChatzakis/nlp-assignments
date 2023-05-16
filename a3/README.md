[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/D6r4m_Tc)
## **Assignment Description**
- In this assignment, you will be looking at natural language generation (NLG), precisely the task of summarization. You will be exploring ways to generate text and how fine-grained decisions of decoding parameters can affect the generations.
    
- You will not need to train any models in this assignment. A pretrained one is provided for you by Huggingface.
    
- In Part 1, you will implement two decoding algorithms (greedy and beam search), as well as two sampling algorithms (top-p and top-k) to replicate (to some extent) what one would get when using Huggingface's `generate` function that you've played with during the Week 7's exercise session.
    
- For Part 2, you will analyze how varying specific parameters of decoding and sampling algorithms can qualitatively affect the generation.

- For Part 3, you will answer some questions on interpreting automatic NLG evaluation metrics.

### Table of Contents
- **[Setup](#setup)**
    - [1) Google Setup](#1-google-colab-setup)
    - [2) Local Setup](#2-local-setup)
    - [3) Rest of the Setup](#3-rest-of-the-setup-colab-and-local)

- **[Introduction: T5 Primer](#introduction-t5-primer)**

- **[PART 1: Natural Language Generation Decoding and Sampling Algorithms](#part-1-natural-language-generation-decoding-and-sampling-algorithms)**
    - [1.1) Implement decoding and sampling algorithms](#11-implement-decoding-and-sampling-algorithms)
    - [1.2) Test your implementations](#12-testing-your-implementation)
    
- **[PART 2: Qualitative Evaluation of Generation Parameters](#part-2-qualitative-evaluation-of-generation-parameters)**
    - [2.1) Beam size for beam-search](#21-beam-size-for-beam-search)
    - [2.2) Length penalty for beam-search](#22-length-penalty-for-beam-search)
    - [2.3) Top-k for top-k](#23-top-k-for-top-k)
    - [2.4) Temperature for top-p](#25-temperature-for-top-p)

- **[PART 3: Reflection on Automatic NLG Evaluation Metrics](#part-3-reflection-on-automatic-nlg-evaluation-metrics)**
    - [3.1) Description](#31-description)
    - [3.2) Task](#32-task)

- **[PART 4: Checklist](#part-4-checklist)**
    
### Deliverables

To give us the deliverables you will have to commit the following files if your github classroom repository:

- ✅ The python files:
    - [ ] `a3_decoding.py`
    - [ ] `a3_sampling.py`
    - [ ] `a3_utils.py`, if you added any helper functions

- ✅ This jupyter notebook `a3_notebook.py` with 
    - [ ] the answers to Part 2 questions written out in their corresponding cells.
        - [ ] Answers to (2.1) questions
        - [ ] Answers to (2.2) questions
        - [ ] Answers to (2.3) questions
        - [ ] Answers to (2.4) questions
    - [ ] the answers to Part 3 questions written out in its corresponding cell.

### Expected Workload

We expect the first part of the assignment, notably Beam search, to take the most out of the complete assignment. 
You can plan your workload according to that. Keep in mind that this is just our expectation, not a guarantee.