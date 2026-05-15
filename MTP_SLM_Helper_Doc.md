# **M.Tech Thesis Project Report**

## **Small Language Models for Hydrodynamics Problem Solving using Reinforcement Learning**

## **1\. Introduction**

Recent developments in **Large Language Models (LLMs)** have shown remarkable capabilities in reasoning, coding, and natural language understanding. However, these models are computationally expensive and often inefficient for domain-specific tasks.

Recent research indicates that **Small Language Models (SLMs)** with **1–8B parameters**, when fine-tuned on specialized datasets, can match or outperform larger models on specific tasks while requiring significantly lower computational resources. Surveys of recent research confirm that task-specific SLMs frequently rival or exceed larger models on focused benchmarks such as reasoning, structured output generation, and domain-specific question answering.

This project explores the use of **SLMs trained with reinforcement learning techniques (RLPT / GRPO)** to solve **marine hydrodynamics problems**. The goal is to demonstrate that a **small model (1–3B parameters)** can achieve comparable or better performance than general-purpose LLMs on hydrodynamics questions.

Unlike traditional training pipelines, this work focuses on **data-scarce domains** where labeled question-answer datasets are limited. The proposed solution involves **synthetic dataset generation and physics-based reward verification**, enabling reinforcement learning without large labeled datasets.

---

## **2\. Problem Statement**

Hydrodynamics is a specialized engineering domain involving problems related to:

* Potential flow theory  
* Wave-body interactions  
* Ship hydrodynamics  
* Boundary layer theory  
* Fluid forces and hydrodynamic coefficients

While general-purpose LLMs possess broad knowledge, they often perform poorly on **specialized engineering problems requiring structured reasoning and equation-based solutions**.

### **Key Challenges**

**Lack of labeled datasets**  
Very few publicly available `<question, answer>` datasets exist for marine hydrodynamics.

**Requirement for numerical and symbolic reasoning**  
Many problems require:

* Equation derivations  
* Numerical calculations  
* Multi-step reasoning

**Evaluation difficulty**  
Solutions must be verified using **physical laws and equations**, rather than subjective evaluation.

This thesis investigates whether **a domain-specialized SLM trained using reinforcement learning with verifiable rewards** can solve hydrodynamics problems more accurately than general-purpose LLMs.

---

## **3\. Research Objectives**

The primary objectives of this project are:

1. Develop a **task-specific Small Language Model** for marine hydrodynamics problem solving.  
2. Create a **hydrodynamics dataset** using synthetic data generation techniques.  
3. Design a **physics-based reward verification system** for reinforcement learning.  
4. Train the model using **Supervised Fine-Tuning (SFT) followed by Reinforcement Learning (RLPT / GRPO)**.  
5. Evaluate the performance of the trained SLM against large language models.

---

## **4\. Literature Background**

Recent research suggests that **task-specialized small models can outperform large models on structured tasks**.

Key findings from the literature include:

* SLMs in the **1–8B parameter range can match or exceed larger models on specific tasks**.  
  [https://arxiv.org/html/2501.05465](https://arxiv.org/html/2501.05465)  
* Small models fine-tuned for structured output tasks have achieved **98%+ formatting accuracy**, outperforming larger models on constrained decoding tasks.  
  [https://techsignal.news/enterprise-ai/red-hat-small-language-models-structured-tasks](https://techsignal.news/enterprise-ai/red-hat-small-language-models-structured-tasks)  
* Reinforcement learning techniques such as **RLHF, GRPO, and RLVR** are increasingly used to improve reasoning capabilities of language models.  
  [https://aws.amazon.com/blogs/machine-learning/fine-tune-large-language-models-with-reinforcement-learning-from-human-or-ai-feedback/](https://aws.amazon.com/blogs/machine-learning/fine-tune-large-language-models-with-reinforcement-learning-from-human-or-ai-feedback/)

These findings suggest that **specialized SLMs can be highly effective in narrow domains such as engineering and scientific problem solving**.

---

## **5\. Proposed System Architecture**

The proposed system follows a multi-stage pipeline:

Textbooks \+ Lecture Notes \+ Exams  
            ↓  
      Text Extraction  
            ↓  
    Synthetic QA Generation  
            ↓  
  Supervised Fine-Tuning (SFT)  
            ↓  
Reinforcement Learning (RLPT / GRPO)  
            ↓  
        Evaluation

---

## **6\. Dataset Construction**

Due to the lack of labeled datasets, a **synthetic dataset generation pipeline** is required.

### **Textbooks**

The following textbooks serve as primary knowledge sources:

* Newman – *Marine Hydrodynamics*  
* Faltinsen – *Sea Loads on Ships and Offshore Structures*

These books provide theoretical explanations, derivations, and worked examples.

### **MIT OpenCourseWare**

Two MIT courses are used as additional sources:

* **MIT OCW 2.20 – Marine Hydrodynamics**  
* **MIT OCW 2.29 – Numerical Marine Hydrodynamics**

These resources provide:

* Lecture notes  
* Problem sets  
* Exams with solutions

Links:

* [https://ocw.mit.edu/courses/2-20-marine-hydrodynamics-13-021-spring-2005/](https://ocw.mit.edu/courses/2-20-marine-hydrodynamics-13-021-spring-2005/)  
* [https://ocw.mit.edu/courses/2-29-numerical-marine-hydrodynamics-13-024-spring-2003/](https://ocw.mit.edu/courses/2-29-numerical-marine-hydrodynamics-13-024-spring-2003/)

### **IIT Exam Papers**

Real exam papers from IIT are included as evaluation problems. These contain authentic graduate-level questions but do not include solutions.

---

## **7\. Data Processing Pipeline**

The collected PDFs are processed through several stages.

### **Step 1: Data Collection**

| Source | Files |
| ----- | ----- |
| Newman textbook | 1 |
| Faltinsen textbook | 1 |
| MIT OCW 2.20 lectures | 22 |
| MIT OCW 2.20 problem sets | 25 |
| MIT OCW exams | 1 |
| MIT OCW 2.29 lecture notes | 14 |
| IIT exam papers | 4 |

Total collected: **\~69 PDFs**

---

### **Step 2: Text Extraction**

Two extraction methods were used.

#### **Digital PDFs**

Digital PDFs were processed using:

* **PyMuPDF**

These include:

* MIT OCW lecture notes  
* Problem sets  
* Newman textbook

#### **Scanned PDFs**

Scanned documents were processed using **OCR**.

Tools used:

* **Tesseract OCR**  
* PyMuPDF page rendering

These include:

* Faltinsen textbook  
* MIT OCW 2.29 lecture notes  
* IIT exam papers

---

### **Extraction Results**

| Source | Output |
| ----- | ----- |
| Newman textbook | 340 chunks |
| Faltinsen textbook | 156 chunks |
| MIT OCW 2.20 lectures | 22 files |
| MIT OCW 2.20 problem sets | 25 files |
| MIT OCW final exam | 1 file |
| MIT OCW 2.29 lecture notes | 14 files |
| IIT exam papers | 4 files |

Total output:

* **563 text files**  
* **2.4 MB of processed text**

---

## **8\. Synthetic Dataset Generation**

To create training data, a **synthetic QA generation pipeline** will be used.

Each extracted text chunk is used as context for generating question-answer pairs.

Example prompt:

Given the following hydrodynamics text:

\[context\]

Generate 3 graduate-level questions and detailed step-by-step solutions.

The generated data will be stored in **JSONL format**.

Example entry:

{  
 "question": "...",  
 "answer": "...",  
 "type": "numerical / conceptual / derivation",  
 "source": "newman"  
}

---

## **9\. Model Training**

The training pipeline consists of two stages.

### **Stage 1: Supervised Fine-Tuning (SFT)**

The SLM is first trained on synthetic QA pairs.

Possible base models include:

* Qwen2.5 (1.5B or 3B)  
* LLaMA-3 small variants  
* DeepSeek small models

Training techniques:

* LoRA  
* QLoRA  
* HuggingFace Transformers

---

### **Stage 2: Reinforcement Learning (RLPT / GRPO)**

After SFT, reinforcement learning improves reasoning accuracy.

Training loop:

1\. Sample a hydrodynamics question  
2\. Model generates N candidate solutions  
3\. Physics verifier scores each solution  
4\. GRPO updates the policy

---

## **10\. Physics-Based Reward Function**

Instead of human labels, a **physics-based verifier** evaluates correctness.

| Problem Type | Verification Method |
| ----- | ----- |
| Numerical problems | Numeric comparison |
| Equation derivation | Symbolic equivalence (SymPy) |
| Dimensional analysis | Unit consistency check |
| Conceptual answers | LLM evaluation |

Reward function:

Reward \=  
α \* numerical correctness  
\+ β \* dimensional consistency  
\+ γ \* LLM evaluation score

---

## **11\. Evaluation Strategy**

### **Benchmark Dataset**

* MIT OCW final exam problems  
* IIT exam questions  
* Held-out textbook examples

### **Baseline Models**

Performance will be compared against:

* GPT-4 class models  
* Other open-source LLMs

### **Metrics**

Evaluation metrics include:

* Numerical accuracy  
* Equation correctness  
* Expert evaluation  
* Solution completeness

---

## **12\. Expected Contributions**

This thesis aims to contribute:

1. A **hydrodynamics QA dataset**  
2. A **domain-specific Small Language Model**  
3. A **physics-grounded reinforcement learning framework**  
4. Evidence that **SLMs can rival larger models in engineering domains**

---

## **13\. Current Progress**

Completed work:

* Raw data collection  
* PDF extraction pipeline  
* OCR processing  
* Text chunk generation

Dataset status:

* **563 extracted files**  
* **496 training chunks from textbooks**

Next step:

* Synthetic QA dataset generation  
* SLM training pipeline

---

