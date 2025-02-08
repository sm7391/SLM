This project leverages the pre-trained BERT model fine-tuned for Question Answering (QA) tasks to automatically answer questions from a given text (such as a book or article). The model processes the context and question, then predicts the most likely answer.

Features:
Uses BERT for Question Answering (bert-large-uncased-whole-word-masking-finetuned-squad).
Tokenizes input questions and contexts (book text) for model inference.
Extracts and decodes the most probable answer.
Saves questions, context, and answers to an output text file for easy reference.
Installation:
Clone the repository.
Install the required libraries:
bash
Copy
Edit
pip install torch transformers
Usage:
Update the book_text variable with your context (book/article).
Modify the questions list with the questions you want to ask.
Run the script to get the answers and save them in output.txt.
python
Copy
Edit
python answer_question.py
Example:
Context:

vbnet
Copy
Edit
Latur is a city in Maharashtra's Marathwada region, known for its Latur Pattern in education...
Question:

css
Copy
Edit
Name a historical site located in Latur?
Answer:

nginx
Copy
Edit
Kharosa Caves# SLM
