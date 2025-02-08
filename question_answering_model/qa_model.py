import torch
from transformers import BertTokenizer, BertForQuestionAnswering

# Load the pre-trained BERT model and tokenizer for Question Answering
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

# Function to get the answer from the book text
def answer_question(book_text, question):
    # Tokenize the input question and context (book text)
    inputs = tokenizer.encode_plus(question, book_text, add_special_tokens=True, return_tensors='pt')

    # Get the model's answer
    with torch.no_grad():
        outputs = model(**inputs)

    # Extract the start and end position of the answer
    start_scores = outputs.start_logits
    end_scores = outputs.end_logits

    # Get the most likely start and end of the answer
    start_index = torch.argmax(start_scores)
    end_index = torch.argmax(end_scores)

    # Convert tokens back to text
    answer_tokens = inputs['input_ids'][0][start_index:end_index + 1]
    answer = tokenizer.decode(answer_tokens)

    return answer

# Function to save the question, context, and answer into a text file
def save_to_txt(question, context, answer):
    with open("output.txt", "a") as file:
        file.write(f"Question: {question}\n")
        file.write(f"Context: {context}\n")
        file.write(f"Answer: {answer}\n\n")

# Example usage
if __name__ == "__main__":
    # Load your book context (can be a long paragraph or entire book text)
    book_text = """
   Latur is a city in Maharashtra's Marathwada region, known for its Latur Pattern in education, agriculture, and historical sites like Kharosa Caves. It is a major producer of pulses, oilseeds, and sugarcane and is developing as an industrial and smart city. The city faced a devastating earthquake in 1993 but has since rebuilt significantly. Latur has good connectivity via road, rail, and air and is growing in the IT and food processing sectors.
    """
    
    # Example questions
    questions = [
        "Name a historical site located in Latur?"
       
    ]
    
    # Process each question and context
    for question in questions:
        answer = answer_question(book_text, question)
        print(f"Question: {question}")
        print(f"Answer: {answer}")
        
        # Save question, context, and answer to a file
        save_to_txt(question, book_text, answer)
