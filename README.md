# **PDFBot**

PDFBot is an intelligent chatbot designed to generate responses enriched by
information from a specific PDF document. Using advanced LLM
techniques through OpenAI's GPT-4 API and a FAISS vector database, PDFBot effectively mitigates hallucinations
and ensures accurate responses.

## **System Architecture**

The system is composed of the following components:

1.  **Chatbot Interface**: A console-based interface using
    prompt_toolkit to interact with the user.

2.  **PDF Parser**: A component that uses PyMuPDF to extract text from a
    PDF document.

3.  **Embedder**: Utilizes SentenceTransformer to make embeddings
    for all of the sentences extracted from the PDF.

4.  **Vector Database**: Uses the FAISS (Facebook AI Similarity Search) library to store and retrieve embeddings. 

5.  **Verifier**: Uses the `cosine_similarity` method from sklearn to validate the chatbot\'s responses by cross-referencing it with the vector
    database to ensure accuracy and prevent hallucinations.

### **System Workflow**

1.  **Parse the PDF**: Extracting text from the PDF document using PyMuPDF.

2.  **Generate Embeddings**: Encoding the extracted text into embeddings
    using SentenceTransformer.

3.  **Store in Vector Database**: Adding the embeddings and corresponding
    sentences to the FAISS vector database.

4.  **User Query**: Accepting user queries through the console interface.

5.  **Generate Response**: Using the GPT-4 model from the OpenAI API to generate a response to the
    user's query.

6.  **Verify Response**: Cross-referencing the generated response with the
    vector database to mitigate hallucinations.

7.  **Return Verified Response**: Presenting the verified response to the
    user.

## **Running the Chatbot**

### **Prerequisites**

-   Python 3.9 or higher

-   Necessary Python packages (see requirements.txt)

### **Installation**

1.  Clone the repository:

`git clone https://github.com/yourusername/pdfbot.git`

`cd pdfbot`

2.  Set up a virtual environment:

`python -m venv venv`

`source venv/bin/activate` (On Windows use `venv\Scripts\activate`)

3.  Install the dependencies:

`pip install -r requirements.txt`

4.  Add your OpenAI API key to your .env file:

`OPENAI_API_KEY=your_openai_api_key`

### **Running the Bot**

Ensure your PDF document is named document.pdf and is located in the
same directory as the main.py file. Then run this:

`python main.py`

### **Using the Bot**

-   Start the bot by running the command above.

-   Type your queries into the console.

-   Type 'exit' or 'quit' to end the session.

## **Hallucination Mitigation Strategy**

To make sure that the responses are accurate and that hallucinations are minimized (preferably removed), the
following strategy is employed:

1.  **Embedding-Based Search**: When a response is generated, its
    embedding is computed using the `sentence_transformers` method from the SentenceTransformer Library.

2.  **Vector Database Search**: This embedding is searched against the
    vector database to find the most relevant sentences from the PDF.

3.  **Cosine Similarity Calculation**: The similarity between the
    response embedding and the retrieved embeddings is calculated by calculating the cosines of the angles between the vectors.

4.  **Dynamic Threshold**: A dynamic threshold based on the mean and
    standard deviation of the similarities is used to verify if the
    response is valid and accurate.

5.  **Response Verification**: If the similarity exceeds the dynamic
    threshold, the response is considered verified. Otherwise, the
    response is flagged as potentially inaccurate.
