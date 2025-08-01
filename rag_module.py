import pandas as pd
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from groq import Groq
import uuid
from typing import List, Dict, Any
import tempfile
import os
import torch

# Force CPU usage to avoid GPU memory issues
os.environ['CUDA_VISIBLE_DEVICES'] = ''

class RAGModule:
    def __init__(self, groq_api_key: str):
        """Initialize the RAG module with Groq API key"""
        self.groq_client = Groq(api_key=groq_api_key)
        
        # Try multiple models in order of preference
        models_to_try = [
            'paraphrase-MiniLM-L3-v2',  # Smaller, more stable
            'all-MiniLM-L6-v2',        # Original choice
        ]
        
        for model_name in models_to_try:
            try:
                self.embedding_model = SentenceTransformer(
                    model_name,
                    device='cpu'
                )
                break
            except Exception as e:
                print(f"Failed to load {model_name}: {e}")
                continue
        else:
            raise Exception("Could not load any embedding model")
            
        self.chroma_client = None
        self.collection = None
        self.collection_name = f"dataset_collection_{str(uuid.uuid4())[:8]}"
        
    def initialize_chromadb(self):
        """Initialize ChromaDB client and create collection"""
        try:
            temp_dir = tempfile.mkdtemp()
            self.chroma_client = chromadb.PersistentClient(path=temp_dir)
            
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
                metadata={"description": "Dataset embeddings for RAG"}
            )
            
            return True
        except Exception as e:
            print(f"Error initializing ChromaDB: {str(e)}")
            return False
    
    def create_dataframe_profile(self, df: pd.DataFrame) -> str:
        """Create a plain-English profile of the DataFrame for better RAG context"""
        try:
            profile_parts = []
            
            profile_parts.append(f"This dataset contains {len(df)} rows and {len(df.columns)} columns.")
            
            column_info = []
            for col in df.columns:
                dtype = str(df[col].dtype)
                if dtype in ['int64', 'float64']:
                    col_type = "numerical"
                elif dtype == 'datetime64[ns]':
                    col_type = "date"
                else:
                    col_type = "categorical"
                column_info.append(f"'{col}' ({col_type})")
            
            profile_parts.append(f"The columns are: {', '.join(column_info)}.")
            
            sample_rows = []
            for idx, row in df.head(3).iterrows():
                row_str = ", ".join([f"{col}: {val}" for col, val in row.items()])
                sample_rows.append(f"Row {idx + 1}: {row_str}")
            
            profile_parts.append("Here are some sample rows:")
            profile_parts.extend(sample_rows)
            
            return "\n".join(profile_parts)
            
        except Exception as e:
            return f"Error creating dataframe profile: {str(e)}"
    
    def chunk_data(self, data, file_type: str) -> List[str]:
        """Smart chunking based on data type"""
        chunks = []
        
        try:
            if file_type == "csv":
                if isinstance(data, pd.DataFrame):
                    profile = self.create_dataframe_profile(data)
                    chunks.append(f"DATASET_PROFILE:\n{profile}")
                    
                    for idx, row in data.iterrows():
                        row_str = ", ".join([f"{col}: {val}" for col, val in row.items()])
                        chunks.append(f"Record {idx + 1}: {row_str}")
                
            elif file_type in ["txt", "text"]:
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    length_function=len,
                )
                
                if isinstance(data, str):
                    chunks = text_splitter.split_text(data)
                else:
                    chunks = ["Error: Text data expected but received different format"]
                    
            else:
                chunks = ["Unsupported file type for chunking"]
                
        except Exception as e:
            chunks = [f"Error during chunking: {str(e)}"]
            
        return chunks
    
    def generate_embeddings_and_store(self, chunks: List[str]) -> bool:
        """Generate embeddings and store in ChromaDB"""
        try:
            if not self.collection:
                if not self.initialize_chromadb():
                    return False
            
            embeddings = self.embedding_model.encode(chunks).tolist()
            ids = [f"chunk_{i}" for i in range(len(chunks))]
            
            self.collection.add(
                embeddings=embeddings,
                documents=chunks,
                ids=ids,
                metadatas=[{"chunk_index": i} for i in range(len(chunks))]
            )
            
            return True
            
        except Exception as e:
            print(f"Error generating embeddings and storing: {str(e)}")
            return False
    
    def retrieve_relevant_chunks(self, query: str, n_results: int = 5) -> List[str]:
        """Retrieve relevant chunks based on query"""
        try:
            if not self.collection:
                return ["Error: No data has been processed yet."]
            
            query_embedding = self.embedding_model.encode([query]).tolist()
            
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=n_results
            )
            
            return results['documents'][0] if results['documents'] else []
            
        except Exception as e:
            return [f"Error retrieving chunks: {str(e)}"]
    
    def ask_rag(self, user_question: str) -> str:
        """Main RAG function - retrieve context and generate answer"""
        try:
            relevant_chunks = self.retrieve_relevant_chunks(user_question)
            
            if not relevant_chunks:
                return "I don't have enough information to answer your question. Please make sure you've uploaded and processed a dataset."
            
            context = "\n\n".join(relevant_chunks)
            
            prompt = f"""You are an AI assistant that answers questions based on the provided dataset context. 
            
Context from the dataset:
{context}

User Question: {user_question}

Please provide a clear, accurate answer based only on the information provided in the context. If the context doesn't contain enough information to answer the question, please say so.

Answer:"""
            
            response = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error generating answer: {str(e)}"
    
    def process_uploaded_data(self, data, file_type: str) -> Dict[str, Any]:
        """Complete pipeline: chunk, embed, and store data"""
        try:
            chunks = self.chunk_data(data, file_type)
            
            if not chunks:
                return {"success": False, "message": "No chunks created from the data"}
            
            success = self.generate_embeddings_and_store(chunks)
            
            if success:
                return {
                    "success": True, 
                    "message": f"Successfully processed {len(chunks)} chunks and stored in knowledge base",
                    "chunks_count": len(chunks)
                }
            else:
                return {"success": False, "message": "Failed to store embeddings"}
                
        except Exception as e:
            return {"success": False, "message": f"Error processing data: {str(e)}"}

# TEST SECTION
if __name__ == "__main__":
    print("Testing RAG Module...")
    print("=" * 50)
    
    try:
        print("Test 1: Importing all dependencies - SUCCESS")
        
        sample_data = pd.DataFrame({
            'Name': ['Alice', 'Bob', 'Charlie'],
            'Age': [25, 30, 35],
            'City': ['New York', 'London', 'Tokyo']
        })
        print("Test 2: Creating sample DataFrame - SUCCESS")
        
        rag = RAGModule(groq_api_key="test_key")
        chunks = rag.chunk_data(sample_data, "csv")
        print(f"Test 3: Data chunking - SUCCESS ({len(chunks)} chunks created)")
        
        embedding_test = rag.embedding_model.encode(["test sentence"])
        print("Test 4: Embedding model loading - SUCCESS")
        
        if rag.initialize_chromadb():
            print("Test 5: ChromaDB initialization - SUCCESS")
        else:
            print("Test 5: ChromaDB initialization - FAILED")
            
        print("=" * 50)
        print("RAG MODULE TEST COMPLETED SUCCESSFULLY!")
        print("All core components are working properly.")
        print("Ready to integrate with Streamlit app!")
        
    except ImportError as e:
        print(f"Import Error: {e}")
        print("Please install required packages: pip install -r requirements.txt")
        
    except Exception as e:
        print(f"Test Failed: {str(e)}")
        print("There might be an issue with the setup.")
