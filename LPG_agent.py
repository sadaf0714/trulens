import os
import json
import re
import time
from langchain.docstore.document import Document
from typing import List, Tuple
from langchain_core.runnables import Runnable
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from neo4j import GraphDatabase
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda, RunnableBranch
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.pydantic_v1 import BaseModel, Field
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Neo4jVector
from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars
from src.core.logger import app_logger
from src.services.chatbot import chat
# from neo4j_graphrag.retrievers import VectorRetriever
# from neo4j_graphrag.retrievers import VectorCypherRetriever
# from neo4j_graphrag.retrievers import HybridRetriever
# from neo4j_graphrag.retrievers import HybridCypherRetriever
# from neo4j_graphrag.retrievers import Text2CypherRetriever
from utils import prompts
# from neo4j_graphrag.retrievers import Text2CypherRetriever
from neo4j_graphrag.llm import OpenAILLM
from utils.functions import log_separator, clean_json_string
from dotenv import load_dotenv
import neo4j
#from src.core.constant import neo4j_schema1,examples 
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.graphs import Neo4jGraph
from langchain.retrievers import (
    MultiQueryRetriever,
    EnsembleRetriever
)
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Neo4jVector
# from neo4j_graphrag import Neo4jGraphRetriever
from tqdm import tqdm
from langchain.chains import RetrievalQA
from langchain.chains import GraphCypherQAChain
from langchain.prompts import PromptTemplate

import os

load_dotenv()
# Initialize LLM
llm = ChatOpenAI(temperature=0, model="gpt-4")
#llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0)

# Load the Gemini embeddings
#embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#t2c_llm = OpenAILLM(model_name="gpt-3.5-turbo")
embeddings = OpenAIEmbeddings()
uri = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")
#database = os.getenv("NEO4J_DATABASE")
AUTH = (username, password)
driver = neo4j.GraphDatabase.driver(uri, auth=AUTH)
# Initialize Neo4jGraph
graph = Neo4jGraph()

LABEL_PROPERTY_MAPPING = {
    "Person": "name",
    "Company": "company_name",
    "Skill": "skill_name",
    "Product": "product_description",
    "Industry": "industry_name",
    # Add more label:property pairs as needed
}
class Entities(BaseModel):
    """Identifying information about entities."""
    names: List[str] = Field(
        ..., description="All the person, organization, or business entities that appear in the text",
    )


def create_lpg_chat_tool(doc_metadata):
    """Creates a RAG tool that includes structured and unstructured retrieval using Neo4j and a vector store."""
   
    # def get_all_labels():
    #     with driver.session() as session:
    #         query = "MATCH (n) RETURN DISTINCT labels(n) AS labels"
    #         result = session.run(query)
    #         labels = []
    #         for record in result:
    #             labels.extend(record["labels"])
    #         return list(set(labels))

    # labels = get_all_labels()
    # print(f"Found {len(labels)} unique labels: {labels}")

    # def get_properties_for_labels(labels):
    #     label_properties = {}
    #     with driver.session() as session:
    #         for label in labels:
    #             query = f"""
    #             MATCH (n:{label})
    #             RETURN DISTINCT keys(n) AS properties
    #             LIMIT 1
    #             """
    #             result = session.run(query)
    #             for record in result:
    #                 label_properties[label] = record["properties"]
    #     return label_properties

    # label_properties = get_properties_for_labels(labels)
    # for label, properties in label_properties.items():
    #     print(f"Label: {label}, Properties: {properties}")

    # def get_text_properties(label_properties):
    #     text_properties = {}
    #     for label, properties in label_properties.items():
    #         # Filter out properties that likely contain text data
    #         text_properties[label] = [prop for prop in properties if 'text' in prop.lower() or 'description' in prop.lower() or 'name' in prop.lower()]
    #     return text_properties

    # text_properties = get_text_properties(label_properties)
    # for label, properties in text_properties.items():
    #     print(f"Label: {label}, Text Properties: {properties}")

    # --- Step 1: Fetch nodes for a specific label ---
    # def fetch_all_nodes(tx):
    #     result = tx.run("""
    #     MATCH (n)
    #     WITH n, keys(n) AS all_props
    #     WITH n, [prop IN all_props WHERE n[prop] IS NOT NULL AND (toLower(prop) CONTAINS 'name' OR toLower(prop) CONTAINS 'title' OR toLower(prop) CONTAINS 'description' OR toLower(prop) CONTAINS 'text')] AS text_props
    #     WHERE size(text_props) > 0
    #     RETURN id(n) AS node_id, n[text_props[0]] AS text
    #     """)
    #     return [{"node_id": record["node_id"], "text": record["text"]} for record in result]

    # def store_embeddings(tx, node_id, embedding):
    #     tx.run("""
    #     MATCH (n)
    #     WHERE id(n) = $node_id
    #     SET n.embedding = $embedding
    #     """, node_id=node_id, embedding=embedding)

    # with driver.session() as session:
    #     print(f"\n🔎 Fetching all nodes with text-like properties...")

    #     nodes = session.execute_read(fetch_all_nodes)

    #     print(f"Found {len(nodes)} nodes.")

    #     for node in tqdm(nodes, desc=f"Embedding all nodes"):
    #         if node["text"]:
    #             vector = embeddings.embed_query(node["text"])
    #             session.execute_write(store_embeddings, node["node_id"], vector)

    # print("✅ All embeddings created!")

    # with driver.session() as session:
    #     print(f"Creating generic vector index...")

    #     session.run("""
    #         CREATE VECTOR INDEX vector_all IF NOT EXISTS FOR (n:Node) ON (n.embedding)
    #         OPTIONS {
    #             indexConfig: {
    #                 `vector.dimensions`: 1536,
    #                 `vector.similarity_function`: 'cosine'
    #             }
    #         }
    #     """)

    # print("✅ Generic vector index created!")

    #Initialize Vector Store (for unstructured retrieval)
    vector_index = Neo4jVector.from_existing_graph(
        OpenAIEmbeddings(),  # Use OpenAI Embeddings
        search_type="hybrid",
        index_name="vector",
        node_label="*",  # Node labels you want to index
        text_node_properties=["label", "data", "comment", "description", "name", "title", "summary", "job_title", "job_description", "skills"],  # Properties for indexing
        embedding_node_property="embedding"  # The property where the embedding is stored
    )

    def fulltext_only_retriever(question):
        # Ensure fulltext index exists
        # graph.query("""
        # CREATE FULLTEXT INDEX entity IF NOT EXISTS FOR (e:__Entity__) ON EACH [e.id]
        # """)

        result = ""

        # Extract entities from the question
        sr_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are extracting organization and person entities from the text."),
            ("human", "Use the given format to extract information from the following input: {question}"),
        ])
    
        entity_chain = sr_prompt | llm.with_structured_output(Entities)
        entities = entity_chain.invoke({"question": question})

        for entity in entities.names:
            # Run full-text search only, no graph traversal
            response = graph.query(
            """
            CALL db.index.fulltext.queryNodes('fulltext_all', $query, {limit:5})
            YIELD node, score
            RETURN node.id + ': ' + coalesce(node.name, node.id) AS output
            """,
            {"query": generate_full_text_query(entity)}
            )

            result += "\n".join([el["output"] for el in response]) + "\n"

        return result.strip()   

    def generate_full_text_query(input: str) -> str:
        cleaned_input = remove_lucene_chars(input) or ""
        words = [el for el in cleaned_input.split() if el and isinstance(el, str)]
        if not words:
            return "*"  # fallback to match all, or return a minimal query
        if len(words) == 1:
            return f"{words[0]}~2"
        return " AND ".join([f"{word}~2" for word in words])


    def structured_retriever(question):
        #graph.query("CREATE FULLTEXT INDEX entity IF NOT EXISTS FOR (e:__Entity__) ON EACH [e.id]")
        result = ""
        sr_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are extracting organization and person entities from the text."),
            ("human", "Use the given format to extract information from the following input: {question}"),
        ])
        
        entity_chain = sr_prompt | llm.with_structured_output(Entities)
        entities = entity_chain.invoke({"question": question})
        
        for entity in entities.names:
            response = graph.query(
                """CALL db.index.fulltext.queryNodes('fulltext_all', $query, {limit:5})
                YIELD node,score
                CALL {
                  WITH node
                  MATCH (node)-[r:!MENTIONS]->(neighbor)
                  RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
                  UNION ALL
                  WITH node
                  MATCH (node)<-[r:!MENTIONS]-(neighbor)
                  RETURN neighbor.id + ' - ' + type(r) + ' -> ' + node.id AS output
                }
                RETURN output LIMIT 50
                """,
                {"query": generate_full_text_query(entity)},
            )
            result += "\n".join([el['output'] for el in response])
        return result

    # vector_retriever = Neo4jVector(
    # embedding=embeddings,
    # url=os.getenv("NEO4J_URI"),
    # username=os.getenv("NEO4J_USERNAME"),
    # password=os.getenv("NEO4J_PASSWORD"),
    # index_name="vector_all",
    # node_label="*",
    # text_node_property="text",
    # embedding_node_property="embedding",
    # )


    # class CustomNeo4jGraphRetriever(BaseRetriever):

    #     def __init__(self, uri, username, password, database="neo4j"):
    #         self.graph = GraphDatabase.driver(uri, auth=(username, password))
    #         self.database = database

    #     def _get_node_text(self, node):
    #         if "name" in node:
    #             return node["name"]
    #         else:
    #             return " ".join([str(v) for v in node.values()])

    #     def _retrieve_nodes(self):
    #         cypher_query = """
    #         MATCH (n)
    #         RETURN n
    #         LIMIT 1000
    #         """
    #         with self.graph.session(database=self.database) as session:
    #             result = session.run(cypher_query)
    #             docs = []
    #             for record in result:
    #                 node = record["n"]
    #                 text = self._get_node_text(node)
    #                 metadata = {
    #                     "id": node.id,
    #                     "label": list(node.labels),
    #                     "properties": dict(node)
    #                 }
    #                 docs.append(Document(page_content=text, metadata=metadata))
    #         return docs

    #     def _get_relevant_documents(self, query: str) -> list[Document]:
    #         # Dummy logic: just returns all nodes (you can improve with keyword filtering)
    #         return self._retrieve_nodes()

    #     async def _aget_relevant_documents(self, query: str) -> list[Document]:
    #         # Optionally implement async version
    #         return self._get_relevant_documents(query)

            
    # --- 2. Graph Retriever ---
    # graph_retriever = CustomNeo4jGraphRetriever(
    #     uri=uri, 
    #     username=username, 
    #     password=password
    # )
    # --- 3. Hybrid Retriever (Ensemble) ---
    # hybrid_retriever = EnsembleRetriever(
    #     retrievers=[vector_retriever, graph_retriever],
    #     weights=[0.5, 0.5]  # you can tune the weighting
    # )

    # --- 4. Vector + Graph Traversal Retriever ---
    # (Retrieve via vector and then traverse neighbors)
    # Simple manual chaining here
    # class VectorWithTraversalRetriever:
    #     def __init__(self, vector_retriever, graph):
    #         self.vector_retriever = vector_retriever
    #         self.graph = graph

    #     def get_relevant_documents(self, query):
    #         # First do vector search
    #         docs = self.vector_retriever.get_relevant_documents(query)
            
    #         # Then for each doc, expand neighbors
    #         expanded_docs = []
    #         for doc in docs:
    #             node_id = doc.metadata.get("id")  # you must ensure id is available in metadata
    #             if node_id:
    #                 neighbors = self.graph.query(f"""
    #                     MATCH (n) WHERE id(n) = {node_id}
    #                     MATCH (n)--(neighbor)
    #                     RETURN neighbor
    #                     LIMIT 5
    #                 """)
    #                 for record in neighbors:
    #                     neighbor = record.get('neighbor')
    #                     if neighbor:
    #                         expanded_docs.append(str(neighbor))
            
    #         return docs + expanded_docs
        
    # # class HybridWithTraversalRetriever:
    # #     def __init__(self, hybrid_retriever, graph):
    # #         self.hybrid_retriever = hybrid_retriever
    # #         self.graph = graph

    # #     def get_relevant_documents(self, query):
    # #         # First do hybrid search
    # #         docs = self.hybrid_retriever.get_relevant_documents(query)
            
    # #         # Then for each doc, expand neighbors
    # #         expanded_docs = []
    # #         for doc in docs:
    # #             node_id = doc.metadata.get("id")
    # #             if node_id:
    # #                 neighbors = self.graph.query(f"""
    # #                     MATCH (n) WHERE id(n) = {node_id}
    # #                     MATCH (n)--(neighbor)
    # #                     RETURN neighbor
    # #                     LIMIT 5
    # #                 """)
    # #                 for record in neighbors:
    # #                     neighbor = record.get('neighbor')
    # #                     if neighbor:
    # #                         expanded_docs.append(str(neighbor))
            
    # #         return docs + expanded_docs
        
    # vector_with_traversal_retriever = VectorWithTraversalRetriever(vector_retriever, graph)
    # hybrid_with_traversal_retriever = HybridWithTraversalRetriever(hybrid_retriever, graph)
    
    # Vector Retrieval Only
    def vector_retriever(question: str) -> str:

        # vector_qa = RetrievalQA.from_chain_type(
        # llm=llm,
        # retriever=vector_retriever,
        # chain_type="stuff"
        # )
        # vector_retriever = VectorRetriever(
        #     driver,
        #     index_name="vector",
        #     embedder=embedder,
        #     return_properties=["text"],
        # )
        # return vector_retriever
        unstructured_data = [el.page_content for el in vector_index.similarity_search(question)]
        app_logger.info(f"Vector-only result: {unstructured_data}")
        return f"""Unstructured data:\n{"#Document ".join(unstructured_data)}"""
        #return vector_qa.run(question)

    # Vector Retrieval + Graph Traversal
    def vector_graph_retriever(question: str) -> str:
        # vector_with_traversal_qa = RetrievalQA.from_chain_type(
        # llm=llm,
        # retriever=vector_with_traversal_retriever,
        # chain_type="stuff"
        # )
        # return vector_with_traversal_qa.run(question)
        #initial_nodes = vector_index.similarity_search(question)
        result = ""
    
        docs = vector_index.similarity_search(question, k=2)
        print("Vector search returned:", len(docs))
        for doc in docs:
            doc_id = (
                doc.metadata.get("doc_id")
                or doc.metadata.get("id")
                or doc.metadata.get("uuid")
            )
            print("Retrieved doc_id:", doc_id)
            

            response = graph.query(
            """
            MATCH (node)
            WHERE node.doc_id = $id OR node.id = $id OR node.uuid = $id
            CALL {
            WITH node
            MATCH (node)-[r]->(neighbor)
            RETURN 'Document -> ' + type(r) + ' -> ' + coalesce(neighbor.name, neighbor.id, neighbor.text, 'unknown') AS output
            UNION ALL
            WITH node
            MATCH (node)<-[r]-(neighbor)
            RETURN coalesce(neighbor.name, neighbor.id, neighbor.text, 'unknown') + ' -> ' + type(r) + ' -> Document' AS output
            }
            RETURN output LIMIT 50
            """,
            {"id": doc_id},
            )
            result += "\n".join([el["output"] for el in response]) + "\n"
            print(f'vectorgraph-context: {result}')
        

        return result


    # Hybrid Retrieval Only
    def hybrid_retriever(question: str) -> str:

    #     hybrid_qa = RetrievalQA.from_chain_type(
    #         llm=llm,
    #         retriever=hybrid_retriever,
    #         chain_type="stuff"
    #         )
    #     return hybrid_qa.run(question)
        # h_retriever = HybridRetriever(
        #     driver=driver,
        #     vector_index_name="vector",
        #     fulltext_index_name="fulltext_all",
        #     embedder=embedder,
        #     return_properties=["text"],
        # )
        # return h_retriever
        structured_data = fulltext_only_retriever(question)
        unstructured_data = [el.page_content for el in vector_index.similarity_search(question)]
        app_logger.info(f'hybrid context-Structured data:\n{structured_data}\nUnstructured data:\n{"#Document ".join(unstructured_data)}')
        return f"""Structured data:\n{structured_data}\nUnstructured data:\n{"#Document ".join(unstructured_data)}"""

    # Hybrid Retrieval with Graph Traversal
    def hybrid_graph_retriever(question: str) -> str:

    #     vector_with_traversal_qa = RetrievalQA.from_chain_type(
    #     llm=llm,
    #     retriever=hybrid_with_traversal_retriever,
    #     chain_type="stuff"
    #     )
    #     return vector_with_traversal_qa.run(question)
        structured_data = structured_retriever(question)
        app_logger.info(f"hybrid graph")
        app_logger.info(f"Full-text result: {structured_data}")
        unstructured_data = [el.page_content for el in vector_index.similarity_search(question)]
        app_logger.info(f"Similarity-search result: {unstructured_data}")
        return f"""Structured data:\n{structured_data}\nUnstructured data:\n{"#Document ".join(unstructured_data)}"""
        

    # Text2Cypher Retrieval
    def text2Cypher_retriever(question: str) -> str:
        # t2c_retriever = Text2CypherRetriever(
        # driver=driver,
        # llm=t2c_llm,
        # neo4j_schema=neo4j_schema1,
        # examples=examples,
        # )
        # return t2c_retriever.search(query_text=question)
        return "xyz"
        # --- 5. Text2Cypher Retriever (Cypher QA) ---
        # cypher_chain = GraphCypherQAChain.from_llm(
        #     graph=graph,
        #     cypher_llm=llm,
        #     qa_llm=llm,
        #     verbose=True
        # )
        # return cypher_chain.run(question)
    # 🔁 Retrieval Selector
    def retriever_factory(mode="hybrid_graph"):
        if mode == "vector":
            return vector_retriever
        elif mode == "vector_graph":
            return vector_graph_retriever
        elif mode == "hybrid":
            return hybrid_retriever
        elif mode == "text2cypher":
            return text2Cypher_retriever
        else:
            return hybrid_graph_retriever
        

    # 🧠 LangChain logic
    def _format_chat_history(chat_history: List[Tuple[str, str]]) -> List:
        buffer = []
        for human, ai in chat_history:
            buffer.append(HumanMessage(content=human))
            buffer.append(AIMessage(content=ai))
        return buffer

    def create_chain(mode="hybrid_graph"):
        retriever_fn = retriever_factory(mode)
        print(f"[{mode}] CONTEXT:\n{retriever_fn}")
        _template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question,
        in its original language.
        Chat History:
        {chat_history}
        Follow Up Input: {question}
        Standalone question:"""

        _search_query = RunnableBranch(
            (
                RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(run_name="HasChatHistoryCheck"),
                RunnablePassthrough.assign(chat_history=lambda x: _format_chat_history(x["chat_history"]))
                | PromptTemplate.from_template(_template)
                | ChatOpenAI(temperature=0)
                | StrOutputParser(),
            ),
            RunnableLambda(lambda x: x["question"]),
        )

        template = """Answer the question based only on the following context:
        {context}
        
        Question: {question}
        Use natural language to answer and be concise.
        Answer:"""
        
        final_prompt = ChatPromptTemplate.from_template(template)

        chain = (
            RunnableParallel({"context": _search_query | retriever_fn, "question": RunnablePassthrough()})
            | final_prompt
            | llm
            | StrOutputParser()
        )
        return chain

    #chain = create_chain(mode=mode)

    @tool
    def LPG_CHAT(state):
        """Chat tool for querying neo4jdb via all five retrieval modes."""
    
        start_time = time.time()
        responses = {}
        questions=[]
        final_result = ""
        try:
            app_logger.info(log_separator("LPG Agent"))
            app_logger.info("Calling LLM to generate exact query...")

            # Generate exact question from LLM
            print(f"state: {state}")
            prompt = prompts.get_exact_query_for_lpg(state)
            query = llm.invoke(prompt)
            app_logger.info(f"LLM returned: {query.content}")

            # try:
            #     questions = json.loads(clean_json_string(query.content))
            #     print(f'questions: {questions}')
            # except json.JSONDecodeError as e:
            #     app_logger.error(f"Error decoding JSON response: {str(e)}")
            #     print(f'questions: {questions}')
            #     questions = []
                
            questions.append("Give me all jobs published by amazon")
            app_logger.info(f'Generated questions: {questions}')

            if questions:
                for ques in questions:
                    app_logger.info(f"Processing question: {ques}")

                    # Run all four modes
                    for mode in ["vector", "vector_graph", "hybrid", "hybrid_graph", "text2cypher"]:
                        app_logger.info(f"Running mode: {mode}")
                        chain = create_chain(mode=mode)
                        #result = retriever_factory(mode=mode)
                        result = chain.invoke({"question": ques})
                        app_logger.info(f"{mode} result: {result}")
                        responses[mode] = result
                    formatted_response = "\n\n".join(
                                f"{mode.upper()}:\n{answer}" for mode, answer in responses.items())   
                    final_result += formatted_response + "\n\n"
            
            else:
                app_logger.info(f"No queries found. State: {state}")

        except Exception as e:
            app_logger.error(f"Error occurred: {str(e)} | State: {state}")

        end_time = time.time()
        app_logger.info(f"LPG Chat agent tool completed in {end_time - start_time:.2f} seconds")
        app_logger.info(f'final response: {final_result}')
        #return 'lpg result'
        
        return final_result

    #app_logger.info(f'LPG_CHAT:{LPG_CHAT}')
    return LPG_CHAT
