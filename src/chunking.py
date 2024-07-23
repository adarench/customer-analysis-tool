import os
import uuid
from typing import Optional
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from rich import print
from pydantic import BaseModel

from src.utils import read_files_in_directory, extract_speaker_info

load_dotenv()

class ChunkID(BaseModel):
    chunk_id: Optional[str]

    class Config:
        arbitrary_types_allowed = True

class AgenticChunker:
    def __init__(self, openai_api_key=None):
        self.chunks = {}
        self.id_truncate_limit = 5

        self.generate_new_metadata_ind = True
        self.print_logging = True

        if openai_api_key is None:
            openai_api_key = os.getenv("OPENAI_API_KEY")

        if openai_api_key is None:
            raise ValueError("API key is not provided and not found in environment variables")

        self.llm = ChatOpenAI(model='gpt-3.5-turbo', openai_api_key=openai_api_key, temperature=0)

    def add_propositions(self, text):
        propositions_with_context = self._extract_propositions_with_context(text)
        for proposition, original_text in propositions_with_context:
            self.add_proposition(proposition, original_text)

    def add_proposition(self, proposition, original_text):
        if self.print_logging:
            print(f"\nAdding: '{proposition}' with original text: '{original_text}'")

        if len(self.chunks) == 0:
            if self.print_logging:
                print("No chunks, creating a new one")
            self._create_new_chunk(proposition, original_text)
            return

        chunk_id = self._find_relevant_chunk(proposition)

        # If a chunk was found then add the proposition to it
        if chunk_id and chunk_id in self.chunks:
            if self.print_logging:
                print(f"Chunk Found ({self.chunks[chunk_id]['chunk_id']}), adding to: {self.chunks[chunk_id]['title']}")
            self.add_proposition_to_chunk(chunk_id, proposition, original_text)
        else:
            if self.print_logging:
                print("No relevant chunks found, creating a new one")
            # If a chunk wasn't found, then create a new one
            self._create_new_chunk(proposition, original_text)

    def add_proposition_to_chunk(self, chunk_id, proposition, original_text):
        self.chunks[chunk_id]['propositions'].append({
            'proposition': proposition,
            'original_text': original_text
        })

        if self.generate_new_metadata_ind:
            self.chunks[chunk_id]['summary'] = self._update_chunk_summary(self.chunks[chunk_id])
            self.chunks[chunk_id]['title'] = self._update_chunk_title(self.chunks[chunk_id])

    def _extract_propositions_with_context(self, text):
        PROMPT = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    You are an AI designed to extract meaningful propositions from text. 
                    A proposition is a statement that conveys an idea or assertion. Your goal is to break down the text into specific, concise propositions that capture the key points and details discussed.

                    For instance, given the text about a conversation on cloud cost management, break down the details into propositions like:
                    - The challenges of understanding and managing cloud costs.
                    - The importance of proper tagging for cost allocation.
                    - Distinctions between cost of goods sold and R&D expenses.
                    - The need for tools to estimate and optimize costs, especially in areas like machine learning.

                    Text: {text}
                    Propositions:
                    """,
                ),
                ("user", "{text}"),
            ]
        )

        runnable = PROMPT | self.llm
        response = runnable.invoke({"text": text}).content

        propositions = [p.strip() for p in response.split("\n") if p.strip()]
        return [(p, text) for p in propositions]  # Capture original text with each proposition

    def _update_chunk_summary(self, chunk):
        PROMPT = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    You are the steward of a group of chunks which represent groups of sentences that talk about a similar topic.
                    A new proposition was just added to one of your chunks. You should generate a very brief 1-sentence summary which will inform viewers what a chunk group is about.

                    A good summary will say what the chunk is about, and give any clarifying instructions on what to add to the chunk.

                    You will be given a group of propositions which are in the chunk and the chunk's current summary.

                    Your summaries should anticipate generalization. If you get a proposition about apples, generalize it to food.
                    Or month, generalize it to "date and times".

                    Example:
                    Input: Proposition: Greg likes to eat pizza
                    Output: This chunk contains information about the types of food Greg likes to eat.

                    Only respond with the chunk's new summary, nothing else.
                    """,
                ),
                ("user", "Chunk's propositions:\n{proposition}\n\nCurrent chunk summary:\n{current_summary}"),
            ]
        )

        runnable = PROMPT | self.llm

        new_chunk_summary = runnable.invoke({
            "proposition": "\n".join([p['proposition'] for p in chunk['propositions']]),
            "current_summary": chunk['summary']
        }).content

        return new_chunk_summary

    def _update_chunk_title(self, chunk):
        PROMPT = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    You are the steward of a group of chunks which represent groups of sentences that talk about a similar topic.
                    A new proposition was just added to one of your chunks. You should generate a very brief updated chunk title which will inform viewers what a chunk group is about.

                    A good title will say what the chunk is about.

                    You will be given a group of propositions which are in the chunk, chunk summary, and the chunk title.

                    Your title should anticipate generalization. If you get a proposition about apples, generalize it to food.
                    Or month, generalize it to "date and times".

                    Example:
                    Input: Summary: This chunk is about dates and times that the author talks about
                    Output: Date & Times

                    Only respond with the new chunk title, nothing else.
                    """,
                ),
                ("user", "Chunk's propositions:\n{proposition}\n\nChunk summary:\n{current_summary}\n\nCurrent chunk title:\n{current_title}"),
            ]
        )

        runnable = PROMPT | self.llm

        updated_chunk_title = runnable.invoke({
            "proposition": "\n".join([p['proposition'] for p in chunk['propositions']]),
            "current_summary": chunk['summary'],
            "current_title": chunk['title']
        }).content

        return updated_chunk_title

    def _get_new_chunk_summary(self, proposition):
        PROMPT = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    You are the steward of a group of chunks which represent groups of sentences that talk about a similar topic.
                    You should generate a very brief 1-sentence summary which will inform viewers what a chunk group is about.

                    A good summary will say what the chunk is about, and give any clarifying instructions on what to add to the chunk.

                    You will be given a proposition which will go into a new chunk. This new chunk needs a summary.

                    Your summaries should anticipate generalization. If you get a proposition about apples, generalize it to food.
                    Or month, generalize it to "date and times".

                    Example:
                    Input: Proposition: Greg likes to eat pizza
                    Output: This chunk contains information about the types of food Greg likes to eat.

                    Only respond with the new chunk summary, nothing else.
                    """,
                ),
                ("user", "Determine the summary of the new chunk that this proposition will go into:\n{proposition}"),
            ]
        )

        runnable = PROMPT | self.llm

        new_chunk_summary = runnable.invoke({
            "proposition": proposition
        }).content

        return new_chunk_summary

    def _get_new_chunk_title(self, summary):
        PROMPT = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    You are the steward of a group of chunks which represent groups of sentences that talk about a similar topic.
                    You should generate a very brief few word chunk title which will inform viewers what a chunk group is about.

                    A good chunk title is brief but encompasses what the chunk is about.

                    You will be given a summary of a chunk which needs a title.

                    Your titles should anticipate generalization. If you get a proposition about apples, generalize it to food.
                    Or month, generalize it to "date and times".

                    Example:
                    Input: Summary: This chunk is about dates and times that the author talks about
                    Output: Date & Times

                    Only respond with the new chunk title, nothing else.
                    """,
                ),
                ("user", "Determine the title of the chunk that this summary belongs to:\n{summary}"),
            ]
        )

        runnable = PROMPT | self.llm

        new_chunk_title = runnable.invoke({
            "summary": summary
        }).content

        return new_chunk_title

    def _create_new_chunk(self, proposition, original_text):
        new_chunk_id = str(uuid.uuid4())[:self.id_truncate_limit]
        new_chunk_summary = self._get_new_chunk_summary(proposition)
        new_chunk_title = self._get_new_chunk_title(new_chunk_summary)

        self.chunks[new_chunk_id] = {
            'chunk_id': new_chunk_id,
            'propositions': [{'proposition': proposition, 'original_text': original_text}],
            'title': new_chunk_title,
            'summary': new_chunk_summary,
            'chunk_index': len(self.chunks)
        }
        if self.print_logging:
            print(f"Created new chunk ({new_chunk_id}): {new_chunk_title}")

    def get_chunk_outline(self):
        chunk_outline = ""

        for chunk_id, chunk in self.chunks.items():
            single_chunk_string = f"""Chunk ({chunk['chunk_id']}): {chunk['title']}\nSummary: {chunk['summary']}\n\n"""

            chunk_outline += single_chunk_string

        return chunk_outline

    def _find_relevant_chunk(self, proposition):
        current_chunk_outline = self.get_chunk_outline()

        PROMPT = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    Determine whether or not the "Proposition" should belong to any of the existing chunks.

                    A proposition should belong to a chunk if their meaning, direction, or intention are similar.
                    The goal is to group similar propositions and chunks.

                    If you think a proposition should be joined with a chunk, return the chunk id.
                    If you do not think an item should be joined with an existing chunk, just return "No chunks".

                    Example:
                    Input:
                        - Proposition: "Greg really likes hamburgers"
                        - Current Chunks:
                            - Chunk ID: 2n4l3d
                            - Chunk Name: Places in San Francisco
                            - Chunk Summary: Overview of the things to do with San Francisco Places

                            - Chunk ID: 93833k
                            - Chunk Name: Food Greg likes
                            - Chunk Summary: Lists of the food and dishes that Greg likes
                    Output: 93833k
                    """,
                ),
                ("user", "Current Chunks:\n--Start of current chunks--\n{current_chunk_outline}\n--End of current chunks--"),
                ("user", "Determine if the following statement should belong to one of the chunks outlined:\n{proposition}"),
            ]
        )

        runnable = PROMPT | self.llm

        chunk_found = runnable.invoke({
            "proposition": proposition,
            "current_chunk_outline": current_chunk_outline
        }).content.strip()

        if chunk_found.startswith("Chunk ID:"):
            chunk_id = chunk_found.split("Chunk ID:")[-1].strip()
            return chunk_id if chunk_id in self.chunks else None
        elif "No chunks" in chunk_found:
            return None
        else:
            return None

    def get_chunks(self, get_type='dict'):
        if get_type == 'dict':
            return self.chunks
        if get_type == 'list_of_strings':
            chunks = []
            for chunk_id, chunk in self.chunks.items():
                chunks.append(" ".join([p['proposition'] for p in chunk['propositions']]))
            return chunks

    def pretty_print_chunks(self):
        print(f"\nYou have {len(self.chunks)} chunks\n")
        for chunk_id, chunk in self.chunks.items():
            print(f"Chunk #{chunk['chunk_index']}")
            print(f"Chunk ID: {chunk_id}")
            print(f"Summary: {chunk['summary']}")
            print(f"Propositions:")
            for prop in chunk['propositions']:
                print(f"    - {prop['proposition']}")
                print(f"      (Original Text: {prop['original_text']})")
        print("\n\n")

    def pretty_print_chunk_outline(self):
        print("Chunk Outline\n")
        print(self.get_chunk_outline())

if __name__ == "__main__":
    ac = AgenticChunker()

    # Read files from the directory
    directory_path = "/Users/arencher/Desktop/customer-feedback-analysis/data"  # Update this path
    combined_text = read_files_in_directory(directory_path)
    speaker_info = extract_speaker_info(combined_text)

    # Process the combined text
    ac.add_propositions(combined_text)
    ac.pretty_print_chunks()
    ac.pretty_print_chunk_outline()
    print(ac.get_chunks(get_type='list_of_strings'))
