import json
import re

import gradio as gr
from textblob import TextBlob

"""
The trail to the summit was absolutely beautiful. Parking was a nightmare. The rangers were friendly and helpful! The bathrooms were disgusting.
"""
# ---------------------------------------------------------------------------
# Helpers (provided). They avoid TextBlob features that need extra downloads.
# ---------------------------------------------------------------------------
def split_sentences(text):
    """Split text into sentences on . ! or ? followed by whitespace."""
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def split_words(text):
    """Return the words in text (letters, digits, and apostrophes)."""
    return re.findall(r"[A-Za-z0-9']+", text)

def sort_rankers(rankers, order):
    for i in range(1, len(rankers)):
        key = rankers[i]
        j = i - 1
        if order == "most_positive":
            while j >= 0 and float(rankers[j]["polarity"]) > float(key["polarity"]):
                rankers[j + 1] = rankers[j]
                j -= 1
        else:
            while j >= 0 and float(rankers[j]["polarity"]) < float(key["polarity"]):
                rankers[j + 1] = rankers[j]
                j -= 1
        rankers[j + 1] = key



# ---------------------------------------------------------------------------
# Tool 1 (provided): from the Hugging Face MCP Course, Unit 2
# ---------------------------------------------------------------------------
def sentiment_analysis(text: str) -> str:
    """
    Analyze the sentiment of the given text.

    Args:
        text (str): The text to analyze

    Returns:
        str: A JSON string containing polarity, subjectivity, and assessment
    """
    sentiment = TextBlob(text).sentiment
    result = {
        "polarity": round(sentiment.polarity, 2),          # -1 (negative) to 1 (positive)
        "subjectivity": round(sentiment.subjectivity, 2),  # 0 (objective) to 1 (subjective)
        "assessment": "positive" if sentiment.polarity > 0
                      else "negative" if sentiment.polarity < 0 else "neutral",
    }
    return json.dumps(result)
# ---------------------------------------------------------------------------
# Tool 2: word and sentence counts
# ---------------------------------------------------------------------------
def text_statistics(text: str) -> str:
    # TODO 1: write a Google-style docstring (summary line, Args:, Returns:)
    # TODO 1: return json.dumps of {"word_count", "sentence_count",
    #         "avg_sentence_length"} using split_words / split_sentences.
    #         Round the average to 1 decimal; guard against 0 sentences.
    """
    Determine the statistics of the given text
    
    Args:
        text (str): The text to analyze
    
    Returns:
            str: A JSON string containing word count, sentence count, and average sentence length
    """
    if text is None:
        return {"error": "INVALID_SENTENCE_LENGTH", "message": "Inputed text cannot be 'None'. Must have aleast one sentence to be valid."}

    sentences = split_sentences(text)
    word_count = 0
    for sentence in sentences:
        word_count += len(split_words(sentence))
    avg_sentence_length = word_count / len(sentences)
    
    return json.dumps({
        "word_count": word_count,
        "sentence_count": round(len(sentences), 1),
        "avg_sentence_length": round(avg_sentence_length, 1),
    })


# ---------------------------------------------------------------------------
# Tool 3: rank sentences by sentiment
# ---------------------------------------------------------------------------
def rank_sentences(text: str, order: str, top_k: int) -> str:
    # TODO 2: write a docstring whose Args: block describes ALL three parameters
    # TODO 2: score each sentence with TextBlob(sentence).sentiment.polarity,
    #         sort by order ("most_positive" = highest first,
    #         "most_negative" = lowest first), and return json.dumps of the
    #         first top_k items as [{"sentence": ..., "polarity": ...}, ...]
    """
    Rank the sentences of the given text based on order
    
    Args:
        text (str): The text to analyze
        order (str): The order in which the sentences will be ranked, 'most_negative' = lowest polarity first, 'most_positive = highest polarity first
        top_k (int): How many sentences that will be returned and ranked
    Returns:
        str: A JSON string containing polarity, subjectivity, and assessment
    """
    rankers = []
    sentences = split_sentences(text)
    for sentence in sentences:
        sentiment = json.loads(sentiment_analysis(sentence))
        polarity = sentiment["polarity"]
        if len(rankers) < top_k:
            rankers.append({"sentence": sentence, "polarity": polarity})
            if len(rankers) == top_k:
                sort_rankers(rankers, order)
        else:
            if order == "most_positive":
                if polarity > rankers[0]["polarity"]:
                    rankers[0] = {"sentence": sentence, "polarity": polarity}
                    sort_rankers(rankers, order)
            else:
                if polarity < rankers[0]["polarity"]:
                    rankers[0] = {"sentence": sentence, "polarity": polarity}
                    sort_rankers(rankers, order)
                    rankers.reverse()
    return json.dumps(rankers)


# ---------------------------------------------------------------------------
# One gr.Interface per tool. Each becomes one tab in the UI AND one MCP tool.
# ---------------------------------------------------------------------------
demo = gr.TabbedInterface(
    [
        gr.Interface(fn=sentiment_analysis, inputs=gr.Textbox(label="Text"), outputs=gr.Textbox()),
        # TODO 3: add a gr.Interface for text_statistics (one Textbox in, one out)
        gr.Interface(fn=text_statistics, inputs=gr.Textbox(label="Text"), outputs=gr.Textbox()),
        # TODO 3: add a gr.Interface for rank_sentences with three inputs:
        #         gr.Textbox, gr.Radio(["most_positive", "most_negative"], value=...),
        #         and gr.Number(value=3, precision=0)
        gr.Interface(fn=rank_sentences, 
                     inputs=[gr.Textbox(label="Text"), 
                             gr.Radio(["most_positive","most_negative"], value="most_positive"),
                             gr.Number(value=3, precision=0)],
                             outputs=gr.Textbox())
    ],  
    tab_names=["Sentiment", "Text Statistics", "Rank Sentences"],  # TODO 3: one name per tab
    title="Review Toolkit",
)

if __name__ == "__main__":
    demo.launch(mcp_server=True, share=True, show_error=True)  # TODO 4: start the MCP server and create a public share link