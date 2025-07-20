import os
import json
import logging
import openai

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Environment variables for OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-3.5-turbo")

_client = None

def _get_client() -> openai.OpenAI:
    """Create a cached OpenAI client."""
    global _client
    if _client is None:
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not set")
        _client = openai.OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
    return _client

def call_openai_api(prompt: str, max_tokens: int = 5) -> str | None:
    """Call the OpenAI chat completion API and return the response.

    Args:
        prompt: Prompt sent to the model.
        max_tokens: Maximum number of tokens in the response.

    Returns:
        The response text, or ``None`` if an error occurs.
    """
    if not OPENAI_API_KEY:
        logging.error("OPENAI_API_KEY is not set. Cannot call the API.")
        return None

    try:
        client = _get_client()
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Error calling OpenAI API: {e}")
        return None

def filter_papers_by_topic(papers: list, topic: str = "image or video or multimodal generation") -> list:
    """Filter papers using the OpenAI API to keep only those related to ``topic``.

    Args:
        papers: List of paper dictionaries, each containing ``title`` and ``summary``.
        topic: Topic string to filter by.

    Returns:
        List of papers that match the topic.
    """
    if not OPENAI_API_KEY:
        logging.error("OPENAI_API_KEY is not set. Skipping filtering.")
        # return the original list so the pipeline can continue
        return papers

    filtered_papers = []
    logging.info(f"Filtering {len(papers)} papers for topic '{topic}' using OpenAI API...")

    for i, paper in enumerate(papers):
        title = paper.get('title', 'N/A')
        summary = paper.get('summary', 'N/A')

        # Build prompt
        prompt = f"Is the following paper primarily about '{topic}'? Answer with only 'yes' or 'no'.\n\nTitle: {title}\nAbstract: {summary}"

        # Call the API
        ai_response = call_openai_api(prompt, max_tokens=5)

        if ai_response is not None:
            logging.info(f"Paper {i+1}/{len(papers)}: '{title[:50]}...' - AI response: {ai_response}")
            if 'yes' in ai_response.lower():
                filtered_papers.append(paper)
            # Add delay here if your API rate limits require it
        else:
            logging.warning(f"Failed to get AI response for '{title[:50]}...', skipping.")
            continue

    logging.info(f"Filtering complete, {len(filtered_papers)} papers matched topic '{topic}'.")
    return filtered_papers


def categorize_papers(papers: list, categories: list[str]) -> list:
    """Assign one or more categories to each paper using the OpenAI API.

    Adds a new field ``ai_categories`` (list of strings) to each paper.

    Args:
        papers: List of paper dictionaries with ``title`` and ``summary``.
        categories: List of allowed category names.

    Returns:
        The list of papers with the ``ai_categories`` field populated.
    """
    if not OPENAI_API_KEY:
        logging.error("OPENAI_API_KEY is not set. Skipping categorization.")
        return papers

    cat_list = ", ".join(categories)
    logging.info(f"Categorizing {len(papers)} papers into: {cat_list}...")

    for i, paper in enumerate(papers):
        title = paper.get('title', 'N/A')
        summary = paper.get('summary', 'N/A')
        prompt = (
            "You are a helpful assistant that assigns one or more categories to a research paper. "
            f"Choose from the following categories: {cat_list}. "
            "Respond only in JSON like {\"categories\": [\"AIGC\"]}.\n\n"
            f"Title: {title}\nAbstract: {summary}"
        )
        ai_response = call_openai_api(prompt, max_tokens=50)
        cats: list[str] = []
        if ai_response is not None:
            try:
                if '```json' in ai_response:
                    ai_response = ai_response.split('```json')[1].split('```')[0]
                resp = json.loads(ai_response)
                if isinstance(resp.get('categories'), list):
                    cats = resp['categories']
            except Exception as e:
                logging.warning(f"Paper {i+1}: failed to parse categories: {e}")
        else:
            logging.warning(f"Paper {i+1}: categorization API returned None")
        paper['ai_categories'] = cats

    return papers


# Template used for rating prompts
rating_prompt_template = """
# Role Setting
You are an experienced researcher in the field of Artificial Intelligence, skilled at quickly evaluating the potential value of research papers.

# Task
Based on the following paper's title and abstract, please summarize it and score it across multiple dimensions (1-10 points, 1 being the lowest, 10 being the highest). Finally, provide an overall preliminary priority score.

# Input
Paper Title: %s
Paper Abstract: %s

# My Research Interests
image generation, video generation, multimodal generation

# Output Requirements
Output should always be in JSON format, strictly compliant with RFC8259.
Please output the evaluation and explanations in the following JSON format:
{
  "tldr": "<summary>", // Too Long; Didn't Read. Summarize the paper in one or two brief sentences.
  "tldr_zh": "<summary>", // Too Long; Didn't Read. Summarize the paper in one or two brief sentences, in Chinese.
  "relevance_score": <score>, // Relevance to my research interests
  "novelty_claim_score": <score>, // Degree of novelty claimed in the abstract
  "clarity_score": <score>, // Clarity and completeness of the abstract writing
  "potential_impact_score": <score>, // Estimated potential impact based on abstract claims
  "overall_priority_score": <score> // Preliminary reading priority score combining all factors above
}

# Scoring Guidelines
- Relevance: Focus on whether it is directly related to the research interests I provided.
- Novelty: Evaluate the degree of innovation claimed in the abstract regarding the method or viewpoint compared to known work.
- Clarity: Evaluate whether the abstract itself is easy to understand and complete with essential elements.
- Potential Impact: Evaluate the importance of the problem it claims to solve and the potential application value of the results.
- Overall Priority: Provide an overall score combining all the above factors. A high score indicates suggested priority for reading.
"""


def rate_papers(papers: list) -> list:
    """Rate papers using the OpenAI API and return the updated list.

    Args:
        papers: List of paper dictionaries that contain ``title`` and ``summary``.

    Returns:
        The list with rating information added to each paper.
    """
    if not OPENAI_API_KEY:
        logging.error("OPENAI_API_KEY is not set. Skipping rating.")
        return papers

    logging.info(f"Rating {len(papers)} papers via OpenAI API...")
    for i, paper in enumerate(papers):
        title = paper.get('title', 'N/A')
        summary = paper.get('summary', 'N/A')
        # Build the evaluation prompt
        prompt = rating_prompt_template % (title, summary)

        # Retry up to two times if the API call fails
        success = False
        for attempt in range(2):
            ai_response = call_openai_api(prompt, max_tokens=1000)

            if ai_response is not None:
                try:
                    if '```json' in ai_response:
                        ai_response = ai_response.split('```json')[1].split('```')[0]
                    rating_data = json.loads(ai_response)
                    logging.info(f"Paper {i+1}/{len(papers)} (attempt {attempt+1}): '{title[:50]}...' - Rating: {rating_data}")
                    papers[i].update(rating_data)
                    success = True
                    break
                except json.JSONDecodeError:
                    logging.warning(f"Paper {i+1}/{len(papers)} (attempt {attempt+1}): invalid JSON: {ai_response[:100]}...")
                except Exception as e:
                    logging.error(f"Paper {i+1}/{len(papers)} (attempt {attempt+1}): error parsing response: {e}", exc_info=True)
            else:
                logging.warning(f"Paper {i+1}/{len(papers)} (attempt {attempt+1}): API returned None")

            if attempt < 1:
                logging.info(f"Paper {i+1}/{len(papers)}: retrying...")

        if not success:
            logging.error(f"Paper {i+1}/{len(papers)}: failed to retrieve rating after retries, skipping.")
            continue

    logging.info("Rating complete.")
    return papers


# Simple manual test
if __name__ == '__main__':
    # Ensure OPENAI_API_KEY is set before running this test
    if OPENAI_API_KEY:
        test_papers = [
            {
                'title': 'Generative Adversarial Networks for Image Synthesis ',
                'summary': 'This paper introduces GANs, a framework for estimating generative models via an adversarial process... focusing on image creation.'
            },
            {
                'title': 'Deep Learning for Natural Language Processing',
                'summary': 'We explore various deep learning architectures like RNNs and Transformers for tasks such as machine translation and sentiment analysis.'
            },
            {
                'title': 'Video Generation using Diffusion Models',
                'summary': 'A novel approach to generating high-fidelity video sequences using latent diffusion models.'
            }
        ]
        logging.info("\n--- Testing filter_papers_by_topic ---")
        filtered = filter_papers_by_topic(test_papers)
        rated = rate_papers(filtered)

        logging.info("\n--- Filtered papers ---")
        for paper in filtered:
            logging.info(f"- {paper['title']}\t{paper.get('overall_priority_score', None)}")
        logging.info("--- Test finished ---")
    else:
        logging.warning("Please set OPENAI_API_KEY to run the test.")
