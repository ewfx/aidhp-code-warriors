from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.llms import HuggingFaceHub
from huggingface_hub import login
import os

def setup_llm(hf_api_key):
    login(hf_api_key)
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_api_key
    return HuggingFaceHub(
        repo_id="mistralai/Mistral-7B-Instruct-v0.1",
        model_kwargs={"temperature": 0.7, "max_new_tokens": 512}
    )

base_template = """
Given the detailed customer profile:

Demographic Information:
{demographics}

Recently Purchased Products:
{transactions}

Social Media Interests and Activities:
{social_media}

Please provide:
{category_prompt}

**Output Requirements:**
- **Recommendation:** clearly provided once.
- **Reason for Recommendation:** provided once.
- Bullet points under each recommendation.
- Highlight product names in **bold**.
- Provide real-world examples with HTTP URLs.
- Format clearly in **Markdown**.

PROMPT ENDED:
"""

def build_recommendation_chain(llm):
    prompt = PromptTemplate.from_template(base_template)
    return (
            {
                "demographics": RunnablePassthrough(),
                "transactions": RunnablePassthrough(),
                "social_media": RunnablePassthrough(),
                "category_prompt": RunnablePassthrough()
            } | prompt | llm | StrOutputParser()
    )
