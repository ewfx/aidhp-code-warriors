# Prompt template definitions
# Enhanced Prompt Template with formatting and links
base_template = """
Given the following detailed customer profile:

Demographic Information:
{demographics}

Recently Purchased Products:
{transactions}

Social Media Interests and Activities:
{social_media}

Please provide the following:
{category_prompt}

**Output Requirements:**
- Start each suggestion with a heading: **Recommendation:** followed by **Reason for Recommendation should be briefly:** heading should not 
be 
repeated.
- Use bullet points under each outcome and reasoning.
- Highlight product names in **bold**.
- Provide specific, real-world **product or service names**.
- Include at least one **online source link (http URL)** for each product or content.
- Format everything using **Markdown** for clarity and readability.

PROMPT ENDED:
"""

categories = {
    "Adaptive Recommendation Engine": "1. Suggest a specific product or service name that adapts to a recent shift in the customer's behavior with brand name and reasoning. Note: Prompt should not be included in the response",
    "Product/Service Suggestions": "2. Recommend at least 5 highly relevant products(electronic,house hold items, online course, ecommorce,etc)/services breif note with examples. Note: Prompt should not be included in the response",
    "Sentiment-Driven Content Recommendations": "3. Recommend content (videos, blogs, etc.) online links based on sentiment analysis of social media posts. Note: Prompt should not be included in the response",
    "Predictive Customer Insights & Business Strategies": "4. Predict churn risks and suggest retention strategies with examples. Note: Prompt should not be included in the response",
    "Multi-Modal Personalization": "5. Use text, image, or voice input to personalize suggestions with examples. Act like a chatbot to respond to the user interactively using provided multi-modal input. Note: Prompt should not be included in the response",
    "Personalized Financial Product Recommendations": "6. Suggest credit plans, various banks products, loans, or investments based on transaction, location, risk profile. Note: Prompt should not be included in the response. Note: Prompt should not be included in the response"
}