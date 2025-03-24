import dash
from dash import html, dcc, Output, Input, callback, dash_table,State
import dash_bootstrap_components as dbc
import pandas as pd
import configparser
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from huggingface_hub import login
from langchain_community.llms import HuggingFaceHub
from newcustomer import get_new_customer_content
import os
import json

# Initialize the app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP],suppress_callback_exceptions=True)


# Read OpenAI API key from config.ini
config = configparser.ConfigParser()
config.read('config.ini')
openai_api_key = config['openai']['api_key']



# Read Hugging Face access token from config_hug.ini
config = configparser.ConfigParser()
config.read('config_hug.ini')
hf_api_key = config['huggingface']['access_token']

# Login to Hugging Face
login(hf_api_key)
os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_api_key

# Initialize Hugging Face LLM (e.g., Mistral-7B)
llm = HuggingFaceHub(
    repo_id="mistralai/Mistral-7B-Instruct-v0.1",
    model_kwargs={"temperature": 0.7, "max_new_tokens": 512}
)

# Load datasets
customer_profiles = pd.read_csv("customer_profiles.csv")
transactions = pd.read_csv("transaction_history.csv")
social_media = pd.read_csv("social_media_activity.csv")
demographics = pd.read_csv("demographic_details.csv")

# Enhanced Prompt Template with product names and formatting
template = """
Given the following detailed customer profile:

Demographic Information:
{demographics}

Recently Purchased Products:
{transactions}

Social Media Interests and Activities:
{social_media}

Please provide the following clearly referencing the provided customer data:

1. *Adaptive Recommendation*  
- Suggest 1 specific product or service that adapts to a recent shift in the customer's behavior.  
- Include a real-world brand or product name (e.g., "Netflix Premium", "Samsung SmartThings Starter Kit").  
- Explain the connection to the customer's latest transactions.

2. *Generated Personalized Suggestions*  
- Recommend at least 2 highly relevant products or services.  
- Include specific examples with names (e.g., "Tata AIA Term Plan", "Amazon Echo Show", etc).  
- Clearly explain how the suggestion connects with demographics or social behavior.

3. *Sentiment-Driven Content Recommendation*  
- Based on social media sentiment, recommend one piece of educational or promotional content (e.g., "YouTube video: 5 Ways to Save in 2024", "Blog: How to Budget with Kids", etc).  
- Explain how it helps the customer based on their social posts.

Format the output using headings and bullet points.
PROMPT ENDED:
"""

prompt = PromptTemplate.from_template(template)

# LangChain Runnable Sequence
recommendation_chain = (
        {"demographics": RunnablePassthrough(),
         "transactions": RunnablePassthrough(),
         "social_media": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
)


# Define the layout
app.layout = html.Div([
    # For handling page navigation
    dcc.Location(id="url", refresh=False),

    # Ribbon at the top with title
    html.Div(
        children="AI-driven Hyper Personalisation",
        style={
            "position": "fixed",
            "top": 0,
            "left": 0,
            "width": "100%",
            "background-color": "#4CAF50",  # Green background color for the title bar
            "color": "white",
            "text-align": "center",
            "padding": "10px",
            "font-size": "20px",
            "font-weight": "bold",
            "z-index": "9999",
            "box-shadow": "0px 4px 8px rgba(0, 0, 0, 0.1)"
        }
    ),
    # Sidebar and Main Content
    dbc.Row([
        # Sidebar
        dbc.Col([
            html.Div([
                dbc.Nav(
                    [
                        dbc.NavLink("New Customer", href="/new-customer" , active="exact",style={"color": "black"}),
                        dbc.NavLink("Existing Customer", href="/existing-customer",  active="exact", style={"color": "black"}),
                        dbc.NavLink("ChatBot", href="/chatbot",active="exact", style={"color": "black"})
                    ],
                    vertical=True,
                    pills=True,


                ),
            ], style={
                "padding": "20px",
                "background-color": "#E0E0E0",  # Light pastel blue for the sidebar
                "height": "100vh",
                "margin-top": "60px",  # Ensures the sidebar does not overlap the title bar
                "overflow-y": "auto"  # Allows scrolling if content overflows
            })
        ], width=2),

        # Main content dynamically displayed based on page selection
        dbc.Col([
            html.Div([
                # Adding padding-top to adjust for the title bar height
                html.Div(id="page-content", style={"padding-top": "80px"})  # Proper spacing below the title bar
            ], style={"padding": "20px"})
        ], width=10),
    ]),
])

# Define content for pages
new_customer_content = html.Div([
    # Main container
    html.Div([


        # Render the "New Customer" page content dynamically
        get_new_customer_content()
    ], style={"padding": "20px"})
])
existing_customer_content = html.Div([
    dbc.Container([
        # Title
        html.H5(
            "Existing Customer Profile",
            style={
                "text-align": "left",
                "margin-bottom": "30px",
                "color": "#000000"
            }
        ),

        # Dropdown with Submit Button on the Right
        dbc.Row([
            # Dropdown Box
            dbc.Col([
                html.Label("Customer ID", style={
                    "font-weight": "bold",
                    "font-family": "sans-serif",
                    "font-size": "14px"
                }),
                dcc.Dropdown(
                    id="customer-id-dropdown",
                    options=[{"label": str(cust_id), "value": str(cust_id)} for cust_id in customer_profiles['Customer_Id'].unique()],
                    value=str(customer_profiles['Customer_Id'].unique()[0]),  # Default value
                    placeholder="Select a customer ID...",
                    style={
                        "width": "100%",  # Full width of the column
                        "margin-bottom": "20px"
                    }
                ),
                html.Div(id="selected-customer-id-output", style={"margin-top": "10px", "color": "#4CAF50"})
            ], width=8),  # Larger column for dropdown

            # Submit Button
            dbc.Col([
                dbc.Button(
                    "Generate Suggestions",
                    id="submit",
                    color="success",
                    className="mt-4",  # Align button with dropdown
                    style={
                        "width": "60%",  # Full width of the column
                        "height": "38px",
                        "background-color": "#4CAF50"
                    }
                )
            ], width=4)  # Button takes 4 columns
        ], style={"margin-bottom": "30px"}),

        # Tables Section
        dbc.Row([
            dbc.Col([
                html.H6("Demographics", style={
                    "text-align": "center",
                    "margin-bottom": "10px",
                    "color": "#000000"
                }),
                dash_table.DataTable(
                    id="demographicsTable",
                    columns=[
                        {"name": "Column Name", "id": "Column Name"},
                        {"name": "Value", "id": "Value"}
                    ],
                    data=[],  # Dynamically populated
                    style_table={"overflowX": "auto"},
                    style_data={
                        'backgroundColor': '#F4F4F4',
                        'color': '#333333',
                        'textAlign': 'left'
                    },
                    style_header={
                        'backgroundColor': '#4CAF50',
                        'color': 'white',
                        'fontWeight': 'bold',
                        'text-align': 'center'
                    },
                    page_size=10
                )
            ], width=4),

            dbc.Col([
                html.H6("Transactions", style={
                    "text-align": "center",
                    "margin-bottom": "10px",
                    "color": "#000000"
                }),
                dash_table.DataTable(
                    id="transactions",
                    columns=[
                        {"name": "Purchase_Date", "id": "Purchase_Date"},
                        {"name": "Category", "id": "Category"},
                        {"name": "Amount (In Dollars)", "id": "Amount (In Dollars)"}
                    ],
                    data=[],  # Dynamically populated
                    style_table={"overflowX": "auto"},
                    style_data={
                        'backgroundColor': '#F4F4F4',
                        'color': '#333333',
                        'textAlign': 'left'
                    },
                    style_header={
                        'backgroundColor': '#4CAF50',
                        'color': 'white',
                        'fontWeight': 'bold',
                        'text-align': 'center'
                    },
                    page_size=10
                )
            ], width=4),

            dbc.Col([
                html.H6("Social Media Activity", style={
                    "text-align": "center",
                    "margin-bottom": "10px",
                    "color": "#000000"
                }),
                dash_table.DataTable(
                    id="socialmedia",
                    columns=[
                        {"name": "Timestamp", "id": "Timestamp"},
                        {"name": "Content", "id": "Content"}
                    ],
                    data=[],  # Dynamically populated
                    style_table={"overflowX": "auto"},
                    style_data={
                        'backgroundColor': '#F4F4F4',
                        'color': '#333333',
                        'textAlign': 'left'
                    },
                    style_header={
                        'backgroundColor': '#4CAF50',
                        'color': 'white',
                        'fontWeight': 'bold',
                        'text-align': 'center'
                    },
                    page_size=10
                )
            ], width=4)
        ], style={"margin-top": "30px"}),

        # Dynamic Output Section
        dbc.Row([
            dbc.Col([

                # Recommendations Section Below Tables
                html.Div([
                    html.H5("Personalized Recommendations", id="recommendations-title", style={
                        "margin-top": "20px",  # Add spacing above the recommendations section
                        "margin-bottom": "10px",
                        "background-color":"DFFFD6",
                        "display": "none"
                    }),
                    dcc.Textarea(
                        id="recommendations-textarea",
                        value="Your recommendations will appear here after submission.",
                        style={
                            "width": "100%",
                            "height": "200px",  # Larger height
                            "padding": "10px",
                            "border-radius": "5px",
                            "font-size": "16px",
                            "border": "1px solid #ccc",
                            "display": "none"
                        } # Make it read-only
                    )
                ])
            ], width=12)  # Full-width column for title and text area
        ], style={"margin-top": "50px"})  # Separation from previous section
    ], fluid=True, style={
        "background-color": "#E0E0E0",
        "padding": "20px",
        "border-radius": "10px"
    })
])



# Callback to render the selected page content
@callback(
    Output("page-content", "children"),
    Input("url", "pathname")
)
def display_page(pathname):
    if pathname == "/new-customer" or pathname == "/":
        return new_customer_content
    elif pathname == "/existing-customer":
        return existing_customer_content
    elif pathname == "/chatbot":
        return html.H3("ChatBot Page (Under Construction)")
    elif pathname == "/logout":
        return html.H3("You have successfully logged out.")
    else:
        return html.H3("404: Page not found")

# Callback to update all tables dynamically
@app.callback(
    [Output("demographicsTable", "data"),
     Output("transactions", "data"),
     Output("socialmedia", "data")],
    [Input("customer-id-dropdown", "value")]
)
def update_tables(selected_customer_id):
    if not selected_customer_id:
        return [], [], []  # Return empty tables if no Customer ID is selected

    # Filter and transform Demographics Table
    demographics_data = customer_profiles[customer_profiles['Customer_Id'] == int(selected_customer_id)]
    transformed_demographics = transform_table_data(demographics_data)

    # Filter and transform Transactions Table
    transactions_data = transactions[transactions['Customer_Id'] == int(selected_customer_id)]
    transformed_transactions = transform_transactions_data(transactions_data)  # Filter columns here

    # Filter and transform Social Media Table
    socialmedia_data = social_media[social_media['Customer_Id'] == int(selected_customer_id)]
    transformed_socialmedia = transform_social_media_data(socialmedia_data)  # Filter 'Timestamp' and 'Content'

    return transformed_demographics, transformed_transactions, transformed_socialmedia




def transform_table_data(data):
    transformed_data = []
    for column in data.columns:
        # Create a row with the column name as the first column and the record value as the second column
        transformed_data.append({"Column Name": column, "Value": data.iloc[0][column]})
    return transformed_data

def transform_transactions_data(transactions_data):
    if not transactions_data.empty:
        transactions_data = transactions_data[['Purchase_Date', 'Category', 'Amount (In Dollars)']]
    return transactions_data.to_dict('records')


def transform_social_media_data(socialmedia_data):
    if not socialmedia_data.empty:
        # Filter for required columns only
        socialmedia_data = socialmedia_data[['Timestamp', 'Content']]
    return socialmedia_data.to_dict('records')

from dash.dependencies import Input, Output, State

@app.callback( [Output("recommendations-textarea", "style"),
                Output("recommendations-title", "style"),
               Output("recommendations-textarea", "value")],
              Input("submit", "n_clicks"),  # Trigger on button click
    State("customer-id-dropdown", "value")  # Get selected Customer ID
)
def update_recommendations(n_clicks, customer_id):
    if n_clicks:
        if customer_id:
            demographics_data = customer_profiles[customer_profiles['Customer_Id'] == int(customer_id)]
            transactions_data = transactions[transactions['Customer_Id'] == int(customer_id)][['Purchase_Date', 'Category', 'Amount (In Dollars)']]
            socialmedia_data = social_media[social_media['Customer_Id'] == int(customer_id)][['Timestamp', 'Content']]

            recommendations = recommendation_chain.invoke({
                "demographics": demographics_data,
                "transactions": ", ".join(transactions_data['Category'].tolist()),
                "social_media": ", ".join(socialmedia_data['Content'].tolist())
            })
            # Show title and text area with personalized recommendations
            return (
                {
                    "width": "100%",
                    "height": "200px",  # Larger height
                    "padding": "10px",
                    "border-radius": "5px",
                    "font-size": "16px",
                    "border": "1px solid #ccc",
                    "display": "block"
                },
                {"display": "block"},
                f"{recommendations.split("PROMPT ENDED:")[1].strip() if "Generated Personalized Suggestions" in recommendations else recommendations}"
            )
        else:
            # Show title and text area with an error message
            return {"display": "none"}, {"display": "none"}, "Please select a Customer ID before submitting."
    # Hide title and text area initially
    return {"display": "none"}, {"display": "none"}, ""



# Run the app
if __name__ == "__main__":
    app.run(debug=True)