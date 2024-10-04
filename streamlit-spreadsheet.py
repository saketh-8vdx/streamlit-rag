import os
import json
import streamlit as st
import openai
import boto3
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain.embeddings import BedrockEmbeddings

# from sklearn.metrics.pairwise import cosine_similarity
aws_access_key = st.secrets["aws"]["AWS_ACCESS_KEY_ID"]
aws_secret_key = st.secrets["aws"]["AWS_SECRET_ACCESS_KEY"]
aws_region = st.secrets["aws"]["AWS_REGION"]
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

bedrock = boto3.client(
    service_name='bedrock-runtime',
    region_name=aws_region,
    aws_access_key_id=aws_access_key,
    aws_secret_access_key=aws_secret_key
)


class CustomBedrockEmbeddings(BedrockEmbeddings):
    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text):
        body = json.dumps({
            "inputText": text,
            "dimensions": 1024,
            "normalize": True
        })

        response = bedrock.invoke_model(
            body=body,
            modelId='amazon.titan-embed-text-v2:0',
            contentType='application/json',
            accept='application/json'
        )

        response_body = json.loads(response['body'].read())
        return response_body['embedding']


faiss_index_path = 'spreadsheet-index-1'
embeddings = CustomBedrockEmbeddings()
vectorstore = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)
response_cache = {}


def retrieve_documents(query, top_k=75):
    documents = vectorstore.similarity_search(query, k=top_k)
    print(len(documents))
    return [doc.page_content for doc in documents]


def generate_response(query, documents):
    try:
        context = "\n\n".join(documents)
        # print(context)

        functions = [
            {
                "name": "list_cusip_numbers",
                "description": "List all CUSIP numbers found in the document. If no CUSIP numbers are found, return an empty list.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "cusip_list": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            },
                            "description": "List of CUSIP numbers. This list will be empty if no CUSIP numbers are found."
                        }
                    },
                    "required": ["cusip_list"]
                }
            }
        ]

        messages = [
            {

                "role": "system",

                "content": """Instructions for Analyzing Complex Spreadsheets:

        1. Understanding Spreadsheet Structure:

        Thorough Review: Carefully examine all aspects of the spreadsheet, including tabs, layouts, formulas, and cell references, to understand how data is organized and interconnected.

        Identification of Key Components: Locate and understand all essential financial statements and models present in the spreadsheet. This includes, but is not limited to, income statements, balance sheets, cash flow statements, forecasts, financial models, valuation models, operational metrics, scenario analyses, sensitivity analyses, and any other relevant financial or analytical data. Ensure that all types of financial information, regardless of format or complexity, are identified and considered.

        2. Accurate Data Extraction:

        Precision in Figures: Extract all financial data accurately, paying close attention to units (e.g., 
        thousands, millions), currencies, and time periods. 

        Formula Interpretation: Understand and, if necessary, replicate key formulas to grasp how figures are calculated, especially in complex models and forecasts. This includes understanding custom metrics, KPIs, and any industry-specific calculations.

        3. Analysis of Financial Models:

        Assumptions and Inputs: Summarize all key assumptions behind the financial models, including growth rates, discount rates, market conditions, operational efficiencies, cost structures, and any variables influencing projections.

        Outputs and Implications: Highlight the results of all financial models, such as projected revenues, EBITDA, cash flows, valuations, break-even analyses, and their implications for the company's financial health and valuation.

        Scenario and Sensitivity Analysis: Explain how changes in key variables impact outcomes if the spreadsheet includes different scenarios or sensitivity analyses. Cover best-case, worst-case, and most likely scenarios where applicable.

        4. Data Integration and Consistency:

        Cross-Verification: Compare and reconcile data from spreadsheets with information in other documents (e.g., PDFs, Word documents, presentations) to ensure consistency across all sources.

        Discrepancy Resolution: Note any inconsistencies or discrepancies between documents. Provide explanations if possible, or highlight them for further review.

        5. Risk Identification in Financial Data:

        Financial Risks: Identify risks evident from the financial data, such as high debt levels, cash flow issues, liquidity concerns, currency risks, market volatility exposure, or unrealistic growth projections.

        Model Limitations: Point out any limitations or potential weaknesses in the financial models, including over-reliance on certain assumptions, lack of consideration for market changes, or overly optimistic projections.

        6. Clear Presentation of Complex Data:

        Simplified Visualization: Use tables, charts, or graphs to present complex financial data clearly and concisely, aiding in the reader's understanding. This may include trend graphs, comparative tables, or visual summaries of key metrics.

        Transparent Methodology: Explain the methodologies used in financial models in straightforward language, avoiding unnecessary technical jargon. Include explanations of any specialized financial techniques or models used.

        7. Highlighting Key Financial Metrics:

        Trends and Patterns: Emphasize important financial trends and patterns, such as revenue growth, margin changes, shifts in working capital, changes in customer acquisition costs, or other significant financial indicators.

        Benchmarking: Where possible, compare the company's financial metrics against industry benchmarks, historical performance, or competitors to provide context and highlight strengths or weaknesses.

        8. Integration with Overall Analysis:

        Cohesive Narrative: Seamlessly incorporate insights from the spreadsheets into the broader investment memo. Ensure that financial data supports and enhances the overall assessment of the company's performance, strategy, and prospects.

        Actionable Insights: Translate complex financial analyses into clear, actionable insights that inform investment decisions. Highlight how the financial data impacts the valuation, risk assessment, and overall attractiveness of the investment opportunity.

        9. Comprehensive Coverage:

        Inclusivity of All Relevant Data: Ensure that the analysis covers all types of financial information present in the spreadsheets, including any non-traditional or industry-specific models and metrics.

        Adaptability: Be prepared to interpret and analyze custom financial models or unique data presentations that may not fit standard templates. Apply financial analysis principles to extract meaningful insights from any type of financial data provided."""

            },
            {
                "role": "user",
                "content": f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
            }

        ]

        response = openai.ChatCompletion.create(
            model="gpt-4-turbo",
            messages=messages,
            functions=functions,
            function_call={"name": "list_cusip_numbers"} if query.lower() == "list all cusip numbers" else "auto",
            temperature=0.2,
            top_p=0.8,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        message = response.choices[0].message

        if message.get("function_call"):
            function_args = json.loads(message["function_call"]["arguments"])
            if message["function_call"]["name"] == "list_cusip_numbers":
                return function_args.get("cusip_list", [])
        else:
            return message['content'].strip()

    except Exception as e:
        print(f"generate response error: {e}")


st.title("Financial Document Analyzer")
query = st.text_input("Enter your query:")
if st.button("Submit"):
    docs = retrieve_documents(query)
    res = generate_response(query, docs)
    st.write(res)
