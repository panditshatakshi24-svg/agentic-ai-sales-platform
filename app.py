import pandas as pd
import json

from langchain_ollama import OllamaLLM
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.prompts import PromptTemplate


# -----------------------------------
# 1. Initialize Local AI Model
# -----------------------------------

llm = OllamaLLM(
    model="llama3",
    temperature=0
)


# -----------------------------------
# 2. Read and Analyze CSV
# -----------------------------------

def analyze_sales_csv(file_path):

    # Load CSV file
    df = pd.read_csv(file_path)

    # Create revenue column
    df["order_revenue"] = (
        df["quantity"]
        * df["unit_price"]
        * (1 - df["discount_percent"] / 100)
    )

    # Return business metrics
    return {

        # Total revenue
        "total_revenue": round(
            df["order_revenue"].sum(),
            2
        ),

        # Average order value
        "average_order_value": round(
            df["order_revenue"].mean(),
            2
        ),

        # Number of orders
        "orders": len(df),

        # Revenue by product category
        "revenue_by_category": (
            df.groupby("product_category")[
                "order_revenue"
            ]
            .sum()
            .round(2)
            .to_dict()
        ),

        # Revenue by region
        "revenue_by_region": (
            df.groupby("region")[
                "order_revenue"
            ]
            .sum()
            .round(2)
            .to_dict()
        ),

        # Top payment method
        "top_payment_method": (
            df.groupby("payment_method")[
                "order_revenue"
            ]
            .sum()
            .idxmax()
        ),
    }


# -----------------------------------
# 3. Web Search Tool
# -----------------------------------

search = DuckDuckGoSearchRun()


def fetch_benchmarks():

    return search.run(
        "average ecommerce order value benchmark"
    )


# -----------------------------------
# 4. Run All Tools
# -----------------------------------

def run_analysis():

    # Read local CSV file
    metrics = analyze_sales_csv(
        "sales.csv"
    )

    # Get industry benchmark
    benchmarks = fetch_benchmarks()

    # Debug print
    print("\nDEBUG: METRICS\n")

    print(
        json.dumps(
            metrics,
            indent=2
        )
    )

    return metrics, benchmarks


# -----------------------------------
# 5. Prompt Template
# -----------------------------------

writer_prompt = PromptTemplate.from_template("""

You are a business analyst.

RULES:
- Use ONLY the numbers provided
- Do NOT invent values
- Do NOT assume currency
- Keep insights short and clear

METRICS:
{metrics}

INDUSTRY CONTEXT:
{benchmarks}

Write:

1. Three business insights
2. Three business recommendations

Use bullet points.
""")


# -----------------------------------
# 6. Main Program
# -----------------------------------

if __name__ == "__main__":

    # Get metrics + benchmarks
    metrics, benchmarks = run_analysis()

    # Send data to AI model
    response = llm.invoke(

        writer_prompt.format(

            metrics=json.dumps(
                metrics,
                indent=2
            ),

            benchmarks=benchmarks
        )
    )

    # Final output
    print("\nFINAL AI REPORT:\n")

    print(response)