"""
Query Clarification Module
==========================
Detects ambiguous queries and generates follow-up questions.

Uses a hybrid approach:
1. Rule-based detection (instant, free) - catches common patterns
2. Optional LLM fallback (smart, costs API call) - catches edge cases

Usage:
    from src.clarifier import check_needs_clarification
    
    result = check_needs_clarification(user_question)
    if result["needs_clarification"]:
        # Show clarification options to user
        print(result["message"])
        print(result["options"])
"""

import re
from typing import Optional
from dataclasses import dataclass


@dataclass
class ClarificationResult:
    needs_clarification: bool
    ambiguity_type: Optional[str] = None
    message: Optional[str] = None
    options: Optional[list[str]] = None
    confidence: float = 1.0  # How confident we are this needs clarification


# ============================================================
# RULE DEFINITIONS
# ============================================================

AMBIGUITY_RULES = {
    # Pattern: (regex, ambiguity_type, message, options)
    
    # ----- Missing Aggregation Level -----
    "revenue_no_level": {
        "patterns": [
            r"^(show|get|what('?s| is| are)?|display|give me|find)?\s*(me\s+)?(the\s+)?(total\s+)?revenue\s*$",
            r"^(show|get|what('?s| is| are)?|display|give me|find)?\s*(me\s+)?(the\s+)?revenue\s*(data|numbers?)?\s*$",
            r"^revenue\s*$",
        ],
        "exclude_patterns": [
            r"(by|per|for each|grouped by|for)\s+(region|nation|customer|supplier|year|month|part)",
            r"(region|nation|customer|supplier|year|month).*revenue",
            r"revenue.*(region|nation|customer|supplier|year|month)",
        ],
        "message": "I can get that for you! Would you like to see revenue broken down by region, nation, customer, or supplier? Or just the grand total?",
        "options": ["By Region", "By Nation", "By Customer", "By Supplier", "Total Overall"],
        "sql_hints": {
            "By Region": "GROUP BY region",
            "By Nation": "GROUP BY nation", 
            "By Customer": "GROUP BY customer",
            "By Supplier": "GROUP BY supplier",
            "Total Overall": "aggregate total"
        }
    },
    
    # ----- Ambiguous "Top" Queries -----
    "top_customers_no_metric": {
        "patterns": [
            r"(top|best|biggest|largest|highest)\s+\d*\s*customers?",
            r"who\s+(are|is)\s+(the\s+)?(top|best|biggest)\s+customers?",
        ],
        "exclude_patterns": [
            r"(by|based on|ranked by|sorted by)\s+(revenue|balance|orders?|spending|value)",
            r"(highest|most)\s+(revenue|balance|orders?|spending)",
        ],
        "message": "Great question! When you say 'top customers', what should I rank them by?",
        "options": ["Total Revenue", "Account Balance", "Order Count", "Avg Order Value"],
        "sql_hints": {
            "Total Revenue": "ORDER BY SUM(revenue) DESC",
            "Account Balance": "ORDER BY c_acctbal DESC",
            "Order Count": "ORDER BY COUNT(orders) DESC",
            "Avg Order Value": "ORDER BY AVG(order_value) DESC"
        }
    },
    
    "top_suppliers_no_metric": {
        "patterns": [
            r"(top|best|biggest|largest|highest)\s+\d*\s*suppliers?",
            r"who\s+(are|is)\s+(the\s+)?(top|best|biggest)\s+suppliers?",
        ],
        "exclude_patterns": [
            r"(by|based on|ranked by)\s+(revenue|parts?|orders?|volume)",
        ],
        "message": "Sure! How would you like me to rank the suppliers?",
        "options": ["By Revenue", "By Part Count", "By Order Volume"],
        "sql_hints": {
            "By Revenue": "ORDER BY SUM(revenue) DESC",
            "By Part Count": "ORDER BY COUNT(parts) DESC",
            "By Order Volume": "ORDER BY SUM(quantity) DESC"
        }
    },
    
    "top_products_no_metric": {
        "patterns": [
            r"(top|best|most popular|highest selling)\s+\d*\s*(products?|parts?|items?)",
        ],
        "exclude_patterns": [
            r"(by|based on|ranked by)\s+(revenue|quantity|orders?|sales)",
        ],
        "message": "I can find that! Should I rank products by revenue generated, quantity sold, or number of orders?",
        "options": ["By Revenue", "By Quantity Sold", "By Order Count"],
        "sql_hints": {
            "By Revenue": "ORDER BY SUM(revenue) DESC",
            "By Quantity Sold": "ORDER BY SUM(quantity) DESC",
            "By Order Count": "ORDER BY COUNT(*) DESC"
        }
    },
    
    # ----- Comparison Without Metric -----
    "compare_no_metric": {
        "patterns": [
            r"compare\s+(the\s+)?(regions?|nations?|suppliers?|customers?)",
            r"(difference|differences?)\s+between\s+(regions?|nations?|suppliers?)",
            r"how\s+do\s+(regions?|nations?|suppliers?)\s+compare",
        ],
        "exclude_patterns": [
            r"(by|based on|in terms of)\s+(revenue|sales|orders?|volume|performance)",
            r"compare.*revenue",
            r"compare.*sales",
        ],
        "message": "Happy to compare them! What metric should I use for the comparison?",
        "options": ["Revenue", "Order Count", "Avg Order Value", "Customer Count"],
        "sql_hints": {
            "Revenue": "compare SUM(revenue)",
            "Order Count": "compare COUNT(orders)",
            "Avg Order Value": "compare AVG(order_value)",
            "Customer Count": "compare COUNT(customers)"
        }
    },
    
    # ----- Time Range Ambiguity -----
    "trend_no_timeframe": {
        "patterns": [
            r"(show|what('?s| is| are)?|display)\s+(me\s+)?(the\s+)?(trend|growth|change|performance)",
            r"how\s+(has|have|did)\s+.*(changed?|grown|performed)",
        ],
        "exclude_patterns": [
            r"(in|for|during|over|last|past)\s+\d+\s*(year|month|quarter|week|day)",
            r"(from|since|between)\s+\d{4}",
            r"(this|last|previous)\s+(year|month|quarter)",
        ],
        "message": "I can show you that trend! What time period are you interested in?",
        "options": ["Last Year", "Last Quarter", "Last Month", "All Time"],
        "sql_hints": {
            "Last Year": "WHERE date >= DATE_SUB(CURRENT_DATE, INTERVAL 1 YEAR)",
            "Last Quarter": "WHERE date >= DATE_SUB(CURRENT_DATE, INTERVAL 3 MONTH)",
            "Last Month": "WHERE date >= DATE_SUB(CURRENT_DATE, INTERVAL 1 MONTH)",
            "All Time": "no date filter"
        }
    },
    
    # ----- Vague Analytical Requests -----
    "analyze_no_focus": {
        "patterns": [
            r"^analyze\s+(the\s+)?(data|sales|business|performance)$",
            r"^give\s+me\s+(an\s+)?analysis$",
            r"^(show|tell)\s+me\s+(some\s+)?insights?$",
        ],
        "exclude_patterns": [
            r"(for|about|regarding|on)\s+(region|customer|supplier|product|revenue)",
        ],
        "message": "I'd love to help you dig into the data! What aspect interests you most?",
        "options": ["Revenue by Region", "Top Customers", "Supplier Performance", "Order Trends"],
        "sql_hints": {
            "Revenue by Region": "SELECT region, SUM(revenue)...",
            "Top Customers": "SELECT customer, SUM(revenue) ORDER BY...",
            "Supplier Performance": "SELECT supplier, metrics...",
            "Order Trends": "SELECT date, COUNT(orders)..."
        }
    },
    
    # ----- Missing Filters -----
    "orders_no_filter": {
        "patterns": [
            r"^(show|list|get|find)\s+(me\s+)?(all\s+)?(the\s+)?orders$",
        ],
        "exclude_patterns": [
            r"(for|from|by|where|with|in)\s+\w+",
            r"(last|this|recent|pending|completed|urgent)",
        ],
        "message": "There are quite a few orders in the system. Want me to narrow it down, or should I show a sample?",
        "options": ["Show Sample (Top 100)", "Recent Orders", "Filter by Status", "Filter by Customer"],
        "sql_hints": {
            "Show Sample (Top 100)": "SELECT * FROM orders LIMIT 100",
            "Recent Orders": "WHERE order_date > ...",
            "Filter by Status": "add status filter",
            "Filter by Customer": "add customer filter"
        }
    },
}


# ============================================================
# CLEAR QUERY PATTERNS (No clarification needed)
# ============================================================

CLEAR_PATTERNS = [
    # Specific counts
    r"how\s+many\s+(customers?|orders?|suppliers?|products?|parts?|nations?|regions?)",
    r"(count|number|total)\s+(of\s+)?(customers?|orders?|suppliers?|products?)",
    
    # Specific aggregations with target
    r"(total|sum|average|avg|max|min)\s+\w+\s+(by|per|for)\s+\w+",
    
    # Specific entity lookups
    r"(show|get|find|list)\s+.*(for|from|in|where)\s+\w+",
    
    # Questions with specific dimensions
    r"(revenue|sales|orders?)\s+(by|per|for each)\s+(region|nation|customer|supplier|year|month)",
    
    # Specific time-bound queries
    r"(in|for|during)\s+(19|20)\d{2}",
    r"(last|this|previous)\s+(year|month|quarter|week)",
    
    # Schema/metadata questions
    r"what\s+(tables?|columns?|fields?)\s+(are|do)",
    r"(show|describe|list)\s+(the\s+)?(schema|tables?|columns?)",
]


def is_clear_query(question: str) -> bool:
    """Check if the query is specific enough to not need clarification."""
    question_lower = question.lower().strip()
    
    for pattern in CLEAR_PATTERNS:
        if re.search(pattern, question_lower):
            return True
    
    return False


def check_needs_clarification(question: str, use_llm_fallback: bool = False) -> ClarificationResult:
    """
    Main function to check if a query needs clarification.
    
    Args:
        question: The user's natural language question
        use_llm_fallback: If True, use LLM for edge cases (costs API call)
    
    Returns:
        ClarificationResult with needs_clarification flag and options if needed
    """
    question_lower = question.lower().strip()
    
    # First, check if it's clearly specific enough
    if is_clear_query(question):
        return ClarificationResult(needs_clarification=False)
    
    # Check against ambiguity rules
    for rule_name, rule in AMBIGUITY_RULES.items():
        # Check if any pattern matches
        pattern_matched = False
        for pattern in rule["patterns"]:
            if re.search(pattern, question_lower):
                pattern_matched = True
                break
        
        if not pattern_matched:
            continue
        
        # Check if any exclude pattern matches (meaning it's actually specific)
        excluded = False
        for exclude_pattern in rule.get("exclude_patterns", []):
            if re.search(exclude_pattern, question_lower):
                excluded = True
                break
        
        if excluded:
            continue
        
        # Found an ambiguity!
        return ClarificationResult(
            needs_clarification=True,
            ambiguity_type=rule_name,
            message=rule["message"],
            options=rule["options"],
            confidence=0.9
        )
    
    # No rule matched - query is either clear or an edge case
    # TODO: Add LLM fallback here if use_llm_fallback is True
    
    return ClarificationResult(needs_clarification=False)


def refine_question(original_question: str, clarification_type: str, selected_option: str) -> str:
    """
    Refine the original question based on user's clarification selection.
    
    Args:
        original_question: The original ambiguous question
        clarification_type: The type of ambiguity that was detected
        selected_option: The option the user selected
    
    Returns:
        A refined, more specific question
    """
    # Mapping of how to refine questions based on selection
    refinements = {
        "revenue_no_level": {
            "By Region": f"{original_question} grouped by region",
            "By Nation": f"{original_question} grouped by nation",
            "By Customer": f"{original_question} grouped by customer",
            "By Supplier": f"{original_question} grouped by supplier",
            "Total Overall": "What is the total overall revenue?",
        },
        "top_customers_no_metric": {
            "Total Revenue": f"{original_question} ranked by total revenue",
            "Account Balance": f"{original_question} ranked by account balance",
            "Order Count": f"{original_question} ranked by number of orders",
            "Avg Order Value": f"{original_question} ranked by average order value",
        },
        "top_suppliers_no_metric": {
            "By Revenue": f"{original_question} ranked by revenue",
            "By Part Count": f"{original_question} ranked by number of parts supplied",
            "By Order Volume": f"{original_question} ranked by order volume",
        },
        "top_products_no_metric": {
            "By Revenue": f"{original_question} ranked by revenue",
            "By Quantity Sold": f"{original_question} ranked by quantity sold",
            "By Order Count": f"{original_question} ranked by order count",
        },
        "compare_no_metric": {
            "Revenue": f"{original_question} by revenue",
            "Order Count": f"{original_question} by order count",
            "Avg Order Value": f"{original_question} by average order value",
            "Customer Count": f"{original_question} by customer count",
        },
        "trend_no_timeframe": {
            "Last Year": f"{original_question} over the last year",
            "Last Quarter": f"{original_question} over the last quarter",
            "Last Month": f"{original_question} over the last month",
            "All Time": f"{original_question} for all available data",
        },
        "analyze_no_focus": {
            "Revenue by Region": "Show me revenue breakdown by region",
            "Top Customers": "Who are the top 10 customers by revenue?",
            "Supplier Performance": "Compare supplier performance by total revenue",
            "Order Trends": "Show me monthly order trends over time",
        },
        "orders_no_filter": {
            "Show Sample (Top 100)": "Show me the first 100 orders",
            "Recent Orders": "Show me the most recent orders",
            "Filter by Status": "Show me orders grouped by their status",
            "Filter by Customer": "Show me order counts by customer",
        },
    }
    
    if clarification_type in refinements:
        if selected_option in refinements[clarification_type]:
            return refinements[clarification_type][selected_option]
    
    # Fallback: just append the selection
    return f"{original_question} ({selected_option})"


# ============================================================
# TESTING
# ============================================================

if __name__ == "__main__":
    test_queries = [
        # Should trigger clarification
        "Show me the revenue",
        "Who are the top customers?",
        "Compare the regions",
        "What's the trend?",
        "Analyze the data",
        "Show me orders",
        "Top 10 suppliers",
        
        # Should NOT trigger clarification (specific enough)
        "How many customers are there?",
        "Show me revenue by region",
        "Top customers by revenue",
        "Revenue for the EUROPE region",
        "Orders from last year",
        "Compare regions by order count",
        "Total revenue for 1995",
    ]
    
    print("=" * 60)
    print("CLARIFICATION MODULE TEST")
    print("=" * 60)
    
    for query in test_queries:
        result = check_needs_clarification(query)
        status = "NEEDS CLARIFICATION" if result.needs_clarification else "CLEAR"
        print(f"\n{status}: \"{query}\"")
        if result.needs_clarification:
            print(f"   Type: {result.ambiguity_type}")
            print(f"   Message: {result.message}")
            print(f"   Options: {result.options}")