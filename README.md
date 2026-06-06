# 🔍 Audit AI: Intelligent Financial Compliance Engine

### 📋 Overview
An AI-driven auditing solution designed to transform traditional financial oversight from **manual sampling** to **100% ledger coverage**. 
This tool bridges the gap between **Statistical Anomaly Detection** and **Semantic Reasoning**, using a hybrid approach to identify accounting inconsistencies, potential fraud, and compliance breaches that human auditors might miss.

### 🚀 Key Features
* **100% Transaction Coverage:** Moves beyond the traditional 5% manual sampling method by analyzing every single journal entry in a fiscal year within seconds.
* **Semantic Reconciliation (LLM):** Utilizes **GPT-4/LLMs** to cross-reference transaction descriptions with accounting codes, detecting "contextual anomalies" (e.g., an "Office Supply" description assigned to a "Capital Investment" account).
* **Statistical Outlier Detection:** Implements **Isolation Forest** and **Z-Score** algorithms to flag numerical outliers, unusual transaction hours, or suspicious rounding patterns (Benford’s Law).
* **Explainable Audit Trail:** Instead of "Black Box" alerts, the system generates natural language justifications for every flagged risk, allowing auditors to understand *why* a specific entry was deemed suspicious.
* **Compliance Guardrails:** Automated checks against specific accounting standards (SNC/IFRS), flagging unauthorized manual overrides or inconsistent VAT applications.

### 🛠️ Tech Stack
* **Core:** Python 3.10+
* **AI/LLM:** OpenAI API (GPT-4), LangChain
* **Machine Learning:** Scikit-Learn (Isolation Forest), Statsmodels
* **Data Engineering:** Pandas, NumPy
* **Frontend/UI:** Streamlit (Audit Dashboard)
* **Testing:** Pytest (for financial logic validation)

### 📊 Methodology: The Hybrid Approach
1. **Quantitative Filter:** The engine first runs a statistical pass to find numerical "needles in the haystack" based on historical distributions.
2. **Qualitative Filter:** Flagged entries are sent to the LLM agent to evaluate the *logic* of the transaction based on business context and accounting rules.
3. **Human-in-the-loop:** Auditors review the "High Risk" dashboard, confirming or dismissing alerts, which refines the model's future precision.
