# 🔍 Audit AI: Intelligent Financial Compliance Engine

### 📋 Visão Geral
Uma solução de auditoria impulsionada por IA, concebida para transformar a fiscalização financeira tradicional, passando da **amostragem manual** para uma **cobertura de 100% do razão contábil**.
Esta ferramenta preenche a lacuna entre a **Deteção Estatística de Anomalias** e o **Raciocínio Semântico**, utilizando uma abordagem híbrida para identificar inconsistências contabilísticas, fraudes potenciais e violações de conformidade que passariam despercebidas por auditores humanos.

### 🚀 Funcionalidades Principais
*   **Cobertura de Transações a 100%:** Supera o método tradicional de amostragem manual (geralmente 5%) ao analisar cada lançamento de diário de um ano fiscal em segundos.
*   **Reconciliação Semântica (LLM):** Utiliza **GPT-4/LLMs** para cruzar as descrições das transações com os códigos contabilísticos, detetando "anomalias contextuais" (ex: uma descrição de "Material de Escritório" atribuída a uma conta de "Investimento de Capital").
*   **Deteção Estatística de Outliers:** Implementa algoritmos de **Isolation Forest** e **Z-Score** para sinalizar valores numéricos atípicos, horários de transação invulgares ou padrões de arredondamento suspeitos (Lei de Benford).
*   **Pista de Auditoria Explicável:** Em vez de alertas de "caixa negra", o sistema gera justificações em linguagem natural para cada risco sinalizado, permitindo que os auditores entendam *por que* um lançamento específico foi considerado suspeito.
*   **Barreiras de Conformidade (Guardrails):** Verificações automáticas contra normas contabilísticas específicas (SNC/IFRS), sinalizando lançamentos manuais não autorizados ou aplicações inconsistentes de IVA.

### 🛠️ Stack Tecnológica
*   **Core:** Python 3.10+
*   **IA/LLM:** OpenAI API (GPT-4), LangChain
*   **Machine Learning:** Scikit-Learn (Isolation Forest), Statsmodels
*   **Processamento de Dados:** Pandas, NumPy
*   **Frontend/UI:** Streamlit (Audit Dashboard)
*   **Testes:** Pytest (para validação de lógica financeira)

### 📊 Metodologia: A Abordagem Híbrida
1.  **Filtro Quantitativo:** O motor executa primeiro uma análise estatística para encontrar "agulhas no palheiro" numérico com base em distribuições históricas.
2.  **Filtro Qualitativo:** Os lançamentos sinalizados são enviados para um agente de LLM para avaliar a *lógica* da transação com base no contexto do negócio.
3.  **Human-in-the-loop:** Os auditores revisam o dashboard de "Alto Risco", confirmando ou descartando alertas, o que refina a precisão futura do modelo.
