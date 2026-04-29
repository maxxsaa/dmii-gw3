# Your Challenge

## Optimizing Production for AgroMill Corp

## Case Context

Your team has been hired to optimize operations for your new client AgroMill Corp.

### Context

AgroMill Corp production and profitability are highly sensitive to external conditions and raw material characteristics, such as:

- **Particle Size (Granulometry)**  
  Affects texture, dissolution time, and flavor (e.g., coffee powder or flour).
- **Density**  
  Crucial for packaging operations to ensure packets are filled accurately by weight.
- **Moisture Content**  
  The primary factor in preventing mold and ensuring a viable shelf-life.

### Goal

Assess which operational components and raw material features most affect AgroMill's production output, providing them with a data-driven strategy to increase efficiency, quality, and throughput.

It's time to demonstrate your skills from the Data Mining II course. Be thoughtful, creative, and show why you're the best fit for the task.

---

## How AgroMill Works

### Shifts

- Morning (06:00 - 14:00)
- Afternoon (14:00 - 22:00)
- Night (22:00 - 06:00)

**Note:** At the end of each shift, a "reset" and a set-up period are required.

### Maintenance

Maintenance and replacement of specific mechanical components, such as the sieves (crivos), can increase set-up time during specific shifts.

### Quality Control

Quality controls are performed continuously during each shift to ensure the product meets AgroMill's client specifications and contractual requirements.

### Illustrative Diagram

Illustrative diagram for the client's processes.

---

## Process Understanding

### Scales

The scales are the key measurement points across the production line, recording how much material enters and leaves as final product. They validate the mass balance, detect process deviations, and ensure each fraction is routed correctly. Reliable scale readings are essential for tracking flow, confirming separation performance, and supporting quality control.

### Production Variables

- **Mechanical Drivers:** Maintenance of sieves and equipment separation settings are the primary drivers of output volume.
- **Shift Operations:** Three daily shifts (Morning, Afternoon, Night) with mandatory "reset" and setup periods.
- **Input/Output Balance:** Critical monitoring to ensure physical laws are met and hardware data collection is accurate.
- **Sensitivity:** High vulnerability to external conditions due to the use of natural raw materials.

### Product Dataset (Output Quality)

- **Moisture:** Essential for shelf-life and mold prevention.
- **Density:** Controls packaging accuracy and weight-based filling.
- **Particle Size:** Determines final texture, flavor, and dissolution.
- **Quality Score:** Continuous analysis focusing on Product 1 specifications.

---

## Provided Data

The dataset you are provided with contains:

- Maintenance & Equipment Setup
- Product Quality Control
- Operational Sensor Data

### A. Maintenance & Equipment Setup

Historical records of mechanical component replacements and "reset" periods across shifts.

**2 datasets:**
- Equipment separation (5 columns)
- Mesh (8 columns)

### B. Product Quality Control

Detailed sample analysis used to determine the Quality Score.

**3 datasets:**
- Product 1, for two different clients (6 columns)
- Product 2 (6 columns)

### C. Operational Sensor Data

Real-time monitoring of factory components and their performance.

**1 dataset:**
- Sensors (3 columns)

---

## Dataset Structures and Fields

### Maintenance, Equipment Setup and Sensors

**Files/tables:**
- `equipment_separation`
- `mesh`
- `sensor_parquet`

**Relevant fields:**
- `id_unidade_prod`
- `id_equipamento_sep`
- `id_malha_ref`
- `espec_abertura_mal`
- `dt_manut_componente`
- `data`
- `tamanho_calibre_malha_3mm`
- `tamanho_calibre_malha_5mm`
- `freq_calibre_malha_3mm`
- `freq_calibre_malha_5mm`
- `chefia`
- `Tag`
- `Value`
- `Date time`

### Product Quality Control

**Files/tables:**
- `Produto_01_2`
- `Product_01`
- `Product_02`

**Shared fields/patterns shown in source instructions:**
- `data_teste`
- `id_ensaio`
- `Detail`
- `data_prod`
- `densidade`
- `humidade`
- `origem`
- `8_236, 25_071, 18_100, 14_140, 10_200, Under_00`
- `8_236, 25_071, 18_100, 14_140, 10_200, Under_000`

**Important note from instructions:**  
`Produto_01_2` and `Product_01` share the same structure and refer to the same product, but they contain records from different clients. `Product_02` also has the same structure, but the measurements correspond to Product 2.

---

## Use Case Guidelines

Feel free to add other analysis if you see fit.

### A. Data Preparation

Examine the dataset's structure for completeness and consistency and ensure it is ready to analyze, by either cleaning or engineering data.

### B. Exploratory Data Analysis

Use descriptive statistics to summarize key variables and visualize distributions for better understanding.

### C. Machine Learning Algorithms

Use a variety of machine learning algorithms to tackle a series of defined challenges, and provide solutions/recommendations.

### D. Open Reflection

Use relevant skills and research for this topic, applying creativity and logic in your analysis. Develop a clear thesis and support it with data.

---

## Deep Dive: Data Cleaning

### Objective

To begin your assessment you should process the data, ensuring it is ready for any further analyses. Hence, your first challenge is to apply the data cleansing techniques you learnt during the Data Mining course.

### Proposed Methodology

Hint: Pay attention to timestamps and physical laws.

Analyze the dataset to identify and address key data quality issues:

- **Missing Data:** Assess the extent and patterns of incomplete records.
- **Illogical/Impossible Data:** Detect and resolve values that violate physical reality or domain constraints.
- **Potential Anomalies:** Flag and investigate extreme outliers that may distort analysis.

Clearly articulate the rationale and logic for every data cleaning method applied to ensure transparency and justify the cleansed dataset's integrity.

---

## Deep Dive: Exploratory Data Analysis

### Proposed Requirement

Example of questions to be answered (not extensive):

- **Insights:** What takeaways can be drawn from descriptive analytics?
- **Visual Analysis:** Are there any underlying concerns or physical bottlenecks presented by the data?
- **Potential Concerns:** Provide visual evidence (e.g., correlation matrices, distributions) to support your claims.

---

## Deep Dive: Machine Learning Algorithms

### A. Quality Score Prediction

Develop a classification or regression model to predict the Quality Score of "Product 1" based on sample analysis.

### B. Throughput Estimation

Build regression models to estimate the production volume of the 5 main products and sub-products.

### C. Feature Importance & Drivers

Identify the most critical variables driving each of the outcomes.

For all requests explain your model outputs in the most logical way possible.

---

## Goals of This Challenge

- **Ability to form and defend a thesis**  
  Create a clear, evidence-based strategy supported by both factory data and external market factors.
- **Critical thinking and data interpretation**  
  Analyze and interpret industrial patterns to draw relevant conclusions about what truly drives production efficiency.
- **Challenge intuition vs. data**  
  Question the existing data collection methods (like sensor glitches) if they don't align with physical reality, actively seeking to prove or disprove operational assumptions.
- **Communication and justification**  
  Assess the ability to express complex ideas in simple, understandable arguments that are easy to follow and debate.

---

## Open Reflection

Based on your modeling, what actions will Volis recommend AgroMill Corp take to extract more financial value from their operations?

Furthermore, what should be the next steps for your data science team, and what is the long-term vision for advanced analytics in this factory?

All these characteristics will be challenged daily in your future work.

---

## Overview and Key Details

- Your groups should have the size of 5 elements.
- The results and proposed solutions should be presented in 7 weeks.
- The presentation should take around 25 minutes and you should expect a 10-minute period for follow-up questions.
- Midway through your journey you will have a session with the Volis team who can guide you and help solve some blockers you may find.
- If any questions arise during the process that require immediate attention, you may contact via email: `pedro@volis.ai` or `andre@volis.ai`.

Thank you. Any questions?

---

## One Last Note

- Although it may not always be the main focus of the evaluation, remember to take pride in your presentation materials.
- Whether consciously or subconsciously, they impact how your work is perceived.
- Invest time to ensure your presentations are distinct, clear, direct, and organized.
- This skill translates to the job market.
