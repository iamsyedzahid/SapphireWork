# Data Cleaning Report – Cafe Sales
**Dataset (Task 1.3)**
**Name:** Syed Zahid Hussain

### 1. Dataset Overview
The Cafe Sales dataset was cleaned and transformed from a raw dataset containing completely string-based columns with dirty text and symbols into a structured, analysis-ready dataset with appropriate data types (numeric, categorical, and datetime) and standardized formatting.

### 2. Before Cleaning (Raw Dataset)
* **Shape:** (10000, 8)
* **Data types:** All columns were stored as `str` (object) type.
* **Missing values present in:** `Item`, `Quantity`, `Price Per Unit`, `Total Spent`, `Payment Method`, `Location`, and `Transaction Date`.

**Sample Columns (Before):**
* `Transaction ID` (string)
* `Item` (string with "Unknown" values)
* `Quantity` (string)
* `Price Per Unit` (string with "ERROR" and "$")
* `Total Spent` (string with "$", ",", spaces)
* `Payment Method` (string)
* `Location` (string)
* `Transaction Date` (string)

**Issues in Raw Data:**
* Numeric features (`Quantity`, `Price Per Unit`, `Total Spent`) contained invalid symbols (e.g., `$`, `,`) and text (e.g., "ERROR"), making them unusable for mathematical operations.
* Categorical features were stored as generic strings with inconsistent whitespaces.
* Dates were stored as raw text rather than standard datetime objects.
* Widespread missing (`NaN`) and unparseable values.

### 3. After Cleaning (Processed Dataset)
* **Shape:** (10000, 8)
* **Data types:** `string`, `Int64`, `float64`, `category`, and `datetime64`
* **Missing Values:** Handled and standardized (invalid text such as "ERROR" was cleanly coerced into `NaN` for numeric columns).

**Final Columns:**
* `Transaction ID` (string)
* `Item` (string)
* `Quantity` (Int64)
* `Price Per Unit` (float64)
* `Total Spent` (float64)
* `Payment Method` (category)
* `Location` (string)
* `Transaction Date` (datetime64)

### 4. Key Transformations Performed
**4.1 String Formatting & Standardization**
* Stripped leading and trailing whitespaces from textual columns: `Transaction ID`, `Item`, `Payment Method`, and `Location`.

**4.2 Type Casting & Feature Coercion**
* Converted `Payment Method` into a `category` data type to save memory and optimize grouping.
* Cleaned numeric columns (`Quantity`, `Price Per Unit`, `Total Spent`) by using Regex to remove unwanted characters (`$`, `,`, `\s`).
* Converted the cleaned numeric columns to `Int64` and `float64`, coercing any remaining unparseable text (like "ERROR") into `NaN`.
* Parsed `Transaction Date` into a proper `datetime64` object.

### 5. Comparison Summary

| Feature | Before | After |
| :--- | :--- | :--- |
| **Rows** | 10000 | 10000 |
| **Columns** | 8 | 8 |
| **Data Types** | Mixed text strings | `string`, `Int64`, `float64`, `category`, `datetime` |
| **Numeric Columns** | Contained `$`, `,`, text | Fully Numeric |
| **Dates** | Raw String | Datetime Object |
| **Analysis Ready** | No | Yes |

### 6. Final Outcome
The dataset is now fully cleaned, structured, and optimized for exploratory data analysis (EDA) tasks such as:
* Time-series analysis (e.g., monthly sales aggregation).
* Most sold items and revenue generation per item.
* Payment method distributions and location-based trends.

### 7. Conclusion
The raw dirty cafe sales dataset was successfully transformed into a clean, structured dataset by handling improper data types, stripping invalid characters via Regex, converting features into their optimal formats, and standardizing text. This ensures accurate aggregations, seamless grouping, and easier data analysis workflows.
