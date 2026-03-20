# Individual Household Electric Power Consumption Dataset

## Source
This dataset is obtained from the UCI Machine Learning Repository:

[Individual Household Electric Power Consumption Dataset](https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip)

## How to Download (Recommended)

Run the following script from the project root:

```bash
python data/get_data.py
```

This script will:  
- Download the dataset  
- Extract it into `data/raw/`

### Alternative (Manual Download)

If the download script is slow or fails:

1. Manually download the dataset from the [link above](https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip).  
2. Extract the `.zip` file.  
3. Place the extracted `.txt` file inside `data/raw/`.

## Notes
- The dataset contains no personally identifiable information (PII).  
- Only anonymized electricity consumption data is included.  
- The repository does **not** store raw data files to comply with licensing.

## File Format
- **File type:** `.txt`  
- **Delimiter:** `;`  
- **Time resolution:** 1-minute intervals

## Citation
Hebrail, G. & Berard, A. (2006). *Individual Household Electric Power Consumption [Dataset]*. UCI Machine Learning Repository. [https://doi.org/10.24432/C58K54](https://doi.org/10.24432/C58K54)