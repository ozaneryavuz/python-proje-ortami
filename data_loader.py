import logging
import pandas as pd


def load_and_validate_data(input_file: str):
    """Load an Excel file and verify required columns are present.

    Parameters
    ----------
    input_file : str
        Path to the Excel file to be read.

    Returns
    -------
    pandas.DataFrame or None
        The loaded data if successful, otherwise ``None``.
    """
    try:
        # Read the data from the specified sheet
        data = pd.read_excel(input_file, sheet_name='Sayfa1')

        # Normalize column names to avoid Turkish characters
        data.columns = (
            data.columns
            .str.replace('ı', 'i')
            .str.replace('Ç', 'C')
            .str.replace('ç', 'c')
            .str.replace('ğ', 'g')
        )

        # Ensure all expected columns exist
        expected_columns = ['Girdi 1', 'Girdi 2', 'Girdi 3', 'Cikti 2']
        for col in expected_columns:
            if col not in data.columns:
                raise ValueError(f"Beklenen sütun eksik: {col}")

        return data
    except Exception as e:
        logging.error(f"Error loading or validating data: {e}")
        return None

